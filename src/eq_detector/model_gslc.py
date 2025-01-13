import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Callable

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SEED=0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Helper function to calculate the Chebyshev polynomial T_k(X)
def chebyshev_polynomial(X, k: int, batch=False):
    """
    Compute Chebyshev polynomial of order k.

    Args:
        X: Adjacency matrix or dynamic weight matrix.
        k: Order of the polynomial.
        batch: Boolean indicating if X includes batch dimension.

    Returns:
        Chebyshev polynomial T_k(X).
    """
    if not batch:
        n, _ = X.size()
        if k == 0:
            return torch.eye(n, device=X.device)
        elif k == 1:
            return X

        T_k_2 = torch.eye(n, device=X.device)
        T_k_1 = X
        for _ in range(2, k + 1):
            T_k = 2 * torch.matmul(X, T_k_1) - T_k_2
            T_k_2, T_k_1 = T_k_1, T_k
        return T_k
    else:
        batch_size, n, _ = X.size()
        if k == 0:
            return torch.eye(n, device=X.device).unsqueeze(0).repeat(batch_size, 1, 1)
        elif k == 1:
            return X

        T_k_2 = torch.eye(n, device=X.device).unsqueeze(0).repeat(batch_size, 1, 1)
        T_k_1 = X
        for _ in range(2, k + 1):
            T_k = 2 * torch.bmm(X, T_k_1) - T_k_2
            T_k_2, T_k_1 = T_k_1, T_k
        return T_k


class GlobalSLCLayer(nn.Module):
    def __init__(self, n: int, c_in: int, c_out: int, cheb_max_order: int):
        """
        Global Structure Learning Convolution (Global SLC) Layer.

        Args:
            n: Number of nodes.
            c_in: Input feature dimension.
            c_out: Output feature dimension.
            cheb_max_order: Maximum order of Chebyshev polynomials.
        """
        super(GlobalSLCLayer, self).__init__()
        self.n = n
        self.c_in = c_in
        self.c_out = c_out
        self.k = cheb_max_order

        # Learnable parameters for static and dynamic graph structure
        self.Ws = Parameter(torch.zeros(n, n))
        self.W_alpha = Parameter(torch.zeros(n, n))
        self.Theta_s = Parameter(torch.zeros(cheb_max_order, c_in, c_out))
        self.Theta_d = Parameter(torch.zeros(cheb_max_order, c_in, c_out))

        torch.nn.init.xavier_uniform_(self.Ws)
        torch.nn.init.xavier_uniform_(self.W_alpha)
        torch.nn.init.xavier_uniform_(self.Theta_s)
        torch.nn.init.xavier_uniform_(self.Theta_d)

    def forward(self, X):
        """
        Forward pass for Global SLC Layer.

        Args:
            X: Input features of shape (batch_size, n, c_in).

        Returns:
            Output features of shape (batch_size, n, c_out).
        """
        batch_size = X.shape[0]
        assert X.shape == (batch_size, self.n, self.c_in)

        static_filter = torch.zeros(batch_size, self.n, self.c_out, device=X.device)
        for k in range(self.k):
            T_k_Ws = chebyshev_polynomial(self.Ws, k)
            static_filter += torch.matmul(T_k_Ws @ X, self.Theta_s[k])

        Wd = torch.matmul(X, (self.W_alpha @ X).permute(0, 2, 1))

        dynamic_filter = torch.zeros(batch_size, self.n, self.c_out, device=X.device)
        for k in range(self.k):
            T_k_Wd = chebyshev_polynomial(Wd, k, batch=True)
            dynamic_filter += torch.matmul((T_k_Wd @ X), self.Theta_d[k])

        output = F.relu(static_filter) + F.relu(dynamic_filter)
        assert output.shape == (batch_size, self.n, self.c_out)
        return output


class GlobalSLCNNLayer(nn.Module):
    def __init__(self, n: int, c_in: int, c_out: int,
                 gl_out: int, cheb_max_order: int):
        """
        SLCNN Layer combining Global SLC and temporal modeling.

        Args:
            n: Number of nodes.
            c_in: Input feature dimension.
            c_out: Output feature dimension.
            gl_out: Output dimension of Global SLC.
            cheb_max_order: Maximum order of Chebyshev polynomials.
        """
        super(GlobalSLCNNLayer, self).__init__()
        self.n = n
        self.c_in = c_in
        self.c_out = c_out
        self.gl_out = gl_out

        self.global_slc = GlobalSLCLayer(n=n, cheb_max_order=cheb_max_order, c_in=c_in, c_out=gl_out)

        self.gru_gslc = nn.GRU(input_size=n * gl_out, hidden_size=n * c_out, num_layers=1, batch_first=True)

    def forward(self, X):
        """
        Forward pass for SLCNN Layer.

        Args:
            X: Input tensor of shape (batch_size, n, t, c_in).

        Returns:
            Tensor of shape (batch_size, n, t, c_out).
        """
        batch_size, n, t_steps, c_in = X.shape
        assert n == self.n and c_in == self.c_in

        global_outputs = torch.empty((batch_size, t_steps, self.n, self.gl_out), device=X.device)
        for t in range(t_steps):
            global_features = self.global_slc(X[:, :, t, :])
            global_outputs[:, t, :, :] = global_features

        global_features, _ = self.gru_gslc(global_outputs.reshape(batch_size, t_steps, -1))
        global_features = global_features.view(batch_size, t_steps, n, self.c_out).permute(0, 2, 1, 3)

        return global_features


class EarthquakeDetector(nn.Module):
    def __init__(self, num_stations: int, hidden_dim: int,
                 kth_chebeshev: int, num_gslcnn_layers=3):
        """
        Earthquake Detector Model using SLCNN Layers.

        Args:
            num_stations: Number of seismic stations (nodes).
            hidden_dim: Hidden dimension for intermediate layers.
            kth_chebeshev: Maximum order of Chebyshev polynomials.
            num_gslcnn_layers: Number of SLCNN layers.
        """
        super(EarthquakeDetector, self).__init__()

        self.layers = nn.ModuleList([
            GlobalSLCNNLayer(
                n=num_stations,
                c_in=3 if i == 0 else 1,
                c_out=1,
                gl_out=hidden_dim,
                cheb_max_order=kth_chebeshev
            ) for i in range(num_gslcnn_layers)
        ])

        self.dropout = nn.Dropout(0.2)
        self.final = nn.Linear(in_features=num_stations, out_features=num_stations)
        self.final_prob = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for Earthquake Detector.

        Args:
            x: Input tensor of shape (batch_size, n, t, c_in).

        Returns:
            Output tensor of shape (batch_size, n, t, 1).
        """
        batch_size, n, t, c_in = x.shape

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)

        x = x.permute(0, 2, 1, 3).reshape(-1, n)
        x = self.final(x)
        x = self.dropout(x)
        x = self.final_prob(x)
        x = x.view(batch_size, t, n).permute(0, 2, 1)

        return x
