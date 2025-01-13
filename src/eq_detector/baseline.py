import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        """
        Args:
            X: Node features, shape (batch_size, n_stations, in_features)
            A: Adjacency matrix, shape (n_stations, n_stations)

        Returns:
            Updated node features, shape (batch_size, n_stations, out_features)
        """
        AX = torch.matmul(A, X)  # Graph convolution: A * X
        return F.relu(self.fc(AX))  # Apply linear transformation and activation


class BaselineGNN(nn.Module):
    def __init__(self, num_stations, c_in, hidden_dim, c_out, num_gnn_layers):
        """
        Args:
            num_stations: Number of seismic stations (nodes).
            c_in: Input feature dimension (number of waveform components, e.g., 3 for N, E, Z).
            hidden_dim: Hidden dimension for GNN layers.
            c_out: Output feature dimension (1 for probability).
            num_gnn_layers: Number of GNN layers.
        """
        super(BaselineGNN, self).__init__()
        self.num_stations = num_stations
        self.num_gnn_layers = num_gnn_layers

        # Define GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnn_layers.append(SimpleGCNLayer(c_in, hidden_dim))
            else:
                self.gnn_layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))

        # GRU for temporal modeling
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Final fully connected layer
        self.fc = nn.Linear(hidden_dim, c_out)
        self.sigmoid = nn.Sigmoid()  # Apply Sigmoid activation for probabilities

        # Adjacency matrix placeholder
        self.A = None  # Initialize as None, will set in forward based on input device

    def _initialize_static_adjacency_matrix(self, n, device):
        """Create a static adjacency matrix."""
        A = torch.ones(n, n, device=device)  # Fully connected graph
        A = A / (n - 1)  # Normalize
        return A

    def forward(self, X):
        """
        Args:
            X: Input tensor of shape (batch_size, n_stations, t_timesteps, c_in).

        Returns:
            Output tensor of shape (batch_size, n_stations, t_timesteps, 1).
        """
        batch_size, n_stations, t_timesteps, c_in = X.shape
        assert n_stations == self.num_stations

        # Ensure adjacency matrix is on the same device as X
        if self.A is None or self.A.device != X.device:
            self.A = self._initialize_static_adjacency_matrix(n_stations, X.device)

        # Reshape for GNN layers
        X = X.permute(0, 2, 1, 3)  # Shape: (batch_size, t_timesteps, n_stations, c_in)
        X = X.reshape(batch_size * t_timesteps, n_stations, c_in)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            X = gnn_layer(X, self.A)

        # Reshape back for GRU
        X = X.reshape(batch_size, t_timesteps, n_stations, -1)  # Shape: (batch_size, t_timesteps, n_stations, hidden_dim)
        X = X.permute(0, 2, 1, 3)  # Shape: (batch_size, n_stations, t_timesteps, hidden_dim)
        X = X.reshape(batch_size * n_stations, t_timesteps, -1)  # Merge batch and nodes

        # GRU for temporal modeling
        H, _ = self.gru(X)  # Shape: (batch_size * n_stations, t_timesteps, hidden_dim)

        # Final linear layer
        H = self.fc(H)  # Shape: (batch_size * n_stations, t_timesteps, c_out)

        # Apply sigmoid activation for probabilities
        H = self.sigmoid(H)  # Probabilities: values between 0 and 1
        H = H.reshape(batch_size, n_stations, t_timesteps)  # Shape: (batch_size, n_stations, t_timesteps, c_out)
        return H

# Example usage:
# num_stations = 13
# c_in = 3  # 3 components (N, E, Z)
# hidden_dim = 32
# c_out = 1  # Probability of earthquake
# num_gnn_layers = 2  # Number of GNN layers
# t_timesteps = 500
#
# baseline_gnn = BaselineGNN(num_stations, c_in, hidden_dim, c_out, num_gnn_layers)
# x = torch.rand(8, num_stations, t_timesteps, c_in)  # Example input: batch_size=8
# output = baseline_gnn(x)
# print(output.shape)  # Expected: (8, 13, 500, 1)
