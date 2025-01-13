# eq_detector

This library is an implementation of the method and baseline presented in "Spatio-Temporal Graph Structure Learning for Earthquake Detection," submitted to IJCNN 2025.
## installation:
```
# parent directory
cd eq_detector
pip install -e .
```
## Usage

`src/eq_detector/model_gslc.py` contains the implementation of our GCN model.\
`src/eq_detector/baseline.py` contains the implementation of the Baseline GCN model.

### Our GCN

### Example Code
```python
import torch
from eq_detector.model import EarthquakeDetector

# Model Parameters
num_stations = 13  # Number of seismic stations
hidden_dim = 32    # Hidden dimension of intermediate layers
kth_chebyshev = 3  # Maximum order of Chebyshev polynomial
num_gslcnn_layers = 3  # Number of spectral SLCNN layers

# Initialize the model
model = EarthquakeDetector(num_stations, hidden_dim, kth_chebyshev, num_gslcnn_layers)

# Create sample input
batch_size = 8
t_timesteps = 500
c_in = 3  # Features per station (N, E, Z waveform components)
x = torch.rand(batch_size, num_stations, t_timesteps, c_in)

# Forward pass
output = model(x)
print("Output shape:", output.shape)
```

### Baseline:

### Example Code

```python
import torch
from eq_detector.baseline import Baseline

# Define the parameters
num_stations = 13  # Number of seismic stations (nodes)
c_in = 3  # Number of input features per station (e.g., N, E, Z components)
hidden_dim = 32  # Hidden dimension for the GCN layers
c_out = 1  # Output feature dimension (e.g., probability of an event)
num_gnn_layers = 2  # Number of GCN layers
t_timesteps = 500  # Number of time steps in the input sequence

# Initialize the model
model = BaselineGNN(num_stations, c_in, hidden_dim, c_out, num_gnn_layers)

# Example input data: Random tensor
batch_size = 8  # Number of samples
x = torch.rand(batch_size, num_stations, t_timesteps, c_in)  # Shape: (8, 13, 500, 3)

# Forward pass
output = model(x)  # Output shape: (8, 13, 500)
print("Output shape:", output.shape)

# Access the output probabilities for each station and timestep
print("Output probabilities:", output)
