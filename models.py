import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class LipschitzFCNN(nn.Module):
    """Fully Connected Neural Network (FCNN) / MLP with spectral norm constraints."""
    def __init__(
        self,
        input_dim,
        output_dim,
        n_hidden_layers=4,
        hidden_dim=128,
        activation_fn=nn.ReLU(),
    ):
        super(LipschitzFCNN, self).__init__()
        layers = []
        
        # First layer
        layer = spectral_norm(nn.Linear(input_dim, hidden_dim))
        layers.append(layer)
        layers.append(activation_fn)
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layer = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
            layers.append(layer)
            layers.append(activation_fn)

        # Output layer
        layer = spectral_norm(nn.Linear(hidden_dim, output_dim))
        layers.append(layer)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FCNN(nn.Module):
    """Fully Connected Neural Network (FCNN) / MLP, minimal implementation"""

    # fully connected neural network
    def __init__(
        self,
        input_dim,
        output_dim,
        n_hidden_layers=4,
        hidden_dim=128,
        activation_fn=nn.ReLU(),
    ):
        super(FCNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FourierFCNN(nn.Module):
    """
    A network that:
      - Takes real input of shape (batch, n_freqs)
      - Applies rFFT along the last dimension
      - Splits real/imag into a single real vector
      - Passes that through a fully connected network
      - Recombines into a complex tensor
      - Applies an inverse rFFT (irFFT)
      - Returns an output of shape (batch, n_freqs)
    
    This is convenient if you want to learn transformations in the Fourier domain,
    but keep your model interface in the original (real) domain.
    """

    def __init__(
        self,
        n_freqs: int,
        n_hidden_layers: int = 4,
        hidden_dim: int = 128,
        activation_fn=nn.ReLU(),
    ):
        super().__init__()

        # For a real input of length n_freqs, rFFT produces (n_freqs//2 + 1) complex bins
        self.freq_dim = n_freqs

        # We flatten real and imaginary parts: that doubles the dimension
        in_dim = 2 * self.freq_dim
        out_dim = 2 * self.freq_dim

        layers = []
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_fn)
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.fcnn = nn.Sequential(*layers)
        self.n_freqs = n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_freqs) – real-valued input (1D domain).
        Returns:
            y: (batch_size, n_freqs) – real-valued output (1D domain).
        """

        # 1) rFFT along the last dimension -> shape (batch, self.freq_dim), complex-valued
        in_shape = x.shape
        X_complex = torch.fft.rfft(x, dim=-1)
        # print(X_complex.shape)

        # 2) Separate real and imaginary parts
        X_real = X_complex.real[...,:self.freq_dim]  # (batch, freq_dim)
        X_imag = X_complex.imag[...,:self.freq_dim]  # (batch, freq_dim)
        # print(X_real.shape)

        # 3) Concatenate real and imaginary along the last dimension
        #    -> shape (batch, 2*freq_dim)
        X_cat = torch.cat([X_real, X_imag], dim=-1)

        # 4) Forward pass through the FCNN (fully connected network)
        Y_cat = self.fcnn(X_cat)  # shape (batch, 2*freq_dim)

        # 5) Split back into real & imaginary parts
        Y_real = Y_cat[..., : self.freq_dim]
        Y_imag = Y_cat[..., self.freq_dim :]

        # 6) Reconstruct complex tensor
        Y_complex = torch.complex(Y_real, Y_imag)

        # 7) Inverse rFFT to get back a real signal of length n_freqs
        y = torch.fft.irfft(Y_complex, n=in_shape[-1]//2, dim=-1)
        # print(y.shape)
        # print(y[0])

        return y


class FCNNWithSkip(nn.Module):  # consider renaming to resnet
    """Fully Connected Neural Network (FCNN) with Skip Connections"""

    def __init__(
        self,
        input_dim,
        output_dim,
        n_hidden_layers=4,
        hidden_dim=128,
        activation_fn=nn.ReLU(),
    ):
        super(FCNNWithSkip, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        self.activation_fn = activation_fn

        # Create hidden layers
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def apply(self, x):
        # Apply input layer
        x = self.activation_fn(self.input_layer(x))

        # Pass through hidden layers with skip connections
        for layer in self.hidden_layers:
            x = x + self.activation_fn(layer(x))  # Add residual connection

        # Apply output layer
        return self.output_layer(x)

    def forward(self, x):
        return self.apply(x)

class TimeEmbedNN(nn.Module):
    def __init__(self, model):
        super(TimeEmbedNN, self).__init__()
        self.model = model

    def forward(self, x):
        if len(x.shape) > 2:
            x_embed = x.view(x.shape[0], -1)
        else:
            x_embed = x.flatten()
        return self.model(x_embed)


class fd_stencil(nn.Module):
    def __init__(self, stencil_size):
        super(fd_stencil, self).__init__()
        self.stencil_size = stencil_size
        self.stencil = nn.Parameter(torch.randn(stencil_size))
    
    def enforce_zero_sum(self):
        with torch.no_grad():
            self.stencil.data -= torch.mean(self.stencil.data)

    def forward(self, x):
        # self.enforce_zero_sum()
        # need data to be batch_dim, channel_dim, spatial_dim
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        kernel = self.stencil.view(1, 1, -1)
        pad_amt = self.stencil_size // 2
        x_padded = F.pad(x, (pad_amt, pad_amt), mode='circular')
        out = F.conv1d(x_padded, kernel, padding=0)
        return out.squeeze()
