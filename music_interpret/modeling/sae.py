import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for token-level activation vectors.

    This module implements a simple linear encoder–decoder architecture
    with ReLU activations and optional decoder weight normalization.
    It is designed for learning sparse latent representations of
    transformer activations or similar high-dimensional features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input activation vectors.
    latent_dim : int
        Dimensionality of the latent (sparse) representation.
    normalize_decoder : bool, optional
        Whether to enforce unit-norm columns on the decoder weights
        after each optimization step, by default True.

    Notes
    -----
    - The encoder uses a bias term and is initialized with small weights
      and a negative bias to encourage sparsity.
    - The decoder has no bias and can optionally be constrained to have
      unit-norm columns, which stabilizes training and improves
      interpretability of latent features.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        normalize_decoder: bool = True,
    ):
        """
        Initialize the sparse autoencoder.

        Parameters
        ----------
        input_dim : int
            Size of the input feature dimension.
        latent_dim : int
            Size of the latent feature dimension.
        normalize_decoder : bool, optional
            If True, decoder columns are normalized to unit norm after
            initialization and after every optimization step, by default True.

        Notes
        -----
        Weight initialization strategy:

        - Encoder weights: normal distribution with small variance.
        - Encoder bias: initialized to a negative value to promote sparsity.
        - Decoder weights: normal distribution, followed by column-wise
          normalization.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.normalize_decoder = bool(normalize_decoder)

        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=False)

        self.activation = nn.ReLU()

        with torch.no_grad():
            nn.init.normal_(self.encoder.weight, mean=0.0, std=0.01)
            self.encoder.bias.fill_(-1.0)

            nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
            self._normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations into a sparse latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Latent representation of shape ``(N, latent_dim)`` after
            linear projection and ReLU activation.
        """
        return self.activation(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back into the input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed activations of shape ``(N, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a full forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, input_dim)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - ``x_hat`` : torch.Tensor
              Reconstructed input of shape ``(N, input_dim)``.
            - ``z`` : torch.Tensor
              Latent representation of shape ``(N, latent_dim)``.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def _normalize_decoder_columns(self, eps: float = 1e-8) -> None:
        """
        Normalize decoder weight columns to unit norm. Each column corresponds to one latent feature.
        Normalizing columns enforces a consistent scale across features and improves training stability.

        Parameters
        ----------
        eps : float, optional
            Small constant used to avoid division by zero when normalizing,
            by default 1e-8.
        """
        w = self.decoder.weight
        col_norms = torch.linalg.norm(w, dim=0, keepdim=True).clamp_min(eps)
        w.div_(col_norms)

    @torch.no_grad()
    def post_step(self) -> None:
        """
        Apply post-optimization constraints.

        Notes
        -----
        This method should be called after ``optimizer.step()`` during
        training. If decoder normalization is enabled, it re-normalizes
        decoder weight columns to enforce the unit-norm constraint.
        """
        if self.normalize_decoder:
            self._normalize_decoder_columns()
