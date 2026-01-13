from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class SAEPlotter:
    """
    TensorBoard logger for monitoring Sparse Autoencoder training.

    It tracks reconstruction loss, sparsity metrics, activation statistics,
    and detects dead or overactive latent units over time.

    Parameters
    ----------
    log_dir : pathlib.Path
        Directory where TensorBoard event files will be written.
    latent_dim : int
        Dimensionality of the latent space.
    dead_window_tokens : int, optional
        Number of tokens over which dead/overactive statistics are
        accumulated before logging, by default 100_000.
    overactive_threshold : float, optional
        Threshold on firing rate above which a latent unit is considered
        overactive, by default 0.05.
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        latent_dim: int,
        dead_window_tokens: int = 100_000,
        overactive_threshold: float = 0.05,
    ):
        """
        Initialize the TensorBoard plotter for sparse autoencoder training.

        Parameters
        ----------
        log_dir : pathlib.Path
            Directory for TensorBoard logs.
        latent_dim : int
            Number of latent features in the autoencoder.
        dead_window_tokens : int, optional
            Token window size used to compute dead and overactive unit
            statistics, by default 100_000.
        overactive_threshold : float, optional
            Firing-rate threshold used to flag overactive latent units,
            by default 0.05.

        Notes
        -----
        The plotter maintains rolling statistics across batches to detect:

        - **Dead units**: latent dimensions that never activate.
        - **Overactive units**: latent dimensions that activate too often.
        """
        self.writer = SummaryWriter(log_dir=log_dir)

        self.latent_dim = latent_dim
        self.dead_window_tokens = dead_window_tokens
        self.overactive_threshold = overactive_threshold

        # step counters
        self.global_step = 0

        # rolling window stats
        self.window_tokens = 0
        self.window_fire_counts = torch.zeros(latent_dim)


    def log_scalars(
        self,
        *,
        recon_loss: float,
        sparsity_loss: float,
        total_loss: float,
        avg_active_frac: float,
        avg_active_count: float,
        mean_active_activation: float,
        mean_decoder_cos_sim: float,
        batch_fire_mask: torch.Tensor,
        batch_size: int,
    ) -> None:
        """
            Log scalar training metrics and update rolling sparsity statistics.

            Parameters
            ----------
            recon_loss : float
                Reconstruction loss value for the current batch.
            sparsity_loss : float
                Sparsity regularization loss for the current batch.
            total_loss : float
                Total training loss for the current batch.
            avg_active_frac : float
                Average fraction of latent units that are active per token.
            avg_active_count : float
                Average number of active latent units per token.
            mean_active_activation : float
                Mean activation value over active latent units.
            mean_decoder_cos_sim : float
                Mean cosine similarity between decoder weight vectors.
            batch_fire_mask : torch.Tensor
                Boolean or binary tensor of shape ``(B, latent_dim)`` indicating
                which latent units fired in the current batch.
            batch_size : int
                Number of tokens in the current batch.
            """
        self.writer.add_scalar("loss/recon", recon_loss, self.global_step)
        self.writer.add_scalar("loss/sparsity", sparsity_loss, self.global_step)
        self.writer.add_scalar("loss/total", total_loss, self.global_step)

        self.writer.add_scalar(
            "sparsity/avg_active_frac", avg_active_frac, self.global_step
        )
        self.writer.add_scalar(
            "sparsity/avg_active_count", avg_active_count, self.global_step
        )

        self.writer.add_scalar(
            "activs/mean_active_activation",
            mean_active_activation,
            self.global_step,
        )

        self.writer.add_scalar(
            "activs/mean_decoder_cos_sim",
            mean_decoder_cos_sim,
            self.global_step,
        )

        with torch.no_grad():
            fires = batch_fire_mask.float().sum(dim=0).cpu()
            self.window_fire_counts += fires
            self.window_tokens += batch_size

        if self.window_tokens >= self.dead_window_tokens:
            fire_rates = self.window_fire_counts / float(self.window_tokens)

            dead_ratio = (fire_rates == 0).float().mean().item()
            overactive_ratio = (
                fire_rates > self.overactive_threshold
            ).float().mean().item()

            self.writer.add_scalar(
                "sparsity/dead_ratio_window",
                dead_ratio,
                self.global_step,
            )
            self.writer.add_scalar(
                "sparsity/overactive_ratio_window",
                overactive_ratio,
                self.global_step,
            )

            self.window_fire_counts.zero_()
            self.window_tokens = 0

        self.global_step += 1


    @staticmethod
    def compute_mean_decoder_cos_sim(decoder_weight: torch.Tensor) -> float:
        """
        Compute the mean cosine similarity between decoder weight vectors.

        Parameters
        ----------
        decoder_weight : torch.Tensor
            Decoder weight matrix of shape ``(input_dim, latent_dim)``.

        Returns
        -------
        float
            Mean absolute cosine similarity between all distinct pairs
            of latent feature vectors.
        """
        with torch.no_grad():
            W = F.normalize(decoder_weight, dim=0)
            cos = W.T @ W
            latent_dim = cos.size(0)
            mask = ~torch.eye(latent_dim, dtype=torch.bool, device=cos.device)
            mean_cos = cos[mask].abs().mean().item()
        return mean_cos

    def close(self) -> None:
        """
        Close the underlying TensorBoard SummaryWriter.

        Notes
        -----
        This should be called at the end of training to ensure that all
        pending events are flushed and file handles are released.
        """
        self.writer.close()
