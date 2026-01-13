from pathlib import Path
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from music_interpret.modeling.sae import SparseAutoencoder
from music_interpret.activations.dataset import ActivationDataset
from music_interpret.modeling.plotting import SAEPlotter
from music_interpret.config import ACTIVATION_DIR, REPORTS_DIR

app = typer.Typer(help="Train sparse autoencoders on extracted activation datasets.")


@app.command()
def train(
    series_name: str = typer.Argument(..., help="Name of the experiment series used for grouping runs."),
    experiment_name: str = typer.Argument(..., help="Name of this specific experiment."),
    layer_dir: Path = typer.Argument(..., help="Relative path to the activation directory."),

    input_dim: int = typer.Option(..., help="Size of activation vectors."),
    latent_dim: int = typer.Option(..., help="Target size of the latent representation."),
    batch_size: int = typer.Option(2048, help="Batch size used during training."),
    lr: float = typer.Option(1e-3, help="Learning rate for the Adam optimizer."),
    sparsity_alpha: float = typer.Option(1e-3, help="Weight of the sparsity regularization term."),
    num_epochs: int = typer.Option(2, help="Number of training epochs."),
    num_workers: int = typer.Option(4, help="Number of worker processes for data loading."),
    normalize_decoder: bool = typer.Option(True, help="Normalize decoder weights to unit norm after each update."),

    save_every: int = typer.Option(1, help="Save a checkpoint every N epochs."),
    dead_window_tokens: int = typer.Option(100_000, help="Token window size used to detect dead latent units."),
    overactive_threshold: float = typer.Option(0.1, help="Firing-rate threshold for detecting overactive latent units."),
):
    """
        Train a sparse autoencoder on activation data.

        This command trains a sparse autoencoder (SAE) on token-level
        activations extracted from a transformer model. The training
        objective combines reconstruction loss with an L1 sparsity
        penalty on the latent activations.

        For each run, the script:

        - Creates an experiment directory under ``REPORTS_DIR``.
        - Saves all training arguments to ``training_args.json``.
        - Logs training metrics to TensorBoard.
        - Periodically saves model checkpoints.

        Parameters
        ----------
        series_name : str
            Name of the experiment series used for grouping runs.
        experiment_name : str
            Name of the specific experiment.
        layer_dir : pathlib.Path
            Relative path to the activation directory to train on.
        input_dim : int
            Dimensionality of the input activation vectors.
        latent_dim : int
            Dimensionality of the latent representation.
        batch_size : int, optional
            Batch size used during training, by default 2048.
        lr : float, optional
            Learning rate for the optimizer, by default 1e-3.
        sparsity_alpha : float, optional
            Weight applied to the sparsity regularization term,
            by default 1e-3.
        num_epochs : int, optional
            Number of training epochs, by default 2.
        num_workers : int, optional
            Number of DataLoader worker processes, by default 4.
        normalize_decoder : bool, optional
            Whether to normalize decoder weights after each update,
            by default True.
        save_every : int, optional
            Save a model checkpoint every ``save_every`` epochs,
            by default 1.
        dead_window_tokens : int, optional
            Number of tokens used to compute rolling dead-unit statistics,
            by default 100_000.
        overactive_threshold : float, optional
            Threshold for classifying a latent unit as overactive,
            by default 0.1.

        Raises
        ------
        RuntimeError
            If the output experiment directory already exists to avoid overwriting.

        Notes
        -----
        Output structure for a training run::

            REPORTS_DIR/
                <series_name>/
                    <experiment_name>/
                        training_args.json
                        tensorboard/
                        sae_ckpt_XXX.pt
        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = REPORTS_DIR / series_name / experiment_name
    if out_dir.exists():
        raise RuntimeError(f"Experiment already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    args = dict(
        series_name=series_name,
        experiment_name=experiment_name,
        input_dim=input_dim,
        latent_dim=latent_dim,
        sparsity_alpha=sparsity_alpha,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_workers=num_workers,
        layer_dir=str(layer_dir),
        dead_window_tokens=dead_window_tokens,
        overactive_threshold=overactive_threshold,
        normalize_decoder=normalize_decoder,
    )

    with open(out_dir / "training_args.json", "w") as f:
        json.dump(args, f, indent=2)

    dataset = ActivationDataset(ACTIVATION_DIR / layer_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    sae = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        normalize_decoder=normalize_decoder,
    ).to(device)

    optimizer = optim.Adam(sae.parameters(), lr=lr)
    recon_loss_fn = nn.MSELoss()

    tb_dir = out_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    tb_logger = SAEPlotter(
        tb_dir,
        latent_dim=latent_dim,
        dead_window_tokens=dead_window_tokens,
        overactive_threshold=overactive_threshold,
    )

    sae.train()

    for epoch in range(num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs - 1}")

        for batch in pbar:
            act = batch["activations"].to(device, non_blocking=True)

            x_hat, z = sae(act)

            recon_loss = recon_loss_fn(x_hat, act)
            sparsity_loss = sparsity_alpha * z.abs().sum(dim=-1).mean()
            total_loss = recon_loss + sparsity_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            sae.post_step()

            # logging progress and stats
            fire_mask = z > 0
            avg_active_frac = fire_mask.float().mean().item()
            avg_active_count = fire_mask.float().sum(dim=1).mean().item()

            if fire_mask.any():
                mean_active_activation = z[fire_mask].mean().item()
            else:
                mean_active_activation = 0.0

            mean_decoder_cos_sim = tb_logger.compute_mean_decoder_cos_sim(
                sae.decoder.weight
            )

            tb_logger.log_scalars(
                recon_loss=recon_loss.item(),
                sparsity_loss=sparsity_loss.item(),
                total_loss=total_loss.item(),
                avg_active_frac=avg_active_frac,
                avg_active_count=avg_active_count,
                mean_active_activation=mean_active_activation,
                mean_decoder_cos_sim=mean_decoder_cos_sim,
                batch_fire_mask=fire_mask,
                batch_size=act.size(0),
            )

            pbar.set_postfix(
                recon=recon_loss.item(),
                sparse=sparsity_loss.item(),
                active=f"{avg_active_count:.1f}/{latent_dim}",
            )

        if epoch % save_every == 0 or epoch == num_epochs - 1:
            ckpt_path = out_dir / f"sae_ckpt_{epoch:03d}.pt"
            torch.save(
                {
                    "model_state": sae.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "sparsity_alpha": sparsity_alpha,
                    "lr": lr,
                    "layer_dir": str(layer_dir),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to: {ckpt_path}")

    tb_logger.close()


if __name__ == "__main__":
    app()
