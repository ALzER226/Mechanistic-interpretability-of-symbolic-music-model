from pathlib import Path
from typing import List

import numpy as np
import typer
from tqdm import tqdm

from music_interpret.config import ACTIVATION_DIR

app = typer.Typer(help="Preprocess activation shards for downstream analysis.")

EPS = 1e-8

def _load_activ_shards(in_dir: Path) -> List[Path]:
    """
    Load activation shard file paths from a directory.

    Parameters
    ----------
    in_dir : pathlib.Path
        Directory containing activation shard files named ``shard_*.npy``.

    Returns
    -------
    List[pathlib.Path]
        Sorted list of shard file paths.

    Raises
    ------
    RuntimeError
        If no activation shards are found in the directory.
    """
    shards = sorted(in_dir.glob("shard_*.npy"))
    if not shards:
        raise RuntimeError(f"No activation shards found in {in_dir}")
    return shards


def _compute_mean_std(shard_paths: List[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray] :
    """
    Compute per-feature mean and standard deviation over activation shards.

    Parameters
    ----------
    shard_paths : List[pathlib.Path]
        List of paths to activation shard files.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, int]
        - ``mean`` : Per-feature mean vector.
        - ``std`` : Per-feature standard deviation vector.
        - ``total`` : Total number of samples across all shards.
    """
    total = 0
    sum_ = None
    sum_sq = None

    for path in tqdm(shard_paths, desc="Computing per-feature mean/std"):
        x = np.load(path, mmap_mode="r")

        if sum_ is None:
            d = x.shape[1]
            sum_ = np.zeros(d, dtype=np.float64)
            sum_sq = np.zeros(d, dtype=np.float64)

        sum_ += x.sum(axis=0)
        sum_sq += (x ** 2).sum(axis=0)
        total += x.shape[0]

    mean = sum_ / total
    var = sum_sq / total - mean**2
    std = np.sqrt(np.maximum(var, 0.0)) + EPS

    return mean.astype(np.float32), std.astype(np.float32), total


def _normalize_activations(
    shard_paths: List[Path],
    out_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
):
    """
    Normalize activation shards and write the results to disk, using global stats per feature.
    Apply transformation::

        x_norm = (x - mean) / std

    Parameters
    ----------
    shard_paths : List[pathlib.Path]
        Paths to raw activation shard files.
    out_dir : pathlib.Path
        Output directory for normalized activation shards.
    mean : numpy.ndarray
        Per-feature mean vector.
    std : numpy.ndarray
        Per-feature standard deviation vector.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(shard_paths, desc=f"Normalizing → {out_dir}"):
        x = np.load(path)
        x_norm = (x - mean) / std
        np.save(out_dir / path.name, x_norm.astype(np.float32))


@app.command()
def normalize(
    dataset_name: str= typer.Argument(..., help="Name of the dataset associated with the activation files."),
    data_repr: str = typer.Argument(..., help="Data representation identifier (e.g. 'event', 'pitch')."),
    layers: list[int] = typer.Option(..., help="Layer indices"),
    overwrite: bool = typer.Option(False, help="Overwrite existing processed data"),
):
    """
    Normalize activation shards using per-feature standardization.

    This command computes the global mean and standard deviation of
    activation vectors for each selected layer and applies the
    transformation::

        x_norm = (x - mean) / std

    Parameters
    ----------
    dataset_name : str
        Name of the dataset associated with the activation files.
    data_repr : str
        Data representation identifier used to locate activation data.
    layers : List[int]
        Indices of layers whose activations should be normalized.
    overwrite : bool, optional
        Whether to overwrite existing normalized data, by default False.

    Raises
    ------
    RuntimeError
        If normalized data already exists and ``overwrite`` is False.

    Notes
    -----
    Output structure for each layer::

        ACTIVATION_DIR/
            <dataset>_<repr>_residual/
                layer_XXX/
                    raw/        # original activations
                    processed/  # normalized activations
                    stats.npz   # mean, std, and sample count
    """
    base_path = ACTIVATION_DIR / f"{dataset_name}_{data_repr}_residual"

    for layer in layers:
        layer_name = f"layer_{layer:03d}"

        activ_dir = base_path / layer_name / "raw"
        activ_norm_dir = base_path / layer_name / "processed"
        stats_path = base_path / layer_name / "stats.npz"

        if activ_norm_dir.exists() and not overwrite:
            raise RuntimeError(f"Data already normalized for {layer_name} (use --overwrite)")

        shards = _load_activ_shards(activ_dir)

        mean, std, count = _compute_mean_std(shards)

        np.savez(
            stats_path,
            mean=mean,
            std=std,
            count=count,
        )

        _normalize_activations(
            shard_paths=shards,
            out_dir=activ_norm_dir,
            mean=mean,
            std=std,
        )


if __name__ == "__main__":
    app()
