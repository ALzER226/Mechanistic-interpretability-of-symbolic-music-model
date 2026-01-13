import torch
import typer
from tqdm import tqdm

from music_interpret.activations.extractor import ActivationExtractor
from music_interpret.model_adapters.mmt_adapter import MMTAdapter
from music_interpret.config import ACTIVATION_DIR

app = typer.Typer()

@app.command()
def extract(
    dataset_name: str = typer.Argument(..., help="Name of the dataset used by the trained model."),
    data_repr: str = typer.Argument(..., help="Data representation used by the model."),
    layers: list[int] = typer.Option(..., help="Indices of transformer layers to extract activations from."),
    batch_size: int = typer.Option(1, help="Batch size for activation extraction"),
    shuffle: bool = typer.Option(True, help="Shuffle data"),
    max_tokens_per_shard: int = typer.Option(200_000, help="Maximum number of tokens per output shard file."),
    num_workers: int = typer.Option(0, help="Number of worker processes for data loading.")
):
    """
    Extract and store activations from selected model layers.

    This script loads a trained MMT model, registers forward hooks on
    the specified layers, and iterates over the dataset to collect
    activations and metadata.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset associated with the trained model.
    data_repr : str
        Data representation identifier used by the model.
    layers : List[int]
        Indices of transformer layers from which activations should
        be extracted.
    batch_size : int, optional
        Batch size used during extraction.
    shuffle : bool, optional
        Whether to shuffle the dataset before extraction,
        by default True.
    max_tokens_per_shard : int, optional
        Maximum number of tokens to store per shard file before
        writing to disk, by default 200_000.
    num_workers : int, optional
        Number of worker processes used by the DataLoader,
        by default 0.

    Notes
    -----
    - The script automatically selects CUDA if available.
    - The model is run in evaluation mode and under ``torch.no_grad()``.
    - All hooks are safely removed even if an error occurs during
      extraction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = MMTAdapter(dataset_name, data_repr)
    model = adapter.load_model(device)
    model.eval()

    hookable_layers = adapter.get_hookable_layers(layers)

    dataloader = adapter.create_dataloader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    extractor = ActivationExtractor(
        root_dir=ACTIVATION_DIR / adapter.target,
        layer_indices=layers,
        max_tokens_per_shard=max_tokens_per_shard,
    )

    extractor.register(hookable_layers)

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Activation extraction"):
                extractor.set_batch_context(
                    mask=batch["mask"],
                    sample_idx=batch["sample_idx"],
                )
                adapter.forward(model, batch)
    finally:
        extractor.close()


if __name__ == "__main__":
    app()
