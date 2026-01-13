import sys
import torch
from pathlib import Path
from typing import Dict, Iterable, Any, Optional
from loguru import logger

from .base import ModelAdapter, IndexedDataset
from music_interpret.config import PROJ_ROOT, BASE_DATA_DIR, BASE_MODEL_DIR

mmt_dir = (PROJ_ROOT / "mmt" / "mmt").resolve()
if str(mmt_dir) not in sys.path:
    sys.path.append(str(mmt_dir))
    logger.debug(f"MMT import path added: {mmt_dir}")

import utils
import representation
import dataset as mmt_dataset
import music_x_transformers

class MMTAdapter(ModelAdapter):
    """
    Adapter for loading, configuring, and interacting with the Multitrack Music
    Transformer (MMT) models used in this project.

    This class is responsible for:
    - Resolving project-specific paths for checkpoints, encodings, and metadata.
    - Loading trained MMT models with their correct configuration.
    - Building PyTorch dataloaders compatible with the MMT training format.
    - Exposing internal transformer layers for hooking or inspection.
    - Providing a unified forward interface and decoding utilities.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset used to train the model.
    data_repr : str
        Data representation identifier (e.g. "rpe", "ape").
    """

    def __init__(self, dataset_name: str, data_repr: str):
        self.dataset_name = dataset_name
        self.data_repr = data_repr
        self.target = f"{dataset_name}_{data_repr}_residual"

        self._model: Optional[torch.nn.Module] = None
        self._train_args: Optional[Dict[str, Any]] = None
        self._encoding: Optional[Any] = None
        self._paths: Optional[Dict[str, Path]] = None


    def _resolve_paths(self):
        """
        Resolve and validate all filesystem paths required by the adapter.

        This includes paths to:
        - Training arguments of base model
        - checkpoint path
        - Encoding definition
        - Notes dataset directory
        - Dataset file name list

        Returns
        -------
        Dict[str, pathlib.Path]
            Mapping of resource names to paths.

        Raises
        ------
        FileNotFoundError
            If any required file or directory does not exist.

        Notes
        -----
        Validation is done only during the first call and assumes that paths are not broken after it.
        """
        if not (self._paths is None):
            return self._paths

        base_model_dir = BASE_MODEL_DIR / self.dataset_name / self.data_repr
        notes_dir = BASE_DATA_DIR / self.dataset_name / "notes"

        paths =  {
            "train_args": base_model_dir / "train-args.json",
            "checkpoint": base_model_dir / "checkpoints" / "best_model.pt",
            "encoding": notes_dir / "encoding.json",
            "notes_dir": notes_dir,
            "names_pth": PROJ_ROOT / "mmt" / "data" / self.dataset_name / "processed" / "names.txt",
        }

        for k, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"{k} not found: {p}")

        self._paths = paths
        return paths


    def _load_train_args_and_encoding(self):
        """
        Load training arguments and token encoding metadata from disk.

        Returns
        -------
        Tuple[Dict[str, Any], Any]
            - Training arguments dictionary.
            - Encoding object used by the representation module.

        Notes
        -----
        The loaded values are cached.
        """
        if self._train_args is not None and self._encoding is not None:
            return self._train_args, self._encoding

        paths = self._resolve_paths()
        self._train_args = utils.load_json(paths["train_args"])
        self._encoding = representation.load_encoding(paths["encoding"])
        return self._train_args, self._encoding


    def _collate(self, batch: Iterable[tuple[Any, int]]) -> Dict[str, torch.Tensor]:
        """
        Collate function used by the PyTorch DataLoader.

        Parameters
        ----------
        batch : Iterable[Tuple[Any, int]]
            Iterable of (sample, index) pairs produced by IndexedDataset.

        Returns
        -------
        Dict[str, torch.Tensor]
            Batch dictionary compatible with MusicDataset output,
            with an additional ``sample_idx`` tensor containing the
            original dataset indices.

        Notes
        -----
        This wraps ``MusicDataset.collate`` and augments the result
        with index tracking for later analysis.
        """
        samples, indices = zip(*batch)
        out = mmt_dataset.MusicDataset.collate(list(samples))
        out["sample_idx"] = torch.tensor(indices, dtype=torch.long)
        return out


    def load_model(self, device: torch.device) -> torch.nn.Module:
        """
        Load the trained Music Transformer model and move it to a device.

        Parameters
        ----------
        device : torch.device
            Target device on which the model should be loaded
            (e.g. CPU or CUDA device).

        Returns
        -------
        torch.nn.Module
            The initialized and weight-loaded MusicXTransformer model.

        Raises
        ------
        FileNotFoundError
            If required checkpoint or configuration files are missing.

        Notes
        -----
        This method:
        - Loads training arguments and encoding.
        - Instantiates the transformer with the original hyperparameters.
        - Loads the checkpoint strictly.
        """
        train_args, encoding = self._load_train_args_and_encoding()
        paths = self._resolve_paths()


        model = music_x_transformers.MusicXTransformer(
            dim=train_args["dim"],
            encoding=encoding,
            depth=train_args["layers"],
            heads=train_args["heads"],
            max_seq_len=train_args["max_seq_len"],
            max_beat=train_args["max_beat"],
            rotary_pos_emb=train_args["rel_pos_emb"],
            use_abs_pos_emb=train_args["abs_pos_emb"],
            emb_dropout=train_args["dropout"],
            attn_dropout=train_args["dropout"],
            ff_dropout=train_args["dropout"],
        ).to(device)

        state = torch.load(paths["checkpoint"], map_location=device)
        load_info = model.load_state_dict(state, strict=True)
        logger.info(f"Loaded MMT checkpoint: {paths['checkpoint']} | {load_info}")

        self._model = model
        return model

    def create_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for the MMT dataset.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default False.
        num_workers : int, optional
            Number of worker processes for data loading, by default 0.

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader yielding indexed batches compatible
            with the MMT model.
        """
        train_args, encoding = self._load_train_args_and_encoding()
        paths = self._resolve_paths()

        base_dataset = mmt_dataset.MusicDataset(
            paths["names_pth"],
            paths["notes_dir"],
            encoding,
            max_seq_len=train_args["max_seq_len"],
            max_beat=train_args["max_beat"],
        )

        dataset = IndexedDataset(base_dataset)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
        )

    def get_hookable_layers(
            self,
            layer_indices: Iterable[int],
    ) -> Dict[int, torch.nn.Module]:
        """
        Retrieve internal transformer layers for hooking or inspection.

        Parameters:
        layer_indices : Iterable[int]
            Indices of layers to retrieve from the decoder stack.

        Returns:
        Dict[int, torch.nn.Module]
            Mapping from layer index to the corresponding module.

        Raises:
        RuntimeError
            If the model has not been loaded yet.
        ValueError
            If any provided index is out of range.

        Notes:
        This method currently inflexibly exposes the third
        submodule of each attention block (``blocks[idx][2]``),
        which in the MMT project architecture corresponds
        to the model's residual stream.
        """

        if self._model is None:
            raise RuntimeError("Model must be loaded before accessing layers.")

        layers = {}
        blocks = self._model.decoder.net.attn_layers.layers

        for idx in layer_indices:
            if idx < 0 or idx >= len(blocks):
                raise ValueError(f"Invalid layer index: {idx}")

            layers[idx] = blocks[idx][2]

        return layers


    def forward(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run a forward pass of the model.

        Parameters
        ----------
        model : torch.nn.Module
            The loaded MMT model.
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing at least:
            - ``seq`` : input token sequence tensor.
            - ``mask`` : attention mask tensor.

        Returns
        -------
        torch.Tensor
            Model output logits or hidden states, depending on the
            transformer configuration.

        Notes
        -----
        The batch tensors are automatically moved to the device of
        the provided model. The attention mask is coerced to boolean
        if required.
        """
        device = next(model.parameters()).device

        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)

        if mask.dtype != torch.bool:
            mask = mask.bool()

        return model(seq, mask=mask)


    def decode_notes(self, tokens: Iterable[int]) -> Any:
        """
            Decode a sequence of token IDs back into a musical representation.

            Parameters
            ----------
            tokens : Iterable[int]
                Sequence of token indices produced by the model.

            Returns
            -------
            Any
                Decoded musical structure as defined by the active encoding in text format.

            Notes
            -----
            This uses the encoding loaded from disk to ensure that decoding
            is consistent with the training configuration.
            """
        _, encoding = self._load_train_args_and_encoding()
        return representation.dump(tokens, encoding)
