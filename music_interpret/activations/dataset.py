from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    """
    PyTorch dataset for token activation storage and access.

    This dataset provides indexed access to activation shards produced by
    the extraction pipeline. Activations are memory-mapped
    to avoid loading large arrays. Optional metadata shards
    provide sample and token position information.

    Parameters
    ----------
    activ_root : pathlib.Path
        Directory containing activation shard files (``shard_*.npy``).
    meta_root : pathlib.Path, optional
        Directory containing metadata shard files (``shard_*.npy``),
        by default None.
    dtype : torch.dtype, optional
        Data type used when converting activations to torch tensors,
        by default ``torch.float32``.

    Raises
    ------
    RuntimeError
        If no activation shards are found or if activation and metadata
        shard counts do not match.
    """

    def __init__(
        self,
        activ_root: Path,
        meta_root: Optional[Path] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the activation dataset from shards.

        Parameters
        ----------
        activ_root : pathlib.Path
            Directory containing activation shard files.
        meta_root : pathlib.Path, optional
            Directory containing metadata shard files, by default None.
        dtype : torch.dtype, optional
            Torch dtype used for returned activation tensors,
            by default ``torch.float32``.

        Notes
        -----
        All shard files are opened using NumPy memory mapping, allowing
        random access to large dataset with small memory usage.
        """
        self.activ_root = Path(activ_root)
        self.meta_root = Path(meta_root) if meta_root is not None else None
        self.dtype = dtype

        self._activ_shards = sorted(self.activ_root.glob("shard_*.npy"))
        if not self._activ_shards:
            raise RuntimeError(f"No activation shards found in {self.activ_root}")

        if self.meta_root is not None:
            self._meta_shards = sorted(self.meta_root.glob("shard_*.npy"))
            if len(self._meta_shards) != len(self._activ_shards):
                raise RuntimeError(
                    "Activation shards and meta shards count mismatch"
                )
        else:
            self._meta_shards = None

        self._activ_mmaps = [
            np.load(p, mmap_mode="r") for p in self._activ_shards
        ]
        self._meta_mmaps = (
            [np.load(p, mmap_mode="r") for p in self._meta_shards]
            if self._meta_shards is not None
            else None
        )

        self._lengths = [m.shape[0] for m in self._activ_mmaps]
        self._offsets = np.cumsum([0] + self._lengths)

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - self._offsets[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx, local_idx = self._locate(idx)

        x = self._activ_mmaps[shard_idx][local_idx]
        out = {
            "activations": torch.tensor(x, dtype=self.dtype),
        }

        if self._meta_mmaps is not None:
            meta = self._meta_mmaps[shard_idx][local_idx]
            out["sample_idx"] = int(meta[0])
            out["token_pos"] = int(meta[1])

        return out
