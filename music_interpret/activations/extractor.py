from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


class ActivationExtractor:
    """
    Collects and stores activations from multiple model layers
    during a forward pass over a dataset.

    The extractor registers forward hooks on selected layers and buffers
    activations together with metadata (sample index and token position).
    Data is periodically flushed to disk in shards to avoid excessive
    memory usage.

    Parameters
    ----------
    root_dir : pathlib.Path
        Root directory where activation shards and metadata will be stored.
    layer_indices : Iterable[int]
        Indices of model layers from which activations should be extracted.
    max_tokens_per_shard : int, optional
        Maximum number of tokens to store before writing a shard to disk,
        by default 200_000.
    dtype : numpy.dtype, optional
        Data type used when saving activations, by default ``np.float32``.

    Notes
        -----
        For each layer index, the following directory structure is created::

            root_dir/
                layer_XXX/
                    raw/    # activation shards
                    meta/   # metadata shards
    """

    def __init__(
            self,
            root_dir: Path,
            layer_indices: Iterable[int],
            max_tokens_per_shard: int = 200_000,
            dtype: np.dtype = np.float32,
    ):
        self.root_dir = Path(root_dir)
        self.layer_indices = list(layer_indices)
        self.max_tokens_per_shard = max_tokens_per_shard
        self.dtype = dtype

        self._mask = None
        self._sample_idx = None

        self._buffers = {}
        self._meta_buffers = {}
        self._buffer_tokens = {}
        self._shard_idx = {}
        self._dirs = {}

        self._handles = []

        for layer_idx in self.layer_indices:
            base = self.root_dir / f"layer_{layer_idx:03d}"
            activ_dir = base / "raw"
            meta_dir = base / "meta"

            activ_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            self._buffers[layer_idx] = []
            self._meta_buffers[layer_idx] = []
            self._buffer_tokens[layer_idx] = 0
            self._shard_idx[layer_idx] = 0
            self._dirs[layer_idx] = {
                "raw": activ_dir,
                "meta": meta_dir,
            }

    def set_batch_context(self, mask: torch.Tensor, sample_idx: torch.Tensor) -> None:
        """
        Set batch-specific context required by the activation hooks.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean tensor of shape ``(B, T)`` indicating which tokens are valid.
        sample_idx : torch.Tensor
            Tensor of shape ``(B,)`` mapping each batch element to its
            original dataset index.

        Notes
        -----
        This method **must** be called before running a forward pass when
        hooks are active. The tensors are detached and moved to CPU for
        safe use inside hooks.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()

        self._mask = mask.detach().cpu()
        self._sample_idx = sample_idx.detach().cpu()

    def _make_hook(self, layer_idx: int):
        """
            Create a forward hook function for a specific layer.

            Parameters
            ----------
            layer_idx : int
                Index of the layer for which the hook is being created.

            Returns
            -------
            Callable
                A hook function compatible with ``torch.nn.Module.register_forward_hook``.

            Raises
            ------
            TypeError
                If the hooked module does not return a tensor.
            RuntimeError
                If batch context has not been set or if dimensions mismatch.
            ValueError
                If the activation tensor does not have shape ``(B, T, D)``.

            Notes
            -----
            This function is compatible with mmt project and requires that model outputs data in specific way.

            The hook performs the following steps:
            - Validates activation shape and batch context.
            - Applies the batch mask to filter valid tokens.
            - Flattens activations to ``(N, D)`` where ``N`` is the number of
              valid tokens.
            - Stores activations and minimal metadata (sample index, token position).
            - Flushes buffered data to disk when the shard size limit is reached.
            """
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Hook for layer {layer_idx} expected Tensor output, got {type(output)} from module {module.__class__.__name__}")

            if self._mask is None or self._sample_idx is None:
                raise RuntimeError("Batch context not set before forward")

            acts = output.detach().float().cpu()

            if acts.ndim != 3:
                raise ValueError(
                    f"Hook for layer {layer_idx} expected activation of shape (B, T, D), got tensor with shape {tuple(acts.shape)}")

            B, T, D = acts.shape

            if self._mask.shape[1] < T:
                raise RuntimeError(
                    f"Activation time dimension T={T} exceeds mask length {self._mask.shape[1]}. Possible memory tokens or model-side token insertion."
                )
            if self._mask.shape[0] != B:
                raise RuntimeError(
                    f"Batch size mismatch: activations B={B}, mask B={self._mask.shape[0]}"
                )

            mask = self._mask[:, :T]
            sample_idx = self._sample_idx

            flat_mask = mask.reshape(-1)
            flat_acts = acts.reshape(-1, D)[flat_mask]

            if flat_acts.shape[0] == 0:
                raise RuntimeError(
                    f"Hook for layer {layer_idx} produced zero valid tokens. Mask may be incorrect or misaligned.")

            token_pos = (
                torch.arange(T)
                .unsqueeze(0)
                .expand(B, T)
                .reshape(-1)[flat_mask]
            )

            flat_sample_idx = (
                sample_idx.unsqueeze(1)
                .expand(B, T)
                .reshape(-1)[flat_mask]
            )

            self._buffers[layer_idx].append(
                flat_acts.numpy().astype(self.dtype)
            )
            meta = torch.stack(
                [flat_sample_idx, token_pos], dim=1
            ).cpu().numpy().astype("int64")
            self._meta_buffers[layer_idx].append(meta)

            self._buffer_tokens[layer_idx] += flat_acts.shape[0]

            if self._buffer_tokens[layer_idx] >= self.max_tokens_per_shard:
                self._flush(layer_idx)

        return hook

    def register(self, layers: Dict[int, torch.nn.Module]) -> None:
        """
        Register forward hooks on the specified model layers.

        Parameters
        ----------
        layers : Dict[int, torch.nn.Module]
            Mapping from layer index to the corresponding module on which
            a forward hook should be registered.

        Notes
        -----
        Hook handles are stored internally and must be removed by calling
        :meth:`close` when extraction is finished.
        """
        for layer_idx, module in layers.items():
            handle = module.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def _flush(self, layer_idx: int):
        """
        Write buffered activations and metadata for a layer to disk.

        Parameters
        ----------
        layer_idx : int
            Index of the layer whose buffers should be flushed.

        Notes
        -----
        This method concatenates all buffered arrays, saves them as NumPy
        ``.npy`` files in the corresponding ``raw`` and ``meta`` directories,
        and resets the in-memory buffers.
        """
        if not self._buffers[layer_idx]:
            return

        activ = np.concatenate(self._buffers[layer_idx], axis=0)
        shard = self._shard_idx[layer_idx]
        np.save(
            self._dirs[layer_idx]["raw"] / f"shard_{shard:04d}.npy",
            activ,
        )

        meta = np.concatenate(self._meta_buffers[layer_idx], axis=0)
        np.save(
            self._dirs[layer_idx]["meta"] / f"shard_{shard:04d}.npy",
            meta,
        )

        self._buffers[layer_idx].clear()
        self._meta_buffers[layer_idx].clear()
        self._buffer_tokens[layer_idx] = 0
        self._shard_idx[layer_idx] += 1

    def flush(self):
        """
        Flush all buffered activations for every registered layer.
        """
        for key in list(self._buffers.keys()):
            self._flush(key)

    def close(self):
        """
        Finalize extraction by flushing buffers and removing all hooks.

        Notes
        -----
        This method:

        - Flushes any remaining buffered activations.
        - Removes all registered forward hooks.
        - Clears internal hook handle references.

        After calling this method, the extractor can no longer collect
        activations unless hooks are registered again.
        """
        self.flush()
        for h in self._handles:
            h.remove()
        self._handles.clear()
