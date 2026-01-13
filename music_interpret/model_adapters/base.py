from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any
import torch


class ModelAdapter(ABC):

    @abstractmethod
    def load_model(self, device: torch.device) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def create_dataloader(
            self,
            batch_size: int,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    @abstractmethod
    def get_hookable_layers(self, layer_indices: Iterable[int]) -> Dict[int, torch.nn.Module]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Any:
        raise NotImplementedError


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, idx: int):
        return self.base_dataset[idx], idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)
