"""CIFAR-100 dataset configuration."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from .base import ZeroShotDataset

DEFAULT_DATA_ROOT = Path("/zssd/home/weiyang/modelselection/datasets/datasets/cifar-100-python")


class CIFAR100Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR100Task(ZeroShotDataset):
    name = "cifar100"

    def __init__(self, data_root: Path = DEFAULT_DATA_ROOT) -> None:
        super().__init__(
            batch_size=128,
            num_workers=8,
            output_csv="cifar100_vlm_eval_results.csv",
        )
        self.data_root = data_root
        self.test_images, self.test_labels = self._load_split("test")
        self._fine_names = self._load_meta()

    def classnames(self):
        return self._fine_names

    def templates(self):
        return ("a photo of a {}",)

    def build_dataset(self, preprocess):
        return CIFAR100Dataset(self.test_images, self.test_labels, transform=preprocess)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_split(self, split: str):
        path = self.data_root / split
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="bytes")
        images = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = entry[b"fine_labels"]
        return images, labels

    def _load_meta(self) -> Tuple[str, ...]:
        path = self.data_root / "meta"
        with open(path, "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        return tuple(x.decode("utf-8") for x in meta[b"fine_label_names"])


__all__ = ["CIFAR100Task", "CIFAR100Dataset", "DEFAULT_DATA_ROOT"]
