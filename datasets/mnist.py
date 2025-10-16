"""MNIST dataset configuration."""
from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms

from .base import ZeroShotDataset

DATA_ROOT = Path("./data")


class MNISTTask(ZeroShotDataset):
    name = "mnist"

    def __init__(self) -> None:
        super().__init__(
            batch_size=64,
            num_workers=2,
            output_csv="mnist_zero_shot_results.csv",
            dataloader_kwargs={"persistent_workers": True},
        )

    def classnames(self):
        return [str(i) for i in range(10)]

    def templates(self):
        return (
            "a photo of the handwritten digit {}",
            "a grayscale image of the number {}",
            "a drawing of the number {}",
            "an MNIST image of the digit {}",
            "a picture of the number {}",
        )

    def build_dataset(self, preprocess):
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda im: im.convert("RGB")),
                preprocess,
            ]
        )
        return datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)


__all__ = ["MNISTTask", "DATA_ROOT"]
