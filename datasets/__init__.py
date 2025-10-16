"""Dataset registry for zero-shot evaluation tasks."""
from __future__ import annotations

from typing import Dict, Type

from .base import ZeroShotDataset
from .mnist import MNISTTask
from .cifar100 import CIFAR100Task

DATASET_REGISTRY: Dict[str, Type[ZeroShotDataset]] = {
    MNISTTask.name: MNISTTask,
    CIFAR100Task.name: CIFAR100Task,
}

__all__ = ["DATASET_REGISTRY", "ZeroShotDataset", "MNISTTask", "CIFAR100Task"]
