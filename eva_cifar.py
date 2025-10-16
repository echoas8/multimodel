"""Compatibility wrapper for CIFAR-100 zero-shot evaluation."""
from __future__ import annotations

from datasets.cifar100 import CIFAR100Task
from eva_model import evaluate_models_on_dataset
from model_config import DEFAULT_MODEL_LIST


if __name__ == "__main__":
    task = CIFAR100Task()
    evaluate_models_on_dataset(task, DEFAULT_MODEL_LIST)
