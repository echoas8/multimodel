"""Compatibility wrapper for MNIST zero-shot evaluation."""
from __future__ import annotations

from datasets.mnist import MNISTTask
from eva_model import evaluate_models_on_dataset
from model_config import DEFAULT_MODEL_LIST


if __name__ == "__main__":
    task = MNISTTask()
    evaluate_models_on_dataset(task, DEFAULT_MODEL_LIST)
