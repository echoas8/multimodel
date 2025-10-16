"""Command-line entry point for zero-shot evaluation."""
from __future__ import annotations

import argparse

from datasets import DATASET_REGISTRY, ZeroShotDataset
from eva_model import evaluate_models_on_dataset, resolve_device
from model_config import DEFAULT_MODEL_LIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation runner")
    parser.add_argument("dataset", choices=sorted(DATASET_REGISTRY.keys()), help="Dataset to evaluate")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on (e.g. cuda, cuda:1, cpu). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for saving the aggregated results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset: ZeroShotDataset = dataset_cls()

    device = resolve_device(args.device)
    evaluate_models_on_dataset(dataset, DEFAULT_MODEL_LIST, device=device, save_to=args.output)


if __name__ == "__main__":
    main()
