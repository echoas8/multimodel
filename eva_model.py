"""Core zero-shot evaluation utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm
import open_clip

from datasets.base import ZeroShotDataset


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve the torch device to run evaluation on."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer, preprocess


def evaluate_models_on_dataset(
    dataset: ZeroShotDataset,
    model_specs: Sequence[Tuple[str, str]],
    *,
    device: str | torch.device | None = None,
    save_to: Path | None = None,
) -> pd.DataFrame:
    """Evaluate a list of models on a dataset configuration."""
    torch_device = resolve_device(str(device) if isinstance(device, torch.device) else device)
    results: List[dict] = []

    for model_name, pretrained in model_specs:
        model, tokenizer, preprocess = load_model(model_name, pretrained, torch_device)
        with torch.no_grad():
            text_features = dataset.build_text_features(model, tokenizer, torch_device)

        data = dataset.build_dataset(preprocess)
        dataloader = dataset.build_dataloader(data)
        correct = 0
        total = 0

        autocast_ctx = torch.cuda.amp.autocast if dataset.amp and torch_device.type == "cuda" else _nullcontext
        with autocast_ctx():
            for images, labels in tqdm(dataloader, desc=f"{dataset.name}:{model_name}"):
                images = images.to(torch_device)
                labels = labels.to(torch_device)

                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logits = 100.0 * image_features @ text_features
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0
        result = {
            "dataset": dataset.name,
            "model": model_name,
            "pretrained": pretrained,
            "top1": accuracy,
        }
        results.append(result)

    df = pd.DataFrame(results)
    output_path = Path(save_to) if save_to else Path(dataset.output_csv)
    df.to_csv(output_path, index=False)
    return df


class _nullcontext:
    """Fallback context manager mimicking contextlib.nullcontext for older Python."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


__all__ = ["evaluate_models_on_dataset", "resolve_device"]
