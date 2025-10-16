"""Base dataset abstraction used by zero-shot evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


class ZeroShotDataset(ABC):
    """Base class shared by dataset configurations."""

    #: Registry key for the dataset. Subclasses should override this.
    name: str

    def __init__(
        self,
        *,
        batch_size: int = 256,
        num_workers: int = 4,
        output_csv: Optional[str] = None,
        amp: bool = True,
        dataloader_kwargs: Optional[Dict[str, object]] = None,
    ) -> None:
        if not getattr(type(self), "name", None):
            raise ValueError("Subclasses must define a non-empty 'name' attribute for registry use.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_csv = output_csv or f"{self.name}_results.csv"
        self.amp = amp
        self._dataloader_kwargs = dataloader_kwargs or {}

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def classnames(self) -> Sequence[str]:
        """Return the class names used to build zero-shot prompts."""

    @abstractmethod
    def build_dataset(self, preprocess) -> Dataset:
        """Create the dataset using the model-specific pre-processing pipeline."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------
    def templates(self) -> Sequence[str]:
        """Prompt templates used for classnames."""

        return ("a photo of a {}",)

    def prompts(self) -> Sequence[str]:
        """Full list of prompts used to produce text embeddings."""

        return [template.format(name) for name in self.classnames() for template in self.templates()]

    def dataloader_kwargs(self) -> Dict[str, object]:
        """Extra keyword arguments passed to :class:`torch.utils.data.DataLoader`."""

        base_kwargs = {"pin_memory": True}
        base_kwargs.update(self._dataloader_kwargs)
        return base_kwargs

    def build_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            **self.dataloader_kwargs(),
        )

    def build_text_features(self, model, tokenizer, device: torch.device) -> torch.Tensor:
        """Create text features following CLIP-style prompt ensembling."""

        prompts = self.prompts()
        if not prompts:
            raise ValueError("prompts() returned an empty list; provide at least one prompt.")

        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        num_classes = len(self.classnames())
        if num_classes == 0:
            raise ValueError("classnames() returned an empty list; unable to build classifier.")

        text_features = text_features.view(len(prompts), num_classes, -1)
        text_features = text_features.mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


__all__ = ["ZeroShotDataset"]
