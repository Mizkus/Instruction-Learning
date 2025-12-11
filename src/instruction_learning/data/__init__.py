"""Data loading utilities."""

from .datamodule import EmbeddingDataModule
from .download import download_data

__all__ = ["EmbeddingDataModule", "download_data"]
