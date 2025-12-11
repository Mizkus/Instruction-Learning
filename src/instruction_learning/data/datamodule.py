from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingDataModule(L.LightningDataModule):
    """LightningDataModule that loads base/instruct embeddings or generates synthetic ones."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data.dataset
        self.batch_cfg = cfg.training.batch_size
        self.loader_cfg = cfg.training.dataloader
        self.input_dim = cfg.model.input_dim
        self.output_dim = cfg.model.output_dim
        self.seed = cfg.project.seed

        self.train_ds: Optional[TensorDataset] = None
        self.val_ds: Optional[TensorDataset] = None
        self.test_ds: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        base_embeddings, target_embeddings = self._load_embeddings()
        labels = self._load_labels(len(base_embeddings))
        splits = self._compute_split_indices(len(base_embeddings))

        self.train_ds = TensorDataset(
            base_embeddings[splits["train"]],
            target_embeddings[splits["train"]],
            labels[splits["train"]],
        )
        self.val_ds = TensorDataset(
            base_embeddings[splits["val"]],
            target_embeddings[splits["val"]],
            labels[splits["val"]],
        )
        self.test_ds = TensorDataset(
            base_embeddings[splits["test"]],
            target_embeddings[splits["test"]],
            labels[splits["test"]],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_cfg.train,
            shuffle=True,
            num_workers=self.loader_cfg.num_workers,
            pin_memory=self.loader_cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_cfg.eval,
            shuffle=False,
            num_workers=self.loader_cfg.num_workers,
            pin_memory=self.loader_cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_cfg.eval,
            shuffle=False,
            num_workers=self.loader_cfg.num_workers,
            pin_memory=self.loader_cfg.pin_memory,
        )

    # ---------------------------------------------------------------------
    def _load_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_path = Path(self.data_cfg.embeddings.base_path)
        instruct_path = Path(self.data_cfg.embeddings.instruct_path)
        if base_path.exists() and instruct_path.exists():
            base, target = self._load_from_disk(base_path, instruct_path)
        else:
            base, target = self._generate_synthetic()
        return torch.from_numpy(base).float(), torch.from_numpy(target).float()

    def _load_labels(self, num_samples: int) -> torch.Tensor:
        labels_path = Path(self.cfg.paths.data_root) / "embeddings" / "labels.npy"
        if labels_path.exists():
            labels = np.load(labels_path)
        else:
            labels = np.arange(num_samples)
        if len(labels) != num_samples:
            raise ValueError("Labels count does not match embedding rows")
        return torch.from_numpy(labels.astype(np.int64))

    def _load_from_disk(self, base_path: Path, instruct_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        base_arrays = self._load_stack(base_path)
        target_arrays = self._load_stack(instruct_path)
        if base_arrays.shape != target_arrays.shape:
            raise ValueError("Base and target embeddings must have identical shapes")
        return base_arrays, target_arrays

    def _load_stack(self, folder: Path) -> np.ndarray:
        npy_files = sorted(folder.glob("*.npy"))
        if npy_files:
            arrays = [np.load(file) for file in npy_files]
            return np.concatenate(arrays, axis=0)
        json_files = sorted(folder.glob("*.json"))
        if json_files:
            vectors = []
            for file in json_files:
                with file.open() as fp:
                    payload = json.load(fp)
                vectors.append(np.asarray(payload["embedding"], dtype=np.float32))
            return np.stack(vectors, axis=0)
        raise FileNotFoundError(f"No embedding files found in {folder}")

    def _generate_synthetic(self) -> Tuple[np.ndarray, np.ndarray]:
        num_samples = self.data_cfg.get("synthetic_samples", 4096)
        rng = np.random.default_rng(self.seed)
        base = rng.standard_normal((num_samples, self.input_dim)).astype(np.float32)
        transform = rng.standard_normal((self.output_dim, self.input_dim)).astype(np.float32)
        target = base @ transform.T
        noise = 0.05 * rng.standard_normal(target.shape).astype(np.float32)
        return base, target + noise

    def _compute_split_indices(self, num_samples: int) -> Dict[str, np.ndarray]:
        split_cfg = self.data_cfg.split
        train_ratio = float(split_cfg.get("train_ratio", 0.2))
        val_ratio = float(split_cfg.get("val_ratio", 0.1))
        test_ratio = float(split_cfg.get("test_ratio", max(0.0, 1.0 - train_ratio - val_ratio)))
        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if ratios.sum() <= 0:
            ratios = np.array([0.8, 0.1, 0.1], dtype=float)
        ratios = ratios / ratios.sum()

        cumulative = np.cumsum(ratios) * num_samples
        train_end = max(1, int(cumulative[0]))
        val_end = max(train_end + 1, int(cumulative[1]))
        if val_end >= num_samples:
            val_end = max(train_end + 1, num_samples - 1)
        rng = np.random.default_rng(split_cfg.get("seed", self.seed))
        indices = np.arange(num_samples)
        if split_cfg.get("shuffle", True):
            rng.shuffle(indices)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        if len(val_idx) == 0 and len(train_idx) > 1:
            val_idx = train_idx[-1:]
            train_idx = train_idx[:-1]
        if len(test_idx) == 0:
            if len(val_idx) > 1:
                test_idx = val_idx[-1:]
                val_idx = val_idx[:-1]
            elif len(train_idx) > 1:
                test_idx = train_idx[-1:]
                train_idx = train_idx[:-1]
            else:
                test_idx = indices[-1:]

        splits = {
            "train": torch.from_numpy(train_idx),
            "val": torch.from_numpy(val_idx),
            "test": torch.from_numpy(test_idx),
        }

        for key in ("train", "val", "test"):
            if splits[key].numel() == 0:
                donor_key = next(
                    (k for k in ("train", "val", "test") if splits[k].numel() > 1),
                    None,
                )
                if donor_key is not None:
                    donor = splits[donor_key]
                    splits[key] = donor[-1:].clone()
                    splits[donor_key] = donor[:-1]
                else:
                    splits[key] = torch.zeros(1, dtype=torch.long)
        return splits
