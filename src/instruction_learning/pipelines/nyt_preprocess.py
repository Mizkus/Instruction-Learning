from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..data import download_data


def _resolve_config_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("configs directory not found relative to nyt_preprocess.py")


CONFIG_DIR = _resolve_config_dir()


def _load_default_cfg() -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    return cfg


class NYTPreprocessor:
    """Prepare topic.csv and pre-compute base/instruct Qwen3 embeddings."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data.dataset
        self.instructions_cfg = cfg.instructions
        self.paths_cfg = cfg.paths
        self.embedding_cfg = cfg.embedding_model

        self.instruction_text = self.instructions_cfg.topic or None

        self.raw_dir = Path(self.paths_cfg.data_root) / "raw"
        self.processed_dir = Path(self.paths_cfg.data_root) / "processed"
        self.embeddings_dir = Path(self.paths_cfg.data_root) / "embeddings"
        self.base_dir = self.embeddings_dir / "base"
        self.instruct_dir = self.embeddings_dir / "instruct"

        for path in [self.raw_dir, self.processed_dir, self.base_dir, self.instruct_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(self.embedding_cfg.name, device=self.embedding_cfg.device)
        if self.embedding_cfg.get("max_seq_length"):
            self.model.max_seq_length = int(self.embedding_cfg.max_seq_length)
        encode_signature = inspect.signature(self.model.encode)
        self.supports_prompt = "prompt" in encode_signature.parameters
        self.supports_prompts = "prompts" in encode_signature.parameters

    def run(self) -> None:
        csv_path = download_data(target_dir=self.raw_dir, cfg=self.cfg)
        df = pd.read_csv(csv_path)
        df = self._clean_dataframe(df)
        max_samples = self.data_cfg.get("max_samples")
        if max_samples is not None:
            df = df.head(int(max_samples))
        cleaned_csv = self.processed_dir / "topic_clean.csv"
        df.to_csv(cleaned_csv, index=False)
        jsonl_path = self.processed_dir / "topic_clean.jsonl"
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        if self._embeddings_cached(len(df)):
            print("[preprocess] Cached embeddings detected for matching instruction; skipping recomputation.")
            return
        self._compute_and_save_embeddings(df)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        text_col = self.data_cfg.text_column
        target_col = self.data_cfg.target_column
        df = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col])
        df = df[[text_col, target_col]].rename(columns={text_col: "text", target_col: "target"})
        return df.reset_index(drop=True)

    def _compute_and_save_embeddings(self, df: pd.DataFrame) -> None:
        texts = df["text"].tolist()
        base_embeddings, base_seconds = self._encode(texts, prompt=None)
        instruct_embeddings, instruct_seconds = self._encode(texts, prompt=self.instruction_text)

        np.save(self.base_dir / f"{self.data_cfg.instructions.field}.npy", base_embeddings)
        np.save(self.instruct_dir / f"{self.data_cfg.instructions.field}.npy", instruct_embeddings)

        labels, uniques = pd.factorize(df["target"])
        np.save(self.embeddings_dir / "labels.npy", labels.astype(np.int64))
        label_map = {int(idx): value for idx, value in enumerate(uniques)}
        with (self.embeddings_dir / "label_mapping.json").open("w", encoding="utf-8") as fp:
            json.dump(label_map, fp, ensure_ascii=False, indent=2)

        # Metadata for debugging/auditing
        stats = {
            "dataset": self.data_cfg.url,
            "instruction": self.instruction_text,
            "model": self.embedding_cfg.name,
            "num_samples": len(texts),
            "embedding_dim": int(base_embeddings.shape[1]) if base_embeddings.ndim == 2 else None,
            "base_prompt": None,
            "instruct_prompt": self.instruction_text,
            "base_encode_seconds": base_seconds,
            "instruct_encode_seconds": instruct_seconds,
            "base_latency_ms": (base_seconds / len(texts) * 1e3) if len(texts) else None,
            "instruct_latency_ms": (instruct_seconds / len(texts) * 1e3) if len(texts) else None,
            "max_samples": self.data_cfg.get("max_samples", len(texts)),
        }
        with (self.embeddings_dir / "metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, ensure_ascii=False, indent=2)

    def _encode(self, sentences: list[str], prompt: Optional[str]) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        encode_kwargs = {}
        if prompt:
            effective_prompt = str(prompt)
            if self.supports_prompt:
                encode_kwargs["prompt"] = effective_prompt
            elif self.supports_prompts:
                encode_kwargs["prompts"] = [effective_prompt] * len(sentences)
        embeddings = (
            self.model.encode(
                sentences,
                batch_size=self.embedding_cfg.batch_size,
                show_progress_bar=True,
                normalize_embeddings=self.embedding_cfg.normalize,
                convert_to_numpy=True,
                **encode_kwargs,
            ).astype(np.float32)
        )
        elapsed = time.perf_counter() - start
        return embeddings, elapsed

    def _embeddings_cached(self, expected_rows: int) -> bool:
        metadata_path = self.embeddings_dir / "metadata.json"
        base_file = self.base_dir / f"{self.data_cfg.instructions.field}.npy"
        instruct_file = self.instruct_dir / f"{self.data_cfg.instructions.field}.npy"
        if not (metadata_path.exists() and base_file.exists() and instruct_file.exists()):
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("instruction") != self.instruction_text:
            return False
        return metadata.get("num_samples") == expected_rows


def preprocess_nyt_dataset(cfg: Optional[DictConfig] = None) -> None:
    cfg = cfg or _load_default_cfg()
    preprocessor = NYTPreprocessor(cfg)
    preprocessor.run()


def main() -> None:
    cfg = _load_default_cfg()
    preprocess_nyt_dataset(cfg)


if __name__ == "__main__":
    main()
