"""
모델 3: KoELECTRA Multi-task 3-head 분류기.

monologg/koelectra-base-v3-discriminator 인코더 위에
대분류 / 중분류 / 카테고리태그 3개 Linear head를 붙인다.

Loss = w_large * CE(large) + w_medium * CE(medium) + w_tag * CE(tag)
Optimizer: AdamW + Linear warmup + Cosine decay
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.error("torch / transformers 미설치")

from .base import BaseClassifier


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _REFDataset(Dataset):
        def __init__(
            self,
            texts: list[str],
            y_large: np.ndarray | None,
            y_medium: np.ndarray | None,
            y_tag: np.ndarray | None,
            tokenizer,
            max_length: int,
        ) -> None:
            self.enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.y_large  = torch.tensor(y_large,  dtype=torch.long) if y_large  is not None else None
            self.y_medium = torch.tensor(y_medium, dtype=torch.long) if y_medium is not None else None
            self.y_tag    = torch.tensor(y_tag,    dtype=torch.long) if y_tag    is not None else None

        def __len__(self) -> int:
            return self.enc["input_ids"].size(0)

        def __getitem__(self, idx: int) -> dict:
            item = {k: v[idx] for k, v in self.enc.items()}
            if self.y_large  is not None: item["y_large"]  = self.y_large[idx]
            if self.y_medium is not None: item["y_medium"] = self.y_medium[idx]
            if self.y_tag    is not None: item["y_tag"]    = self.y_tag[idx]
            return item


    # ──────────────────────────────────────────────────────────────
    # Model Architecture
    # ──────────────────────────────────────────────────────────────

    class _KoElectraMultiTask(nn.Module):
        def __init__(
            self,
            backbone_name: str,
            n_large: int,
            n_medium: int,
            n_tag: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(backbone_name)
            hidden = self.encoder.config.hidden_size

            self.dropout = nn.Dropout(dropout)
            self.head_large  = nn.Linear(hidden, n_large)
            self.head_medium = nn.Linear(hidden, n_medium)
            self.head_tag    = nn.Linear(hidden, n_tag)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            out = self.encoder(**kwargs)
            # [CLS] 벡터
            cls = out.last_hidden_state[:, 0, :]
            cls = self.dropout(cls)
            return (
                self.head_large(cls),
                self.head_medium(cls),
                self.head_tag(cls),
            )


class KoElectraMultiTaskClassifier(BaseClassifier):
    """
    KoELECTRA-base-v3 Multi-task 3-head 분류기.

    backbone은 monologg/koelectra-base-v3-discriminator 고정.
    """

    BACKBONE = "monologg/koelectra-base-v3-discriminator"

    def __init__(
        self,
        n_large: int = 2,
        n_medium: int = 2,
        n_tag: int = 2,
        # 학습 파라미터
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 5,
        max_length: int = 64,
        dropout: float = 0.1,
        freeze_layers: int = 6,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        # Loss 가중치
        w_large: float = 0.4,
        w_medium: float = 0.4,
        w_tag: float = 0.2,
        seed: int = 42,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch / transformers 패키지가 필요합니다.")

        self._hparams = dict(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            max_length=max_length,
            dropout=dropout,
            freeze_layers=freeze_layers,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            w_large=w_large,
            w_medium=w_medium,
            w_tag=w_tag,
        )
        self._n_large  = n_large
        self._n_medium = n_medium
        self._n_tag    = n_tag
        self._seed     = seed

        torch.manual_seed(seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("KoELECTRA — 디바이스: %s", self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.BACKBONE)
        self._model: _KoElectraMultiTask | None = None

    # ──────────────────────────────────────────────────────────────
    # BaseClassifier 구현
    # ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
        y_large: np.ndarray,
        y_medium: np.ndarray,
        y_tag: np.ndarray,
        **params,
    ) -> None:
        if params:
            self._hparams.update(params)
        hp = self._hparams

        self._model = _KoElectraMultiTask(
            backbone_name=self.BACKBONE,
            n_large=self._n_large,
            n_medium=self._n_medium,
            n_tag=self._n_tag,
            dropout=hp["dropout"],
        ).to(self._device)

        # 하위 레이어 Freeze
        if hp["freeze_layers"] > 0:
            self._freeze_bottom_layers(hp["freeze_layers"])

        texts = X_refined.fillna("").tolist()
        dataset = _REFDataset(texts, y_large, y_medium, y_tag, self._tokenizer, hp["max_length"])
        loader  = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True, num_workers=0)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
        )
        total_steps  = len(loader) * hp["epochs"]
        warmup_steps = int(total_steps * hp["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        criterion = nn.CrossEntropyLoss()
        w_large  = hp["w_large"]
        w_medium = hp["w_medium"]
        w_tag    = hp["w_tag"]

        self._model.train()
        for epoch in range(hp["epochs"]):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                input_ids      = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self._device)

                logits_large, logits_medium, logits_tag = self._model(
                    input_ids, attention_mask, token_type_ids
                )
                loss = (
                    w_large  * criterion(logits_large,  batch["y_large"].to(self._device))
                    + w_medium * criterion(logits_medium, batch["y_medium"].to(self._device))
                    + w_tag    * criterion(logits_tag,   batch["y_tag"].to(self._device))
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            logger.debug("Epoch %d/%d — loss: %.4f", epoch + 1, hp["epochs"], epoch_loss / len(loader))

    def predict(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        probas = self.predict_proba(X_refined, X_nouns)
        return (
            np.argmax(probas[0], axis=1),
            np.argmax(probas[1], axis=1),
            np.argmax(probas[2], axis=1),
        )

    def predict_proba(
        self,
        X_refined: pd.Series,
        X_nouns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        hp = self._hparams
        texts   = X_refined.fillna("").tolist()
        dataset = _REFDataset(texts, None, None, None, self._tokenizer, hp["max_length"])
        loader  = DataLoader(dataset, batch_size=hp["batch_size"] * 2, shuffle=False, num_workers=0)

        all_large, all_medium, all_tag = [], [], []
        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self._device)

                logits_l, logits_m, logits_t = self._model(input_ids, attention_mask, token_type_ids)
                all_large.append(torch.softmax(logits_l, dim=-1).cpu().numpy())
                all_medium.append(torch.softmax(logits_m, dim=-1).cpu().numpy())
                all_tag.append(torch.softmax(logits_t,   dim=-1).cpu().numpy())

        return (
            np.vstack(all_large),
            np.vstack(all_medium),
            np.vstack(all_tag),
        )

    def get_params(self) -> dict:
        return dict(**self._hparams, backbone=self.BACKBONE)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict() if self._model else None,
            "hparams":     self._hparams,
            "n_large":     self._n_large,
            "n_medium":    self._n_medium,
            "n_tag":       self._n_tag,
        }, path)
        logger.info("모델 저장: %s", path)

    @classmethod
    def load(cls, path: str) -> "KoElectraMultiTaskClassifier":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(
            n_large=ckpt["n_large"],
            n_medium=ckpt["n_medium"],
            n_tag=ckpt["n_tag"],
            **ckpt["hparams"],
        )
        if ckpt["model_state"] is not None:
            obj._model = _KoElectraMultiTask(
                cls.BACKBONE, ckpt["n_large"], ckpt["n_medium"], ckpt["n_tag"]
            ).to(obj._device)
            obj._model.load_state_dict(ckpt["model_state"])
        return obj

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _freeze_bottom_layers(self, n: int) -> None:
        """ELECTRA 인코더 하위 n개 레이어를 동결한다."""
        encoder = self._model.encoder
        # discriminator_predictions를 제외한 레이어 접근
        layers = None
        if hasattr(encoder, "electra") and hasattr(encoder.electra, "encoder"):
            layers = encoder.electra.encoder.layer
        elif hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            layers = encoder.encoder.layer

        if layers is None:
            logger.warning("레이어 Freeze 경로를 찾지 못함. Freeze 생략.")
            return

        for i, layer in enumerate(layers):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
        logger.debug("하위 %d개 레이어 Freeze 완료", n)
