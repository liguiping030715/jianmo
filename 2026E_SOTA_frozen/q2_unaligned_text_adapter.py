"""Frozen raw-text -> BERT interface for a future 2026E unaligned runner.

This module encodes supplied text. It does not load any competition split,
train/fine-tune a model, infer hidden missingness, or run M0 predictions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from transformers import BertModel, BertTokenizerFast

BERT_ROOT = Path("workspace/references/bert-base-uncased")
WEIGHTS_SHA256 = "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a"
VOCAB_SHA256 = "b49e80874c5efbc7f4cde245a812673b05ef1360ced56b8bc8c750eaeef3fe0f"
MAX_LENGTH = 50
BERT_BATCH_SIZE = 16
TOKEN_FIELDS = ("input_ids", "attention_mask", "token_type_ids")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FrozenUnalignedTextAdapter:
    """Pinned tokenizer and frozen BERT producing [N,50,768] float32 states."""

    def __init__(self):
        if file_sha256(BERT_ROOT / "pytorch_model.bin") != WEIGHTS_SHA256:
            raise RuntimeError("frozen BERT weights checksum mismatch")
        if file_sha256(BERT_ROOT / "vocab.txt") != VOCAB_SHA256:
            raise RuntimeError("frozen BERT vocabulary checksum mismatch")
        self.tokenizer = BertTokenizerFast.from_pretrained(str(BERT_ROOT), local_files_only=True)
        if self.tokenizer.padding_side != "right" or self.tokenizer.truncation_side != "right":
            raise RuntimeError("tokenizer side settings changed")
        if (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id, self.tokenizer.unk_token_id) != (101, 102, 0, 100):
            raise RuntimeError("BERT special-token IDs changed")
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        self.bert = BertModel.from_pretrained(str(BERT_ROOT), local_files_only=True)
        self.bert.eval()
        self.bert.requires_grad_(False)
        if self.bert.config.hidden_size != 768:
            raise RuntimeError("BERT hidden width changed")

    def tokenize(self, raw_texts: Sequence[str]) -> np.ndarray:
        texts = list(raw_texts)
        if not all(isinstance(value, str) for value in texts):
            raise TypeError("raw_text must be a string for every sample")
        if not texts:
            return np.empty((0, 3, MAX_LENGTH), dtype=np.int64)
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="np",
        )
        triplet = np.stack([encoded[field] for field in TOKEN_FIELDS], axis=1).astype(np.int64, copy=False)
        if triplet.shape != (len(texts), 3, MAX_LENGTH):
            raise RuntimeError("token triplet shape mismatch")
        if not np.isin(triplet[:, 1], (0, 1)).all():
            raise RuntimeError("attention mask is not binary")
        return triplet

    def encode_triplets(self, triplets: np.ndarray) -> np.ndarray:
        values = np.asarray(triplets)
        if values.ndim != 3 or values.shape[1:] != (3, MAX_LENGTH) or values.dtype != np.int64:
            raise ValueError("expected int64 [N,3,50] token triplets")
        output = np.empty((len(values), MAX_LENGTH, 768), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(values), BERT_BATCH_SIZE):
                tokens = torch.from_numpy(values[start:start+BERT_BATCH_SIZE].copy())
                hidden = self.bert(input_ids=tokens[:, 0],
                                   attention_mask=tokens[:, 1],
                                   token_type_ids=tokens[:, 2]).last_hidden_state
                output[start:start+len(tokens)] = hidden.numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite BERT hidden states")
        return output

    def encode_raw_text(self, raw_texts: Sequence[str]) -> Dict[str, np.ndarray]:
        triplet = self.tokenize(raw_texts)
        # The supplied raw text is available content. Attention=0 is verified
        # tokenizer padding, not a feature-magnitude missingness decision.
        p = triplet[:, 1, :].astype(np.uint8, copy=True)
        return {"text_bert": triplet,
                "text_hidden": self.encode_triplets(triplet),
                "text_P": p,
                "text_O": p.copy()}

    def provenance(self) -> dict:
        names = ("pytorch_model.bin", "vocab.txt", "tokenizer.json",
                 "tokenizer_config.json", "config.json")
        return {
            "local_model": str(BERT_ROOT),
            "model_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "file_sha256": {name: file_sha256(BERT_ROOT / name) for name in names},
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_settings": {
                "add_special_tokens": True, "padding": "max_length",
                "truncation": True, "max_length": MAX_LENGTH,
                "padding_side": self.tokenizer.padding_side,
                "truncation_side": self.tokenizer.truncation_side,
                "do_lower_case": bool(self.tokenizer.do_lower_case),
                "return_attention_mask": True,
                "return_token_type_ids": True,
                "triplet_row_order": list(TOKEN_FIELDS),
                "special_token_ids": {
                    "CLS": self.tokenizer.cls_token_id,
                    "SEP": self.tokenizer.sep_token_id,
                    "PAD": self.tokenizer.pad_token_id,
                    "UNK": self.tokenizer.unk_token_id}},
            "bert_eval_mode": not self.bert.training,
            "bert_parameters_require_grad": any(p.requires_grad for p in self.bert.parameters()),
            "bert_batch_size": BERT_BATCH_SIZE,
            "cpu_threads": torch.get_num_threads(),
            "output_shape_per_sample": [MAX_LENGTH, 768],
            "output_dtype": "float32"}
