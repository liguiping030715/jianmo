
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T,D]
    mask: [B,T] bool
    """
    w = mask.to(x.dtype).unsqueeze(-1)
    den = w.sum(dim=1).clamp_min(1.0)
    return (x * w).sum(dim=1) / den


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    scores: [B,T]
    mask: [B,T] bool
    Returns all-zero weights for rows with no eligible positions.
    """
    neg = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~mask, neg)
    any_valid = mask.any(dim=dim, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    w = torch.softmax(safe, dim=dim)
    return torch.where(any_valid, w * mask.to(w.dtype), torch.zeros_like(w))


class PositionStateEncoder(nn.Module):
    """
    Shared per-modality encoder:
      content projection + position embedding + tri-state P/O embeddings.

    P/O codes:
      0 = known padding/unavailable
      1 = known valid/usable
      2 = unresolved

    Unknown=2 is NOT hard-masked.
    """
    def __init__(self, in_dim: int, hidden: int = 128, max_len: int = 50):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.p_emb = nn.Embedding(3, hidden)
        self.o_emb = nn.Embedding(3, hidden)
        self.state_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,       # [B,T,D]
        p: torch.Tensor,       # [B,T]
        o: torch.Tensor,       # [B,T]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        z = self.proj(x) + self.pos(pos) + self.p_emb(p) + self.o_emb(o)
        z = F.gelu(z)

        # Primary policy from Stage 4 B1:
        # P=0 => hard padding exclusion
        # O=0 => known unavailable exclusion
        # O=2/P=2 => retained as unknown
        content_mask = (p != 0) & (o != 0)
        state_mask = (p != 0)

        state_z = self.state_proj(self.p_emb(p) + self.o_emb(o) + self.pos(pos))
        state_summary = masked_mean(state_z, state_mask)

        return z, content_mask, state_summary


class MeanPoolModality(nn.Module):
    """B1-style state-aware mean pooling, used by D0."""
    def __init__(self, in_dim: int, hidden: int = 128, max_len: int = 50):
        super().__init__()
        self.enc = PositionStateEncoder(in_dim, hidden, max_len)

    def forward(self, x, p, o):
        z, content_mask, state_summary = self.enc(x, p, o)
        content = masked_mean(z, content_mask)
        h = content + state_summary
        return h, {"content_mask": content_mask}


class LocalStatePoolModality(nn.Module):
    """
    D1 local state-aware temporal pooling.

    It operates BEFORE modality pooling. A shallow temporal scorer uses
    local context over the encoded content+state representation.

    This is deliberately not a global modality gate.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        scorer_hidden: int = 32,
        max_len: int = 50,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.enc = PositionStateEncoder(in_dim, hidden, max_len)
        pad = kernel_size // 2
        self.local_scorer = nn.Sequential(
            nn.Conv1d(hidden, scorer_hidden, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(scorer_hidden, 1, kernel_size=1),
        )

    def forward(self, x, p, o):
        z, content_mask, state_summary = self.enc(x, p, o)

        scores = self.local_scorer(z.transpose(1, 2)).squeeze(1)  # [B,T]
        weights = masked_softmax(scores, content_mask, dim=1)    # [B,T]

        content = torch.sum(z * weights.unsqueeze(-1), dim=1)
        h = content + state_summary

        return h, {
            "content_mask": content_mask,
            "local_scores": scores,
            "local_weights": weights,
        }


class TaskSpecificFusion(nn.Module):
    """
    Separate fusion pathways for classification and regression.
    No reliability gate; no MoE; no cross-modal Transformer.
    """
    def __init__(self, hidden: int = 128):
        super().__init__()
        in_dim = hidden * 3
        self.f_cls = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        self.f_reg = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(hidden, 3)
        self.reg_head = nn.Linear(hidden, 1)

    def forward(self, ht, ha, hv):
        cat = torch.cat([ht, ha, hv], dim=-1)
        h_cls = self.f_cls(cat)
        h_reg = self.f_reg(cat)
        logits = self.cls_head(h_cls)
        intensity = 3.0 * torch.tanh(self.reg_head(h_reg)).squeeze(-1)
        return logits, intensity, {"h_cls": h_cls, "h_reg": h_reg}


class D0TaskSpecificFusion(nn.Module):
    """
    D0 = exact Stage-4 B1-style modality encoding/pooling
         + task-specific fusion paths.

    Direct scientific comparator:
      frozen R1/B1 vs D0
    isolates H8 (shared-fusion negative transfer).
    """
    def __init__(self, hidden: int = 128, max_len: int = 50):
        super().__init__()
        self.text = MeanPoolModality(768, hidden, max_len)
        self.audio = MeanPoolModality(74, hidden, max_len)
        self.vision = MeanPoolModality(35, hidden, max_len)
        self.fusion = TaskSpecificFusion(hidden)

    def forward(self, batch: Dict[str, torch.Tensor]):
        ht, dt = self.text(batch["text_states"], batch["p_text"], batch["o_text"])
        ha, da = self.audio(batch["audio"], batch["p_audio"], batch["o_audio"])
        hv, dv = self.vision(batch["vision"], batch["p_vision"], batch["o_vision"])
        logits, intensity, df = self.fusion(ht, ha, hv)
        return {
            "logits": logits,
            "intensity": intensity,
            "diagnostics": {"text": dt, "audio": da, "vision": dv, **df},
        }


class D1LocalStateTemporalPooling(nn.Module):
    """
    D1 = D0 task-specific fusion
         + position-level/local-window state-aware temporal pooling.

    Direct scientific comparator:
      D0 vs D1
    isolates H9 (local selection before pooling).
    """
    def __init__(
        self,
        hidden: int = 128,
        scorer_hidden: int = 32,
        max_len: int = 50,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.text = LocalStatePoolModality(768, hidden, scorer_hidden, max_len, kernel_size)
        self.audio = LocalStatePoolModality(74, hidden, scorer_hidden, max_len, kernel_size)
        self.vision = LocalStatePoolModality(35, hidden, scorer_hidden, max_len, kernel_size)
        self.fusion = TaskSpecificFusion(hidden)

    def forward(self, batch: Dict[str, torch.Tensor]):
        ht, dt = self.text(batch["text_states"], batch["p_text"], batch["o_text"])
        ha, da = self.audio(batch["audio"], batch["p_audio"], batch["o_audio"])
        hv, dv = self.vision(batch["vision"], batch["p_vision"], batch["o_vision"])
        logits, intensity, df = self.fusion(ht, ha, hv)
        return {
            "logits": logits,
            "intensity": intensity,
            "diagnostics": {"text": dt, "audio": da, "vision": dv, **df},
        }


@dataclass
class LossWeights:
    cls: float = 1.0
    reg: float = 1.0


def multitask_loss(
    output: Dict[str, torch.Tensor],
    y_cls: torch.Tensor,
    y_reg: torch.Tensor,
    w: LossWeights,
):
    loss_cls = F.cross_entropy(output["logits"], y_cls)
    loss_reg = F.huber_loss(output["intensity"], y_reg)
    total = w.cls * loss_cls + w.reg * loss_reg
    return total, {
        "loss": float(total.detach()),
        "loss_cls": float(loss_cls.detach()),
        "loss_reg": float(loss_reg.detach()),
    }


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
