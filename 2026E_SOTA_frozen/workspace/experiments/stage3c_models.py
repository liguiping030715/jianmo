"""
Stage 3C E0 model.

E0 = frozen R1/B1 architecture with ONE delta: content temporal pooling is
replaced by a local state-aware (Conv1d kernel=3) scorer + masked-softmax
weighted pooling. The existing state summary, shared fusion, heads, losses,
and all encoder/embedding components are identical in structure to B1.

Direct comparator: frozen R1 (R1 vs E0 differ only in temporal pooling).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODS = ("text", "audio", "vision")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(scores.dtype).min
    m = scores.masked_fill(~mask, neg)
    any_valid = mask.any(dim=-1, keepdim=True)
    safe = torch.where(any_valid, m, torch.zeros_like(m))
    w = torch.softmax(safe, dim=-1)
    return torch.where(any_valid, w * mask.to(w.dtype), torch.zeros_like(w))


class E0Head(nn.Module):
    def __init__(self, kernel_size: int = 3, scorer_hidden: int = 32):
        super().__init__()
        # Encoder components identical to PilotHead('B1').
        self.proj = nn.ModuleList(
            [nn.Linear(768, 128), nn.Linear(74, 128), nn.Linear(35, 128)])
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.randn(50, 128) * 0.02) for _ in range(3)])
        self.p_embed = nn.ModuleList([nn.Embedding(3, 128) for _ in range(3)])
        self.o_embed = nn.ModuleList([nn.Embedding(3, 128) for _ in range(3)])
        # Only new sub-module: shallow local temporal scorer per modality.
        pad = kernel_size // 2
        self.scorer = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, scorer_hidden, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv1d(scorer_hidden, 1, kernel_size=1))
            for _ in range(3)])
        # Shared fusion + heads identical to B1.
        self.fuse = nn.Sequential(nn.Linear(384, 128), nn.GELU())
        self.cls = nn.Linear(128, 3)
        self.reg = nn.Linear(128, 1)

    @staticmethod
    def pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float().unsqueeze(-1)
        return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(self, text_state, audio, vision, p, o, unknown_excluded=False):
        p = p.long()
        o = o.long()
        xs = (text_state, audio, vision)
        hs, local = [], {}
        for m in range(3):
            base = self.proj[m](xs[m])
            pos = self.pos[m].unsqueeze(0)
            state = self.p_embed[m](p[:, m]) + self.o_embed[m](o[:, m]) + pos
            z = F.gelu(base + state)
            # B1 content mask (unknown_excluded=False default).
            cq = ((p[:, m] == 1) & (o[:, m] == 1)) if unknown_excluded \
                else ((p[:, m] != 0) & (o[:, m] != 0))
            # ONLY delta: learned local weights instead of masked mean.
            scores = self.scorer[m](z.transpose(1, 2)).squeeze(1)
            w = masked_softmax(scores, cq)
            content = (z * w.unsqueeze(-1)).sum(dim=1)
            # Existing state summary, unchanged from B1.
            state_summary = self.pool(F.gelu(state), p[:, m] != 0)
            hs.append(content + state_summary)
            local[m] = {"weights": w, "scores": scores, "content_mask": cq}
        h = self.fuse(torch.cat(hs, dim=1))
        logits = self.cls(h)
        intensity = 3.0 * torch.tanh(self.reg(h)).squeeze(1)
        return logits, intensity, local
