"""Q2 E1: R1 shared-fusion head plus small latent restoration modules."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pilot_stage4 import PilotHead


class E1Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = PilotHead("B1")
        self.restore = nn.ModuleList([
            nn.Sequential(nn.Linear(264, 128), nn.GELU(), nn.Linear(128, 128))
            for _ in range(3)
        ])

    @staticmethod
    def neighbor_context(z: torch.Tensor, available: torch.Tensor):
        total = torch.zeros_like(z)
        count = torch.zeros_like(available, dtype=z.dtype)
        for shift in (-2, -1, 1, 2):
            shifted_z = torch.roll(z, shifts=shift, dims=1)
            shifted_ok = torch.roll(available, shifts=shift, dims=1)
            if shift > 0:
                shifted_ok[:, :shift] = False
            else:
                shifted_ok[:, shift:] = False
            total = total + shifted_z * shifted_ok.unsqueeze(-1)
            count = count + shifted_ok.to(z.dtype)
        return total / count.clamp_min(1.0).unsqueeze(-1), count / 4.0

    def forward(self, text_state, audio, vision, p, o, injected=None,
                restoration_enabled=True, diagnostics=False):
        p, o = p.long(), o.long()
        xs = (text_state, audio, vision)
        if injected is None:
            injected = torch.zeros_like(p, dtype=torch.bool)
        else:
            injected = injected.bool()
        if torch.any(injected & ~((p == 1) & (o == 0))):
            raise RuntimeError("restoration mask outside known P=1/O=0 synthetic gap")
        state, z, available = [], [], []
        for m in range(3):
            s = self.core.p_embed[m](p[:, m]) + self.core.o_embed[m](o[:, m]) \
                + self.core.pos[m].unsqueeze(0)
            state.append(s)
            z.append(F.gelu(self.core.proj[m](xs[m]) + s))
            available.append((p[:, m] == 1) & (o[:, m] == 1))
        restored = [None, None, None]
        if restoration_enabled and torch.any(injected):
            for m in range(3):
                if not torch.any(injected[:, m]):
                    continue
                local_mean, local_fraction = self.neighbor_context(z[m], available[m])
                others = [j for j in range(3) if j != m]
                cross_count = sum(available[j].to(z[m].dtype) for j in others)
                cross_sum = sum(z[j] * available[j].unsqueeze(-1) for j in others)
                cross_mean = cross_sum / cross_count.clamp_min(1.0).unsqueeze(-1)
                context = torch.cat([
                    local_mean, cross_mean, local_fraction.unsqueeze(-1),
                    (cross_count / 2.0).unsqueeze(-1),
                    F.one_hot(p[:, m], 3).to(z[m].dtype),
                    F.one_hot(o[:, m], 3).to(z[m].dtype),
                ], dim=-1)
                restored[m] = self.restore[m](context)
        hs, hs_untreated = [], []
        for m in range(3):
            q = (p[:, m] != 0) & (o[:, m] != 0)  # frozen R1 unknown-retained policy
            state_summary = self.core.pool(F.gelu(state[m]), p[:, m] != 0)
            untreated_content = self.core.pool(z[m], q)
            hs_untreated.append(untreated_content + state_summary)
            if restoration_enabled and restored[m] is not None:
                zeff = torch.where(injected[:, m].unsqueeze(-1), restored[m], z[m])
                content = self.core.pool(zeff, q | injected[:, m])
            else:
                content = untreated_content
            hs.append(content + state_summary)
        h = self.core.fuse(torch.cat(hs, dim=1))
        logits = self.core.cls(h)
        intensity = 3.0 * torch.tanh(self.core.reg(h)).squeeze(1)
        out = {"logits": logits, "intensity": intensity, "H": h}
        if diagnostics:
            out["z"] = z
            out["z_hat"] = restored
            out["H_untreated"] = self.core.fuse(torch.cat(hs_untreated, dim=1))
        return out


def local_loss(gap: dict, clean: dict, injected: torch.Tensor) -> torch.Tensor:
    parts = []
    counts = []
    for m in range(3):
        mask = injected[:, m]
        if gap["z_hat"][m] is not None and torch.any(mask):
            per = F.huber_loss(gap["z_hat"][m], clean["z"][m].detach(),
                               delta=1.0, reduction="none").mean(dim=-1)
            parts.append(per[mask].sum())
            counts.append(mask.sum())
    if not parts:
        return gap["H"].sum() * 0.0
    return sum(parts) / sum(counts)


def global_loss(gap: dict, clean: dict, injected: torch.Tensor) -> torch.Tensor:
    active = injected.any(dim=(1, 2))
    if not torch.any(active):
        return gap["H"].sum() * 0.0
    return (1.0 - F.cosine_similarity(gap["H"][active],
                                       clean["H"][active].detach(), dim=-1)).mean()
