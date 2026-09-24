# Stage 1.3 — refreshed negative-transfer audit

Sources: official statement (current requirements), expanded SOTA matrix (scientific mechanisms), and exact full-library retrieval in `historical_transfer_plan.md` (structural experience). Status is for architectural admission, not a final model choice.

| Transfer | Structural match / current evidence | Transferable / excluded | Falsifier, feasibility and integrity | Status |
|---|---|---|---|---|
| Source-linked coordinate, valid/observed/reliability states | Official A/U lengths, padding, L gaps, Q3 media evidence | Explicit provenance and three distinct states; no invented timestamps | PKL/media cannot support claimed resolution; low cost, no leakage | ALLOW_TO_ARCHITECTURE |
| Hard aligned versus modality-native asynchronous | Both official versions supplied | Compare information loss, compute and localization; no presumed winner | Stage 2/pilot shows failure; one version throughout pipeline | HOLD_FOR_DATA_EVIDENCE |
| Soft/elastic or text-anchored alignment | Heterogeneous rates, official U | Test learned correspondence; do not force text to explain every acoustic/visual cue | Gap/evidence smearing or no gain; medium/high cost | HOLD_FOR_DATA_EVIDENCE |
| Direct local-mask fusion | Official L continuous zeros | Condition on genuine observed intervals; do not interpret zero as neutral | No improvement over simple masked baseline; low/medium cost | ALLOW_TO_ARCHITECTURE |
| Full-to-partial distillation, CMAD + 2025D historical card | Full official training inputs can simulate partial gaps | Teacher consistency only; W source protocol and meteorological weights excluded | Weak teacher, no L robustness gain; official train teacher only | ALLOW_COMPONENT_ONLY |
| Reliability/proxy, P-RMF/EBMC | Local loss creates uneven evidence | Calibrated source quality concept; no automatic Gaussian proxy | Miscalibration/no gain; medium/high cost | HOLD_FOR_DATA_EVIDENCE |
| Dynamic expert/gate, EMOE | Variable sample modality informativeness | Adaptive fusion; weights do not prove explanatory contribution | Unstable or no gain; medium/high cost | ALLOW_COMPONENT_ONLY |
| Reconstruction/diffusion, FUSE-Net/HyperEF | Missing inputs broadly | At most inexpensive reconstruction diagnostic; whole-dialogue diffusion excluded | Hallucinated gaps, high cost, no benefit; no extra data | REJECT |
| Invariant factors, CmIR/MISR | Potential conflict/noise | Consider only after observing actual environment shift/conflict | No shift or label-preservation evidence; medium/high cost | HOLD_FOR_DATA_EVIDENCE |
| Modality/interval deletion, retention, stability | Q3 demands quantifiable source-linked evidence | Model dependence tests and equal-budget controls; no psychological causality | Weak/unstable effects, deletion artifacts; inference-only on special set | ALLOW_TO_ARCHITECTURE |
| Gradient/counterfactual attribution | Differentiable predictive output possible | Compare as diagnostic if feasible; no raw-signal counterfactual fabricated from features | Saturation, implausible changes or no source mapping | ALLOW_COMPONENT_ONLY |
| Historical 2024E raw-video chain / 2025E explanation chain | Intermediate evidence + progressive questions | ID/provenance and split-safe evidence audit; no traffic/bearing algorithm | Fails independent source check; historical scores not benchmarks | ALLOW_TO_ARCHITECTURE |
| Historical 2025F “multimodal” keyword | Only broad name overlaps | Nothing specific | Spatial aesthetics not sentiment/temporal gaps | REJECT |
| Transformer because multimodal; larger/newer because prestigious | No structural evidence | None | Simpler baseline can test same need; budget risk | REJECT |
| Attention as explanation | Q3 requires intervention faithfulness | Attention may visualize internals only | No prediction response under deletion | REJECT |
| `text_bert` as fourth modality; zero as observed/neutral | Contradicts official semantics | None | Double-counted text or missing/padding confusion | REJECT |
| DANN/MMD because modalities differ | 2025E actually had source-target shift; no such finding yet here | Shift-detection discipline only | No measured shift; special-set tuning forbidden | REJECT |
| Extra sentiment datasets; tune on Attachment 3/4 | Explicitly prohibited | None | Competition integrity violation | REJECT |

Every admitted component must survive its listed falsifier on official train/valid data. Rejected components cannot silently reappear in Stage 3.
