# Stage 1.2 — full Pattern Library retrieval for 2026E

## Library verification

`workspace/knowledge/INDEX.md` records 45 L1 paper cards, including 12 L3 deep reads, 12 same-problem syntheses, six reusable families and three writing guides. This workspace initially contained only seed files. `python scripts/sync_jianmo.py` completed successfully and merged the Pattern Library from `https://github.com/liguiping030715/jianmo.git`; while it was downloading, the same library was also copied from the adjacent checkout of that repository. Local verification found 45 paper cards plus `_TEMPLATE.md`, 12 syntheses, six families plus the seed, and three writing files plus the seed. The following retrieval uses full-library files. L1 cards provide structural cues; detailed mechanism critique relies on marked L3 cards.

Fingerprint searched: heterogeneous synchronized/asynchronous signals; source-to-feature traceability; supervised classification and regression; local observation failure; uncertainty; progressive question chain; source-linked explanations. Retrieval was not based on problem letter.

## Exact files retrieved and bounded transfer

| Files | Structural match and transfer | Non-transferable detail / falsifier |
|---|---|---|
| `reusable_patterns/signal_image_space.md`; `pattern_cards/2024/E24102870008.md`; `same_problem_synthesis/2024_E.md` | Raw video → calibrated intermediate coordinate → downstream decision; independent video segment checks. Q1 needs source-linked positions. | Traffic tracking, road thresholds and perspective geometry do not describe emotion. Reject if positions cannot map to media. |
| `reusable_patterns/prediction.md`; `pattern_cards/2025/D题-2-多源融合.md`; `same_problem_synthesis/2025_D.md` | Heterogeneous sources with variable availability; full-source/limited-source comparison and teacher limits. Q2 distillation hypothesis. | Atmospheric physics, weights and performance. Reject if teacher is weak or no local-gap benefit. |
| `reusable_patterns/evidence_gate.md`; `pattern_cards/2025/E题-3-智能诊断.md`; `same_problem_synthesis/2025_E.md` | Split-safe normalization, prediction versus proxy evidence, layered explanation checks. | Bearing frequencies; DANN/MMD without measured shift; SHAP as proof. Reject explanation failing perturbation. |
| `reusable_patterns/evaluation.md`; `pattern_cards/2025/F题-量化分析.md`; `same_problem_synthesis/2025_F.md` | Define modality importance operationally and test sensitivity. | Garden weights/AHP/aesthetic proxies. Reject unstable contribution ranking. |
| `reusable_patterns/domain_adaptation.md`; `pattern_cards/2025/B题-机器学习MIMO.md`; `same_problem_synthesis/2025_B.md` | Check grouped dependence and leakage before adapting. | Wireless equations and automatic domain adaptation. Reject absent verified shift. |
| `reusable_patterns/prediction.md`; `pattern_cards/2024/E24102870008.md` | Baseline then component ablation; avoid stacked complexity without evidence. | Its specific CNN/GRU/attention and thresholds. Reject additions without gain. |

All six families were searched; `optimization.md` has no positive analogue because 2026E has no resource/routing decision. `writing_patterns/question_section_patterns.md` and `writing_patterns/validation_patterns.md` were retrieved for claim-to-evidence organization, not wording. `pattern_cards/2025/F题-多模态融合.md` was explicitly rejected as a keyword match: garden spatial aesthetics do not imply sentiment fusion.

Open current-data questions: masks, ID overlap, time-to-media map, grouped dependence, teacher reliability, and attribution stability. Historical papers cannot establish these 2026E facts.
