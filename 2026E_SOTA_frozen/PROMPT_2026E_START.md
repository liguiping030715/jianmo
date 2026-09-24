# 2026 E — 第一轮 Codex Prompt

把下面内容整段发给 Codex。第一轮只做 Stage 1–1.5。

---

We are now in `COMPETITION_SOLVING` mode for the official 2026 Huawei Cup Problem E, “复杂场景下多模态情感预测的数学建模与算法设计”.

Read `AGENTS.md` and the relevant skills.

Use only:

1. the official 2026E problem statement and official attachments/data;
2. legitimate general academic literature about the underlying scientific problems;
3. the frozen/synced historical Pattern Library as structural experience.

Do **not** search for or use 2026E solution posts, other teams' approaches, answer summaries, target-specific solution repositories, or target-specific code produced after release.

## Stage 1 — Problem Analysis

Create:

- `workspace/analysis/problem_breakdown.md`
- `workspace/analysis/variables_and_constraints.md`
- `workspace/analysis/assumptions.md`
- `workspace/analysis/task_fingerprint.md`

For Q1/Q2/Q3 record exact deliverables, official inputs, outputs, labels/targets, constraints, intermediate representations, uncertainty, leakage risks, validation requirements, and handoff to the next question.

Pay special attention to:

- Q1: text/audio/vision feature definition, heterogeneous temporal resolution, alignment, valid length, padding/mask semantics, sample-ID traceability, reproducibility, and mapping back to raw video/text/audio;
- Q2: **local continuous temporal missingness**, not merely complete-modality absence; missing modality/type/location/duration/ratio; polarity classification + intensity regression; robustness degradation curves; zero-valued missing intervals must not be treated as ordinary observations;
- Q3: sample-level modality contribution, major reference modality, local temporal evidence, traceability to text phrase/audio interval/visual frame, and faithfulness validation beyond attention visualization.

Do not select a final model.

## Stage 1.1 — Current SOTA Literature Scan

Load `.agents/skills/sota-literature-scan/SKILL.md`.

Search the **underlying scientific problems only**, not the competition solution.
Prioritize 2024–2026 peer-reviewed work and the official problem's cited papers, then expand through primary sources.

For Q1 search mechanisms for multimodal temporal representation/alignment and aligned vs unaligned/asynchronous sequences.

For Q2 search mechanisms for incomplete multimodal learning, **partial temporal missing intervals**, mask-aware/dynamic reliability fusion, distillation/reconstruction/proxy/MoE/uncertainty/invariant representations. For every paper distinguish whole-modality missingness from local temporal missingness.

For Q3 search mechanisms for multimodal explainability, modality attribution, temporal evidence localization, perturbation/counterfactual faithfulness, sufficiency/comprehensiveness, and evidence deletion/retention tests. Do not equate attention weight with explanation.

Create:

- `workspace/references/sota_literature_scan.md`
- `workspace/references/sota_method_matrix.md`

Classify each serious method as `DIRECTLY_USEFUL`, `USEFUL_COMPONENT_ONLY`, or `NOT_SUITABLE_FOR_THIS_COMPETITION`.

Do not implement yet.

## Stage 1.2 — Historical Pattern Retrieval

Load `.agents/skills/historical-pattern-retrieval/SKILL.md`.

Retrieve by structure, not topic name.
Use historical cases only for decomposition, intermediate representations, baseline discipline, validation, robustness, question-chain design, and writing logic.

Create:

`workspace/analysis/historical_transfer_plan.md`

Include both useful and rejected analogies.

## Stage 1.3 — Negative-Transfer & Evidence Audit

Load `.agents/skills/negative-transfer-audit/SKILL.md`.

Create:

`workspace/analysis/negative_transfer_audit.md`

Explicitly audit/reject unsupported moves such as:

- multimodal -> Transformer automatically;
- missing data -> diffusion/GAN automatically;
- attention -> explanation;
- larger/newer model -> better;
- aligned must be better than unaligned;
- zero feature -> neutral information;
- pretrained representation -> a fourth modality;
- DANN/MMD merely because modalities differ;
- extra sentiment datasets when official rules prohibit them;
- tuning on unlabeled final special test sets.

Every allowed component needs a falsification condition.

## Stage 1.5 — Solution Architecture

Load `.agents/skills/solution-architecture/SKILL.md`.

Create:

- `workspace/analysis/solution_story.md`
- `workspace/analysis/question_dependency_map.md`
- `workspace/analysis/intermediate_representations.md`
- `workspace/analysis/validation_blueprint.md`

Test whether the natural cross-question representation is approximately:

`raw text/audio/video -> modality-specific temporal representation -> common temporal coordinate + masks -> modality availability/reliability -> robust fusion -> shared emotional representation -> polarity/intensity -> modality contribution + local evidence`

Do not assume this is correct; verify it against the official data interface.

For `aligned_50` vs `unaligned_50`, do not decide by intuition. Compare temporal information loss, alignment convenience, local-missingness localization, explanation localization, masking semantics, compute cost, complexity, and reproducibility. Leave the final choice to Stage 2/pilot unless the official data make it unambiguous.

Validation blueprint must include:

- Q1 coverage/ID/length/mask/alignment traceability + at least one manual typical-sample verification;
- Q2 Accuracy/F1 and MAE/Pearson plus robustness curves over missing type/location/duration/ratio;
- Q3 faithfulness checks such as modality deletion, important-interval masking, evidence retention/sufficiency, comprehensiveness, perturbation stability, and prediction drop after deleting claimed evidence.

Then STOP.

Do not enter Stage 2 automatically.
Do not train a final architecture.
Do not draft the paper.

Print a final summary of:

1. hardest issue in Q1;
2. hardest issue in Q2;
3. hardest issue in Q3;
4. proposed cross-question representation;
5. top recent-literature mechanism candidates;
6. top historical patterns;
7. top negative-transfer risks;
8. unresolved questions requiring actual PKL/video/XLSX inspection.
