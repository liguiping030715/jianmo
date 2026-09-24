# Validation blueprint (prespecified before experiments)

| Claim | Evidence / metric | Split and leakage guard |
|---|---|---|
| Q1 covers all originals | 100-row inventory: `(video_id,clip_id)`, media and output presence, feature dimensions, valid lengths, logs; duplicate/missing report | Attachment 1 only; retain all records |
| Q1 temporal correspondence | One or more manually checked typical samples showing phrase, audio interval, frame interval, source offsets and feature positions; alignment error/uncertainty where possible | Do not assert word timestamps from Attachment 2 PKL |
| Q1 reproducibility | Versioned tools, parameters, deterministic IDs, hashes and rerun instructions | Raw immutable |
| Q2 predicts | Accuracy, declared F1 average, MAE, Pearson; class confusion and intensity error by class | Attachment 2 valid; train-only scaling, valid-only selection |
| Q2 handles local gaps | Fixed-seed synthetic contiguous gaps varying modality, position, duration, ratio and simultaneous modalities; full-input versus gap degradation curves, sample counts and confidence intervals | Inject only inside valid lengths; keep missing separate from padding/true zero; no Attachment 3 tuning |
| Q2 mechanism helps | Same splits, gap draws, metrics and budget for simple masked baseline and candidate; ablations on mask and reliability | Reject if no stable gain |
| Correspondence choice is justified | Compare A/U on temporal information retained, gap localization, source mapping resolution, predictive metrics and cost; compare hard/soft/native only where feasible | Same official splits and feature version through training and special inference |
| Teacher helps partial input | Check full-input teacher validation error/calibration, then partial-input consistency and held-out local-gap performance | Teacher trained on train only; no teacher claims from Attachment 3 labels |
| Q3 predicts | Same Accuracy/F1 and MAE/Pearson on complete valid inputs; error cases | Attachment 2 valid only |
| Q3 contribution is faithful | Per-modality deletion/change in target probability or intensity, important-interval deletion, retained-evidence sufficiency, removed-evidence comprehensiveness, perturbation stability; compare random/equal-length intervals | Recompute through same inference pipeline; control deletion artifacts; no label-guided evidence on special test |
| Q3 evidence is traceable | Audit source locator against text phrase/audio time/frame for representative valid and Attachment 4 samples; record ambiguous mappings | Unlabeled Attachment 4 for final inference only; no performance claims from it |
| Attribution is not an artifact | Compare ablation, gradient and perturbation rankings when feasible; equal-length random intervals, realistic replacements, repeated small perturbations and modality interactions | Report disagreement; attention/gate weights alone do not count |

Pearson undefined for constant predictions/labels must be reported explicitly. Macro versus weighted F1, intervention baseline, interval budget and aggregation must be fixed before comparison. Deletion measures model reliance, not psychological causality. All quantitative claims later require saved machine-readable outputs.
