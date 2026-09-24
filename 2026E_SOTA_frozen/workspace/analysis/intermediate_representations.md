# Intermediate representations and unresolved interfaces

1. **Sample provenance**: immutable ID, raw filename, clip duration, transcript, source version and processing log. Required by Q1/Q3 and all result CSVs.
2. **Modality-native sequence**: feature array, dimensionality, native rate or position, valid length and padding mask. Q1 creates from raw material; Attachment 2 supplies standardized arrays. Never equate these feature spaces without evidence.
3. **Temporal correspondence**: map from feature position to `[start,end]` seconds or word/frame identifier, with uncertainty if reconstructed. Aligned 50 provides corresponding sequence positions, not guaranteed exact timestamps. Unaligned 500 preserves separate positions; mapping must be established.
4. **Observation state**: separate valid `P`, observed `O`, and potential quality/reliability `R`. A feature vector of zeros can mean missing, padding, or a valid numeric value; context/metadata must disambiguate.
5. **Predictive state**: multimodal representation and outputs `(class probabilities, intensity)`. Algorithm unspecified. Include class/intensity consistency checks and calibrated class decision threshold chosen on valid.
6. **Explanation record**: declared modality intervention contribution, dominant modality, ranked local intervals, corresponding text/audio/video locator, and deletion/retention response. Normalize contributions only after specifying sign and denominator; negative evidence may exist.

The expanded mechanism scan introduces **correspondence uncertainty** as an optional field in object 3, and a **teacher confidence** diagnostic only if full-to-partial distillation is piloted. These do not change the core object. A counterfactual feature edit is not automatically a feasible raw-video edit. Modality contributions may be nonadditive because interactions exist; any allocation rule must state how interaction effects are handled.

Stage 2 questions: exact PKL keys/dtypes and lengths; padding side/value; whether local zero runs overlap padding/natural zeros; fields available in Attachments 3–4; split and ID overlap; alignment positions versus wall-clock time; sample XLSX/video matching; transcript token-to-time recoverability; whether Q1 100 clips overlap Attachment 2; tool and packaging feasibility. No answer is inferred merely from file names.
