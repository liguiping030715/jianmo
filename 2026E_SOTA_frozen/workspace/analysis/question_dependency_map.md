# Question dependency map

```mermaid
flowchart LR
 A1[Attachment 1 raw clips + transcript] --> Q1[Q1 extraction and alignment rules]
 Q1 --> P[ID, time/position, valid mask, provenance contract]
 A2[Attachment 2 train/valid, one version] --> P2[Equivalent feature interface]
 P --> P2
 P2 --> Q2[Q2 local-gap predictive design]
 P2 --> C[Compare A/U, hard/soft/native correspondence]
 C --> Q2
 Q2 --> H[Availability/reliability + prediction interface]
 P2 --> Q3[Q3 complete-input prediction]
 H --> Q3
 A3[Attachment 3 unlabeled gaps] --> F2[Final Q2 inference only]
 Q2 --> F2
 A4[Attachment 4 unlabeled features + video] --> F3[Final Q3 inference and evidence mapping only]
 Q3 --> F3
```

The correspondence comparison is a deferred evidence gate, not a Stage 1.5 selection. Full-to-partial teaching is optional and must first pass a complete-input teacher check. Q3 attribution is computed on the selected predictive interface and checked by interventions; neither a gate nor attention value is accepted as its definition.

Q1's independently produced 100 features document extraction/traceability; official Q2/Q3 parameter fitting remains on Attachment 2 train. Shared objects are specifications, not automatically identical numerical features. Model selection and thresholds: Attachment 2 valid only. Attachment 2 `test` and Attachments 3–4 must not guide selection unless official instructions later explicitly permit it.
