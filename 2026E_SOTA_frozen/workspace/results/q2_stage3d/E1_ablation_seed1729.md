# E1 seed-1729 diagnostic ablations

The same frozen training split, mask, optimizer protocol and train-calibrated lambda values were used; disabling a component only sets its weight/path to zero. Ablations cannot promote E1.

| Run | Restoration | Local loss | Global consistency | Clean F1 | Clean MAE | Gap F1 gain vs R1 | Gap MAE reduction vs R1 |
|---|---|---|---|---:|---:|---:|---:|
| E1-full | yes | yes | yes | 0.4908 | 0.6225 | -0.1224 | -0.0291 |
| E1-local-only | yes | yes | no | 0.5678 | 0.6170 | -0.0606 | -0.0153 |
| E1-consistency-only | no | no | yes | 0.4705 | 0.6174 | -0.1392 | -0.0140 |

Local-only has better clean F1 than full (0.5678 versus 0.4908) but still worse gap F1 than R1 by about 0.0606. Consistency-only predicts only 5 neutral cases among 728 clean valid samples and is not a viable rescue. These are diagnostics, not a post-result architecture search.
