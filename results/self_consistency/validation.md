# Self-consistency validation

Expected: model `gpt-4o-mini-2024-07-18`, split `test`, mode `benchmark`, 3 reps, settings from `configs/experiment.toml` `[approaches.self_consistency]`. Same run checks as the main results, plus: the number of samples is as configured and a recomputed majority vote gives the recorded final answer.

- Runs used: 180
- Items: 60
- Runs left out: 0
- Items dropped (a rep missing): 0
- Runs not used by design (dev split or demo): 2

