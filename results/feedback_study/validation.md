# Feedback study validation

Expected: model `gpt-4o-mini-2024-07-18`, split `test`, mode `benchmark`, 3 reps, settings from `configs/experiment.toml` (`[approaches.self_refine]`, `[approaches.debate]`, `[approaches.retry]`). Same run checks as the main results. Each approach keeps only items with a valid run for every rep.

- Runs used: 540
  - self_refine: 180
  - debate: 180
  - retry: 180
- Runs left out: 0
- Items dropped (a rep missing): 0
- Runs not used by design (dev split or demo): 23

