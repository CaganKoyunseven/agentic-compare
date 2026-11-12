# Result validation

Generated 2026-11-12 12:47 +03 by `python -m scripts.update_results`.

Expected: model `gpt-4o-mini-2024-07-18`, split `test`, mode `benchmark`, 3 reps, approaches baseline, reflective, multi_agent, tot, settings from `configs/experiment.toml`.

Checks on every run: finished with status ok; mode, split and model as expected; settings hash equals the frozen settings; seed as the runner sets it; item in the split; token and call totals equal the LLM call events; exactly one final answer, equal to the scored answer; a fresh official re-score gives the same verdict. An item is kept only if every approach has a valid run for every rep.

## Summary

- Runs in the results: 1440
- game24: 60 items
- humaneval: 60 items
- Runs left out: 361
- Items dropped from all approaches: 0

## Experiments used

| Experiment | Task | Runs in trace | Runs that passed the checks |
|---|---|---|---|
| game24-test-gpt-4o-mini-2024-07-18-openai-api-test-main | game24 | 720 | 540 |
| game24-test-gpt-4o-mini-2024-07-18-openai-api-test-tot-paper | game24 | 180 | 180 |
| humaneval-test-gpt-4o-mini-2024-07-18-openai-api-test-main | humaneval | 720 | 540 |
| humaneval-test-gpt-4o-mini-2024-07-18-openai-api-test-tot-paper | humaneval | 181 | 180 |

## Runs left out

| Task | Approach | Runs | Reason |
|---|---|---|---|
| game24 | tot | 180 | earlier settings (superseded by configs/experiment.toml) |
| humaneval | tot | 180 | earlier settings (superseded by configs/experiment.toml) |

| Experiment | Approach | Item | Rep | Run | Reason |
|---|---|---|---|---|---|
| humaneval-test-gpt-4o-mini-2024-07-18-openai-api-test-tot-paper | tot | HumanEval/23 | 0 | tot-HumanEval-23-r0-6eff05 | the run ended with status cancelled: KeyboardInterrupt:  |

## Items dropped from all approaches

None.

## Experiments not used (by design)

| Reason | Experiments |
|---|---|
| dev split (tuning only) | 13 |
| separate study (see self_consistency/) | 1 |
