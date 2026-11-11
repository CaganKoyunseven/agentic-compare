# Results report

- Input file: `runs.csv`
- Model: gpt-4o-mini-2024-07-18
- Machine used for time: `openai-api`
- Split: test
- Items per task: Game of 24 60, HumanEval 60
- Repetitions per item: Game of 24 3, HumanEval 3

## Success rate

Mean accuracy over all repetitions; 95% bootstrap CI over items.

### Game of 24

| Approach | Success | 95% CI | n items |
|---|---|---|---|
| Baseline | 21.7% | [13.3, 30.6] | 60 |
| Reflective | 39.4% | [30.0, 49.4] | 60 |
| Multi-agent | 28.3% | [19.4, 37.8] | 60 |
| Tree-of-Thoughts | 32.8% | [23.3, 42.2] | 60 |

### HumanEval

| Approach | Success | 95% CI | n items |
|---|---|---|---|
| Baseline | 74.4% | [66.1, 82.2] | 60 |
| Reflective | 89.4% | [82.2, 95.6] | 60 |
| Multi-agent | 92.2% | [85.6, 97.2] | 60 |
| Tree-of-Thoughts | 95.6% | [90.6, 99.4] | 60 |

## Paired comparison with Baseline

Accuracy difference in percentage points over the items both approaches ran; 95% paired bootstrap CI. Solved = correct in at least half the repetitions.

### Game of 24

- Reflective minus Baseline = +17.8 points (95% CI [+11.1, +25.6]), n = 60 items
  - Items solved: only Reflective 11, only Baseline 0, both 11, neither 38
- Multi-agent minus Baseline = +6.7 points (95% CI [-1.1, +15.0]), n = 60 items (within noise, below 15 points)
  - Items solved: only Multi-agent 7, only Baseline 2, both 9, neither 42
- Tree-of-Thoughts minus Baseline = +11.1 points (95% CI [+0.6, +22.2]), n = 60 items (within noise, below 15 points)
  - Items solved: only Tree-of-Thoughts 13, only Baseline 4, both 7, neither 36

### HumanEval

- Reflective minus Baseline = +15.0 points (95% CI [+6.1, +23.9]), n = 60 items
  - Items solved: only Reflective 9, only Baseline 4, both 45, neither 2
- Multi-agent minus Baseline = +17.8 points (95% CI [+8.9, +26.7]), n = 60 items
  - Items solved: only Multi-agent 9, only Baseline 1, both 48, neither 2
- Tree-of-Thoughts minus Baseline = +21.1 points (95% CI [+13.9, +28.9]), n = 60 items
  - Items solved: only Tree-of-Thoughts 9, only Baseline 0, both 49, neither 2

## Cost

Means per run. Wall time uses only machine `openai-api`; tokens per correct answer is all tokens divided by the number of correct runs.

### Game of 24

| Approach | Mean tokens | Mean LLM calls | Mean wall time (s) | Tokens per correct |
|---|---|---|---|---|
| Baseline | 335 | 1.0 | 4.2 | 1,547 |
| Reflective | 1,503 | 3.9 | 15.3 | 3,811 |
| Multi-agent | 1,416 | 4.5 | 11.3 | 4,999 |
| Tree-of-Thoughts | 9,346 | 29.0 | 43.0 | 28,513 |

### HumanEval

| Approach | Mean tokens | Mean LLM calls | Mean wall time (s) | Tokens per correct |
|---|---|---|---|---|
| Baseline | 594 | 1.0 | 6.2 | 797 |
| Reflective | 2,802 | 4.1 | 16.0 | 3,133 |
| Multi-agent | 3,027 | 4.5 | 15.9 | 3,282 |
| Tree-of-Thoughts | 3,135 | 8.5 | 19.9 | 3,281 |

## Figures

- `success.png`
- `cumulative.png`
- `rep_spread.png`
- `paired.png`
- `cost_vs_accuracy.png`
- `time_and_calls.png`


Only runs that passed the checks are included: see `validation.md` (0 runs left out, 0 items dropped).
