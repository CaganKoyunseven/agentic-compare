# agentic-compare

One small local LLM, four control flows: **Baseline**, **Reflective**, **Multi-agent** and
**Tree-of-Thoughts**. The same model and settings are used for each approach on Game of 24 and
HumanEval. We measure accuracy, cost (tokens, LLM calls) and wall time. The goal is to find
which approach is worth its cost for which kind of task, not to crown a single winner. A live
web demo shows the four approaches side by side and can also replay recorded traces without a
model.

**Status:** work in progress.

## Requirements

- Python 3.11 or 3.12
- [Ollama](https://ollama.com) (only for live mode; replay mode works without it)

## Quick start

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Replay demo (no Ollama needed): TBD

Live demo: TBD

## Repository layout

```
agentic/     Python package: LLM client, approaches, tasks, sandbox, runner, analysis, server
configs/     Experiment and demo settings (TOML)
data/        Datasets and dev/test splits
traces/      Recorded event traces (small examples are committed)
results/     Figures and summary tables
scripts/     Data download, speed test, smoke test, mock traces
tests/       Unit tests
```

## Data sources

- Game of 24: `data/game24/24.csv`, 1362 puzzles from the Tree of Thoughts authors
  (Yao et al., 2023, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
  repository [princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm),
  MIT licence (copy in `data/game24/LICENSE`). The authors collected the puzzles from 4nums.com.
  Re-download with `python -m scripts.fetch_data` (pinned commit, SHA-256 checked).
  Our approaches and prompts are our own. The Tree of Thoughts prompts for Game of 24
  (`agentic/prompts/game24_tot_*.txt`) follow the step format of the paper
  ("a op b = c (left: ...)", sure/likely/impossible) with our own wording and examples.
- HumanEval: `data/humaneval/HumanEval.jsonl.gz`, 164 Python problems from OpenAI
  (Chen et al., 2021, "Evaluating Large Language Models Trained on Code"), repository
  [openai/human-eval](https://github.com/openai/human-eval), MIT licence (copy in
  `data/humaneval/LICENSE`). Re-download with `python -m scripts.fetch_humaneval`.
  Final scores come only from the official tests, which no approach ever sees.

## License

MIT
