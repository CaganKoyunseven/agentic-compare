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

- Game of 24: puzzle set from the Tree-of-Thoughts paper authors (Yao et al., 2023). Details TBD.
- HumanEval: OpenAI (Chen et al., 2021). Details TBD.

## License

MIT
