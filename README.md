# agentic-compare

One LLM, four control flows: **Baseline**, **Reflective**, **Multi-agent** and
**Tree-of-Thoughts**. The same model and settings are used for each approach on Game of 24 and
HumanEval. We measure accuracy, cost (tokens, LLM calls) and wall time. The goal is to find
which approach is worth its cost for which kind of task, not to crown a single winner. A live
web demo shows the four approaches side by side and can also replay recorded traces without a
model.

**Status:** work in progress.

## Model and main metric

- **Model:** GPT-4o mini through the OpenAI API, pinned snapshot `gpt-4o-mini-2024-07-18`,
  for every benchmark run and the live demo. Our first plan was a 3B model running locally
  with Ollama, but on the Game of 24 dev set it solved almost no puzzles, so there was no
  difference left to compare between the approaches. The local Ollama backend is still
  supported (`LLM_BACKEND=ollama`).
- **Model check on the dev set (before any test-set run):** the baseline must solve roughly
  5-30% of Game of 24 puzzles (0% gives no signal, close to 100% leaves no room to improve).
  GPT-4o mini solved 4 of 36 baseline runs (11%) on the 12 dev puzzles.
- **Main metric:** Game of 24 success rate. HumanEval is a secondary task. We fixed this
  before running the test set: Game of 24 needs search and backtracking, which is what the
  four approaches do differently, and the model has room to improve on it. On HumanEval this
  model is expected to be close to its ceiling, so differences there are likely small.
- **Cost:** reported as tokens and LLM calls per run, as read from the traces. Wall time with
  a cloud API also depends on the network and server load, so treat it as approximate.
- **Prompts:** the Tree of Thoughts propose prompt for Game of 24 contains a short format
  example (the step format described under Data sources); the other approaches' prompts do
  not.

## Requirements

- Python 3.11 or 3.12
- For live mode and benchmark runs: an OpenAI API key, or [Ollama](https://ollama.com) with a
  local model. Replay mode needs neither.

## Quick start

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Replay demo (no Ollama needed):

```sh
python -m scripts.make_mock_traces   # only needed while traces/examples/ is empty
python -m agentic.demo --replay-only
```

Live demo: copy `.env.example` to `.env` and fill it in. For the OpenAI backend:

```
LLM_BACKEND=openai
OPENAI_API_KEY=<your key>
MODEL_NAME=gpt-4o-mini-2024-07-18
MACHINE_ID=openai-api
```

For a local model instead, set `LLM_BACKEND=ollama`, start Ollama and set `MODEL_NAME` to an
installed model. Then:

```sh
python -m agentic.demo
```

The launcher checks the backend, warms the model, starts the server on `http://127.0.0.1:8000`
and opens the browser. If the backend or the model is not available it starts in replay-only
mode. Live mode with the OpenAI backend needs internet; replay mode works offline.

Benchmark runs write traces to `traces/runs/`, for example:

```sh
python -m agentic.runner --task game24 --split dev --approach baseline --reps 3
```
Options: `--port`, `--no-browser`. Live runs are also saved to `traces/live/`. Demo settings
(lighter than the benchmark, fixed scenes, draw-card pool) are in `configs/demo.toml`; demo
timings are never used in the results.

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
