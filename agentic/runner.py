"""Run benchmark experiments and write traces.

Usage:
    python -m agentic.runner --task game24 --split dev
    python -m agentic.runner --task game24 --split test --limit 30 --approach baseline,tot
    python -m agentic.runner --task humaneval --split test --limit 30

For each repetition and each item, the chosen approaches run back to back in a fixed order
(interleaved), so slow drift of the machine (heat, background load) does not favour one
approach. All events of one experiment go to one JSONL trace:
    traces/runs/<task>-<split>-<model>-<machine_id>[-<tag>]/trace.jsonl
and runs.csv next to it (one row per finished run, contract 6.3) is derived from that trace.

Resumable: running the same command again skips every (approach, item, rep) that already
ended with status "ok". Ctrl-C ends the current run as "cancelled" and stops. If the settings
of an approach changed since the earlier runs, the runner refuses to mix them; use --tag.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic.approaches.base import Approach
from agentic.approaches.baseline import Baseline
from agentic.approaches.multi_agent import MultiAgent
from agentic.approaches.reflective import Reflective
from agentic.approaches.tot import TreeOfThoughts
from agentic.config import ROOT, load_toml, machine_id, model_name
from agentic.events import FinalAnswer, LLMCall, RunEnd, RunStamp, RunStart
from agentic.llm.base import LLMClient
from agentic.llm.ollama import OllamaClient, OllamaError
from agentic.tasks.base import Item, Task
from agentic.tasks.game24 import Game24Task
from agentic.tasks.humaneval import HumanEvalTask
from agentic.traces import TraceWriter, group_runs, read_trace

ORDER = ["baseline", "reflective", "multi_agent", "tot"]
APPROACHES: dict[str, Approach] = {"baseline": Baseline(), "reflective": Reflective(),
                                   "multi_agent": MultiAgent(), "tot": TreeOfThoughts()}
TASKS: dict[str, type] = {"game24": Game24Task, "humaneval": HumanEvalTask}
TRACES_DIR = ROOT / "traces" / "runs"
CSV_COLUMNS = ["run_id", "task", "approach", "item_id", "difficulty", "rep", "split", "correct",
               "tokens_in", "tokens_out", "llm_calls", "wall_ms", "model", "machine_id"]


@dataclass(frozen=True)
class Setup:
    """What stays the same for every run of one experiment."""
    task: Task
    split: str
    model: str
    machine_id: str
    base_seed: int
    llm_settings: dict[str, Any]
    approach_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    ollama_version: str | None = None
    model_digest: str | None = None

    def config(self, approach: str) -> dict[str, Any]:
        """Settings recorded on run_start: shared LLM settings plus the approach's own."""
        return {"llm": self.llm_settings, "approach": self.approach_configs.get(approach, {})}


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def run_seed(base_seed: int, rep: int, index: int) -> int:
    """Shared by all approaches for the same (item, rep); different for every rep."""
    return base_seed + 1000 * rep + index


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def experiment_dir(task: str, split: str, model: str, machine: str, tag: str = "") -> Path:
    name = "-".join(slug(part) for part in (task, split, model, machine, tag) if part)
    return TRACES_DIR / name


# ---- One run ----

class RunCancelled(Exception):
    """Raised inside an approach when somebody stopped the run on purpose (not a failure)."""


def run_one(setup: Setup, approach: Approach, item: Item, index: int, rep: int,
            llm: LLMClient, writer: TraceWriter, mode: str = "benchmark") -> RunEnd:
    """Run one approach on one item and write all its events. Returns the run_end event.
    An error inside the approach is recorded and the experiment goes on; Ctrl-C and a lost
    Ollama connection are recorded and then stop the experiment."""
    seed = run_seed(setup.base_seed, rep, index)
    config = setup.config(approach.name)
    run_id = f"{approach.name}-{slug(item.item_id)}-r{rep}-{uuid.uuid4().hex[:6]}"
    stamp = RunStamp(run_id=run_id, approach=approach.name, task=setup.task.name,
                     item_id=item.item_id, rep=rep, model=setup.model,
                     machine_id=setup.machine_id)
    writer.write(stamp.next(), RunStart(
        split=setup.split, mode=mode, seed=seed, config=config,
        config_hash=config_hash(config), ollama_version=setup.ollama_version,
        model_digest=setup.model_digest, payload=item.payload, difficulty=item.difficulty))

    totals = {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0}
    answer = None
    start = time.perf_counter()
    try:
        for body in approach.run(item, setup.task, llm, seed=seed,
                                 config=config["approach"]):
            writer.write(stamp.next(), body)
            if isinstance(body, LLMCall):
                totals["tokens_in"] += body.tokens_in
                totals["tokens_out"] += body.tokens_out
                totals["llm_calls"] += 1
            elif isinstance(body, FinalAnswer):
                answer = body.answer
        if answer is None:
            raise RuntimeError(f"{approach.name} ended without a final_answer event")
    except (KeyboardInterrupt, Exception) as e:
        status = "cancelled" if isinstance(e, (KeyboardInterrupt, RunCancelled)) else "error"
        end = RunEnd(status=status, answer=answer, error=f"{type(e).__name__}: {e}",
                     wall_ms=_ms_since(start), **totals)
        writer.write(stamp.next(), end)
        if isinstance(e, (KeyboardInterrupt, OllamaError)):
            raise
        return end

    # Approach time ends here: official scoring is not part of any approach's cost.
    wall_ms = _ms_since(start)
    score = setup.task.score(item, answer)
    end = RunEnd(status="ok", answer=answer, correct=score.correct,
                 score_message=score.message, wall_ms=wall_ms, **totals)
    writer.write(stamp.next(), end)
    return end


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# ---- An experiment: many runs, resumable ----

def finished_runs(trace_path: Path) -> dict[tuple[str, str, int], str]:
    """(approach, item_id, rep) -> config_hash for every run that ended with status ok."""
    if not trace_path.exists():
        return {}
    done = {}
    for run in group_runs(read_trace(trace_path)).values():
        env, start = run[0]
        end = run[-1][1]
        if start.type == "run_start" and end.type == "run_end" and end.status == "ok":
            done[(env.approach, env.item_id, env.rep)] = start.config_hash
    return done


def run_experiment(setup: Setup, approaches: list[Approach], items: list[Item], reps: int,
                   llm: LLMClient, trace_path: Path,
                   log: Callable[[str], None] = print) -> None:
    """Run every (rep, item, approach) that is not finished yet. Items must be in split
    order (their position sets the seed). runs.csv is rewritten at the end, also after
    Ctrl-C or an error."""
    done = finished_runs(trace_path)
    for approach in approaches:
        current = config_hash(setup.config(approach.name))
        if any(a == approach.name and h != current for (a, _, _), h in done.items()):
            raise RuntimeError(
                f"The settings of {approach.name} changed since the runs in {trace_path}. "
                "Start a new experiment folder with --tag, or restore the settings.")

    todo = [(rep, index, item, approach)
            for rep in range(reps)
            for index, item in enumerate(items)
            for approach in approaches
            if (approach.name, item.item_id, rep) not in done]
    total = reps * len(items) * len(approaches)
    log(f"{trace_path}\n{len(todo)} runs to do, {total - len(todo)} already done")
    if not todo:
        write_csv(trace_path, trace_path.with_name("runs.csv"))
        return

    with TraceWriter(trace_path) as writer:
        try:
            for n, (rep, index, item, approach) in enumerate(todo, 1):
                end = run_one(setup, approach, item, index, rep, llm, writer)
                verdict = {True: "correct", False: "wrong"}.get(end.correct, end.status)
                log(f"[{n}/{len(todo)}] rep {rep} {item.item_id} {approach.name:<11} "
                    f"{verdict:<9} {end.llm_calls} calls {end.tokens_in + end.tokens_out} tok "
                    f"{end.wall_ms / 1000:.1f}s" + (f"  {end.error}" if end.error else ""))
        finally:
            write_csv(trace_path, trace_path.with_name("runs.csv"))


def write_csv(trace_path: Path, csv_path: Path) -> int:
    """Derive the results table (contract 6.3) from a trace. Only benchmark runs with status
    ok are rows, so demo timings can never enter the results. Returns the number of rows."""
    rows: dict[tuple, dict] = {}
    for run in group_runs(read_trace(trace_path)).values():
        env, start = run[0]
        end = run[-1][1]
        if start.type != "run_start" or end.type != "run_end" or end.status != "ok":
            continue
        if start.mode != "benchmark":
            continue
        # A later run of the same (model, approach, item, rep) replaces an earlier one.
        rows[(env.model, env.approach, env.item_id, env.rep)] = {
            "run_id": env.run_id, "task": env.task, "approach": env.approach,
            "item_id": env.item_id, "difficulty": start.difficulty or "", "rep": env.rep,
            "split": start.split, "correct": int(bool(end.correct)),
            "tokens_in": end.tokens_in, "tokens_out": end.tokens_out,
            "llm_calls": end.llm_calls, "wall_ms": end.wall_ms, "model": env.model,
            "machine_id": env.machine_id,
        }
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())
    return len(rows)


# ---- Command line ----

def choose_approaches(spec: str) -> list[Approach]:
    """Approaches to run, in the fixed ORDER."""
    names = ORDER if spec == "all" else [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in ORDER]
    if unknown:
        raise SystemExit(f"Unknown approach {unknown}; choose from {ORDER} or 'all'")
    return [APPROACHES[n] for n in ORDER if n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark and write traces.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--approach", default="all",
                        help="'all' (default) or a comma list, e.g. baseline,tot")
    parser.add_argument("--reps", type=int, help="repetitions (default from experiment.toml)")
    parser.add_argument("--limit", type=int, help="only the first N items of the split")
    parser.add_argument("--tag", default="", help="suffix for a new experiment folder")
    args = parser.parse_args(argv)

    settings = load_toml("experiment")
    approaches = choose_approaches(args.approach)
    task = TASKS[args.task]()
    items = task.load_items(args.split)[:args.limit]
    model, machine = model_name(), machine_id()
    llm = OllamaClient(model, **settings["llm"])
    version, digest = llm.server_info()
    if digest is None:
        raise SystemExit(f"Model {model} is not installed in Ollama (ollama pull {model})")
    setup = Setup(task=task, split=args.split, model=model, machine_id=machine,
                  base_seed=settings["seed"], llm_settings=settings["llm"],
                  approach_configs=settings.get("approaches", {}),
                  ollama_version=version, model_digest=digest)
    trace_path = experiment_dir(args.task, args.split, model, machine, args.tag) / "trace.jsonl"

    # Warm-up call (not recorded), so the first run does not pay the model load time.
    warm = llm.chat([{"role": "user", "content": "Reply with OK."}])
    print(f"Model {model} warm (load {warm.load_ms} ms), Ollama {version}")
    try:
        run_experiment(setup, approaches, items, args.reps or settings["reps"], llm,
                       trace_path)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
        return 130
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
