// One column per approach: status chip, counters, model output box, approach view, and the
// answer card with the official verdict. Columns only show recorded numbers from the events.

import { officialVerdict } from "./badges.js";
import { ChatView } from "./chat.js";
import { clear, el, formatInt, formatSeconds, scrollToEnd } from "./dom.js";
import { TreeView } from "./tree.js";
import { BaselineView, ReflectiveView } from "./views.js";

export const APPROACHES = [
  { id: "baseline", name: "Baseline", short: "Baseline", color: "#6B7280", View: BaselineView },
  { id: "reflective", name: "Reflective", short: "Reflective", color: "#2563EB",
    View: ReflectiveView },
  { id: "multi_agent", name: "Multi-agent", short: "Multi-agent", color: "#D97706",
    View: ChatView },
  { id: "tot", name: "Tree-of-Thoughts", short: "ToT", color: "#059669", View: TreeView },
];

const CHIP_TEXT = { waiting: "waiting", running: "running", correct: "correct",
  wrong: "wrong", error: "error" };
const TYPE_CHARS_PER_S = 90;

export function statusOf(runEnd) {
  if (runEnd.status !== "ok") return "error";
  if (runEnd.correct === true) return "correct";
  if (runEnd.correct === false) return "wrong";
  return "error";
}

// Types out a model response in replay; appends token text in live mode.
class OutputBox {
  constructor(getSpeed) {
    this.getSpeed = getSpeed;
    this.root = el("div", "output");
    this.head = el("div", "output-head", "Model output");
    this.text = el("pre", "output-text");
    this.root.append(this.head, this.text);
    this.callId = null;
    this.streamed = new Set();   // call_ids that arrived as live tokens
    this.frame = null;
  }

  stopTyping() {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
      this.text.textContent = this.full;
    }
  }

  setHead(role) {
    this.head.textContent = role ? `Model output · ${role}` : "Model output";
  }

  token(ev) {
    if (this.callId !== ev.call_id) {
      this.stopTyping();
      this.callId = ev.call_id;
      this.text.textContent = "";
      this.setHead(null);
    }
    this.streamed.add(ev.call_id);
    this.text.textContent += ev.text;
    scrollToEnd(this.text);
  }

  call(ev) {
    this.stopTyping();
    this.setHead(ev.role);
    this.callId = ev.call_id;
    this.full = ev.response;
    if (this.streamed.has(ev.call_id)) {
      this.text.textContent = ev.response;   // live: already typed by the tokens
      scrollToEnd(this.text);
      return;
    }
    const start = performance.now();
    const step = (now) => {
      const speed = Math.max(this.getSpeed(), 0.25);
      const n = Math.floor(((now - start) / 1000) * TYPE_CHARS_PER_S * speed);
      this.text.textContent = this.full.slice(0, n);
      scrollToEnd(this.text);
      this.frame = n < this.full.length ? requestAnimationFrame(step) : null;
    };
    this.text.textContent = "";
    this.frame = requestAnimationFrame(step);
  }

  reset() {
    this.stopTyping();
    this.callId = null;
    this.full = "";
    this.streamed.clear();
    this.text.textContent = "";
    this.setHead(null);
  }
}

export class Column {
  constructor(approach, getSpeed) {
    this.approach = approach;
    this.root = el("section", "column");
    this.root.style.setProperty("--approach", approach.color);

    const head = el("header", "column-head");
    this.chip = el("span", "chip");
    head.append(el("h2", "column-title", approach.name), this.chip);

    const counters = el("div", "counters");
    this.tokens = this.counter(counters, "tokens");
    this.calls = this.counter(counters, "LLM calls");
    this.time = this.counter(counters, "time");

    this.output = new OutputBox(getSpeed);
    this.body = el("div", "column-body");
    this.root.append(head, counters, this.output.root, this.body);
    this.reset();
  }

  counter(parent, label) {
    const box = el("div", "counter");
    const value = el("div", "counter-value", "0");
    box.append(value, el("div", "counter-label", label));
    parent.append(box);
    return value;
  }

  setStatus(status) {
    this.status = status;
    this.chip.className = `chip ${status}`;
    this.chip.textContent = CHIP_TEXT[status];
    this.root.classList.toggle("active", status === "running");
  }

  showCounters(tokens, calls, ms) {
    this.tokens.textContent = formatInt(tokens);
    this.calls.textContent = formatInt(calls);
    this.time.textContent = formatSeconds(ms);
  }

  reset() {
    if (this.view && this.view.destroy) this.view.destroy();
    this.view = new this.approach.View();
    this.final = el("div", "final");
    this.answer = null;
    clear(this.body);
    this.body.append(this.view.root, this.final);
    this.output.reset();
    this.task = null;
    this.sumTokens = 0;
    this.sumCalls = 0;
    this.runEnd = null;
    this.showCounters(0, 0, 0);
    this.setStatus("waiting");
  }

  handle(ev) {
    switch (ev.type) {
      case "run_start":
        this.task = ev.task;
        this.setStatus("running");
        break;
      case "llm_call":
        this.sumTokens += ev.tokens_in + ev.tokens_out;
        this.sumCalls += 1;
        this.output.call(ev);
        break;
      case "token":
        this.output.token(ev);
        break;
      case "final_answer":
        this.showAnswer(ev.answer);
        break;
      case "run_end":
        this.finish(ev);
        break;
      default:
        this.view.handle(ev);
    }
    if (ev.type !== "run_end" && ev.type !== "token") {
      // While playing: running sums and the recorded run time of the latest event.
      this.showCounters(this.sumTokens, this.sumCalls, ev.t_ms);
    }
    if (this.approach.id !== "tot") scrollToEnd(this.body);
  }

  showAnswer(text) {
    if (!this.answer) {
      const card = el("div", "answer-card");
      this.answer = el("pre", "answer-text");
      card.append(el("div", "answer-label", "Final answer"), this.answer);
      this.final.prepend(card);
    }
    this.answer.textContent = text || "(no answer)";
    this.answer.classList.toggle("code", String(text).includes("\n"));
    this.answer.classList.toggle("muted", !text);
  }

  // After the page layout changed (summary strip shown): keep the end of the column visible.
  afterLayout() {
    if (this.approach.id !== "tot") scrollToEnd(this.body);
  }

  finish(ev) {
    this.runEnd = ev;
    this.output.stopTyping();
    if (!this.answer && ev.answer) this.showAnswer(ev.answer);
    this.final.append(officialVerdict(ev, this.task));
    this.showCounters(ev.tokens_in + ev.tokens_out, ev.llm_calls, ev.wall_ms);
    this.setStatus(statusOf(ev));
  }
}
