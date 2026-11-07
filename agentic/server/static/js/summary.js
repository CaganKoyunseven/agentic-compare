// Summary strip after the last run_end: the four approaches side by side.

import { clear, el, formatInt, formatSeconds } from "./dom.js";
import { statusOf } from "./columns.js";

export function renderSummary(container, columns) {
  clear(container);
  for (const column of columns) {
    const cell = el("div", "summary-cell");
    cell.style.setProperty("--approach", column.approach.color);
    const name = el("div", "summary-head");
    name.append(el("div", "summary-name", column.approach.short));
    cell.append(name);
    const end = column.runEnd;
    if (!end) {
      name.append(el("div", "summary-missing", "not run"));
    } else {
      const status = statusOf(end);
      name.append(el("span", `chip ${status}`, status));
      const stats = el("div", "summary-stats");
      stats.append(
        stat(formatInt(end.tokens_in + end.tokens_out), "tokens"),
        stat(formatInt(end.llm_calls), "calls"),
        stat(formatSeconds(end.wall_ms), "time"),
      );
      cell.append(stats);
    }
    container.append(cell);
  }
  container.hidden = false;
}

function stat(value, label) {
  const box = el("div", "summary-stat");
  box.append(el("span", "summary-value", value), el("span", "summary-label", label));
  return box;
}
