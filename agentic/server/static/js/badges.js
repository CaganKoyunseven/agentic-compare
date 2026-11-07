// Feedback badges. Feedback an approach may see (verifier, own tests) is a small outlined pill;
// the official verdict from run_end is a large filled block, so the two never look the same.

import { el } from "./dom.js";

const SOURCE_LABELS = { verifier: "Verifier", own_tests: "Own tests" };

export function testBadge(ev, prefix) {
  const badge = el("div", `badge ${ev.ok ? "ok" : "fail"}`);
  const source = SOURCE_LABELS[ev.source] || ev.source;
  badge.append(
    el("span", "badge-source", prefix ? `${prefix} · ${source}` : source),
    el("span", "badge-mark", ev.ok ? "✓" : "✗"),
    el("span", "badge-text", ev.message),
  );
  return badge;
}

export function officialVerdict(runEnd, task) {
  let kind = "error";
  let word = runEnd.status === "cancelled" ? "Cancelled" : "Error";
  if (runEnd.status === "ok" && runEnd.correct === true) {
    kind = "correct";
    word = "Correct";
  } else if (runEnd.status === "ok" && runEnd.correct === false) {
    kind = "wrong";
    word = "Wrong";
  } else if (runEnd.status === "ok") {
    word = "No verdict";
  }
  const box = el("div", `official ${kind}`);
  const label = task === "humaneval" ? "Official tests" : "Official check";
  box.append(el("div", "official-label", `${label} · never seen by the approach`),
    el("div", "official-word", word));
  const detail = runEnd.error || runEnd.score_message;
  if (detail) box.append(el("div", "official-detail", detail));
  return box;
}
