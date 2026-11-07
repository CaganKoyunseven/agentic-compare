// Approach views for Baseline and Reflective. Each view has a root element and handle(ev).
// The answer card and the official verdict are drawn by the column for every approach.

import { testBadge } from "./badges.js";
import { el } from "./dom.js";

export class BaselineView {
  constructor() {
    this.root = el("div", "view view-baseline");
    this.root.append(el("p", "view-hint", "One call, first answer."));
  }

  handle() {
    // Baseline has no steps besides the model call and the final answer.
  }
}

export class ReflectiveView {
  constructor() {
    this.root = el("div", "view view-reflective");
    this.cards = new Map();
  }

  card(attemptNo) {
    const key = attemptNo ?? this.lastAttempt ?? 0;
    if (!this.cards.has(key)) {
      const card = el("div", "attempt");
      const head = el("div", "attempt-head", key ? `Attempt ${key}` : "Feedback");
      const answer = el("pre", "attempt-answer");
      const notes = el("div", "attempt-notes");
      card.append(head, answer, notes);
      this.root.append(card);
      this.cards.set(key, { card, answer, notes });
    }
    return this.cards.get(key);
  }

  handle(ev) {
    if (ev.type === "attempt") {
      this.lastAttempt = ev.attempt_no;
      this.card(ev.attempt_no).answer.textContent = ev.answer;
    } else if (ev.type === "test_result") {
      this.card(ev.attempt_no).notes.append(testBadge(ev));
    } else if (ev.type === "critique") {
      const box = el("div", "critique");
      box.append(el("span", "critique-label", "Critique"), el("span", "critique-text", ev.text));
      this.card(ev.attempt_no).notes.append(box);
    }
  }
}
