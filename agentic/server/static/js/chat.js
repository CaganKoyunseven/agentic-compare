// Multi-agent view: one chat bubble per agent_message, one color and label per role.

import { testBadge } from "./badges.js";
import { el } from "./dom.js";

const ROLE_LABELS = { planner: "Planner", solver: "Solver", critic: "Critic", tester: "Tester" };

export class ChatView {
  constructor() {
    this.root = el("div", "view view-chat");
    this.round = null;
  }

  markRound(round) {
    if (round !== null && round !== undefined && round !== this.round) {
      this.round = round;
      this.root.append(el("div", "round-mark", `Round ${round}`));
    }
  }

  handle(ev) {
    if (ev.type === "agent_message") {
      this.markRound(ev.round);
      const known = ROLE_LABELS[ev.role] ? ev.role : "other";
      const bubble = el("div", `bubble role-${known}`);
      const text = el(ev.text.includes("\n") ? "pre" : "div", "bubble-text", ev.text);
      bubble.append(el("div", "bubble-role", ROLE_LABELS[ev.role] || ev.role), text);
      this.root.append(bubble);
    } else if (ev.type === "test_result") {
      this.markRound(ev.attempt_no);
      this.root.append(testBadge(ev));
    }
  }
}
