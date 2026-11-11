// Results view: the report figures one by one, large, and the report text.
// section: null for the main results, or a separate study (its own subfolder, its own tab).

import { clear, el } from "./dom.js";

const FAKE_MARK = "FAKE DATA, NOT A RESULT";

export class ResultsView {
  constructor(root, { onFake, onError, section = null, emptyText = "No results yet" }) {
    this.root = root;
    this.api = section ? `/api/results/${section}` : "/api/results";
    this.files = section ? `/results/${section}/` : "/results/";
    this.onFake = onFake;
    this.onError = onError;
    this.figures = [];
    this.index = 0;
    this.stamp = 0;
    this.root.append(
      this.figureBox = el("div", "results-figure"),
      this.report = el("pre", "results-report"),
      this.empty = el("p", "results-empty", emptyText),
    );
    const nav = el("div", "figure-nav");
    this.prevButton = el("button", "", "←");
    this.prevButton.type = "button";
    this.prevButton.title = "Previous figure (Left arrow)";
    this.nextButton = el("button", "", "→");
    this.nextButton.type = "button";
    this.nextButton.title = "Next figure (Right arrow)";
    this.caption = el("span", "figure-caption");
    nav.append(this.prevButton, this.caption, this.nextButton);
    this.image = el("img", "figure-image");
    this.image.alt = "";
    this.figureBox.append(nav, this.image);
    for (const [button, step] of [[this.prevButton, -1], [this.nextButton, 1]]) {
      button.addEventListener("click", (e) => {
        e.currentTarget.blur();
        this.move(step);
      });
    }
  }

  // Read the folder again each time the tab opens, so new runs show up.
  async load() {
    this.stamp = Date.now();
    let listing;
    try {
      const response = await fetch(this.api);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      listing = await response.json();
    } catch (err) {
      this.onError(`Could not load the results: ${err.message}`);
      return;
    }
    this.figures = listing.figures;
    this.index = Math.min(this.index, Math.max(0, this.figures.length - 1));
    let text = "";
    if (listing.report) {
      try {
        const response = await fetch(`${this.files}report.md?v=${this.stamp}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        text = await response.text();
      } catch (err) {
        this.onError(`Could not load report.md: ${err.message}`);
      }
    }
    this.report.textContent = text;
    this.report.hidden = !text;
    this.figureBox.hidden = this.figures.length === 0;
    this.empty.hidden = this.figures.length > 0 || Boolean(text);
    this.root.classList.toggle("no-report", !text);
    this.onFake(text.includes(FAKE_MARK));
    this.show();
  }

  show() {
    if (!this.figures.length) {
      clear(this.caption);
      this.image.removeAttribute("src");
      return;
    }
    const name = this.figures[this.index];
    this.image.src = `${this.files}${encodeURIComponent(name)}?v=${this.stamp}`;
    this.image.alt = name;
    this.caption.textContent = `${this.index + 1} / ${this.figures.length}  ·  ${name}`;
    this.prevButton.disabled = this.index === 0;
    this.nextButton.disabled = this.index === this.figures.length - 1;
  }

  move(step) {
    if (!this.figures.length) return;
    this.index = Math.min(this.figures.length - 1, Math.max(0, this.index + step));
    this.show();
  }
}
