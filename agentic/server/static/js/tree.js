// Tree-of-Thoughts view: a d3 tree that grows as node events arrive.
// Depth runs left to right; each node has its step above and its state below the dot, so the
// levels can sit close together. The tree is scaled to fit the column (never above 1:1).
// Pruned nodes turn grey and take less room; the chosen path turns green.
// Text never shrinks below a size readable on a projector. When a tree is too large for that,
// pruned nodes lose their text (it stays in the tooltip) and any label with no free space is
// dropped, most important labels first, so labels never touch.

import { testBadge } from "./badges.js";
import { el, truncate } from "./dom.js";

const d3 = window.d3;
const ROW_H = 58;          // vertical space per node
const LEVEL_W = 124;       // horizontal space per depth level
const PAD = 12;
const HALF_LABEL = 62;     // room for half a centered label left and right of a dot
const MAX_LABEL = 13;      // characters per text line before "..."
const ANIM_MS = 350;
const STEP_PX = 17;        // text sizes at full scale, in tree units
const STATE_PX = 14;
const MIN_STEP_PX = 14;    // smallest text allowed on screen, in real pixels
const MIN_STATE_PX = 13;
const CHAR_W = 0.61;       // width of a monospace character, relative to the font size
const LABEL_GAP = 6;       // free space between labels on neighbouring levels
const LINE_GAP = 4;        // free space kept above and below a label
const DOT_R = 8;
const STEP_LIFT = 14;      // step text baseline above the dot center
const STATE_DROP = 14;     // top of the state text below the dot center
const COMPACT_ROW = 46;    // row height in a large tree, for a node with text
const COMPACT_DOT_ROW = 24; // and for a node that shows only its dot

export class TreeView {
  constructor() {
    this.root = el("div", "view view-tree");
    this.svgHost = el("div", "tree-host");
    this.badges = el("div", "tree-badges");
    this.root.append(this.svgHost, this.badges);

    this.nodes = new Map();      // node_id -> {id, parent, step, state, label, score, pruned}
    this.chosen = new Set();
    this.lastPos = new Map();    // node_id -> {x, y}, for growing from the parent
    this.pending = false;

    this.svg = d3.select(this.svgHost).append("svg").attr("class", "tree");
    this.gLinks = this.svg.append("g").attr("class", "links");
    this.gNodes = this.svg.append("g").attr("class", "nodes");
    // Refit whenever the space for the tree changes (cards added below it, summary, window).
    this.size = { w: 0, h: 0 };
    this.observer = new ResizeObserver(() => {
      const w = this.svgHost.clientWidth;
      const h = this.svgHost.clientHeight;
      if (w !== this.size.w || h !== this.size.h) {
        this.size = { w, h };
        this.render(false);
      }
    });
    this.observer.observe(this.svgHost);
  }

  destroy() {
    this.observer.disconnect();
  }

  handle(ev) {
    const node = this.nodes.get(ev.node_id);
    if (ev.type === "node_added") {
      this.nodes.set(ev.node_id, { id: ev.node_id, parent: ev.parent_id, step: ev.step,
        state: ev.state, label: "", score: null, pruned: false });
    } else if (ev.type === "node_scored" && node) {
      node.score = ev.score;
      node.label = ev.label;
    } else if (ev.type === "node_pruned" && node) {
      node.pruned = true;
    } else if (ev.type === "path_chosen") {
      this.chosen = new Set(ev.node_ids);
      if (ev.node_ids.length === 0) {
        // No kept leaf reached 24: ToT gives no answer (nothing turns green).
        const note = el("div", "badge fail");
        note.append(el("span", "badge-mark", "✗"), el("span", "badge-text", "No path reached 24"));
        this.badges.append(note);
      }
    } else if (ev.type === "test_result") {
      this.badges.append(testBadge(ev, this.nodeName(ev.node_id)));
      return;
    } else {
      return;
    }
    this.schedule();
  }

  // A short name for a node in a badge; the parent step tells apart nodes with equal steps.
  nodeName(nodeId) {
    const n = this.nodes.get(nodeId);
    if (!n) return null;
    const step = firstLine(n.step) || n.id;
    const same = [...this.nodes.values()].filter((m) => firstLine(m.step) === step).length;
    const parent = this.nodes.get(n.parent);
    if (same > 1 && parent && parent.step) {
      return `${truncate(firstLine(parent.step), 16)} → ${truncate(step, 10)}`;
    }
    return truncate(step, 24);
  }

  // Several events often arrive in the same moment: draw once per animation frame.
  schedule() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.render(true);
    });
  }

  hierarchy() {
    const rootNode = [...this.nodes.values()].find((n) => n.parent === null);
    if (!rootNode) return null;
    const children = new Map();
    for (const n of this.nodes.values()) {
      if (n.parent !== null && this.nodes.has(n.parent)) {
        if (!children.has(n.parent)) children.set(n.parent, []);
        children.get(n.parent).push(n);
      }
    }
    return d3.hierarchy(rootNode, (n) => children.get(n.id) || []);
  }

  // Lay the nodes out, then work out the zoom that fits them in the column.
  fit(root, hasText, compact) {
    const row = (d) => (hasText(d) ? COMPACT_ROW : COMPACT_DOT_ROW);
    const gap = (a, b) => {
      if (compact) return (row(a) + row(b)) / (2 * ROW_H);
      const tight = a.data.pruned && b.data.pruned && !a.children && !b.children;
      return (tight ? 0.5 : 1) * (a.parent === b.parent ? 1 : 1.15);
    };
    d3.tree().nodeSize([ROW_H, LEVEL_W]).separation(gap)(root);

    // Screen y follows breadth; screen x follows depth (set below, once the zoom is known).
    const all = root.descendants();
    for (const d of all) d.sy = d.x;
    const top = d3.min(all, (d) => d.sy);
    const bottom = d3.max(all, (d) => d.sy);
    const levels = d3.max(all, (d) => d.depth);
    const width = Math.max(1, this.svgHost.clientWidth);
    const height = Math.max(1, this.svgHost.clientHeight);

    const baseWidth = levels * LEVEL_W + 2 * (HALF_LABEL + PAD);
    const span = bottom - top + 2 * PAD;
    // In a compact tree the text has its minimum size in real pixels, so reserve that first.
    const fitHeight = compact
      ? (height - 0.8 * MIN_STEP_PX - MIN_STATE_PX) / (span + STEP_LIFT + STATE_DROP)
      : height / (span + ROW_H);
    const scale = Math.max(0.05, Math.min(1, width / baseWidth, fitHeight));

    // A tree limited by its height has free width: give it to the levels, for longer labels.
    const freeLevel = (width / scale + LABEL_GAP - 2 * PAD) / (levels + 1);
    const levelW = scale < 1 ? Math.max(LEVEL_W, freeLevel) : LEVEL_W;
    for (const d of all) d.sx = d.depth * levelW;

    const halfLabel = Math.max(HALF_LABEL, (levelW - LABEL_GAP) / 2);
    const marginY = compact ? { top: STEP_LIFT + 0.8 * MIN_STEP_PX / scale,
      bottom: STATE_DROP + MIN_STATE_PX / scale } : { top: ROW_H / 2, bottom: ROW_H / 2 };
    return { scale, levelW, minX: -halfLabel - PAD, maxX: levels * levelW + halfLabel + PAD,
      minY: top - marginY.top - PAD, maxY: bottom + marginY.bottom + PAD, width, height };
  }

  render(animate) {
    const root = this.hierarchy();
    if (!root) return;
    const isChosen = (d) => this.chosen.has(d.data.id);
    const hasText = (d) => isChosen(d) || !d.data.pruned;

    // Fit with the normal layout first; a tree that would need too small text is laid out compact.
    let fit = this.fit(root, hasText, false);
    const compact = fit.scale * STEP_PX < MIN_STEP_PX;
    if (compact) fit = this.fit(root, hasText, true);
    const { scale, levelW, minX, maxX, minY, maxY, width, height } = fit;

    const all = root.descendants();
    const sizes = placeLabels(all, scale, levelW, (d) => !compact || hasText(d), isChosen);
    const vbW = width / scale;
    const vbH = height / scale;
    // Center the tree in the free space.
    const vbX = minX - (vbW - (maxX - minX)) / 2;
    const vbY = minY - (vbH - (maxY - minY)) / 2;
    const viewBox = [vbX, vbY, vbW, vbH].join(" ");

    const zoom = animate && this.svg.attr("viewBox");
    // Without animation, stop a running zoom first, or its next frame would overwrite the view.
    if (!zoom) this.svg.interrupt().attr("viewBox", viewBox);
    const t = this.svg.transition().duration(animate ? ANIM_MS : 0);
    if (zoom) t.attr("viewBox", viewBox);

    const startOf = (d) => {
      const p = d.parent && this.lastPos.get(d.parent.data.id);
      return p || { x: d.sx, y: d.sy };
    };
    const link = d3.linkHorizontal().x((p) => p.x).y((p) => p.y);

    // Links: one per non-root node, keyed by the child id.
    this.gLinks.selectAll("path")
      .data(root.links(), (l) => l.target.data.id)
      .join((enter) => enter.append("path").attr("d", (l) => {
        const s = startOf(l.target);
        return link({ source: s, target: s });
      }))
      .attr("class", (l) => {
        if (isChosen(l.source) && isChosen(l.target)) return "link chosen";
        return l.target.data.pruned ? "link pruned" : "link";
      })
      .transition(t)
      .attr("d", (l) => link({ source: { x: l.source.sx, y: l.source.sy },
        target: { x: l.target.sx, y: l.target.sy } }));

    const nodes = this.gNodes.selectAll("g.node")
      .data(all, (d) => d.data.id)
      .join((enter) => {
        const g = enter.append("g")
          .attr("transform", (d) => {
            const s = startOf(d);
            return `translate(${s.x},${s.y})`;
          })
          .attr("opacity", 0);
        g.append("title");
        g.append("circle").attr("r", DOT_R);
        g.append("text").attr("class", "node-step").attr("text-anchor", "middle").attr("y", -14);
        g.append("text").attr("class", "node-state").attr("text-anchor", "middle");
        return g;
      });

    nodes.attr("class", (d) => {
      if (isChosen(d)) return "node chosen";
      return d.data.pruned ? "node pruned" : "node";
    });
    nodes.select("title").text((d) => {
      const n = d.data;
      return [n.step || "start", n.state, n.label].filter(Boolean).join("\n");
    });
    nodes.select(".node-step").style("font-size", `${sizes.step}px`).text((d) => d.stepText);
    nodes.select(".node-state").style("font-size", `${sizes.state}px`)
      .attr("y", stateBaseline(sizes.state)).text((d) => d.stateText);
    nodes.transition(t)
      .attr("opacity", 1)
      .attr("transform", (d) => `translate(${d.sx},${d.sy})`);

    for (const d of all) this.lastPos.set(d.data.id, { x: d.sx, y: d.sy });
  }
}

// Decide the text of every node (d.stepText, d.stateText) and the font sizes, in tree units.
function placeLabels(all, scale, levelW, showsText, isChosen) {
  const room = levelW - LABEL_GAP;
  const stepOf = (d) => firstLine(d.data.step) || "start";
  const longest = d3.max(all.filter(showsText), (d) => Math.min(stepOf(d).length, MAX_LABEL)) || 1;
  const step = Math.max(MIN_STEP_PX / scale, Math.min(STEP_PX, room / (longest * CHAR_W)));
  const state = Math.max(MIN_STATE_PX / scale, STATE_PX);
  const stepChars = Math.min(MAX_LABEL, Math.floor(room / (step * CHAR_W)));
  const stateChars = Math.min(MAX_LABEL, Math.floor(room / (state * CHAR_W)));

  const candidates = [];
  for (const d of all) {
    d.stepText = "";
    d.stateText = "";
    const rank = isChosen(d) ? 0 : 1;
    if (showsText(d)) {
      candidates.push({ d, key: "stepText", text: truncate(stepOf(d), stepChars), rank,
        size: step, baseline: -STEP_LIFT });
    }
    // Grey already says pruned; the state and label are in the tooltip.
    if (!d.data.pruned) {
      candidates.push({ d, key: "stateText", text: stateText(d.data, stateChars), rank: rank + 2,
        size: state, baseline: stateBaseline(state) });
    }
  }
  candidates.sort((a, b) => a.rank - b.rank);

  // The dots are taken first, then each label in order of importance if its place is free.
  const taken = all.map((d) => ({ owner: d, left: d.sx - DOT_R, right: d.sx + DOT_R,
    top: d.sy - DOT_R, bottom: d.sy + DOT_R }));
  for (const c of candidates) {
    const half = (c.text.length * c.size * CHAR_W) / 2;
    const box = { owner: c.d, left: c.d.sx - half, right: c.d.sx + half,
      top: c.d.sy + c.baseline - c.size * 0.8 - LINE_GAP,
      bottom: c.d.sy + c.baseline + c.size * 0.2 + LINE_GAP };
    if (!taken.some((t) => t.owner !== box.owner && overlaps(t, box))) {
      c.d[c.key] = c.text;
      taken.push(box);
    }
  }
  return { step, state };
}

function stateText(node, maxChars) {
  const state = firstLine(node.state);
  if (!node.label) return truncate(state, maxChars);
  const label = truncate(node.label, 8);
  return `${truncate(state, Math.max(3, maxChars - label.length - 3))} · ${label}`;
}

function stateBaseline(size) {
  return STATE_DROP + size * 0.8;
}

function overlaps(a, b) {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

function firstLine(text) {
  return String(text ?? "").split("\n")[0];
}
