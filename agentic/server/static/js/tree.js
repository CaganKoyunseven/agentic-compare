// Tree-of-Thoughts view: a d3 tree that grows as node events arrive.
// Depth runs left to right; each node has its step above and its state below the dot, so the
// levels can sit close together. The tree is scaled to fit the column (never above 1:1).
// Pruned nodes turn grey and take less room; the chosen path turns green.

import { testBadge } from "./badges.js";
import { el, truncate } from "./dom.js";

const d3 = window.d3;
const ROW_H = 58;          // vertical space per node
const LEVEL_W = 124;       // horizontal space per depth level
const PAD = 12;
const HALF_LABEL = 62;     // room for half a centered label left and right of a dot
const MAX_LABEL = 13;      // characters per text line before "..."
const ANIM_MS = 350;

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

  render(animate) {
    const root = this.hierarchy();
    if (!root) return;
    const gap = (a, b) => {
      const tight = a.data.pruned && b.data.pruned && !a.children && !b.children;
      return (tight ? 0.5 : 1) * (a.parent === b.parent ? 1 : 1.15);
    };
    d3.tree().nodeSize([ROW_H, LEVEL_W]).separation(gap)(root);

    // Screen x follows depth, screen y follows breadth.
    const all = root.descendants();
    for (const d of all) {
      d.sx = d.y;
      d.sy = d.x;
    }
    const minX = d3.min(all, (d) => d.sx) - HALF_LABEL - PAD;
    const maxX = d3.max(all, (d) => d.sx) + HALF_LABEL + PAD;
    const minY = d3.min(all, (d) => d.sy) - ROW_H / 2 - PAD;
    const maxY = d3.max(all, (d) => d.sy) + ROW_H / 2 + PAD;
    const width = Math.max(1, this.svgHost.clientWidth);
    const height = Math.max(1, this.svgHost.clientHeight);
    const scale = Math.min(1, width / (maxX - minX), height / (maxY - minY));
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

    const isChosen = (d) => this.chosen.has(d.data.id);
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
        g.append("circle").attr("r", 8);
        g.append("text").attr("class", "node-step").attr("text-anchor", "middle").attr("y", -14);
        g.append("text").attr("class", "node-state").attr("text-anchor", "middle").attr("y", 25);
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
    nodes.select(".node-step").text((d) => truncate(firstLine(d.data.step) || "start", MAX_LABEL));
    nodes.select(".node-state").text((d) => {
      const n = d.data;
      if (n.pruned) return "";    // grey already says pruned; the label is in the tooltip
      const state = truncate(firstLine(n.state), MAX_LABEL);
      return n.label ? `${truncate(state, 7)} · ${truncate(n.label, 8)}` : state;
    });
    nodes.transition(t)
      .attr("opacity", 1)
      .attr("transform", (d) => `translate(${d.sx},${d.sy})`);

    for (const d of all) this.lastPos.set(d.data.id, { x: d.sx, y: d.sy });
  }
}

function firstLine(text) {
  return String(text ?? "").split("\n")[0];
}
