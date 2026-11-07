// Demo page entry: scene controls, replay or live stream, routing events to the columns.

import { APPROACHES, Column } from "./columns.js";
import { clear, el } from "./dom.js";
import { installShortcuts } from "./shortcuts.js";
import { ResultsView } from "./results.js";
import { openStream } from "./stream.js";
import { renderSummary } from "./summary.js";

const SCENE_NAMES = { easy24: "Easy 24", hard24: "Hard 24", humaneval: "HumanEval" };
const SPEEDS = [0.5, 1, 2, 4, 8];

const $ = (id) => document.getElementById(id);

const state = {
  scenes: [],
  scene: null,
  mode: "replay",
  speedIndex: SPEEDS.indexOf(1),
  stream: null,
  playing: false,
  finished: false,
  questionShown: false,
  drawn: null,
  currentApproach: null,
  tab: "demo",
  labels: { model: null, mode: null },
};

const speed = () => SPEEDS[state.speedIndex];
const columns = APPROACHES.map((a) => new Column(a, speed));
const byApproach = new Map(columns.map((c) => [c.approach.id, c]));
for (const c of columns) $("columns").append(c.root);

// ---- Banner ----

let bannerTimer = null;

function showBanner(text, kind = "error") {
  clearTimeout(bannerTimer);
  $("banner").className = `banner ${kind}`;
  $("banner-text").textContent = text;
  $("banner").hidden = false;
  if (kind === "info") bannerTimer = setTimeout(hideBanner, 6000);
}

function hideBanner() {
  clearTimeout(bannerTimer);
  $("banner").hidden = true;
}

// ---- Labels from the data ----

function setDataLabels(model, mode) {
  state.labels = { model, mode };
  if (state.tab !== "demo") return;
  $("model-name").textContent = model ? `Model: ${model}` : "";
  $("label-mock").hidden = model !== "mock";
  $("label-demo").hidden = mode !== "demo";
}

// ---- Question panel ----

function sceneName(scene) {
  return SCENE_NAMES[scene.id] || scene.id;
}

function showQuestionPlaceholder() {
  const scene = state.scene;
  const meta = $("question-meta");
  clear(meta);
  if (scene) {
    const parts = [sceneName(scene), scene.task, scene.difficulty].filter(Boolean);
    meta.textContent = parts.join(" · ");
  }
  const body = $("question-body");
  clear(body);
  if (state.drawn) {
    showNumbers(state.drawn.numbers, `Drawn card (${state.drawn.difficulty})`);
  } else {
    body.append(el("p", "muted", "Press Start (Space) to play this scene."));
  }
  state.questionShown = false;
}

function showNumbers(numbers, caption) {
  const body = $("question-body");
  clear(body);
  const row = el("div", "numbers");
  for (const n of numbers) row.append(el("span", "number", n));
  body.append(row);
  body.append(el("span", "question-caption", caption || "Make 24 with + - * /, each number once"));
}

function showQuestion(runStart) {
  const payload = runStart.payload || {};
  const body = $("question-body");
  clear(body);
  if (Array.isArray(payload.numbers)) {
    showNumbers(payload.numbers);
  } else if (typeof payload.prompt === "string") {
    body.append(el("pre", "prompt", payload.prompt));
  } else {
    body.append(el("pre", "prompt", JSON.stringify(payload, null, 2)));
  }
  const meta = [state.scene ? sceneName(state.scene) : null, runStart.task, runStart.item_id,
    runStart.difficulty].filter(Boolean);
  $("question-meta").textContent = meta.join(" · ");
  state.questionShown = true;
}

// ---- Playing ----

function routeEvent(ev) {
  if (ev.type === "run_start") {
    state.currentApproach = ev.approach;
    setDataLabels(ev.model, ev.mode);
    if (!state.questionShown) showQuestion(ev);
  }
  const column = byApproach.get(ev.approach || state.currentApproach);
  if (!column) {
    console.error("Event for an unknown approach", ev);
    return;
  }
  column.handle(ev);
}

function streamUrl() {
  const scene = encodeURIComponent(state.scene.id);
  if (state.mode === "live") {
    const item = state.drawn ? `?item_id=${encodeURIComponent(state.drawn.item_id)}` : "";
    return `/api/live/${scene}${item}`;
  }
  return `/api/replay/${scene}?speed=${speed()}`;
}

function start() {
  if (!state.scene || state.playing) return;
  if (state.finished) reset();
  hideBanner();
  state.playing = true;
  updateControls();
  state.stream = openStream(streamUrl(), {
    onEvent: routeEvent,
    onEnd: () => {
      state.stream = null;
      state.playing = false;
      state.finished = true;
      renderSummary($("summary"), columns);
      requestAnimationFrame(() => columns.forEach((c) => c.afterLayout()));
      updateControls();
    },
    onError: (message) => {
      state.stream = null;
      state.playing = false;
      state.finished = true;
      showBanner(message);
      updateControls();
    },
  });
}

function reset() {
  if (state.stream) state.stream.close();
  state.stream = null;
  state.playing = false;
  state.finished = false;
  state.currentApproach = null;
  for (const c of columns) c.reset();
  $("summary").hidden = true;
  clear($("summary"));
  hideBanner();
  if (state.scene) setDataLabels(state.scene.model, state.scene.mode);
  showQuestionPlaceholder();
  updateControls();
}

function selectScene(index) {
  const scene = state.scenes[index];
  if (!scene) return;
  state.scene = scene;
  state.drawn = null;
  reset();
  document.querySelectorAll("#scene-buttons button").forEach((b, i) => {
    b.classList.toggle("selected", i === index);
  });
}

// ---- Live mode and draw card ----

async function toggleLive() {
  if (state.playing) return;
  if (state.mode === "live") {
    state.mode = "replay";
    state.drawn = null;
    reset();
    return;
  }
  if (!state.scene) return;
  // Ask the live endpoint once; stop reading as soon as the status is known.
  const controller = new AbortController();
  try {
    const response = await fetch(`/api/live/${encodeURIComponent(state.scene.id)}`,
      { signal: controller.signal });
    controller.abort();
    if (response.status === 501) {
      showBanner("Live mode is not connected, staying in replay", "info");
      return;
    }
    if (!response.ok) {
      showBanner(`Live mode is not available (HTTP ${response.status}), staying in replay`);
      return;
    }
    state.mode = "live";
    reset();
  } catch (err) {
    showBanner(`Live mode check failed: ${err.message}`);
  }
}

async function drawCard() {
  if (state.mode !== "live" || state.playing || !canDraw()) return;
  const difficulty = $("draw-difficulty").value;
  try {
    const response = await fetch(`/api/draw?difficulty=${encodeURIComponent(difficulty)}`);
    const data = await response.json();
    if (!response.ok) {
      showBanner(data.detail || `Draw card failed (HTTP ${response.status})`);
      return;
    }
    state.drawn = data;
    reset();
  } catch (err) {
    showBanner(`Draw card failed: ${err.message}`);
  }
}

function canDraw() {
  return state.scene && state.scene.task === "game24";
}

// ---- Demo and results tabs ----

const resultsView = new ResultsView($("results"), {
  onFake: (fake) => {
    if (state.tab === "results") $("label-mock").hidden = !fake;
  },
  onError: (message) => showBanner(message),
});

function showTab(tab) {
  state.tab = tab;
  const demo = tab === "demo";
  for (const id of ["question", "columns", "scene-buttons", "demo-controls"]) {
    $(id).hidden = !demo;
  }
  $("summary").hidden = !(demo && state.finished && $("summary").childElementCount > 0);
  $("results").hidden = demo;
  $("tab-demo").classList.toggle("selected", demo);
  $("tab-results").classList.toggle("selected", !demo);
  if (demo) {
    setDataLabels(state.labels.model, state.labels.mode);
    requestAnimationFrame(() => columns.forEach((c) => c.afterLayout()));
  } else {
    $("model-name").textContent = "";
    $("label-demo").hidden = true;
    $("label-mock").hidden = true;
    resultsView.load();
  }
}

function toggleTab() {
  showTab(state.tab === "demo" ? "results" : "demo");
}

// Demo shortcuts do nothing while the results are on screen.
function demoOnly(action) {
  return (...args) => {
    if (state.tab === "demo") action(...args);
  };
}

function resultsOnly(action) {
  return (...args) => {
    if (state.tab === "results") action(...args);
  };
}

// ---- Controls ----

function updateControls() {
  const live = state.mode === "live";
  $("mode-label").textContent = live ? "LIVE" : "REPLAY";
  $("mode-label").classList.toggle("live", live);
  $("btn-live").textContent = live ? "Replay" : "Live";
  $("btn-start").disabled = !state.scene || state.playing;
  $("btn-live").disabled = state.playing;
  const drawOk = live && canDraw() && !state.playing;
  $("btn-draw").disabled = !drawOk;
  $("draw-difficulty").disabled = !drawOk;
  $("draw-wrap").title = live ? (canDraw() ? "Draw a dev puzzle (D)" : "Game of 24 scenes only")
    : "Live mode only";
  $("speed-label").textContent = `${speed()}×`;
  $("btn-slower").disabled = live || state.speedIndex === 0;
  $("btn-faster").disabled = live || state.speedIndex === SPEEDS.length - 1;
}

function changeSpeed(delta) {
  if (state.mode === "live") return;
  state.speedIndex = Math.min(SPEEDS.length - 1, Math.max(0, state.speedIndex + delta));
  updateControls();
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen();
  } else {
    document.documentElement.requestFullscreen().catch((err) => {
      showBanner(`Full screen is not available: ${err.message}`);
    });
  }
}

function toggleHelp(force) {
  const help = $("help");
  help.hidden = force === undefined ? !help.hidden : !force;
}

function bindButtonEl(button, action) {
  button.addEventListener("click", (e) => {
    e.currentTarget.blur();     // Space must start the demo, not click the focused button
    action();
  });
}

function bindButton(id, action) {
  bindButtonEl($(id), action);
}

bindButton("btn-start", start);
bindButton("btn-reset", reset);
bindButton("btn-live", toggleLive);
bindButton("btn-draw", drawCard);
bindButton("btn-slower", () => changeSpeed(-1));
bindButton("btn-faster", () => changeSpeed(1));
bindButton("btn-help", () => toggleHelp());
bindButton("banner-close", hideBanner);
bindButton("tab-demo", () => showTab("demo"));
bindButton("tab-results", () => showTab("results"));
window.addEventListener("resize", () => columns.forEach((c) => c.afterLayout()));
$("help").addEventListener("click", (e) => {
  if (e.target === $("help")) toggleHelp(false);
});

installShortcuts({
  scene: demoOnly(selectScene),
  start: demoOnly(start),
  reset: demoOnly(reset),
  toggleLive: demoOnly(toggleLive),
  draw: demoOnly(drawCard),
  faster: demoOnly(() => changeSpeed(1)),
  slower: demoOnly(() => changeSpeed(-1)),
  tab: toggleTab,
  prev: resultsOnly(() => resultsView.move(-1)),
  next: resultsOnly(() => resultsView.move(1)),
  fullscreen: toggleFullscreen,
  help: () => toggleHelp(),
  closeHelp: () => toggleHelp(false),
});

// ---- Start-up ----

async function loadScenes() {
  try {
    const response = await fetch("/api/scenes");
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${response.status}`);
    }
    state.scenes = await response.json();
  } catch (err) {
    showBanner(`Could not load the scenes: ${err.message}`);
    updateControls();
    return;
  }
  const holder = $("scene-buttons");
  clear(holder);
  state.scenes.forEach((scene, i) => {
    const button = el("button", "scene", sceneName(scene));
    button.type = "button";
    button.title = i < 9 ? `Scene (${i + 1})` : "Scene";
    bindButtonEl(button, () => selectScene(i));
    holder.append(button);
  });
  if (state.scenes.length) selectScene(0);
  else {
    showBanner("No scenes found. Run: python -m scripts.make_mock_traces");
    updateControls();
  }
}

updateControls();
loadScenes();
