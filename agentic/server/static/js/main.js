// Demo page entry: scene controls, replay or live stream, routing events to the columns.

import { APPROACHES, Column } from "./columns.js";
import { clear, el } from "./dom.js";
import { installShortcuts } from "./shortcuts.js";
import { ResultsView } from "./results.js";
import { openStream } from "./stream.js";
import { renderSummary } from "./summary.js";

const SCENE_NAMES = { easy24: "Easy 24", hard24: "Hard 24", humaneval: "HumanEval" };
const SPEEDS = [0.5, 1, 2, 4, 8];
const POOL_PREFIX = "pool-";     // recordings of draw-card puzzles, not scenes of their own
const FALLBACK_SECONDS = 10;

const $ = (id) => document.getElementById(id);

const state = {
  scenes: [],           // the scene buttons
  recordings: [],       // every recording: the scenes and the draw-card puzzles
  scene: null,
  mode: "replay",
  liveStatus: null,     // /api/status once known: {live, model, reason}
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
  stopCountdown();
  $("banner").className = `banner ${kind}`;
  $("banner-text").textContent = text;
  $("banner").hidden = false;
  if (kind === "info") bannerTimer = setTimeout(hideBanner, 6000);
}

function hideBanner() {
  clearTimeout(bannerTimer);
  stopCountdown();
  $("banner").hidden = true;
}

// ---- Fallback from live to replay ----

let countdownTimer = null;
let fallbackScene = null;

function stopCountdown() {
  clearInterval(countdownTimer);
  countdownTimer = null;
  $("banner-actions").hidden = true;
}

// A live scene failed: replay the same puzzle after a countdown, unless the presenter stays.
function startFallback(message, replayScene) {
  // A drawn card that was not recorded falls back to the selected scene's recording.
  fallbackScene = state.recordings.find((s) => s.id === replayScene) || state.scene;
  showBanner("", "error");
  $("banner-actions").hidden = false;
  let left = FALLBACK_SECONDS;
  const show = () => {
    $("banner-text").textContent =
      `Live failed: ${message}. Replay of ${sceneName(fallbackScene)} in ${left} s`;
  };
  show();
  countdownTimer = setInterval(() => {
    left -= 1;
    if (left > 0) show();
    else switchToReplay();
  }, 1000);
}

function switchToReplay() {
  if (countdownTimer === null) return;
  state.mode = "replay";
  state.drawn = null;
  setScene(fallbackScene);
  start();
  showBanner("Live failed, showing the recorded replay", "info");
}

function stayLive() {
  if (countdownTimer === null) return;
  showBanner("Staying in live mode. Press L for replay.", "info");
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

const isPoolRecording = (scene) => scene.id.startsWith(POOL_PREFIX);

function sceneName(scene) {
  if (isPoolRecording(scene)) return "Drawn card";
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
    if (state.drawn) return `/api/live/draw?item_id=${encodeURIComponent(state.drawn.item_id)}`;
    return `/api/live/${scene}`;
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
    onError: (message, details = {}) => {
      state.stream = null;
      state.playing = false;
      state.finished = true;
      if (state.mode === "live") {
        const running = byApproach.get(state.currentApproach);
        if (running && running.status === "running") running.setStatus("error");
        startFallback(message, details.replay_scene);
      } else {
        showBanner(message);
      }
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
  if (state.mode === "live") setDataLabels(state.liveStatus.model, "demo");
  else if (state.scene) setDataLabels(state.scene.model, state.scene.mode);
  showQuestionPlaceholder();
  updateControls();
}

function setScene(scene) {
  state.scene = scene;
  state.drawn = null;
  reset();
  document.querySelectorAll("#scene-buttons button").forEach((b, i) => {
    b.classList.toggle("selected", state.scenes[i] === scene);
  });
}

function selectScene(index) {
  if (state.scenes[index]) setScene(state.scenes[index]);
}

// ---- Live mode and draw card ----

function toggleLive() {
  if (state.playing) return;
  if (state.mode === "live") {
    state.mode = "replay";
    state.drawn = null;
    reset();
    return;
  }
  if (!state.scene) return;
  if (!isLiveAvailable()) {
    const reason = state.liveStatus ? state.liveStatus.reason : "status not known yet";
    showBanner(`Live mode is not available (${reason}), staying in replay`, "info");
    return;
  }
  state.mode = "live";
  // A drawn card's recording (after a fallback) cannot run live; go back to a real scene.
  if (isPoolRecording(state.scene)) selectScene(0);
  else reset();
}

function isLiveAvailable() {
  return state.liveStatus !== null && state.liveStatus.live;
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
// "results" shows the main comparison; "sc" the separate self-consistency study (own folder).

const TABS = ["demo", "results", "sc"];
const resultViews = {
  results: { section: "results", tab: "tab-results" },
  sc: { section: "results-sc", tab: "tab-sc" },
};
resultViews.results.view = new ResultsView($("results"), {
  onFake: (fake) => {
    if (state.tab === "results") $("label-mock").hidden = !fake;
  },
  onError: (message) => showBanner(message),
});
resultViews.sc.view = new ResultsView($("results-sc"), {
  section: "self_consistency",
  emptyText: "No self-consistency results yet",
  onFake: (fake) => {
    if (state.tab === "sc") $("label-mock").hidden = !fake;
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
  $("tab-demo").classList.toggle("selected", demo);
  for (const [name, r] of Object.entries(resultViews)) {
    $(r.section).hidden = name !== tab;
    $(r.tab).classList.toggle("selected", name === tab);
  }
  if (demo) {
    setDataLabels(state.labels.model, state.labels.mode);
    requestAnimationFrame(() => columns.forEach((c) => c.afterLayout()));
  } else {
    $("model-name").textContent = "";
    $("label-demo").hidden = true;
    $("label-mock").hidden = true;
    resultViews[tab].view.load();
  }
}

function toggleTab() {
  showTab(TABS[(TABS.indexOf(state.tab) + 1) % TABS.length]);
}

// Demo shortcuts do nothing while the results are on screen.
function demoOnly(action) {
  return (...args) => {
    if (state.tab === "demo") action(...args);
  };
}

// Figure navigation acts on the results tab that is on screen.
function resultsOnly(action) {
  return (...args) => {
    if (resultViews[state.tab]) action(resultViews[state.tab].view, ...args);
  };
}

// ---- Controls ----

function updateControls() {
  const live = state.mode === "live";
  const replayOnly = state.liveStatus !== null && !state.liveStatus.live;
  $("mode-label").textContent = live ? "LIVE" : (replayOnly ? "REPLAY ONLY" : "REPLAY");
  $("mode-label").classList.toggle("live", live);
  $("btn-live").textContent = live ? "Replay" : "Live";
  $("btn-live").title = replayOnly ? `Replay only: ${state.liveStatus.reason}` : "Live or replay (L)";
  $("btn-start").disabled = !state.scene || state.playing;
  $("btn-live").disabled = state.playing || replayOnly;
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
bindButton("btn-switch-now", switchToReplay);
bindButton("btn-stay", stayLive);
bindButton("tab-demo", () => showTab("demo"));
bindButton("tab-results", () => showTab("results"));
bindButton("tab-sc", () => showTab("sc"));
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
  switchNow: switchToReplay,
  stay: stayLive,
  tab: toggleTab,
  prev: resultsOnly((view) => view.move(-1)),
  next: resultsOnly((view) => view.move(1)),
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
    state.recordings = await response.json();
    state.scenes = state.recordings.filter((scene) => !isPoolRecording(scene));
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

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.liveStatus = await response.json();
  } catch (err) {
    state.liveStatus = { live: false, model: null, reason: `status check failed: ${err.message}` };
  }
  updateControls();
}

updateControls();
loadStatus();
loadScenes();
