// Presenter keyboard shortcuts. Handlers come from main.js.

export function installShortcuts(actions) {
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = document.activeElement ? document.activeElement.tagName : "";
    if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;

    const key = e.key;
    let handled = true;
    if (key >= "1" && key <= "9") actions.scene(Number(key) - 1);
    else if (key === " ") actions.start();
    else if (key === "r" || key === "R") actions.reset();
    else if (key === "l" || key === "L") actions.toggleLive();
    else if (key === "d" || key === "D") actions.draw();
    else if (key === "+" || key === "=") actions.faster();
    else if (key === "-" || key === "_") actions.slower();
    else if (key === "f" || key === "F") actions.fullscreen();
    else if (key === "t" || key === "T") actions.tab();
    else if (key === "ArrowLeft") actions.prev();
    else if (key === "ArrowRight") actions.next();
    else if (key === "n" || key === "N") actions.switchNow();
    else if (key === "s" || key === "S") actions.stay();
    else if (key === "?") actions.help();
    else if (key === "Escape") actions.closeHelp();
    else handled = false;
    if (handled) e.preventDefault();
  });
}
