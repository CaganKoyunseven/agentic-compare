// Small DOM helpers. Data from events is always set as text, never as HTML.

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function scrollToEnd(node) {
  node.scrollTop = node.scrollHeight;
}

export function formatInt(n) {
  return Number(n || 0).toLocaleString("en-US");
}

export function formatSeconds(ms) {
  return `${(Number(ms || 0) / 1000).toFixed(1)} s`;
}

export function truncate(text, max) {
  const s = String(text ?? "");
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}
