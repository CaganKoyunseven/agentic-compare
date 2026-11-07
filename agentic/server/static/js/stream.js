// One SSE connection. Replay and live use the same message format:
//   data: <event>          one event dict per message
//   event: end             stream finished
//   event: error           server reported a failure, data.message says why

export function openStream(url, { onEvent, onEnd, onError }) {
  const source = new EventSource(url);
  let closed = false;

  const close = () => {
    if (!closed) {
      closed = true;
      source.close();
    }
  };

  source.onmessage = (msg) => {
    if (closed) return;
    let event;
    try {
      event = JSON.parse(msg.data);
    } catch (err) {
      close();
      onError(`Bad message from the server: ${err.message}`);
      return;
    }
    onEvent(event);
  };

  source.addEventListener("end", () => {
    close();
    onEnd();
  });

  // The server's "error" event carries data; the browser's own connection error does not.
  source.addEventListener("error", (msg) => {
    if (closed) return;
    close();
    if (msg.data) {
      let text = msg.data;
      try {
        text = JSON.parse(msg.data).message || text;
      } catch {
        // keep the raw text
      }
      onError(text);
    } else {
      onError("Connection to the demo server was lost");
    }
  });

  return { close };
}
