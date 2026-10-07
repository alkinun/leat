// leat agent's app. It keeps nothing of its own: it shows the conversations on the box as the
// agent's events change them, the turns running there too, and sends what the user writes.

import { markdown } from "/markdown.mjs";

const $ = (id) => document.getElementById(id);
let conversations = []; // the latest updated first: {id, title, updated, running}
let shown = null; // the conversation shown, with its messages; null for a new one
let models = [], loading = null, unreachable = null; // the engine's, a model it loads, or why not
let lost = false; // the events' connection, until it is back

$("new").onclick = () => {
  open(null);
  $("input").focus();
};
$("menu").onclick = () => document.body.classList.toggle("menu");
$("model").onchange = () => load($("model").value);
$("send").onclick = () => (shown?.running ? stop() : send());
$("input").oninput = controls;
$("input").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!shown?.running) send();
  }
};
window.onpopstate = () => open(page(), false);

const events = new EventSource("/api/events");
events.onmessage = (event) => handle(JSON.parse(event.data));
events.onerror = () => {
  lost = true;
  status("Reconnecting to the box…", true);
};
events.onopen = () => {
  if (!lost) return;
  lost = false;
  status("");
  if (shown) open(shown.id, false); // what changed meanwhile
};
open(page(), false);

function handle(event) {
  switch (event.type) {
    case "conversations":
      conversations = event.conversations;
      break;
    case "conversation":
      update(event.conversation);
      break;
    case "deleted":
      conversations = conversations.filter((c) => c.id !== event.id);
      if (shown?.id === event.id) open(null);
      break;
    case "message":
      if (showing(event.conversation)) put(event.index, event.message);
      return;
    case "delta":
      if (showing(event.conversation)) grow(event);
      return;
    case "error": // the turn taken back, its message to send again
      if (showing(event.conversation)) {
        shown.messages.length = event.start;
        renderLog();
      }
      if (shown?.id === event.conversation) {
        status(event.error, true);
        if (!$("input").value) $("input").value = event.content;
      }
      break;
    case "loading":
      loading = event.model;
      break;
    case "models":
      models = event.models;
      loading = null;
      unreachable = event.error ?? null;
      if (unreachable) status(unreachable, true);
      renderModels();
      break;
  }
  renderList();
  controls();
}

// a conversation's summary, new or changed
function update(c) {
  conversations = [c, ...conversations.filter((other) => other.id !== c.id)];
  conversations.sort((a, b) => b.updated - a.updated);
  if (shown?.id !== c.id) return;
  const ended = shown.running && !c.running;
  Object.assign(shown, { title: c.title, running: c.running });
  if (ended) $("log").lastChild?.update?.(); // its reply no longer thinking
}

// whether a conversation is the one shown, its messages here
function showing(id) {
  return shown?.id === id && shown.messages !== null;
}

// a message, new or whole again
function put(index, m) {
  if (index > shown.messages.length) return open(shown.id, false); // one was missed
  shown.messages[index] = m;
  const log = $("log"), element = message(m);
  follow(() => (log.children[index] ? log.children[index].replaceWith(element) : log.append(element)));
}

// more of a message's text: its reasoning or content from `at`
function grow({ index, key, at, text }) {
  const m = shown.messages[index];
  if (!m || (m[key] ?? "").length < at) return open(shown.id, false); // some was missed
  m[key] = (m[key] ?? "").slice(0, at) + text;
  follow(() => $("log").children[index].update());
}

// shows a conversation, or a new one, at its own address
async function open(id, push = true) {
  if (push) history.pushState(null, "", id ? `/c/${id}` : "/");
  document.body.classList.remove("menu");
  shown = id ? { id, title: "", messages: null, running: false } : null; // until it comes
  render();
  if (!id) return;
  try {
    const response = await fetch(`/api/conversations/${id}`);
    if (!response.ok) throw new Error((await response.json()).error.message);
    const c = await response.json();
    if (shown?.id !== id) return;
    shown = c;
    render();
  } catch (error) {
    if (shown?.id !== id) return;
    history.replaceState(null, "", "/");
    shown = null;
    render();
    status(error.message, true);
  }
}

async function send() {
  const content = $("input").value.trim();
  if (!content || !ready()) return;
  $("input").value = "";
  status("");
  controls();
  try {
    const path = shown ? `/api/conversations/${shown.id}/messages` : "/api/conversations";
    const { id } = await (await post(path, { content })).json();
    if (shown?.id !== id) open(id);
  } catch (error) {
    if (!$("input").value) $("input").value = content;
    status(error.message, true);
    controls();
  }
}

async function stop() {
  try {
    await post(`/api/conversations/${shown.id}/stop`, {});
  } catch (error) {
    status(error.message, true);
  }
}

async function remove(c) {
  try {
    await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
  } catch (error) {
    status(error.message, true);
  }
}

async function load(model) {
  loading = model;
  controls();
  status("");
  try {
    await post("/api/models/load", { model });
  } catch (error) {
    status(error.message, true);
  }
}

async function post(path, body) {
  const response = await fetch(path, { method: "POST", body: JSON.stringify(body) });
  if (!response.ok) throw new Error((await response.json()).error.message);
  return response;
}

// the conversation the address shows, or null for a new one
function page() {
  return location.pathname.match(/^\/c\/([0-9a-f]+)$/)?.[1] ?? null;
}

// the loaded model's id, once it is ready for a message
function ready() {
  return loading ? null : (models.find((m) => m.status === "loaded")?.id ?? null);
}

function render() {
  renderList();
  renderLog();
  controls();
}

function renderList() {
  $("conversations").replaceChildren(...conversations.map((c) => {
    const item = element("div", "conversation");
    item.classList.toggle("shown", c.id === shown?.id);
    item.classList.toggle("running", c.running);
    const remover = element("button", "", "×");
    remover.title = "Delete";
    remover.onclick = (event) => {
      event.stopPropagation();
      remove(c);
    };
    item.append(element("span", "", c.title), remover);
    item.onclick = () => open(c.id);
    return item;
  }));
}

function renderModels() {
  const none = new Option(models.length ? "Choose a model" : "No models", "");
  none.disabled = true;
  $("model").replaceChildren(none, ...models.map((m) => new Option(m.id)));
}

function renderLog() {
  $("log").replaceChildren(...(shown?.messages ?? []).map(message));
  $("log").scrollTop = $("log").scrollHeight;
}

// shows the controls as the state has them, the input as tall as its text
function controls() {
  const running = Boolean(shown?.running), input = $("input"), model = ready();
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
  input.style.overflowY = input.scrollHeight > 240 ? "auto" : "hidden"; // its max-height
  document.title = shown?.title || "leat";
  $("main").classList.toggle("empty", !shown);
  $("greeting").textContent = unreachable ? "The engine is not reachable"
    : loading ? "Loading…" : model ? "How can I help?" : "Choose a model";
  $("send").classList.toggle("stop", running);
  $("send").title = running ? "Stop" : "Send";
  $("send").disabled = !running && !(model && input.value.trim());
  $("model").value = loading ?? model ?? "";
  $("model").disabled = loading !== null || conversations.some((c) => c.running);
}

function status(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "";
}

// a message's element, whose update() shows it again as it grows; a system message's is hidden
function message(m) {
  const item = element("div", `message ${m.role}`);
  const thinking = element("details"), reasoning = element("div");
  const text = element("div"), note = element("div", "note");
  const summary = element("summary", "", "Thinking");
  item.hidden = m.role === "system";
  thinking.append(summary, reasoning);
  item.append(thinking, text, note);
  item.update = () => {
    thinking.hidden = !m.reasoning_content;
    const thinks = m.reasoning_content && !m.content && shown?.running;
    summary.textContent = thinks ? "Thinking…" : "Thinking";
    markdown(reasoning, m.reasoning_content ?? "");
    if (m.role === "user") text.textContent = m.content;
    else markdown(text, m.content ?? "");
    item.querySelectorAll(":not(.code) > pre").forEach(codeBar); // the blocks new since
    note.textContent = m.info ? describe(m.info) : "";
  };
  item.update();
  return item;
}

// what a reply's info tells people: its model, its tokens, how fast they came, and whether it
// was stopped
function describe({ model, tokens, rate, first, stopped }) {
  let speed = tokens ? `${tokens} ${tokens === 1 ? "token" : "tokens"}` : "";
  if (rate) speed += `, ${rate.toFixed(1)} tok/s`;
  if (speed && first !== undefined) speed += `, the first after ${first.toFixed(1)} s`;
  return [model, speed, stopped && "stopped"].filter(Boolean).join(" · ");
}

// sets a code block below a bar of its language and a button that copies it
function codeBar(pre) {
  const block = element("div", "code"), bar = element("div", "bar");
  const copy = element("button", "", "Copy");
  copy.onclick = async () => {
    await clipboard(pre.textContent);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 2000);
  };
  bar.append(element("span", "", pre.firstChild.className.replace("language-", "")), copy);
  pre.replaceWith(block);
  block.append(bar, pre);
}

// copies text, without the clipboard's API where the page is not secure: over plain HTTP
async function clipboard(text) {
  if (navigator.clipboard) return navigator.clipboard.writeText(text);
  const area = element("textarea");
  area.value = text;
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

// runs a change to the log, keeping it scrolled to the end if it was
function follow(change) {
  const log = $("log"), end = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  change();
  if (end) log.scrollTop = log.scrollHeight;
}

function element(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text) e.textContent = text;
  return e;
}
