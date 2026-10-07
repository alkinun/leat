// leat agent's app. It keeps nothing of its own: it shows the conversations on the box as the
// agent's events change them, the turns running there too, and sends what the user writes.

import { markdown } from "/markdown.mjs";

const $ = (id) => document.getElementById(id);
let conversations = []; // the latest updated first: {id, title, updated, running}
let shown = null; // the conversation shown, with its messages; null for a new one
let models = [], loading = null, unreachable = null; // the engine's, a model it loads, or why not
let lost = false; // the events' connection, until it is back
let think = localStorage.getItem("leat.think") === "true"; // as the user last chose

$("new").onclick = () => {
  open(null);
  $("input").focus();
};
$("menu").onclick = () => document.body.classList.toggle("menu");
$("think").onclick = () => {
  think = !think;
  localStorage.setItem("leat.think", think);
  controls();
  $("input").focus();
};
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
    const { id } = await (await post(path, { content, think })).json();
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
  $("think").classList.toggle("on", think);
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

// a message's element, whose update() shows it again as it changes: a tool's is a line of the
// work it did, a system message's is hidden
function message(m) {
  if (m.role === "tool") return work(m);
  const item = element("div", `message ${m.role}`);
  const thinking = element("details"), reasoning = element("div");
  const text = element("div"), pages = element("div", "sources"), note = element("div", "note");
  const summary = element("summary", "", "Thinking");
  thinking.append(summary, reasoning);
  item.append(thinking, text, pages, note);
  item.update = () => {
    const answer = m.role === "assistant" && !m.tool_calls; // not a step to the tools it calls
    item.hidden = m.role === "system" || (!answer && !m.content && !m.reasoning_content);
    item.classList.toggle("step", Boolean(m.tool_calls));
    thinking.hidden = !m.reasoning_content;
    const thinks = !m.content && answer && shown?.running && shown.messages.at(-1) === m;
    summary.textContent = thinks ? "Thinking…" : "Thinking";
    markdown(reasoning, m.reasoning_content ?? "");
    if (m.role === "user") text.textContent = m.content;
    else markdown(text, m.content ?? "");
    item.querySelectorAll(":not(.code) > pre").forEach(codeBar); // the blocks new since
    if (answer) pages.replaceChildren(...sources(m).map(source));
    note.textContent = answer ? describe(m.info ?? {}) : "";
    const { cached, read } = m.info ?? {};
    note.title = read === undefined ? "" : `the prompt's tokens: ${cached} cached, ${read} read`;
  };
  item.update();
  return item;
}

// a tool's message, as a line of the work it did, which opens on what it found
function work(m) {
  const item = element("details", "message tool");
  const summary = element("summary"), found = element("div");
  item.append(summary, found);
  item.update = () => {
    const { arguments: args, results, url, title, error, stopped } = m.info ?? {};
    const running = !m.content, query = args?.query, address = url ?? args?.url ?? "";
    let line = `${m.name}`, done = [];
    if (m.name === "search") {
      line = running ? `Searching for “${query}”…` : error ? `Couldn't search for “${query}”`
        : `Searched for “${query}”`;
      done = (results ?? []).map((r) => link(r.url, r.title));
    } else if (m.name === "fetch") {
      line = running ? `Reading ${host(address)}…` : error ? `Couldn't read ${host(address)}`
        : `Read ${title || host(address)}`;
      if (!error) done = [link(address, title || address)];
    }
    if (stopped) line += " · stopped";
    item.classList.toggle("running", running);
    summary.replaceChildren(icon(m.name), element("span", "", line));
    if (error || !done.length) done = [element("p", "", error ?? m.content)];
    found.replaceChildren(...(done.length > 1 ? [listed(done)] : done));
  };
  item.update();
  return item;
}

// the pages a reply's turn read before it, its sources
function sources(m) {
  const messages = shown?.messages ?? [], pages = [];
  for (let i = messages.indexOf(m) - 1; i >= 0 && messages[i].role !== "user"; i--) {
    const { url, title } = (messages[i].role === "tool" && messages[i].info) || {};
    if (url && !pages.some((p) => p.url === url)) pages.unshift({ url, title });
  }
  return pages;
}

function source({ url, title }) {
  const a = link(url, host(url));
  a.className = "source";
  a.title = title || url;
  return a;
}

function link(url, text) {
  const a = element("a", "", text);
  Object.assign(a, { href: url, target: "_blank", rel: "noopener noreferrer" });
  return a;
}

function listed(items) {
  const list = element("ol");
  list.append(...items.map((item) => {
    const li = element("li");
    li.append(item, element("span", "host", host(item.href)));
    return li;
  }));
  return list;
}

// a web address's site, as people name it
function host(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

// the icon of a tool's line
const ICONS = {
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  fetch: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  tool: '<circle cx="12" cy="12" r="3"/>',
};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.innerHTML = ICONS[name] ?? ICONS.tool;
  return svg;
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
