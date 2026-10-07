// leat agent's app. It keeps nothing of its own: it shows the conversations on the box as the
// agent's events change them, the turns running there too, and sends what the user writes.

import { markdown } from "/markdown.mjs";

const $ = (id) => document.getElementById(id);
let conversations = []; // the latest updated first: {id, title, updated, running}
let shown = null; // the conversation shown, with its messages; null for a new one
let memories = []; // what the agent remembers of the user, the oldest first
let files = []; // the workspace's, the latest changed first: {name, size, modified}
let attached = []; // the files the next message attaches: {name, uploading}
let tasks = []; // the scheduled, the next due first: {id, prompt, schedule, conversation}
let telegram = null; // its bot, if connected, and the people allowed and asking
// the conversations whose turns ended while another was shown, as this device saw them
const unread = new Set(JSON.parse(localStorage.getItem("leat.unread") ?? "[]"));
let models = [], loading = null, unreachable = null; // the engine's, a model it loads, or why not
let lost = false; // the events' connection, until it is back
let think = localStorage.getItem("leat.think") === "true"; // as the user last chose
let views = []; // the shown messages' elements, by their indexes
const opened = new Map(); // whether each turn's work is open, as the user left it
// the pages beside the conversations, each a section of its own name, and their titles
const PAGES = ["memory", "files", "tasks", "settings"];
const TITLES = { memory: "Memory", files: "Files", tasks: "Tasks", settings: "Settings" };

$("new").onclick = () => {
  open(null);
  $("input").focus();
};
$("menu").onclick = () => document.body.classList.toggle("menu");
$("remembered").onclick = () => turnTo("memory");
$("filed").onclick = () => turnTo("files");
$("timed").onclick = () => turnTo("tasks");
$("set").onclick = () => turnTo("settings");
$("attach").onclick = () => pick(attach);
$("upload").onclick = () => pick(upload);
window.ondragover = (event) => event.preventDefault();
window.ondrop = (event) => { // files dropped, attached to the next message, or on their page uploaded
  event.preventDefault();
  const dropped = [...event.dataTransfer.files];
  if (document.body.classList.contains("files")) dropped.forEach(upload);
  else dropped.forEach(attach);
};
$("remember").onsubmit = async (event) => {
  event.preventDefault();
  const text = $("memorable").value.trim();
  if (!text) return;
  try {
    await post("/api/memories", { text, category: $("category").value });
    $("memorable").value = "";
  } catch (error) {
    status(error.message, true);
  }
};
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
window.onpopstate = route;

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
route();

// shows what the address names: a page, as the memory's, a conversation, or a new one
function route() {
  const name = location.pathname.slice(1);
  if (PAGES.includes(name)) turnTo(name, false);
  else open(addressed(), false);
}

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
    case "compacted": // earlier messages summarized, to make room
      if (showing(event.conversation)) {
        shown.summarized = event.summarized;
        renderLog();
      }
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
    case "memories":
      memories = event.memories;
      renderMemories();
      return;
    case "files":
      files = event.files;
      renderFiles();
      return;
    case "tasks":
      tasks = event.tasks;
      renderTasks();
      return;
    case "telegram":
      telegram = event;
      renderTelegram();
      return;
    case "done": // a scheduled task's turn, ended
      notify(event.task, event.conversation);
      return;
    case "loading":
      loading = event.model;
      break;
    case "models":
      models = event.models;
      loading = models.find((m) => m.status === "loading")?.id ?? null;
      unreachable = event.error ?? null;
      if (unreachable) status(unreachable, true);
      renderModels();
      break;
  }
  renderList();
  controls();
}

// a conversation's summary, new or changed: one whose turn ended unseen, unread
function update(c) {
  const before = conversations.find((other) => other.id === c.id);
  if (before?.running && !c.running && shown?.id !== c.id) {
    unread.add(c.id);
    localStorage.setItem("leat.unread", JSON.stringify([...unread]));
  }
  conversations = [c, ...conversations.filter((other) => other.id !== c.id)];
  conversations.sort((a, b) => b.updated - a.updated);
  if (shown?.id !== c.id) return;
  const ended = shown.running && !c.running;
  Object.assign(shown, { title: c.title, running: c.running });
  if (ended && shown.messages?.length) refresh(shown.messages.length - 1); // done working
}

// whether a conversation is the one shown, its messages here
function showing(id) {
  return shown?.id === id && shown.messages !== null;
}

// a message, new or whole again
function put(index, m) {
  if (index > shown.messages.length) return open(shown.id, false); // one was missed
  shown.messages[index] = m;
  follow(() => refresh(index));
}

// more of a message's text: its reasoning or content from `at`
function grow({ index, key, at, text }) {
  const m = shown.messages[index];
  if (!m || (m[key] ?? "").length < at) return open(shown.id, false); // some was missed
  m[key] = (m[key] ?? "").slice(0, at) + text;
  follow(() => views[index].update());
}

// shows a conversation, or a new one, at its own address
async function open(id, push = true) {
  if (push) history.pushState(null, "", id ? `/c/${id}` : "/");
  if (unread.delete(id)) localStorage.setItem("leat.unread", JSON.stringify([...unread]));
  document.body.classList.remove("menu", ...PAGES);
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
  if (!content || !ready() || attached.some((a) => a.uploading)) return;
  $("input").value = "";
  status("");
  controls();
  try {
    const path = shown ? `/api/conversations/${shown.id}/messages` : "/api/conversations";
    const names = attached.map((a) => a.name);
    const { id } = await (await post(path, { content, think, files: names })).json();
    attached = [];
    renderAttached();
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

// shows a page, in place of the conversation
function turnTo(name, push = true) {
  if (push) history.pushState(null, "", `/${name}`);
  document.body.classList.remove("menu", ...PAGES);
  document.body.classList.add(name);
  shown = null;
  render();
  document.title = TITLES[name];
}

// the memories by category, each its own list, and how full their room is
const CATEGORIES = { about: "About you", preferences: "Preferences", people: "People",
  work: "Work", plans: "Plans" };
const ROOM = 3000; // characters, as the agent bounds them
function renderMemories() {
  const used = memories.reduce((n, m) => n + m.text.length, 0);
  $("room").textContent = memories.length ? `It is ${Math.round((100 * used) / ROOM)}% full.` : "";
  $("memories").replaceChildren(...Object.entries(CATEGORIES).flatMap(([category, name]) => {
    const of = memories.filter((m) => m.category === category);
    if (!of.length) return [];
    const list = element("ul");
    list.append(...of.map((m) => {
      const item = element("li"), remover = element("button", "", "×");
      remover.title = "Forget";
      remover.onclick = () => fetch(`/api/memories/${m.id}`, { method: "DELETE" });
      item.append(element("span", "", m.text), remover);
      return item;
    }));
    return [element("h2", "", name), list];
  }));
}

function renderTasks() {
  $("scheduled").replaceChildren(...tasks.map((t) => {
    const item = element("li"), about = element("div"), remover = element("button", "", "×");
    remover.title = "Cancel";
    remover.onclick = () => fetch(`/api/tasks/${t.id}`, { method: "DELETE" });
    about.append(element("span", "", t.prompt), element("span", "meta", t.schedule));
    if (t.conversation) about.append(conversationLink({ id: t.conversation, title: "Its chat" }));
    item.append(about, remover);
    return item;
  }));
  const secure = window.isSecureContext && "Notification" in window;
  $("notifying").replaceChildren();
  if (secure && Notification.permission === "default") {
    const ask = element("button", "", "Notify me when one is done");
    ask.onclick = async () => (await Notification.requestPermission(), renderTasks());
    $("notifying").append(ask);
  }
}

// Telegram's settings: how to make a bot and connect it, or the bot connected, and its people
function renderTelegram() {
  const box = $("telegram");
  if (!telegram?.bot) {
    const steps = element("ol", "steps");
    steps.append(...["In Telegram, open @BotFather and send it /newbot.",
      "Give your bot a name, then a username that ends in “bot”.",
      "Paste the token BotFather gives you here."].map((step) => element("li", "", step)));
    const form = element("form"), token = element("input"), button = element("button", "", "Connect");
    Object.assign(token, { placeholder: "123456:ABC-…", autocomplete: "off" });
    form.append(token, button);
    form.onsubmit = async (event) => {
      event.preventDefault();
      try {
        await post("/api/telegram", { token: token.value });
        status("");
      } catch (error) {
        status(error.message, true);
      }
    };
    return box.replaceChildren(element("p", "", "Talk to Leat from Telegram, anywhere."), steps, form);
  }
  const bot = link(`https://t.me/${telegram.bot}`, `@${telegram.bot}`);
  const unlink = element("button", "", "Disconnect");
  unlink.onclick = () => fetch("/api/telegram", { method: "DELETE" });
  const connected = element("p", "", "Connected as ");
  connected.append(bot, ". Only the people you allow can talk to it. ", unlink);
  const person = (p, asking) => {
    const item = element("li"), about = element("div");
    about.append(element("span", "", p.name), element("span", "meta", asking ? "asks to talk to Leat" : "allowed"));
    const yes = element("button", "allow", "Allow"), no = element("button", "", "×");
    yes.onclick = () => post("/api/telegram/people", { id: p.id });
    no.title = asking ? "Turn down" : "Remove";
    no.onclick = () => fetch(`/api/telegram/people/${p.id}`, { method: "DELETE" });
    item.append(about, ...(asking ? [yes] : []), no);
    return item;
  };
  const people = element("ul");
  people.append(...telegram.requests.map((p) => person(p, true)), ...telegram.allowed.map((p) => person(p, false)));
  const none = element("p", "meta", `No one yet: write to @${telegram.bot}, then allow yourself here.`);
  box.replaceChildren(connected, telegram.requests.length + telegram.allowed.length ? people : none);
}

// says a scheduled task is done, in the page, and the system's notification if allowed and the
// page is out of sight; either opens its conversation
function notify(task, id) {
  const toast = $("toast");
  toast.textContent = `Done: ${task}`;
  toast.hidden = false;
  toast.onclick = () => {
    toast.hidden = true;
    open(id);
  };
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => (toast.hidden = true), 8000);
  if (document.hidden && window.Notification?.permission === "granted") {
    new Notification("Leat", { body: task }).onclick = () => (window.focus(), open(id));
  }
}

function renderFiles() {
  $("workspace").replaceChildren(...files.map((f) => {
    const item = element("li"), remover = element("button", "", "×");
    remover.title = "Delete";
    remover.onclick = () => fetch(`/api/files/${encodeURIComponent(f.name)}`, { method: "DELETE" });
    const day = { day: "numeric", month: "short" };
    const when = new Date(f.modified * 1000).toLocaleDateString(undefined, day);
    item.append(fileLink(f.name), element("span", "meta", `${bytes(f.size)} · ${when}`), remover);
    return item;
  }));
  if (shown?.messages) renderLog(); // the cards of the files its answers made
}

// asks the user for files, and hands them to `take`, each
function pick(take) {
  const picker = element("input");
  Object.assign(picker, { type: "file", multiple: true });
  picker.onchange = () => [...picker.files].forEach(take);
  picker.click();
}

// uploads a file to the workspace; returns the name it got there, or null if it did not
async function upload(file) {
  try {
    const path = `/api/files/${encodeURIComponent(file.name)}`;
    const response = await fetch(path, { method: "PUT", body: file });
    if (!response.ok) throw new Error((await response.json()).error.message);
    return (await response.json()).name;
  } catch (error) {
    status(`${file.name} was not uploaded: ${error.message}`, true);
    return null;
  }
}

// attaches a file to the next message, once it is uploaded
async function attach(file) {
  const chip = { name: file.name, uploading: true };
  attached.push(chip);
  renderAttached();
  chip.name = await upload(file);
  chip.uploading = false;
  attached = attached.filter((a) => a.name);
  renderAttached();
}

function renderAttached() {
  $("attached").replaceChildren(...attached.map((a) => {
    const chip = element("span", "chip", a.uploading ? `${a.name}…` : a.name);
    const remover = element("button", "", "×");
    remover.title = "Remove";
    remover.onclick = () => {
      attached = attached.filter((other) => other !== a);
      renderAttached();
    };
    chip.append(remover);
    return chip;
  }));
  controls();
}

// the conversation the address shows, or null for a new one
function addressed() {
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
    item.classList.toggle("unread", unread.has(c.id) && !c.running);
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
  views = [];
  $("log").replaceChildren(...turns().map((indexes) => fill(element("div", "turn"), indexes)));
  $("log").scrollTop = $("log").scrollHeight;
}

// the shown conversation's turns, each the indexes of a user's message and those that came of it;
// the system's message, before them, a turn of its own
function turns() {
  const all = [];
  (shown?.messages ?? []).forEach((m, i) => {
    if (m.role === "user" || !all.length) all.push([]);
    all.at(-1).push(i);
  });
  return all;
}

// shows a message's turn again, or as a new one
function refresh(index) {
  const indexes = turns().find((t) => t.includes(index));
  const old = [...$("log").children].find((turn) => turn.start === indexes[0]);
  const turn = fill(old ?? element("div", "turn"), indexes);
  if (!old) $("log").append(turn);
}

// shows a turn in its element: the user's message, the work that came of it folded into a line,
// and the answer
function fill(turn, [start, ...rest]) {
  const view = (i) => (views[i] = message(shown.messages[i]));
  const last = shown.messages[rest.at(-1)], answered = last?.role === "assistant" && !last.tool_calls;
  const work = answered ? rest.slice(0, -1) : rest;
  turn.start = start;
  turn.classList.toggle("summarized", [start, ...rest].includes(shown.summarized));
  turn.replaceChildren(view(start));
  if (work.length) turn.append(fold(start, work));
  if (answered) turn.append(view(rest.at(-1)));
  return turn;
}

// a turn's work, its steps and calls, folded into a line: what it does as it runs, then what it did
function fold(start, indexes) {
  const box = element("details", "work"), key = `${shown.id} ${start}`;
  const running = shown.running && turns().at(-1)[0] === start; // the turn running
  const calls = indexes.map((i) => shown.messages[i]).filter((m) => m.role === "tool");
  box.open = opened.get(key) ?? false;
  box.ontoggle = () => opened.set(key, box.open);
  box.classList.toggle("running", running);
  const said = running ? (calls.length ? line(calls.at(-1)) : "Working…") : summary(calls);
  box.append(element("summary", "", said), ...indexes.map((i) => (views[i] = message(shown.messages[i]))));
  return box;
}

// what each tool's calls did, in a few words: one call, and n
const DID = {
  recall: ["recalled earlier chats", () => "recalled earlier chats"],
  search: ["searched the web", (n) => `searched the web ${n} times`],
  fetch: ["read a page", (n) => `read ${n} pages`],
  read: ["read a file", (n) => `read ${n} files`],
  run: ["ran code", (n) => `ran code ${n} times`],
  write: ["wrote a file", (n) => `wrote ${n} files`],
  edit: ["edited a file", (n) => `edited ${n} files`],
  remember: ["remembered something", (n) => `remembered ${n} things`],
  forget: ["forgot something", (n) => `forgot ${n} things`],
  weather: ["checked the weather", () => "checked the weather"],
  schedule: ["scheduled a task", (n) => `scheduled ${n} tasks`],
  unschedule: ["cancelled a task", (n) => `cancelled ${n} tasks`],
  tasks: ["looked at the tasks", () => "looked at the tasks"],
};

// what a turn's calls did, in a few words, in the order it began them; those that failed not
function summary(all) {
  const calls = all.filter((m) => !m.info?.error);
  const parts = [...new Set(calls.map((m) => m.name))].map((name) => {
    const n = calls.filter((m) => m.name === name).length;
    const [one, many] = DID[name] ?? [`used ${name}`, () => `used ${name}`];
    return n > 1 ? many(n) : one;
  });
  const said = parts.join(", ") || "worked";
  return said[0].toUpperCase() + said.slice(1);
}

// shows the controls as the state has them, the input as tall as its text
function controls() {
  const running = Boolean(shown?.running), input = $("input"), model = ready();
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
  input.style.overflowY = input.scrollHeight > 240 ? "auto" : "hidden"; // its max-height
  const paged = PAGES.some((name) => document.body.classList.contains(name));
  if (!paged) document.title = shown?.title || "leat";
  $("main").classList.toggle("empty", !shown && !paged);
  $("greeting").textContent = unreachable ? "The engine is not reachable"
    : loading ? "Loading…" : model ? "How can I help?" : "Choose a model";
  $("think").classList.toggle("on", think);
  $("send").classList.toggle("stop", running);
  $("send").title = running ? "Stop" : "Send";
  const uploading = attached.some((a) => a.uploading);
  $("send").disabled = !running && !(model && input.value.trim() && !uploading);
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
  const summary = element("summary", "", "Thinking"), cards = element("div", "cards");
  thinking.append(summary, reasoning);
  item.append(thinking, text, cards, pages, note);
  item.update = () => {
    const answer = m.role === "assistant" && !m.tool_calls; // not a step to the tools it calls
    item.hidden = m.role === "system" || (!answer && !m.content && !m.reasoning_content);
    item.classList.toggle("step", Boolean(m.tool_calls));
    thinking.hidden = !m.reasoning_content;
    const thinks = !m.content && answer && shown?.running && shown.messages.at(-1) === m;
    summary.textContent = thinks ? "Thinking…" : "Thinking";
    markdown(reasoning, m.reasoning_content ?? "");
    if (m.role === "user" && m.info?.task) { // a scheduled task's, marked so
      item.classList.add("scheduled");
      text.replaceChildren(icon("schedule"), element("span", "", m.content));
    } else if (m.role === "user") text.textContent = m.content;
    else markdown(text, m.content ?? "");
    item.querySelectorAll(":not(.code) > pre").forEach(codeBar); // the blocks new since
    cite(item);
    if (answer) pages.replaceChildren(...sources(m).map(source));
    if (m.info?.via === "telegram") note.textContent = "via Telegram";
    const names = m.role === "user" ? (m.info?.files ?? []) : answer ? made(m) : [];
    cards.replaceChildren(...names.map(card));
    if (m.role !== "user") note.textContent = answer ? describe(m.info ?? {}) : "";
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
    item.classList.toggle("running", !m.content);
    summary.replaceChildren(icon(m.name), element("span", "", line(m)));
    found.replaceChildren(...what(m));
    found.querySelectorAll(":not(.code) > pre:not(.output)").forEach(codeBar);
  };
  item.update();
  return item;
}

// each tool's call in a line, of its arguments and info: what it is doing, what it did, and that
// it failed
const LINES = {
  search: (a) => [`Searching for “${a.query}”…`, `Searched for “${a.query}”`,
    `Couldn't search for “${a.query}”`],
  fetch: (a, i) => [`Reading ${host(a.url)}…`, `Read ${i.title || host(i.url ?? a.url)}`,
    `Couldn't read ${host(a.url)}`],
  recall: (a) => [`Recalling “${a.query}”…`, `Recalled “${a.query}”`, `Couldn't recall “${a.query}”`],
  remember: (a, i) => ["Remembering…", `${i.replaced ? "Changed" : "Remembered"}: ${i.memory?.text}`,
    "Couldn't remember"],
  forget: (a, i) => ["Forgetting…", `Forgot: ${i.memory?.text}`, "Couldn't forget"],
  weather: (a, i) => [`Checking the weather in ${a.place}…`, `Checked the weather in ${i.place}`,
    `Couldn't check the weather in ${a.place}`],
  read: (a) => [`Reading ${named(a.path)}…`, `Read ${named(a.path)}`, `Couldn't read ${named(a.path)}`],
  write: (a) => [`Writing ${a.path}…`, `Wrote ${a.path}`, `Couldn't write ${a.path}`],
  edit: (a) => [`Editing ${a.path}…`, `Edited ${a.path}`, `Couldn't edit ${a.path}`],
  schedule: (a, i) => ["Scheduling…", `Scheduled: ${i.task?.prompt} · ${i.task?.schedule}`,
    "Couldn't schedule"],
  unschedule: (a, i) => ["Cancelling…", `Cancelled: ${i.task?.prompt}`, "Couldn't cancel"],
  tasks: () => ["Looking at the tasks…", "Looked at the tasks", "Couldn't look at the tasks"],
  run: (a, i) => ["Running code…", i.status === 0 ? "Ran code"
    : i.status === null ? "Ran code, out of time" : "Ran code, which failed", "Couldn't run code"],
};

function line(m) {
  const { arguments: args, error, stopped } = m.info ?? {};
  const lines = LINES[m.name] ?? (() => [`${m.name}…`, m.name, `${m.name} failed`]);
  const [running, done, failed] = lines(typeof args === "object" ? args : {}, m.info ?? {});
  const said = !m.content ? running : error ? failed : done;
  return stopped ? `${said} · stopped` : said;
}

// what a tool's call found, as its line opens on it
function what(m) {
  const { arguments: args, results, url, title, error, conversations } = m.info ?? {};
  if (error || !m.content) return [element("p", "", error ?? "")];
  if (m.name === "search" && results?.length) {
    return [listed(results.map((r) => link(r.url, r.title)), true)];
  }
  if (m.name === "fetch") return [link(url, title || url)];
  if (m.name === "recall" && conversations?.length) return [listed(conversations.map(conversationLink))];
  if (m.name === "write" || m.name === "edit") return [fileLink(args.path)];
  if (m.name === "run") {
    const code = element("div");
    markdown(code, `\`\`\`python\n${args?.code ?? ""}\n\`\`\``);
    return [code, element("pre", "output", m.content)];
  }
  const text = m.content.length > 600 ? `${m.content.slice(0, 600)}…` : m.content;
  return [element("p", "", text)];
}

// a file's name, or a skill's, as people say it
function named(path = "") {
  return path.startsWith("skills/") ? `the ${path.split("/")[1]} skill` : path;
}

// the files a reply's turn made or changed, that are still there
function made(m) {
  const messages = shown?.messages ?? [], names = [];
  for (let i = messages.indexOf(m) - 1; i >= 0 && messages[i].role !== "user"; i--) {
    for (const name of messages[i].info?.files ?? []) {
      if (!names.includes(name) && files.some((f) => f.name === name)) names.unshift(name);
    }
  }
  return names;
}

// a file's card, which opens or downloads it
function card(name) {
  const a = fileLink(name);
  a.className = "card";
  a.prepend(icon("read"));
  return a;
}

function fileLink(name) {
  const a = element("a", "", name);
  Object.assign(a, { href: `/files/${encodeURIComponent(name)}`, target: "_blank" });
  return a;
}

// a size in bytes, as people read it
function bytes(n) {
  return n < 1000 ? `${n} B` : n < 1e6 ? `${Math.round(n / 1e3)} KB` : `${(n / 1e6).toFixed(1)} MB`;
}

// the sources the conversation's tools numbered, by their numbers
function numbered() {
  const all = new Map();
  for (const m of shown?.messages ?? []) {
    const info = m.role === "tool" ? (m.info ?? {}) : {};
    for (const s of [info, ...(info.results ?? [])]) {
      if (s.n && s.url && !all.has(s.n)) all.set(s.n, { n: s.n, url: s.url, title: s.title });
    }
  }
  return all;
}

// links each citation in an element, [1], to its source, those linked before but
function cite(element) {
  const all = numbered();
  for (const sup of element.querySelectorAll("sup.cite:not(.linked)")) {
    const s = all.get(Number(sup.textContent));
    if (!s) continue;
    sup.replaceChildren(Object.assign(link(s.url, sup.textContent), { title: s.title || s.url }));
    sup.classList.add("linked");
  }
}

// a reply's sources: those it cites, in the order it first does, or the pages its turn read
function sources(m) {
  const all = numbered(), cited = [];
  for (const [, n] of (m.content ?? "").matchAll(/\[(\d{1,3})\](?!\()/g)) {
    const s = all.get(Number(n));
    if (s && !cited.includes(s)) cited.push(s);
  }
  if (cited.length) return cited;
  const messages = shown?.messages ?? [], pages = [];
  for (let i = messages.indexOf(m) - 1; i >= 0 && messages[i].role !== "user"; i--) {
    const { url, title, n } = (messages[i].role === "tool" && messages[i].info) || {};
    if (url && !pages.some((p) => p.url === url)) pages.unshift({ url, title, n });
  }
  return pages;
}

function source({ url, title, n }) {
  const a = link(url, n ? `${n} · ${host(url)}` : host(url));
  a.className = "source";
  a.title = title || url;
  return a;
}

// a link to an earlier conversation, which opens it here
function conversationLink({ id, title }) {
  const a = element("a", "", title);
  a.href = `/c/${id}`;
  a.onclick = (event) => {
    event.preventDefault();
    open(id);
  };
  return a;
}

// a link to a page of the web; of an address of another kind, as javascript:, its text alone
function link(url, text) {
  if (!/^https?:\/\//i.test(url ?? "")) return element("span", "", text);
  const a = element("a", "", text);
  Object.assign(a, { href: url, target: "_blank", rel: "noopener noreferrer" });
  return a;
}

// a list of links, each with its site if `hosts`
function listed(items, hosts = false) {
  const list = element("ol");
  list.append(...items.map((item) => {
    const li = element("li");
    li.append(item);
    if (hosts) li.append(element("span", "host", host(item.href)));
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
  recall: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
  remember: '<path d="M6 3h12v18l-6-4-6 4z"/>',
  read: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  write: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  run: '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
  schedule: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  unschedule: '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6m0-6-6 6"/>',
  tasks: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  weather: '<path d="M17.5 19H9a7 7 0 1 1 6.7-9h1.8a4.5 4.5 0 1 1 0 9z"/>',
  forget: '<path d="M6 3h12v18l-6-4-6 4z"/><path d="m10 8 4 4m0-4-4 4"/>',
  tool: '<circle cx="12" cy="12" r="3"/>',
};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.innerHTML = ICONS[name] ?? ICONS.tool;
  return svg;
}

// what a reply's info tells people: its model, its tokens, how fast they came, and whether it
// was stopped, or cut off
function describe({ model, tokens, rate, first, stopped, cut }) {
  let speed = tokens ? `${tokens} ${tokens === 1 ? "token" : "tokens"}` : "";
  if (rate) speed += `, ${rate.toFixed(1)} tok/s`;
  if (speed && first !== undefined) speed += `, the first after ${first.toFixed(1)} s`;
  const out = cut && "cut off: the conversation is out of room";
  return [model, speed, stopped && "stopped", out].filter(Boolean).join(" · ");
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
