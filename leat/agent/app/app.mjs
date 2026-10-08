// leat agent's app. It keeps nothing of its own: it shows the conversations on the box as the
// agent's events change them, the turns running there too, and sends what the user writes. A device
// is one of the household's, its person's, once paired: the first sets the home up, any other asks
// the owner to let it in.

import { markdown } from "/markdown.mjs";
import { THEMES, complete, file, properties, read, scheme, stylesheet } from "/themes.mjs";

const $ = (id) => document.getElementById(id);
let conversations = []; // the latest updated first: {id, title, updated, running}
let shown = null; // the conversation shown, with its messages; null for a new one
let memories = []; // what the agent remembers of the user, the oldest first
let forgotten = []; // the memories forgotten or changed, as they were, the latest first
let files = []; // the workspace's, the latest changed first: {name, size, modified}
let attached = []; // the files the next message attaches: {name, uploading}
let tasks = []; // the scheduled, the next due first: {id, prompt, schedule, conversation}
let telegram = null; // its bot, if connected, and the people allowed and asking
let me = null; // the person whose this device is: {person, name, owner, device}
let household = null; // the owner's to manage: its people and their devices, and those asking
let characters = []; // the household's, which a new chat may be with
let lists = []; // the household's, each with its items: {id, name, items: [{id, text}]}
let cast = null; // the character the next new chat is with, if one
// the conversations whose turns ended while another was shown, as this device saw them
const unread = new Set(JSON.parse(localStorage.getItem("leat.unread") ?? "[]"));
let models = [], loading = null, unreachable = null; // the engine's, a model it loads, or why not
let lost = false; // the events' connection, until it is back
let think = localStorage.getItem("leat.think") === "true"; // as the user last chose
// the themes of this device, Leat's own and those opened here, and the mode they are in
let themes = [...THEMES, ...kept()];
let mode = localStorage.getItem("leat.mode") ?? "system";
let views = []; // the shown messages' elements, by their indexes
const opened = new Map(); // whether each turn's work is open, as the user left it
const quietly = new Set(); // the conversations a check withdrew its turn from, finding nothing
// the pages beside the conversations, each a section of its own name, and their titles
const PAGES = ["memory", "files", "tasks", "lists", "characters", "settings"];
const TITLES = { memory: "Memory", files: "Files", tasks: "Tasks", lists: "Lists",
  characters: "Characters", settings: "Settings" };

$("new").onclick = () => {
  cast = null;
  open(null);
  $("input").focus();
};
$("menu").onclick = () => document.body.classList.toggle("menu");
$("remembered").onclick = () => turnTo("memory");
$("filed").onclick = () => turnTo("files");
$("timed").onclick = () => turnTo("tasks");
$("set").onclick = () => turnTo("settings");
$("cast").onclick = () => turnTo("characters");
$("listed").onclick = () => turnTo("lists");
$("new-list").onsubmit = async (event) => {
  event.preventDefault();
  const name = $("list-name").value.trim();
  if (!name) return;
  try {
    await post("/api/lists", { name });
    $("list-name").value = "";
  } catch (error) {
    status(error.message, true);
  }
};
$("character").onsubmit = async (event) => {
  event.preventDefault();
  const name = $("character-name").value.trim(), about = $("character-about").value.trim();
  if (!name || !about) return;
  try {
    await post("/api/characters", { name, about });
    $("character-name").value = $("character-about").value = "";
  } catch (error) {
    status(error.message, true);
  }
};
$("attach").onclick = () => pick(attach);
$("upload").onclick = () => pick(upload);
$("input").onpaste = (event) => { // images pasted, as a screenshot, attached to the next message
  const pasted = [...event.clipboardData.files];
  if (!pasted.length) return;
  event.preventDefault();
  pasted.forEach(attach);
};
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
$("save").onclick = () => {
  const theme = chosen(), a = element("a");
  Object.assign(a, { download: `${theme.name}.json`,
    href: URL.createObjectURL(new Blob([file(theme)], { type: "application/json" })) });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href));
};
$("open").onclick = () => pick(wear);
matchMedia("(prefers-color-scheme: dark)").onchange = renderAppearance; // the samples', in "system"

dress();
start();

// the app, of the person whose this device is; or, of a device not yet the household's, the gate
async function start() {
  const response = await fetch("/api/me").catch(() => null);
  if (!response?.ok && response?.status !== 401) { // the box away: tried again, in a while
    status("Reconnecting to the box…", true);
    return setTimeout(start, 3000);
  }
  status("");
  if (response.status === 401) return gate((await response.json()).empty);
  me = await response.json();
  document.body.classList.toggle("owner", me.owner);
  renderHousehold();
  const events = new EventSource("/api/events");
  events.onmessage = (event) => handle(JSON.parse(event.data));
  events.onerror = async () => {
    lost = true;
    status("Reconnecting to the box…", true);
    const response = await fetch("/api/me").catch(() => null);
    if (response?.status === 401) location.reload(); // unpaired: the gate
  };
  events.onopen = () => {
    if (!lost) return;
    lost = false;
    status("");
    if (shown) open(shown.id, false); // what changed meanwhile
  };
  route();
}

// the gate of a device not yet the household's: its first person sets the home up, and is its
// owner; any other asks to join, showing a code the owner's device shows too
function gate(empty) {
  document.body.classList.add("gated");
  $("welcome").textContent = empty
    ? "Welcome! This Leat is new. What's your name? You'll be the one who lets the others in."
    : "This device is not one of your home's yet. What's your name?";
  $("joining").textContent = empty ? "Start" : "Ask to join";
  $("join").onsubmit = async (event) => {
    event.preventDefault();
    const name = $("who").value.trim();
    if (!name) return;
    $("refusal").textContent = "";
    try {
      if (empty) return (await post("/api/setup", { name }), location.reload());
      const { id, code } = await (await post("/api/pairings", { name })).json();
      $("join").querySelector(".row").hidden = true;
      $("code").replaceChildren("Ask whoever set Leat up to let this device in, in its Settings, " +
        "where they will see this code:", element("b", "", `${code.slice(0, 3)} ${code.slice(3)}`));
      await waitToJoin(id);
    } catch (error) {
      $("refusal").textContent = error.message;
    }
  };
}

// asks after a request to join until it is answered: the app once let in
async function waitToJoin(id) {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const response = await fetch(`/api/pairings/${id}`).catch(() => null);
    if (response?.status === 200) return location.reload();
    if (response?.status === 404) {
      $("join").querySelector(".row").hidden = false;
      $("code").replaceChildren();
      throw new Error("The request was turned down, or waited too long: ask again.");
    }
  }
}

// the themes opened on this device, those that still are
function kept() {
  return JSON.parse(localStorage.getItem("leat.themes") ?? "[]").flatMap((theme) => {
    try {
      return [complete(theme)];
    } catch {
      return [];
    }
  });
}

// keeps the themes opened here, for the loads to come
function keep() {
  localStorage.setItem("leat.themes", JSON.stringify(themes.filter((t) => !THEMES.includes(t))));
}

function chosen() {
  return themes.find((t) => t.name === localStorage.getItem("leat.theme")) ?? THEMES[0];
}

// makes the app the theme chosen, in its mode, from the next load's first paint on too
function dress() {
  const css = stylesheet(chosen(), mode);
  $("theme").textContent = css;
  localStorage.setItem("leat.theme.css", css);
  renderAppearance();
}

// a theme from a file, Leat's, VS Code's or shadcn/ui's, kept and chosen; one of Leat's own
// names is theirs alone
async function wear(picked) {
  try {
    const theme = read(await picked.text(), picked.name);
    if (THEMES.some((t) => t.name === theme.name)) {
      throw new Error(`${theme.name} is one of Leat's own; give yours a name of its own`);
    }
    themes = [...themes.filter((t) => t.name !== theme.name), theme];
    keep();
    localStorage.setItem("leat.theme", theme.name);
    status("");
    dress();
  } catch (error) {
    status(`${picked.name} is no theme: ${error.message}`, true);
  }
}

// the modes, and the themes, each a sample of itself: a sidebar, a greeting, a message, the
// reply's lines and the composer, in its own fonts, corners and colors
function renderAppearance() {
  const theme = chosen(), alone = !theme.light || !theme.dark;
  $("modes").title = alone ? `${theme.name} is ${theme.light ? "light" : "dark"} alone` : "";
  $("modes").replaceChildren(...["system", "light", "dark"].map((m) => {
    const button = element("button", m === mode && !alone ? "on" : "", m[0].toUpperCase() + m.slice(1));
    button.disabled = alone;
    button.onclick = () => {
      mode = m;
      localStorage.setItem("leat.mode", m);
      dress();
    };
    return button;
  }));
  $("themes").replaceChildren(...themes.map((t) => {
    const item = element("div", t === theme ? "theme chosen" : "theme");
    const sample = element("button", "sample"), chat = element("span", "chat");
    const box = element("span", "box"), name = element("span", "name", t.name);
    for (const [property, value] of Object.entries(properties(t, scheme(t, mode)))) {
      sample.style.setProperty(property, value);
    }
    box.append(element("span", "send"));
    chat.append(element("span", "hello", "Aa"), element("span", "ask"), element("span", "said"),
      element("span", "said"), box);
    sample.append(element("span", "strip"), chat);
    sample.onclick = () => {
      localStorage.setItem("leat.theme", t.name);
      dress();
    };
    if (!THEMES.includes(t)) { // one opened here, to forget
      const remover = element("button", "", "×");
      remover.title = "Remove";
      remover.onclick = () => {
        themes = themes.filter((other) => other !== t);
        keep();
        dress();
      };
      name.append(remover);
    }
    item.append(sample, name);
    return item;
  }));
}

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
    case "withdrawn": // a check's turn, which found nothing to tell: as if it never ran
      quietly.add(event.conversation);
      if (showing(event.conversation)) {
        shown.messages.length = event.start;
        renderLog();
      }
      break;
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
      ({ memories, forgotten } = event);
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
    case "characters":
      characters = event.characters;
      renderCharacters();
      break;
    case "lists":
      lists = event.lists;
      renderLists();
      return;
    case "household":
      household = event;
      renderHousehold();
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
      if (event.error) status(event.error, true);
      else if (unreachable) status(""); // back
      unreachable = event.error ?? null;
      renderModels();
      break;
  }
  renderList();
  controls();
}

// a conversation's summary, new or changed: one whose turn ended unseen, unread
function update(c) {
  const before = conversations.find((other) => other.id === c.id);
  if (before?.running && !c.running && !quietly.delete(c.id) && shown?.id !== c.id) {
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
    const body = { content, think, files: names, ...(!shown && cast ? { character: cast } : {}) };
    const { id } = await (await post(path, body)).json();
    attached = [];
    cast = null;
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
  work: "Work", plans: "Plans", household: "Our home, everyone's" };
const ROOM = 3000; // characters, as the agent bounds them
function renderMemories() {
  const used = memories.reduce((n, m) => n + m.text.length, 0);
  $("room").textContent = memories.length ? `It is ${Math.round((100 * used) / ROOM)}% full.` : "";
  const lists = Object.entries(CATEGORIES).flatMap(([category, name]) => {
    const of = memories.filter((m) => m.category === category);
    if (!of.length) return [];
    const list = element("ul");
    list.append(...of.map((m) => {
      const item = element("li"), remover = element("button", "", "×");
      remover.title = "Forget";
      remover.onclick = () => fetch(`/api/memories/${m.id}`, { method: "DELETE" });
      item.append(element("span", "", m.text), element("span", "meta", dated(m)), remover);
      return item;
    }));
    return [element("h2", "", name), list];
  });
  if (forgotten.length) { // what was forgotten or changed, as it was, to undo
    const list = element("ul");
    list.append(...forgotten.map((f) => {
      const item = element("li"), undo = element("button", "undo", "Undo");
      undo.onclick = () => post("/api/memories/restore", { id: f.id }).catch((e) => status(e.message, true));
      const how = `${f.change === "replaced" ? "changed" : "forgotten"} ${BY[f.by] ?? ""}`;
      item.append(element("span", "", f.text), element("span", "meta", `${how} · ${day(f.at)}`), undo);
      return item;
    }));
    lists.push(element("h2", "", "Recently forgotten or changed"), list);
  }
  $("memories").replaceChildren(...lists);
}

// who forgot or changed a memory, in words
const BY = { app: "by you", conversation: "in a chat", review: "after a chat", tidy: "while tidying" };

// a memory's date: when it was last said, or of a plan, until when it holds, or that it passed
function dated(m) {
  if (!m.until) return day(m.confirmed ?? m.created);
  const until = new Date(`${m.until}T23:59:59`);
  return `${until < new Date() ? "passed" : "until"} ${day(until / 1000)}`;
}

// a time, in seconds, as a day people read
function day(seconds) {
  return new Date(seconds * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

// tasks one may schedule in a click: a morning brief, the week ahead, and a check for rain
const SUGGESTED = [
  ["A morning brief, each day at 8:00", "Give the user a short morning brief: today's weather where they live, and the main news.", "08:00", "daily"],
  ["The week ahead, on Sundays at 18:00", "Help the user plan the week ahead: ask what is coming up, and suggest what to prepare.", sunday(), "weekly"],
  ["A word the evening before rain, at 19:00", "Look at tomorrow's weather where the user lives.", "19:00", "daily", "it will rain"],
];

function renderTasks() {
  $("scheduled").replaceChildren(...tasks.map((t) => {
    const item = element("li"), about = element("div"), remover = element("button", "", "×");
    remover.title = "Cancel";
    remover.onclick = () => fetch(`/api/tasks/${t.id}`, { method: "DELETE" });
    const now = element("button", "talk", "Run now");
    now.onclick = async () => {
      try {
        open((await (await post(`/api/tasks/${t.id}/run`, {})).json()).id);
      } catch (error) {
        status(error.message, true);
      }
    };
    about.append(element("span", "", t.prompt), element("span", "meta", t.schedule));
    if (t.conversation) about.append(conversationLink({ id: t.conversation, title: "Its chat" }));
    item.append(about, now, remover);
    return item;
  }));
  const unset = SUGGESTED.filter(([, prompt]) => !tasks.some((t) => t.prompt === prompt));
  $("suggested").replaceChildren(...unset.map(([label, prompt, at, repeat, only_if]) => {
    const add = element("button", "", `+ ${label}`);
    add.onclick = () => post("/api/tasks", { prompt, at, repeat, ...(only_if ? { only_if } : {}) })
      .catch((error) => status(error.message, true));
    return add;
  }));
  const secure = window.isSecureContext && "Notification" in window;
  $("notifying").replaceChildren();
  if (secure && Notification.permission === "default") {
    const ask = element("button", "", "Notify me when one is done");
    ask.onclick = async () => (await Notification.requestPermission(), renderTasks());
    $("notifying").append(ask);
  }
}

// the household's people, their devices, and the devices asking to join, which the owner lets in
// as a person known or new, once the code they show is the one their device shows; for any other,
// who they are here
function renderHousehold() {
  const box = $("household");
  const unpair = element("button", "", "Unpair this device");
  unpair.onclick = () => confirm("Unpair this device? Using Leat on it again takes asking to join.")
    && fetch(`/api/devices/${me.device}`, { method: "DELETE" }).then(() => location.reload());
  const you = element("p", "meta", `You are ${me?.name ?? ""} here. `);
  you.append(unpair);
  if (!me?.owner || !household) return box.replaceChildren(you);
  const asking = household.requests.map((r) => {
    const item = element("li"), about = element("div");
    const code = element("b", "", `${r.code.slice(0, 3)} ${r.code.slice(3)}`);
    const meta = element("span", "meta", `${r.device} · code `);
    meta.append(code);
    about.append(element("span", "", `${r.name} asks to join`), meta);
    const as = element("select");
    as.append(new Option(`as someone new, ${r.name}`, ""),
      ...household.people.map((p) => new Option(`as ${p.name}`, p.id)));
    const allow = element("button", "allow", "Let in"), no = element("button", "", "×");
    allow.onclick = () => post(`/api/pairings/${r.id}/allow`, as.value ? { person: Number(as.value) } : {})
      .catch((error) => status(error.message, true));
    no.title = "Turn down";
    no.onclick = () => fetch(`/api/pairings/${r.id}`, { method: "DELETE" });
    item.append(about, as, allow, no);
    return item;
  });
  const people = household.people.flatMap((p) => {
    const devices = element("ul");
    devices.append(...p.devices.map((d) => {
      const item = element("li"), remover = element("button", "", "×");
      remover.title = "Unpair";
      remover.hidden = d.id === me.device;
      remover.onclick = () => fetch(`/api/devices/${d.id}`, { method: "DELETE" });
      item.append(element("span", "", d.name), element("span", "meta", `seen ${day(d.seen)}`), remover);
      return item;
    }));
    const heading = element("h3", "", p.owner ? `${p.name} (owner)` : p.name);
    if (!p.owner) { // whether a child, whose new chats keep to a child's rules
      const child = element("label", "child"), box = element("input");
      Object.assign(box, { type: "checkbox", checked: Boolean(p.child) });
      box.onchange = () => post(`/api/people/${p.id}`, { child: box.checked }).catch((e) => status(e.message, true));
      child.append(box, " a child");
      child.title = "Leat keeps to what suits a child, in every new chat";
      heading.append(" ", child);
    }
    if (!p.owner) {
      const remover = element("button", "", "Remove");
      remover.onclick = () => confirm(`Remove ${p.name}, with all their chats, memories and tasks?`)
        && fetch(`/api/people/${p.id}`, { method: "DELETE" });
      heading.append(" ", remover);
    }
    return [heading, devices];
  });
  const requests = element("ul");
  requests.append(...asking);
  const how = element("p", "meta", "To add someone, open Leat on their device: it asks to join, and shows here.");
  box.replaceChildren(...(asking.length ? [requests] : []), ...people, how);
}

// the household's lists: each its things, to tick off, and a line to add one
function renderLists() {
  $("shared").replaceChildren(...lists.flatMap((list) => {
    const heading = element("h2", "", list.name), remover = element("button", "", "Remove");
    remover.onclick = () => confirm(`Remove the list ${list.name}?`)
      && fetch(`/api/lists/${list.id}`, { method: "DELETE" });
    heading.append(remover);
    const items = element("ul");
    items.append(...list.items.map((item) => {
      const row = element("li"), tick = element("button", "tick");
      tick.title = "Tick off";
      tick.onclick = () => fetch(`/api/lists/items/${item.id}`, { method: "DELETE" });
      row.append(tick, element("span", "", item.text));
      return row;
    }));
    const adding = element("form"), input = element("input");
    Object.assign(input, { placeholder: `Add to ${list.name}`, maxLength: 120, autocomplete: "off" });
    adding.append(input);
    adding.onsubmit = async (event) => {
      event.preventDefault();
      if (!input.value.trim()) return;
      await post(`/api/lists/${list.id}/items`, { text: input.value }).catch((e) => status(e.message, true));
      input.value = "";
    };
    return [heading, items, adding];
  }));
}

// characters one may add in a click, as a start
const PRESETS = [
  ["Tutor", "A patient tutor for any school subject. Asks what the learner knows already, explains one step at a time with an example, and asks a question to check before going on. Helps with homework without simply giving the answers."],
  ["Language partner", "A friendly partner to practise a language with. Speaks only the language the user wants to practise, in simple sentences, and gently corrects their mistakes after each reply."],
  ["Storyteller", "Makes up stories together with the user: begins one in the world they ask for, stops at the moments where they choose what happens next, and keeps every story kind and fit for children."],
];

// the household's characters, each to talk to in a new chat, and those to add in a click
function renderCharacters() {
  $("cast-list").replaceChildren(...characters.map((c) => {
    const item = element("li"), about = element("div");
    const brief = c.about.length > 140 ? `${c.about.slice(0, 140)}…` : c.about;
    about.append(element("span", "", c.name), element("span", "meta", brief));
    const talk = element("button", "talk", "Talk"), remover = element("button", "", "×");
    talk.onclick = () => talkTo(c.id);
    remover.title = "Remove";
    remover.onclick = () => fetch(`/api/characters/${c.id}`, { method: "DELETE" });
    item.append(about, talk, remover);
    return item;
  }));
  const unadded = PRESETS.filter(([name]) => !characters.some((c) => c.name === name));
  $("presets").replaceChildren(...unadded.map(([name, about]) => {
    const add = element("button", "", `+ ${name}`);
    add.type = "button";
    add.onclick = () => post("/api/characters", { name, about }).catch((e) => status(e.message, true));
    return add;
  }));
  renderWith();
}

// begins a new chat with a character
function talkTo(id) {
  open(null);
  cast = id;
  renderWith();
  controls();
  $("input").focus();
}

// whom the next new chat is with, if not Leat itself
function renderWith() {
  const c = !shown && characters.find((other) => other.id === cast);
  if (!c) return $("with").replaceChildren();
  const leave = element("button", "", "×");
  leave.title = "Talk to Leat instead";
  leave.onclick = () => {
    cast = null;
    renderWith();
    controls();
  };
  $("with").replaceChildren(`With ${c.name}`, leave);
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
  const people = household?.people ?? [];
  const whose = (id) => people.find((p) => p.id === id)?.name ?? "no one yet";
  const person = (p, asking) => { // one asking is let in as a person of the household's
    const item = element("li"), about = element("div");
    about.append(element("span", "", p.name),
      element("span", "meta", asking ? "asks to talk to Leat" : `talks as ${whose(p.person)}`));
    const as = element("select");
    as.append(...people.map((q) => new Option(`as ${q.name}`, q.id)));
    const yes = element("button", "allow", "Allow"), no = element("button", "", "×");
    yes.onclick = () => post("/api/telegram/people", { id: p.id, person: Number(as.value) })
      .catch((error) => status(error.message, true));
    no.title = asking ? "Turn down" : "Remove";
    no.onclick = () => fetch(`/api/telegram/people/${p.id}`, { method: "DELETE" });
    item.append(about, ...(asking ? [as, yes] : []), no);
    return item;
  };
  const list = element("ul");
  list.append(...telegram.requests.map((p) => person(p, true)), ...telegram.allowed.map((p) => person(p, false)));
  const none = element("p", "meta", `No one yet: write to @${telegram.bot}, then allow yourself here.`);
  box.replaceChildren(connected, telegram.requests.length + telegram.allowed.length ? list : none);
}

// the next Sunday at 18:00, as a task's first time: "YYYY-MM-DD 18:00"
function sunday() {
  const day = new Date();
  day.setDate(day.getDate() + ((7 - day.getDay()) % 7 || 7));
  const pad = (n) => String(n).padStart(2, "0");
  return `${day.getFullYear()}-${pad(day.getMonth() + 1)}-${pad(day.getDate())} 18:00`;
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
    item.append(fileLink(f.name), element("span", "meta", `${bytes(f.size)} · ${day(f.modified)}`), remover);
    return item;
  }));
  // the cards of the files its answers made, where the user reads
  if (shown?.messages) follow(() => views.forEach((view) => view.update()));
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
    if (!a.uploading && PICTURE.test(a.name)) chip.prepend(picture(a.name));
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
  renderWith();
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
    const title = element("span", "", c.title), played = characters.find((p) => p.id === c.character);
    if (played || c.shared) title.prepend(element("span", "with", `${played?.name ?? "In a group"} · `));
    item.append(title, remover);
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
  add_to_list: ["added to a list", (n) => `added to lists ${n} times`],
  check_off: ["checked things off", () => "checked things off"],
  lists: ["looked at the lists", () => "looked at the lists"],
};

// what a turn's calls did, in a few words, in the order it began them; those that failed not
function summary(all) {
  const calls = all.filter((m) => !m.info?.error);
  const parts = [...new Set(calls.map((m) => m.name))].map((name) => {
    const n = calls.filter((m) => m.name === name).length;
    const [one, many] = own(DID, name) ?? [`used ${name}`, () => `used ${name}`];
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
  const played = characters.find((c) => c.id === cast);
  $("greeting").textContent = unreachable ? "The engine is not reachable"
    : loading ? "Loading…" : !model ? "Choose a model" : played ? `Talk to ${played.name}` : "How can I help?";
  $("think").classList.toggle("on", think);
  $("send").classList.toggle("stop", running);
  $("send").title = running ? "Stop" : "Send";
  const uploading = attached.some((a) => a.uploading);
  $("send").disabled = !running && !(model && input.value.trim() && !uploading);
  $("model").value = loading ?? model ?? "";
  $("model").disabled = !me?.owner || loading !== null || conversations.some((c) => c.running);
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
  add_to_list: (a, i) => [`Adding to ${a.list}…`, `Added to ${i.list ?? a.list}: ${(i.added ?? []).join(", ")}`,
    `Couldn't add to ${a.list}`],
  check_off: (a, i) => [`Checking off ${a.list}…`, `Checked off ${(i.done ?? []).join(", ")}`,
    `Couldn't check off ${a.list}`],
  lists: (a) => ["Looking at the lists…", a.list ? `Looked at ${a.list}` : "Looked at the lists",
    "Couldn't look at the lists"],
  run: (a, i) => ["Running code…", i.status === 0 ? "Ran code"
    : i.status === null ? "Ran code, out of time" : "Ran code, which failed", "Couldn't run code"],
};

function line(m) {
  const { arguments: args, error, stopped } = m.info ?? {};
  const lines = own(LINES, m.name) ?? (() => [`${m.name}…`, m.name, `${m.name} failed`]);
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
  if (m.name === "fetch") { // and the question it was read for, if one
    const asked = m.info?.question ? [element("p", "meta", `Read for: ${m.info.question}`)] : [];
    return [link(url, title || url), ...asked];
  }
  if (m.name === "recall" && conversations?.length) return [listed(conversations.map(conversationLink))];
  if (m.name === "write" || m.name === "edit") return [fileLink(args.path)];
  if (m.info?.images) return [cards(m.info.images)]; // an image read, which the model saw
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

// a file's card, which opens or downloads it: an image's, the image
function card(name) {
  const a = fileLink(name);
  if (PICTURE.test(name)) {
    a.className = "picture";
    a.title = name;
    a.replaceChildren(picture(name));
    return a;
  }
  a.className = "card";
  a.prepend(icon("read"));
  return a;
}

function cards(names) {
  const shown = element("div", "cards");
  shown.append(...names.map(card));
  return shown;
}

// the files a page shows as images, by their names
const PICTURE = /\.(png|jpe?g|gif|webp)$/i;

function picture(name) {
  const img = element("img");
  Object.assign(img, { src: `/files/${encodeURIComponent(name)}`, alt: name, loading: "lazy" });
  return img;
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
  add_to_list: '<path d="M9 6h12M9 12h12M9 18h6M3 6h.01M3 12h.01M18 15v6M15 18h6"/>',
  check_off: '<path d="M9 6h12M9 12h12M9 18h12"/><path d="m3 12 1.5 1.5L7 11"/>',
  lists: '<path d="M9 6h12M9 12h12M9 18h12M3 6h.01M3 12h.01M3 18h.01"/>',
  weather: '<path d="M17.5 19H9a7 7 0 1 1 6.7-9h1.8a4.5 4.5 0 1 1 0 9z"/>',
  forget: '<path d="M6 3h12v18l-6-4-6 4z"/><path d="m10 8 4 4m0-4-4 4"/>',
  tool: '<circle cx="12" cy="12" r="3"/>',
};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.innerHTML = own(ICONS, name) ?? ICONS.tool;
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

// a table's entry of a name, as a tool's the model gave; none of what every object has, as
// "constructor"
function own(table, name) {
  return Object.hasOwn(table, name) ? table[name] : undefined;
}

function element(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text) e.textContent = text;
  return e;
}
