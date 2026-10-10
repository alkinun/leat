// leat agent's app. It keeps nothing of its own: it shows the conversations on the box as the
// agent's events change them, the turns running there too, and sends what the user writes. A device
// is its person's once paired: the first sets the box up, any other asks the owner to let it in.

import { markdown } from "/markdown.mjs";

const $ = (id) => document.getElementById(id);
let conversations = []; // the latest updated first: {id, title, updated, project, running}
let shown = null; // the conversation shown, with its messages; null for a new one
// the projects this person sees, the latest updated first, once told: {id, name, instructions,
// person, shared, updated}
let projects = null;
let viewing = null; // the project whose page is shown
// the files of each space, the latest changed first, by its project's id, "" for the person's own:
// {name, size, modified}
const spaces = {};
let attached = []; // the files the next message attaches: {name, uploading}
let me = null; // the person whose this device is: {person, name, owner, device}
let accounts = null; // the owner's to manage: the people and their devices, and those asking
// the conversations whose turns ended while another was shown, as this device saw them
const unread = new Set(JSON.parse(localStorage.getItem("leat.unread") ?? "[]"));
let models = [], loading = null, unreachable = null; // the engine's, a model it loads, or why not
let lost = false; // the events' connection, until it is back
let mode = localStorage.getItem("leat.mode") ?? "system"; // light, dark, or as the system is
let views = []; // the shown messages' elements, by their indexes
let heading = "leat"; // the shown page's title, as the window's says it
const opened = new Map(); // whether each fold is open, as the user left it: work, calls, reasoning
// the pages beside the conversations, each a section of its own name, and their titles; a
// project's page, "project", has its name
const PAGES = ["projects", "project", "files", "settings"];
const TITLES = { projects: "Projects", files: "Files", settings: "Settings" };

$("new").onclick = () => {
  open(null);
  $("input").focus();
};
$("menu").onclick = () => document.body.classList.toggle("menu");
$("held").onclick = () => turnTo("projects");
$("filed").onclick = () => turnTo("files");
$("set").onclick = () => turnTo("settings");
$("attach").onclick = () => pick(attach);
$("upload").onclick = () => pick((file) => upload(file, null));
$("adding").onclick = () => pick((file) => upload(file, viewing));
$("create").onsubmit = create;
$("back").onclick = (event) => {
  event.preventDefault();
  turnTo("projects");
};
$("crumb").onclick = (event) => {
  event.preventDefault();
  showProject(shown.project);
};
$("title").onkeydown = (event) => event.key === "Enter" && $("title").blur();
$("title").onchange = () => { // renamed, unless to nothing
  const name = $("title").value.trim();
  if (name) change(viewing, { name });
  else renderProject();
};
let saving = null; // the instructions' save, a moment after the user stops typing
$("instructions").oninput = () => {
  clearTimeout(saving);
  const id = viewing, instructions = $("instructions").value;
  saving = setTimeout(async () => {
    if (await change(id, { instructions })) $("saved").textContent = "Saved. They hold for chats begun from now on.";
  }, 600);
};
$("discard").onclick = async () => {
  const p = project(viewing);
  if (!confirm(`Delete “${p.name}”, with its files and every chat in it? It cannot be undone.`)) return;
  if (await del(`/api/projects/${p.id}`)) turnTo("projects");
};
$("input").onpaste = (event) => { // images pasted, as a screenshot, attached to the next message
  const pasted = [...event.clipboardData.files]; // but text pasted with a picture of it, as Office's
  if (!pasted.length || event.clipboardData.getData("text/plain")) return;
  event.preventDefault();
  pasted.forEach(attach);
};
window.ondragover = (event) => event.preventDefault();
window.ondrop = (event) => { // files dropped, attached to the next message, or on their page uploaded
  event.preventDefault();
  const dropped = [...event.dataTransfer.files];
  if (paged("files")) dropped.forEach((file) => upload(file, null));
  else if (paged("project")) dropped.forEach((file) => upload(file, viewing));
  else dropped.forEach(attach);
};
$("effort").onchange = () => { // kept for the model, on this device
  localStorage.setItem(`leat.effort.${ready()}`, $("effort").value);
  renderEffort();
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

renderModes();
start();

// the app, of the person whose this device is; or, of a device not yet paired, the gate
async function start() {
  const response = await fetch("/api/me").catch(() => null);
  if (!response?.ok && response?.status !== 401) { // the box away: tried again, in a while
    status("Reconnecting to the box…", true);
    return setTimeout(start, 3000);
  }
  status("");
  if (response.status === 401) return gate((await response.json()).empty);
  me = await response.json();
  renderAccounts();
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

// the gate of a device not yet paired: the box's first person sets it up, and is its owner; any
// other asks to join, showing a code the owner's device shows too
function gate(empty) {
  document.body.classList.add("gated");
  $("welcome").textContent = empty
    ? "Welcome! This Leat is new. What's your name? You'll be the one who lets the others in."
    : "This device has not joined this Leat yet. What's your name?";
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
    const response = await fetch(`/api/pairings/${id}`, { method: "POST" }).catch(() => null);
    if (response?.status === 200) return location.reload();
    if (response?.status === 404) {
      $("join").querySelector(".row").hidden = false;
      $("code").replaceChildren();
      throw new Error("The request was turned down, or waited too long: ask again.");
    }
  }
}

// the modes Leat's look may be in, the system's by default, as this device last chose
function renderModes() {
  document.documentElement.dataset.mode = mode;
  $("modes").replaceChildren(...["system", "light", "dark"].map((m) => {
    const button = element("button", m === mode ? "on" : "", m[0].toUpperCase() + m.slice(1));
    button.onclick = () => {
      mode = m;
      localStorage.setItem("leat.mode", m);
      renderModes();
    };
    return button;
  }));
}

// shows what the address names: a page, as the files', a project's, a conversation, or a new one
function route() {
  const name = location.pathname.slice(1);
  const held = location.pathname.match(/^\/projects\/([0-9a-f]{12})$/)?.[1];
  if (held) showProject(held, false);
  else if (PAGES.includes(name) && name !== "project") turnTo(name, false);
  else open(addressed(), false);
}

function handle(event) {
  // the shown conversation's, while it loads: taken after it, which may have come before them
  const of = event.conversation?.id ?? event.conversation;
  if (shown?.messages === null && of === shown.id) return shown.later.push(event);
  switch (event.type) {
    case "conversations":
      conversations = event.conversations;
      for (const id of unread) if (!conversations.some((c) => c.id === id)) markRead(id); // gone
      break;
    case "conversation":
      update(event.conversation);
      break;
    case "deleted":
      conversations = conversations.filter((c) => c.id !== event.id);
      markRead(event.id);
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
        shown.messages.length = Math.min(event.start, shown.messages.length);
        for (const key of opened.keys()) { // the folds of those taken back, for the next not to open
          const [id, , index] = key.split(" ");
          if (id === event.conversation && Number(index) >= event.start) opened.delete(key);
        }
        renderLog();
      }
      if (shown?.id === event.conversation) {
        status(event.error, true);
        if (!$("input").value) $("input").value = event.content;
      }
      break;
    case "projects":
      projects = event.projects;
      if (viewing && !project(viewing)) { // gone, or no longer shared
        history.replaceState(null, "", "/projects");
        turnTo("projects", false);
      }
      renderProjects();
      break;
    case "files":
      spaces[event.project ?? ""] = event.files;
      renderFiles();
      renderProjects(); // as they count them
      return;
    case "accounts":
      accounts = event;
      renderAccounts();
      return;
    case "loading":
      loading = event.model;
      renderEffort(); // none, till it is loaded
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
  if (before?.running && !c.running && shown?.id !== c.id) {
    unread.add(c.id);
    localStorage.setItem("leat.unread", JSON.stringify([...unread]));
  }
  conversations = [c, ...conversations.filter((other) => other.id !== c.id)];
  conversations.sort((a, b) => b.updated - a.updated);
  if (shown?.id !== c.id) return;
  const ended = shown.running && !c.running;
  Object.assign(shown, { title: c.title, running: c.running });
  // done working: its turn shown again, kept scrolled to the end if it was
  if (ended && shown.messages?.length) follow(() => refresh(shown.messages.length - 1));
}

// a conversation no longer unread, as this device keeps them: seen, or gone
function markRead(id) {
  if (unread.delete(id)) localStorage.setItem("leat.unread", JSON.stringify([...unread]));
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
  const was = answers(m);
  m[key] = (m[key] ?? "").slice(0, at) + text;
  // shown again where it now belongs, once its text shows it the answer rather than a step
  follow(() => (answers(m) === was ? views[index].update() : refresh(index)));
}

// shows a conversation, or a new one, at its own address
async function open(id, push = true) {
  if (push) history.pushState(null, "", id ? `/c/${id}` : "/");
  markRead(id);
  document.body.classList.remove("menu", ...PAGES);
  viewing = null;
  const running = conversations.find((c) => c.id === id)?.running ?? false;
  // until it comes; then shown, unless another was opened since, or this one again
  const pending = id ? { id, title: "", messages: null, running, later: [] } : null;
  shown = pending;
  render();
  if (!id) return;
  try {
    const response = await fetch(`/api/conversations/${id}`);
    if (!response.ok) throw new Error((await response.json()).error.message);
    const c = await response.json();
    if (shown !== pending) return;
    shown = c;
    render();
    pending.later.forEach(handle);
  } catch (error) {
    if (shown !== pending) return;
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
    const sent = attached, from = location.pathname;
    const files = sent.map((a) => a.name);
    const effort = chosen(models.find((m) => m.id === ready())) ?? undefined; // none, if it has none
    const project = shown ? undefined : (viewing ?? undefined); // a new one's, begun on its page
    const { id } = await (await post(path, { content, effort, files, project })).json();
    attached = attached.filter((a) => !sent.includes(a)); // not those added meanwhile
    renderAttached();
    if (location.pathname === from && shown?.id !== id) open(id); // unless another was opened meanwhile
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

// deletes a conversation, once the user says so: it cannot be had back
function remove(c) {
  if (confirm(`Delete “${c.title}”? It cannot be undone.`)) del(`/api/conversations/${c.id}`);
}

// deletes what a path names; whether it did, saying why not if not
async function del(path) {
  try {
    const response = await fetch(path, { method: "DELETE" });
    if (!response.ok) throw new Error((await response.json()).error.message);
    return true;
  } catch (error) {
    status(error.message, true);
    return false;
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

async function post(path, body, method = "POST") {
  const response = await fetch(path, { method, body: JSON.stringify(body) });
  if (!response.ok) throw new Error((await response.json()).error.message);
  return response;
}

// makes a project of the name the user gave, and shows its page
async function create(event) {
  event.preventDefault();
  const name = $("naming").value.trim();
  if (!name) return;
  try {
    const made = await (await post("/api/projects", { name })).json();
    $("naming").value = "";
    projects = [made, ...(projects ?? []).filter((p) => p.id !== made.id)]; // before its event
    showProject(made.id);
  } catch (error) {
    status(error.message, true);
  }
}

// changes a project's name, instructions or sharing; whether it did, saying why not if not
async function change(id, fields) {
  try {
    await post(`/api/projects/${id}`, fields, "PATCH");
    return true;
  } catch (error) {
    status(error.message, true);
    return false;
  }
}

// shows a page, in place of the conversation
function turnTo(name, push = true) {
  if (push) history.pushState(null, "", `/${name}`);
  document.body.classList.remove("menu", ...PAGES);
  document.body.classList.add(name);
  shown = viewing = null;
  heading = TITLES[name];
  render();
}

// shows a project's page: its chats, files and instructions, and the composer, which begins a chat
// in it
function showProject(id, push = true) {
  if (push) history.pushState(null, "", `/projects/${id}`);
  document.body.classList.remove("menu", ...PAGES);
  document.body.classList.add("project");
  if (viewing !== id) $("saved").textContent = "They hold for chats begun from now on.";
  shown = null;
  viewing = id;
  render();
  $("input").focus();
}

// a project this person sees, by its id
function project(id) {
  return projects?.find((p) => p.id === id);
}

// whether a page is shown
function paged(name) {
  return document.body.classList.contains(name);
}

// a time, in seconds, as a day people read
function day(seconds) {
  return new Date(seconds * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

// the box's people, their devices, and the devices asking to join, which the owner lets in as a
// person known or new, once the code they show is the one their device shows; for any other, who
// they are here
function renderAccounts() {
  const box = $("accounts");
  const unpair = element("button", "", "Unpair this device");
  unpair.onclick = () => confirm("Unpair this device? Using Leat on it again takes asking to join.")
    && del(`/api/devices/${me.device}`).then((done) => done && location.reload());
  const you = element("p", "meta", `You are ${me?.name ?? ""} here. `);
  you.append(unpair);
  if (!me?.owner || !accounts) return box.replaceChildren(you);
  const asking = accounts.requests.map((r) => {
    const item = element("li"), about = element("div");
    const code = element("b", "", `${r.code.slice(0, 3)} ${r.code.slice(3)}`);
    const meta = element("span", "meta", `${r.device} · code `);
    meta.append(code);
    about.append(element("span", "", `${r.name} asks to join`), meta);
    const as = element("select");
    as.append(new Option(`as someone new, ${r.name}`, ""),
      ...accounts.people.map((p) => new Option(`as ${p.name}`, p.id)));
    const allow = element("button", "allow", "Let in"), no = element("button", "", "×");
    allow.onclick = () => post(`/api/pairings/${r.id}/allow`, as.value ? { person: Number(as.value) } : {})
      .catch((error) => status(error.message, true));
    no.title = "Turn down";
    no.onclick = () => del(`/api/pairings/${r.id}`);
    item.append(about, as, allow, no);
    return item;
  });
  const people = accounts.people.flatMap((p) => {
    const devices = element("ul");
    devices.append(...p.devices.map((d) => {
      const item = element("li"), remover = element("button", "", "×");
      remover.title = "Unpair";
      remover.hidden = d.id === me.device;
      remover.onclick = () => del(`/api/devices/${d.id}`);
      item.append(element("span", "", d.name), element("span", "meta", `seen ${day(d.seen)}`), remover);
      return item;
    }));
    const heading = element("h3", "", p.owner ? `${p.name} (owner)` : p.name);
    if (!p.owner) {
      const remover = element("button", "", "Remove");
      remover.onclick = () => confirm(`Remove ${p.name}, with all their chats?`)
        && del(`/api/people/${p.id}`);
      heading.append(" ", remover);
    }
    return [heading, devices];
  });
  const requests = element("ul");
  requests.append(...asking);
  const how = element("p", "meta", "To add someone, open Leat on their device: it asks to join, and shows here.");
  box.replaceChildren(...(asking.length ? [requests] : []), ...people, how);
}

function renderFiles() {
  $("workspace").replaceChildren(...listFiles(null));
  $("kept").replaceChildren(...(viewing ? listFiles(viewing) : []));
  // the cards of the files its answers made, where the user reads
  if (shown?.messages) follow(() => views.forEach((view) => view.update()));
}

// a space's files, a project's or the person's own, each a row that opens it and deletes it
function listFiles(project) {
  return filesOf(project).map((f) => {
    const item = element("li"), remover = element("button", "", "×");
    remover.title = "Delete";
    remover.onclick = () => confirm(`Delete “${f.name}”? It cannot be undone.`)
      && del(`/api${place(project)}/files/${encodeURIComponent(f.name)}`);
    const about = element("span", "meta", [STATES[f.state], bytes(f.size), day(f.modified)].filter(Boolean).join(" · "));
    about.title = f.error ?? (f.state === "scanned" ? "Its scanned pages are read once a model that sees images is loaded" : "");
    about.classList.toggle("failed", f.state === "failed");
    item.append(fileLink(f.name, project), about, remover);
    return item;
  });
}

// what a file's state in the index says, while it is not read: being read, or not readable
const STATES = { reading: "Reading…", failed: "Couldn't read", scanned: "Scanned" };

// the files of a project's space, or of the person's own
function filesOf(project) {
  return spaces[project ?? ""] ?? [];
}

// where a space's files are, of a project's or the person's own: their addresses' start
function place(project) {
  return project ? `/projects/${project}` : "";
}

// the space the user works in: the shown conversation's, or the project's whose page is shown
function here() {
  return shown ? (shown.project ?? null) : viewing;
}

// the projects, as their page lists them
function renderProjects() {
  $("listed").replaceChildren(...(projects ?? []).map((p) => {
    const item = element("li"), a = element("a", "", p.name);
    a.href = `/projects/${p.id}`;
    a.onclick = (event) => {
      event.preventDefault();
      showProject(p.id);
    };
    const n = filesOf(p.id).length;
    const about = [p.shared ? "Shared" : "Only you", `${n} ${n === 1 ? "file" : "files"}`, day(p.updated)];
    item.append(a, element("span", "meta", about.join(" · ")));
    return item;
  }));
  renderProject();
}

// the shown project's page: its name, whom it is shared with, its chats, files and instructions
function renderProject() {
  const p = project(viewing);
  if (!p) return;
  heading = p.name;
  if (document.activeElement !== $("title")) $("title").value = p.name;
  if (document.activeElement !== $("instructions")) $("instructions").value = p.instructions;
  const mine = p.person === me?.person;
  $("sharing").replaceChildren(...[[false, "Only me"], [true, "Everyone"]].map(([shared, said]) => {
    const button = element("button", shared === p.shared ? "on" : "", said);
    button.disabled = !mine;
    button.title = mine ? "" : "Only whoever made the project can change whom it is shared with";
    button.onclick = () => shared !== p.shared && change(p.id, { shared });
    return button;
  }));
  $("discard").hidden = !mine && !me?.owner;
  $("chats").replaceChildren(...conversations.filter((c) => c.project === p.id).map((c) => {
    const item = element("li"), a = element("a", "", c.title);
    a.href = `/c/${c.id}`;
    a.onclick = (event) => {
      event.preventDefault();
      open(c.id);
    };
    item.append(a, element("span", "meta", day(c.updated)));
    return item;
  }));
}

// asks the user for files, and hands them to `take`, each
function pick(take) {
  const picker = element("input");
  Object.assign(picker, { type: "file", multiple: true });
  picker.onchange = () => [...picker.files].forEach(take);
  picker.click();
}

// uploads a file to a project's files, or the person's own; returns the name it got there, or null
// if it did not
async function upload(file, project) {
  try {
    const path = `/api${place(project)}/files/${encodeURIComponent(file.name)}`;
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
  chip.name = await upload(file, here());
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
  renderFiles();
  controls();
}

function renderList() {
  renderProject(); // its chats
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
  renderEffort();
}

// the efforts the model reasons at, as the composer names them: of two, the first none, off and on
const EFFORTS = { none: "Off", minimal: "Minimal", low: "Low", medium: "Medium", high: "High", xhigh: "Max" };

// the efforts the loaded model takes, as leat serve lists them, to choose among: none where the
// model has no choice, as one that never reasons, or always does
function renderEffort() {
  const model = models.find((m) => m.id === ready()), efforts = model?.reasoning?.efforts ?? [];
  const toggle = efforts.length === 2 && efforts[0] === "none";
  $("effort").replaceChildren(...efforts.map((e) =>
    new Option(toggle ? (e === "none" ? "Off" : "On") : (EFFORTS[e] ?? e), e)));
  $("effort").value = chosen(model) ?? "";
  $("reasoning").hidden = efforts.length < 2;
  $("reasoning").classList.toggle("on", (chosen(model) ?? "none") !== "none");
}

// the effort the model is to reason at: as last chosen for it on this device, else its default;
// none of a model that takes no effort
function chosen(model) {
  const { efforts = [], default: given = null } = model?.reasoning ?? {};
  const kept = localStorage.getItem(`leat.effort.${model?.id}`);
  return efforts.includes(kept) ? kept : given;
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

// whether a message is its turn's answer, rather than a step of its work: the model's, calling no
// tools, and, while it is written, saying something, as until then it may yet call them, which
// the engine sends as the reply ends
function answers(m) {
  if (m?.role !== "assistant" || m.tool_calls) return false;
  return !live(m) || Boolean(m.content?.trim());
}

// whether a message is the reply the model is writing
function live(m) {
  return Boolean(shown?.running) && shown.messages.at(-1) === m;
}

// shows a turn in its element: the user's message, the work that came of it folded into a line,
// and the answer
function fill(turn, [start, ...rest]) {
  const view = (i) => (views[i] = message(shown.messages[i], i));
  const answered = answers(shown.messages[rest.at(-1)]);
  const work = answered ? rest.slice(0, -1) : rest;
  turn.start = start;
  turn.classList.toggle("summarized", [start, ...rest].includes(shown.summarized));
  turn.replaceChildren(view(start));
  if (work.length) turn.append(fold(start, work, !answered));
  const long = shown.running && !answered && longCall(work);
  if (long) turn.append(meter(long));
  if (answered) turn.append(view(rest.at(-1)));
  return turn;
}

// the call of a turn's work that runs long, telling how far it is, while it runs; none if none
function longCall(indexes) {
  const m = shown.messages[indexes.at(-1)];
  return m?.role === "tool" && !m.content && m.info?.total > 1 ? m : null;
}

// how far a long call is, below its turn's work: a bar, and that the box works on without the page
function meter(m) {
  const box = element("div", "meter"), track = element("div", "track"), done = element("span");
  done.style.width = `${(100 * (m.info.done ?? 0)) / m.info.total}%`;
  track.append(done);
  box.append(track, element("p", "", "Leat keeps working if you leave or close this page, and marks the chat when it's done."));
  return box;
}

// a turn's work, its steps and calls, folded into a line: what it does as it runs, then what it did
function fold(start, indexes, unanswered) {
  const box = kept(element("details", "work"), `work ${start}`);
  const running = unanswered && shown.running && turns().at(-1)[0] === start; // working still
  const calls = indexes.map((i) => shown.messages[i]).filter((m) => m.role === "tool");
  box.classList.toggle("running", running);
  const said = running ? (calls.length ? line(calls.at(-1)) : "Thinking…") : summary(calls);
  box.append(element("summary", "", said), ...indexes.map((i) => (views[i] = message(shown.messages[i], i))));
  return box;
}

// what each tool's calls did, in a few words: one call, and n
const DID = {
  ask_files: ["read the files", (n) => `read the files ${n} times`],
  search: ["searched the web", (n) => `searched the web ${n} times`],
  search_files: ["searched the files", (n) => `searched the files ${n} times`],
  fetch: ["read a page", (n) => `read ${n} pages`],
  read: ["read a file", (n) => `read ${n} files`],
  run: ["ran code", (n) => `ran code ${n} times`],
  write: ["wrote a file", (n) => `wrote ${n} files`],
  edit: ["edited a file", (n) => `edited ${n} files`],
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
  const page = PAGES.some(paged);
  // the page's title, after how many chats' work ended unseen, as a tab shows it
  const named = page ? heading : shown?.title || "leat";
  document.title = unread.size ? `(${unread.size}) ${named}` : named;
  $("main").classList.toggle("empty", !shown && !page);
  const held = project(here());
  input.placeholder = viewing && held ? `Start a chat in ${held.name}` : "Message";
  $("crumb").hidden = !shown || !held; // the project the conversation is in
  $("crumb").lastElementChild.textContent = held?.name ?? "";
  $("crumb").href = held ? `/projects/${held.id}` : "";
  $("greeting").textContent = unreachable ? "The engine is not reachable"
    : loading ? "Loading…" : !model ? "Choose a model" : "How can I help?";
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
function message(m, index) {
  if (m.role === "tool") return work(m, index);
  const item = element("div", `message ${m.role}`);
  const thinking = kept(element("details"), `thinking ${index}`), reasoning = element("div");
  const text = element("div"), pages = element("div", "sources"), note = element("div", "note");
  const summary = element("summary", "", "Thinking"), cards = element("div", "cards");
  thinking.append(summary, reasoning);
  item.append(thinking, text, cards, pages, note);
  item.update = () => {
    const answer = answers(m); // not a step of the work, as a reply that calls tools is
    item.hidden = m.role === "system" || (!answer && !m.content && !m.reasoning_content);
    item.classList.toggle("step", m.role === "assistant" && !answer);
    thinking.hidden = !m.reasoning_content;
    const thinks = !m.content && !m.tool_calls && live(m);
    summary.textContent = thinks ? "Thinking…" : "Thinking";
    markdown(reasoning, m.reasoning_content ?? "");
    if (m.role === "user") text.textContent = m.content;
    else markdown(text, m.content ?? "");
    item.querySelectorAll(":not(.code) > pre").forEach(codeBar); // the blocks new since
    cite(item);
    if (answer) pages.replaceChildren(...chips(sources(m)));
    const names = m.role === "user" ? (m.info?.files ?? []) : answer ? made(m) : [];
    cards.replaceChildren(...names.map(card));
    if (m.role !== "user") note.textContent = answer ? describe(m.info ?? {}) : "";
    const { cached, read } = m.info ?? {};
    note.title = read === undefined ? "" : `the prompt's tokens: ${cached} cached, ${read} read`;
  };
  item.update();
  return item;
}

// a fold, open or not as the user left it, by its key in the shown conversation, through the
// renders that make it again
function kept(details, key) {
  const where = `${shown.id} ${key}`;
  details.open = opened.get(where) ?? false;
  details.ontoggle = () => opened.set(where, details.open);
  return details;
}

// a tool's message, as a line of the work it did, which opens on what it found
function work(m, index) {
  const item = kept(element("details", "message tool"), `call ${index}`);
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
  ask_files: (a, i) => [i.total ? `Reading ${i.done ?? 0} of ${i.total} files…` : "Reading the files…",
    i.done < i.total ? `Read ${i.done} of ${i.total} files` : `Read ${i.total === 1 ? "1 file" : `${i.total ?? 0} files`}`,
    "Couldn't read the files"],
  search: (a) => [`Searching for “${a.query}”…`, `Searched for “${a.query}”`,
    `Couldn't search for “${a.query}”`],
  search_files: (a) => [`Searching the files for “${a.query}”…`, `Searched the files for “${a.query}”`,
    `Couldn't search the files for “${a.query}”`],
  fetch: (a, i) => [`Reading ${host(a.url)}…`, `Read ${i.title || host(i.url ?? a.url)}`,
    `Couldn't read ${host(a.url)}`],
  read: (a) => [`Reading ${named(a.path)}…`, `Read ${named(a.path)}`, `Couldn't read ${named(a.path)}`],
  write: (a) => [`Writing ${a.path}…`, `Wrote ${a.path}`, `Couldn't write ${a.path}`],
  edit: (a) => [`Editing ${a.path}…`, `Edited ${a.path}`, `Couldn't edit ${a.path}`],
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
  const { arguments: args, results, url, title, error } = m.info ?? {};
  if (error || !m.content) return [element("p", "", error ?? "")];
  if (m.name === "search" && results?.length) {
    return [cited(listed(results.map((r) => link(r.url, r.title)), true), results)];
  }
  if (m.name === "ask_files") { // what it asked of each file, and the spreadsheet of the answers
    const asked = element("p", "", `Asked of each file: ${args?.question ?? ""}`);
    const columns = m.info?.columns?.length > 1 ? [element("p", "meta", `Columns: ${m.info.columns.join(", ")}`)] : [];
    return [asked, ...columns, cards(m.info?.files ?? [])];
  }
  if (m.name === "search_files" && results?.length) {
    return [cited(listed(results.map((r) => sourceLink(r, r.title))), results)];
  }
  if (m.name === "fetch") { // and the question it was read for, if one
    const asked = m.info?.question ? [element("p", "meta", `Read for: ${m.info.question}`)] : [];
    return [link(url, title || url), ...asked];
  }
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
  const messages = shown?.messages ?? [], names = [], files = filesOf(here());
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

// an image of a space's: a project's, or the person's own, the space worked in by default
function picture(name, project = here()) {
  const img = element("img");
  Object.assign(img, { src: `${place(project)}/files/${encodeURIComponent(name)}`, alt: name, loading: "lazy" });
  return img;
}

function fileLink(name, project = here()) {
  const a = element("a", "", name);
  Object.assign(a, { href: `${place(project)}/files/${encodeURIComponent(name)}`, target: "_blank" });
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
      if (s.n && s.url && !all.has(s.n)) all.set(s.n, s);
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
    sup.replaceChildren(Object.assign(sourceLink(s, sup.textContent), { title: s.title || s.url }));
    sup.classList.add("linked");
  }
}

// a reply's sources: those it cites, in the order it first does, or the pages its turn read
function sources(m) {
  const all = numbered(), cited = [];
  const cites = /\[(\d{1,3})\](?!\()|【(\d{1,3})(?:†[^】\n]*)?】/g; // [1], or gpt-oss's 【1】
  for (const [, n, oss] of (m.content ?? "").matchAll(cites)) {
    const s = all.get(Number(n ?? oss));
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

// a reply's sources as chips, the first CHIPS of many, then how many more
const CHIPS = 8;
function chips(all) {
  const more = all.length > CHIPS + 1 ? [element("span", "more", `and ${all.length - CHIPS} more`)] : [];
  return [...all.slice(0, more.length ? CHIPS : all.length).map(source), ...more];
}

function source(s) {
  const named = s.file ? s.title : host(s.url);
  const a = sourceLink(s, s.n ? `${s.n} · ${named}` : named);
  a.className = "source";
  a.title = s.title || s.url;
  return a;
}

// a link to a source: a page of the web, or a passage of a file, which opens it, a PDF at its page
function sourceLink(s, text) {
  if (!s.file) return link(s.url, text);
  const a = fileLink(s.file);
  a.textContent = text;
  const page = s.place?.match(/^page (\d+)$/)?.[1];
  if (page && /\.pdf$/i.test(s.file)) a.href += `#page=${page}`;
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

// a list of sources, each numbered as the conversation cites it, rather than by its place there
function cited(list, sources) {
  [...list.children].forEach((item, i) => sources[i].n && (item.value = sources[i].n));
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
  ask_files: '<path d="M3 5h18M3 12h18M3 19h18M9 5v14"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  search_files: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  fetch: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  read: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  write: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  run: '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
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
