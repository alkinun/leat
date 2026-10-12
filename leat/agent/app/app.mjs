// leat agent's app. It keeps nothing of its own: it shows the conversations on the box as the
// agent's events change them, the turns running there too, and sends what the user writes. A device
// is its person's once paired: the first sets the box up, any other asks to be let in, by its
// person from a device of theirs, or by an admin.

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
let attached = []; // the files and folders the next message attaches: {name, uploading, files}
const unfolded = new Set(); // the folders whose files the lists show, by space and name
let workflows = []; // the requests saved to make again, by name: {id, name, prompt, project, schedule}
let me = null; // the person whose this device is: {person, name, admin, device}
// the accounts as this device sees them: the people, each whether an admin, with their devices, of
// an admin's, or else this person's own; and the devices asking to join
let accounts = null;
// the box's settings, as its admins chose them: whether it is kept from the internet, its office's
// name, and whether its screen shows its address
let settings = { offline: false, office: null, address: true, timezone: null };
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
const PAGES = ["projects", "project", "files", "settings", "activity"];
const TITLES = { projects: "Projects", files: "Files", settings: "Settings", activity: "Activity" };

$("new").onclick = () => {
  open(null);
  $("input").focus();
};
$("menu").onclick = () => document.body.classList.toggle("menu");
$("held").onclick = () => turnTo("projects");
$("filed").onclick = () => turnTo("files");
$("set").onclick = () => turnTo("settings");
$("attach").onclick = () => pick(attach);
$("upload").onclick = () => pick((items) => uploads(items, null));
$("uploadFolder").onclick = () => pick((items) => uploads(items, null), true);
$("adding").onclick = () => pick((items) => uploads(items, viewing));
$("addingFolder").onclick = () => pick((items) => uploads(items, viewing), true);
$("workspaceHistory").ontoggle = () => renderHistory($("workspaceHistory"), null);
$("keptHistory").ontoggle = () => renderHistory($("keptHistory"), viewing);
$("create").onsubmit = create;
$("back").onclick = (event) => {
  event.preventDefault();
  turnTo("projects");
};
for (const [link, page] of [["toSettings", "settings"], ["seeActivity", "activity"]]) {
  $(link).onclick = (event) => {
    event.preventDefault();
    turnTo(page);
  };
}
$("older").onclick = () => renderActivity(true);
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
  attach(pasted.map((file) => ({ file, path: file.name })));
};
window.ondragover = (event) => event.preventDefault();
window.ondrop = async (event) => { // files and folders dropped, attached to the next message, or on
  event.preventDefault(); // a page of files uploaded there
  const items = await dropped(event.dataTransfer);
  if (paged("files")) uploads(items, null);
  else if (paged("project")) uploads(items, viewing);
  else attach(items);
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
  if (response.status === 401) {
    const { empty, locked, recovering } = await response.json();
    return gate(empty, locked, recovering);
  }
  me = await response.json();
  renderAccounts();
  if (me.admin) watchHealth();
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

// the gate of a device not yet paired: the box's first person sets it up, and is its first admin;
// any other asks to join, showing a code the devices paired show too
function gate(empty, locked, recovering) {
  document.body.classList.add("gated");
  if (empty) return setup(locked);
  const scanned = new URLSearchParams(location.hash.slice(1)).get("recover");
  if (scanned) history.replaceState(null, "", location.pathname); // kept out of the address bar
  if (scanned || recovering) { // the box's power button pressed: its recovery code, to give
    const link = element("a", "", "Use the recovery code the box's screen shows");
    link.href = "#";
    link.onclick = (event) => (event.preventDefault(), recover(""));
    $("join").append(element("p", "recovering", "Has everyone who looks after this Leat lost " +
      "their devices? "), link);
    if (scanned) return recover(scanned);
  }
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
      $("code").replaceChildren("Let this device in from Leat on a device of yours, in Settings, " +
        "or ask an admin to. They'll see this code:", element("b", "", `${code.slice(0, 3)} ${code.slice(3)}`));
      await waitToJoin(id);
    } catch (error) {
      $("refusal").textContent = error.message;
    }
  };
}

// the box set up by its first person, its first admin, a step at a time: of a box, the code its
// screen shows, which its QR code gives the address it opens, else typed; its office's name, theirs,
// and whether it may reach the internet, sent at once at the end
function setup(locked) {
  const scanned = new URLSearchParams(location.hash.slice(1)).get("setup");
  if (scanned) history.replaceState(null, "", location.pathname); // kept out of the address bar
  const chosen = { code: scanned ?? "", office: "", name: "", offline: false };
  const typing = { title: "The setup code", about: "Type the setup code the box's screen shows, " +
    "below its address. It shows that you're in the office with the box.", key: "code",
    placeholder: "K7X 2QM", autocomplete: "off", max: 7, needed: true };
  const steps = [
    { title: "Welcome to Leat", about: "Your office's own assistant. It runs on this box, and what " +
      "you give it stays here. Setting it up takes a minute.", next: "Begin" },
    ...(locked && !scanned ? [typing] : []),
    { title: "Your office", about: "What is your office called? The box's screen shows its name. " +
      "You can change it later in Settings.", key: "office", placeholder: "Kanzlei Weber",
      autocomplete: "organization", max: 60 },
    { title: "Your name", about: "You'll be this Leat's admin: you let the others in, and look " +
      "after it in Settings. You can make others admins too.", key: "name", placeholder: "Your name", autocomplete: "name", max: 40,
      needed: true },
    { title: "The internet", about: null, next: "Set up Leat" },
  ];
  const show = (at) => {
    const step = steps[at];
    const parts = [element("h1", "", step.title)];
    if (step.about) parts.push(element("p", "", step.about));
    let input = null;
    if (step.key) {
      input = Object.assign(element("input", step.key), { value: chosen[step.key], placeholder: step.placeholder,
        autocomplete: step.autocomplete, maxLength: step.max });
      parts.push(input);
    }
    if (at === steps.length - 1) { // the internet, on or off, and what each means
      const said = element("p");
      const choice = element("div", "choice");
      const pick = (off) => {
        chosen.offline = off;
        choice.replaceChildren(...[[false, "On"], [true, "Off"]].map(([value, label]) => {
          const button = element("button", value === off ? "on" : "", label);
          button.type = "button";
          button.onclick = () => pick(value);
          return button;
        }));
        said.textContent = off
          ? "Leat answers from what it knows and your files alone: nothing it does reaches the internet."
          : "Leat may search the web and read pages to answer. What it searches for leaves the " +
            "office; your files never do.";
      };
      pick(chosen.offline);
      parts.push(choice, said);
    }
    const refusal = element("p", "error");
    const actions = element("div", "actions");
    actions.append(element("span", "steps", `${at + 1} of ${steps.length}`));
    if (at > 0) {
      const back = element("button", "previous", "Back");
      back.type = "button";
      back.onclick = () => show(at - 1);
      actions.append(back);
    }
    actions.append(element("button", "", step.next ?? "Next"));
    parts.push(actions, refusal);
    $("join").replaceChildren(...parts);
    input?.focus();
    $("join").onsubmit = async (event) => {
      event.preventDefault();
      if (input) chosen[step.key] = input.value.trim();
      if (step.needed && !chosen[step.key]) return input.focus();
      if (at < steps.length - 1) return show(at + 1);
      try {
        const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone; // this device's, the box's too
        await post("/api/setup", { code: chosen.code || null, name: chosen.name, office: chosen.office || null,
          offline: chosen.offline, timezone });
        location.reload();
      } catch (error) {
        if (error.status !== 403 || !locked) return (refusal.textContent = error.message);
        // the code wrong, or a new one shown: typed again, at its step
        if (!steps.includes(typing)) steps.splice(1, 0, typing);
        chosen.code = "";
        show(1);
        $("join").querySelector(".error").textContent = error.message[0].toUpperCase() + error.message.slice(1) + ".";
      }
    };
  };
  show(0);
}

// a new admin, of the recovery code the box's screen shows once its power button is pressed, and a
// name, as an office whose every admin's device is gone gets back in
function recover(code) {
  const title = element("h1", "", "Recovery");
  const about = element("p", "", "Type the recovery code the box's screen shows, and your name. " +
    "This device becomes an admin of this Leat, as someone new.");
  const typed = Object.assign(element("input", "code"), { value: code, placeholder: "K7X 2QM",
    autocomplete: "off", maxLength: 7, ariaLabel: "The recovery code" });
  const name = Object.assign(element("input"), { placeholder: "Your name", autocomplete: "name",
    maxLength: 40, ariaLabel: "Your name" });
  const refusal = element("p", "error"), actions = element("div", "actions");
  const back = element("button", "previous", "Back");
  back.type = "button";
  back.onclick = () => location.reload();
  actions.append(element("span", "steps"), back, element("button", "", "Get back in"));
  $("join").replaceChildren(title, about, typed, name, actions, refusal);
  (code ? name : typed).focus();
  $("join").onsubmit = async (event) => {
    event.preventDefault();
    if (!typed.value.trim()) return typed.focus();
    if (!name.value.trim()) return name.focus();
    try {
      await post("/api/recover", { code: typed.value, name: name.value.trim() });
      location.reload();
    } catch (error) {
      refusal.textContent = error.message[0].toUpperCase() + error.message.slice(1) + ".";
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
    case "workflows":
      workflows = event.workflows;
      renderWorkflows();
      return;
    case "files":
      spaces[event.project ?? ""] = event.files;
      renderFiles();
      renderProjects(); // as they count them
      return;
    case "accounts":
      if (event.me.admin !== me.admin) return location.reload(); // made an admin, or no longer
      me = { ...me, ...event.me };
      accounts = event;
      renderAccounts();
      renderRecovering();
      return;
    case "settings":
      settings = { offline: event.offline, office: event.office, address: event.address,
        timezone: event.timezone };
      renderInternet();
      renderBox();
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
  if (!response.ok) {
    throw Object.assign(new Error((await response.json()).error.message), { status: response.status });
  }
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
  if (name === "settings") renderOverview();
  if (name === "activity") renderActivity();
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

// whether a project's files are synced with a folder of the box's, and how the last sync went; the
// an admin's to set up, with the folder's path, or to stop
let choosing = false; // an admin choosing the folder, its form shown
function renderSyncing(p) {
  const box = $("syncing"), admin = me?.admin;
  if (choosing && admin && !p.source) {
    if (box.querySelector("form")) return; // as they type
    const form = element("form"), path = element("input"), ok = element("button", "", "Sync"), cancel = element("button", "", "Cancel");
    Object.assign(path, { placeholder: "The folder's path, as /mnt/office/Yılmaz", spellcheck: false });
    cancel.type = "button";
    cancel.onclick = () => (choosing = false, renderSyncing(p));
    form.onsubmit = async (event) => {
      event.preventDefault();
      try {
        await post(`/api/projects/${p.id}/sync`, { source: path.value.trim() });
        choosing = false;
      } catch (error) {
        status(error.message, true);
      }
    };
    form.append(path, ok, cancel);
    box.replaceChildren(form);
    return path.focus();
  }
  const parts = [];
  if (p.source) {
    const said = p.unsynced ? `Couldn't sync with ${p.source}: ${p.unsynced}`
      : p.synced ? `Synced from ${p.source} · ${ago(p.synced)}` : `Syncing with ${p.source}…`;
    parts.push(element("span", p.unsynced ? "warning" : "meta", said));
    if (admin) {
      const stop = element("button", "", "Stop syncing");
      stop.onclick = () => confirm(`Stop syncing with ${p.source}? The files copied so far stay.`)
        && post(`/api/projects/${p.id}/sync`, { source: null }).catch((e) => status(e.message, true));
      parts.push(stop);
    }
  } else if (admin) {
    const start = element("button", "", "Sync with a folder on this computer…");
    start.title = "Keep a copy of a folder here, as a network drive's, in this project's files";
    start.onclick = () => (choosing = true, renderSyncing(p));
    parts.push(start);
  }
  box.replaceChildren(...parts);
}

// a time, in seconds, as how long ago it was
function ago(seconds) {
  const minutes = Math.round((Date.now() / 1000 - seconds) / 60);
  return minutes < 1 ? "just now" : minutes < 60 ? `${minutes} min ago`
    : minutes < 1440 ? `${Math.round(minutes / 60)} h ago` : day(seconds);
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

// the accounts: who this person is here, and their own devices, each to unpair; the devices asking
// to join, which they let in as their own, once the code shown is the one the device shows, and an
// admin lets in as someone new, or as someone whose every device is gone; and, of an admin, the
// box's people, each to rename, make an admin or not, or remove, with their devices
function renderAccounts() {
  const box = $("accounts");
  if (!me || !accounts) return box.replaceChildren();
  const self = accounts.people.find((p) => p.id === me.person);
  const you = element("p", "", `You are ${me.name} here`);
  you.append(element("span", "meta", me.admin ? " · Admin" : ""), " ", renaming(self ?? { id: me.person, name: me.name }));
  const mine = element("ul");
  mine.append(...(self?.devices ?? []).map((d) => deviceRow(d)));
  const asking = accounts.requests.map((r) => {
    const item = element("li"), about = element("div");
    const code = element("b", "", `${r.code.slice(0, 3)} ${r.code.slice(3)}`);
    const meta = element("span", "meta", `${r.device} · code `);
    meta.append(code);
    about.append(element("span", "", `${r.name} asks to join`), meta);
    const as = element("select");
    as.append(new Option(`as you, ${me.name}`, me.person));
    if (me.admin) {
      as.append(new Option(`as someone new, ${r.name}`, ""), ...accounts.people
        .filter((p) => p.id !== me.person && !p.devices.length)
        .map((p) => new Option(`as ${p.name}, who has no device`, p.id)));
      as.value = r.name === me.name ? me.person : "";
    }
    as.hidden = !me.admin;
    const allow = element("button", "allow", me.admin ? "Let in" : "It's mine");
    allow.onclick = () => post(`/api/pairings/${r.id}/allow`, as.value ? { person: Number(as.value) } : {})
      .catch((error) => status(error.message, true));
    item.append(about, as, allow);
    if (me.admin) {
      const no = element("button", "", "×");
      no.title = "Turn down";
      no.onclick = () => del(`/api/pairings/${r.id}`);
      item.append(no);
    }
    return item;
  });
  const requests = element("ul");
  requests.append(...asking);
  const parts = [you, element("h3", "", "Your devices"), mine];
  if (asking.length) parts.push(element("h3", "", "Asking to join"), requests);
  const others = accounts.people.filter((p) => p.id !== me.person);
  if (me.admin) {
    const people = element("ul", "people");
    if (others.length) parts.push(element("h3", "", "People"), people);
    people.append(...others.map((p) => {
      const item = element("li"), about = element("div");
      const seen = p.devices.length ? `${p.devices.length} device${p.devices.length > 1 ? "s" : ""}` : "no device";
      about.append(element("span", "", p.name), element("span", "meta", `${p.admin ? "Admin" : "Member"} · ${seen}`));
      const devices = element("ul");
      devices.append(...p.devices.map((d) => deviceRow(d)));
      about.append(devices);
      const making = element("button", "", p.admin ? "Remove admin" : "Make admin");
      making.onclick = () => patchPerson(p.id, { admin: !p.admin });
      const remover = element("button", "", "Remove");
      remover.onclick = () => confirm(`Remove ${p.name}, with their own chats, projects and files? ` +
        "The projects they shared stay, as yours.") && del(`/api/people/${p.id}`);
      item.append(about, making, renaming(p), remover);
      return item;
    }));
    parts.push(element("p", "meta", "To add someone, open Leat on their device: it asks to join, " +
      "and shows here for you to let in. Admins let people in and look after the box; no one sees " +
      "another's own chats or files."));
  } else {
    const admins = accounts.people.filter((p) => p.admin).map((p) => p.name).join(", ");
    parts.push(element("p", "meta", "To add a device of yours, open Leat on it: it asks to join, and " +
      `shows here for you to let in. To add someone else, ask an admin: ${admins}.`));
  }
  box.replaceChildren(...parts);
}

// of an admin's: that the box's power button was pressed, its screen showing a recovery code, which
// makes whoever gives it an admin, for as long as it holds; theirs to cancel, if it was not them
let recoveryEnds = 0, recoveryTimer = null;
function renderRecovering() {
  const left = me?.admin ? accounts?.recovering : null;
  if (left) recoveryEnds = Date.now() + left * 1000;
  const minutes = Math.ceil((recoveryEnds - Date.now()) / 60000);
  $("recovering").hidden = !left || minutes <= 0;
  clearTimeout(recoveryTimer);
  if ($("recovering").hidden) return;
  $("recovering").querySelector("span").textContent = "Someone pressed the box's power button: its " +
    `screen shows a recovery code, which makes whoever types it an admin, for ${minutes} more ` +
    `minute${minutes > 1 ? "s" : ""}. Not you?`;
  recoveryTimer = setTimeout(() => {
    if (accounts) accounts.recovering = Math.max(0, (recoveryEnds - Date.now()) / 1000) || null;
    renderRecovering();
  }, 15000);
}
$("cancelRecovery").onclick = () => del("/api/recovery");

// of an admin's: what of the box needs attention, as its screen says it too, each a few words and
// what to do; looked at every few minutes
async function watchHealth() {
  const answer = await fetch("/api/health").then((r) => r.ok ? r.json() : null).catch(() => null);
  $("warnings").replaceChildren(...(answer?.warnings ?? []).map((w) => {
    const item = element("div", "warning");
    item.append(element("b", "", w.title), " ", w.said);
    return item;
  }));
  setTimeout(watchHealth, 300000);
}

// a device's row: its name, when it was last seen, and a button that unpairs it; this device's
// signs it out
function deviceRow(d) {
  const item = element("li"), about = element("div"), self = d.id === me.device;
  about.append(element("span", "", self ? `${d.name} (this device)` : d.name),
    element("span", "meta", `seen ${day(d.seen)}`));
  const remover = element("button", "", self ? "Sign out" : "Unpair");
  remover.onclick = () => self
    ? confirm("Sign this device out? Using Leat on it again takes asking to join.")
      && del(`/api/devices/${d.id}`).then((done) => done && location.reload())
    : confirm(`Unpair ${d.name}? Using Leat on it again takes asking to join.`) && del(`/api/devices/${d.id}`);
  item.append(about, remover);
  return item;
}

// a button that renames a person, asking for the name
function renaming(p) {
  const button = element("button", "", "Rename");
  button.onclick = () => {
    const name = prompt(`A new name for ${p.name}:`, p.name)?.trim();
    if (name && name !== p.name) patchPerson(p.id, { name });
  };
  return button;
}

async function patchPerson(id, change) {
  try {
    await post(`/api/people/${id}`, change, "PATCH");
  } catch (error) {
    status(error.message, true);
  }
}

function renderFiles() {
  $("workspace").replaceChildren(...listFiles(null));
  $("kept").replaceChildren(...(viewing ? listFiles(viewing) : []));
  renderHistory($("workspaceHistory"), null);
  if (viewing) renderHistory($("keptHistory"), viewing);
  // the cards of the files its answers made, where the user reads
  if (shown?.messages) follow(() => views.forEach((view) => view.update()));
}

// a space's files, a project's or the person's own: each folder at its top a row that opens on its
// files, then each file a row that opens it; each deleted from its row
function listFiles(project) {
  const folders = new Map(), loose = [];
  for (const f of filesOf(project)) {
    const top = f.name.split("/")[0];
    if (f.name.includes("/")) folders.set(top, [...(folders.get(top) ?? []), f]);
    else loose.push(f);
  }
  const sorted = [...folders].sort(([a], [b]) => a.localeCompare(b));
  return [...sorted.map(([name, files]) => folderRow(name, files, project)), ...loose.map((f) => fileRow(f, project))];
}

// a file's row, by its name in its folder if it is shown in one
function fileRow(f, project, label = f.name) {
  const item = element("li"), link = fileLink(f.name, project);
  link.textContent = label;
  const about = element("span", "meta", [STATES[f.state], bytes(f.size), day(f.modified)].filter(Boolean).join(" · "));
  about.title = f.error ?? (f.state === "scanned" ? "Its scanned pages are read once a model that sees images is loaded" : "");
  about.classList.toggle("failed", f.state === "failed");
  item.append(link, about, remover(`“${f.name}”`, f.name, project));
  return item;
}

// a folder's row, which opens on its files, saying how many and how big, and whether some are
// being read or cannot be
function folderRow(name, files, project) {
  const key = `${project ?? ""}:${name}`, open = unfolded.has(key);
  const item = element("li", open ? "folder open" : "folder"), toggle = element("button", "toggle");
  toggle.append(icon("folder"), element("span", "", name));
  toggle.onclick = () => {
    if (!unfolded.delete(key)) unfolded.add(key);
    renderFiles();
  };
  const reading = files.filter((f) => f.state === "reading").length;
  const failed = files.filter((f) => f.state === "failed").length;
  const size = files.reduce((sum, f) => sum + f.size, 0);
  const about = element("span", "meta", [reading && `Reading ${reading}…`, failed && `${failed} couldn't be read`,
    `${files.length} ${files.length === 1 ? "file" : "files"}`, bytes(size)].filter(Boolean).join(" · "));
  about.classList.toggle("failed", failed > 0 && !reading);
  const inside = element("ul");
  if (open) inside.append(...files.map((f) => fileRow(f, project, f.name.slice(name.length + 1))));
  item.append(toggle, about, remover(`the folder “${name}”, with its ${files.length} ${files.length === 1 ? "file" : "files"}`, name, project), inside);
  return item;
}

// a row's button that deletes a file or a folder, once the user says so
function remover(what, name, project) {
  const button = element("button", "", "×");
  button.title = "Delete";
  button.onclick = () => confirm(`Delete ${what}? It can be restored from Earlier versions for 30 days.`)
    && del(`/api${place(project)}/files/${encodeURIComponent(name)}`);
  return button;
}

// a space's files' earlier versions, a project's or the person's own, asked while their list is
// open: each a row of its file, whether the file is deleted, and when it was kept, restored from
// its row, which keeps the file as it is now as one more
async function renderHistory(details, project) {
  if (!details.open) return;
  try {
    const response = await fetch(`/api${place(project)}/history`);
    if (!response.ok) throw new Error((await response.json()).error.message);
    const { versions } = await response.json();
    details.querySelector("ul").replaceChildren(...(versions.length ? versions.map((v) => versionRow(v, project))
      : [element("li", "meta", "Nothing changed or deleted in the last 30 days.")]));
  } catch (error) {
    status(error.message, true);
  }
}

function versionRow(v, project) {
  const item = element("li"), restore = element("button", "restore", "Restore");
  const when = new Date(v.kept * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  restore.onclick = () => post(`/api${place(project)}/history`, { name: v.name, version: v.version })
    .catch((error) => status(error.message, true));
  item.append(element("span", "", v.name),
    element("span", "meta", [v.deleted ? "Deleted" : "Earlier version", `${day(v.kept)} ${when}`, bytes(v.size)].join(" · ")), restore);
  return item;
}

// what a file's state in the index says, while it is not read: being read, or not readable
const STATES = { reading: "Reading…", failed: "Couldn't read", scanned: "Scanned" };

// the box as its admins look after it, asked as Settings opens: the room its files take and that
// left on their disk, a warning if little is, and each person's use this month, counted
async function renderOverview() {
  $("overview").hidden = !me?.admin;
  if (!me?.admin) return;
  try {
    const response = await fetch("/api/overview");
    if (!response.ok) throw new Error((await response.json()).error.message);
    const { files, versions, disk, people } = await response.json();
    const low = disk && disk.free < disk.total / 10;
    $("room").textContent = `Files take ${bytes(files)}` + (versions ? `, and their earlier versions ${bytes(versions)}` : "") + (disk ? `; ${bytes(disk.free)} of ${bytes(disk.total)} is free on this computer's disk${low ? ": it is nearly full" : ""}.` : ".");
    $("room").className = low ? "warning" : "meta";
    $("use").replaceChildren(...people.map((p) => {
      const item = element("li");
      item.append(element("span", "", p.name), element("span", "meta", `${p.chats} ${p.chats === 1 ? "chat" : "chats"}, ${p.messages} ${p.messages === 1 ? "message" : "messages"} this month`));
      return item;
    }));
  } catch (error) {
    status(error.message, true);
  }
}

// what was done on the box, the admins' to look back on, by day, the latest first: each at its time,
// by whom, or by Leat for whom, what, and where; asked anew, or `older`, after what is shown
let activity = [];
async function renderActivity(older = false) {
  const before = older && activity.length ? `?before=${activity.at(-1).id}` : "";
  try {
    const response = await fetch(`/api/activity${before}`);
    if (!response.ok) throw new Error((await response.json()).error.message);
    const more = (await response.json()).activity;
    activity = older ? [...activity, ...more] : more;
    $("older").hidden = more.length < 100;
  } catch (error) {
    status(error.message, true);
  }
  const days = new Map();
  for (const a of activity) {
    const day = dayOf(a.at);
    days.set(day, [...(days.get(day) ?? []), a]);
  }
  $("done").replaceChildren(...[...days].flatMap(([day, done]) => {
    const list = element("ul");
    list.append(...done.map((a) => {
      const item = element("li"), time = new Date(a.at * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
      const who = a.leat ? (a.person ? `Leat, for ${a.person}` : "Leat") : (a.person ?? "Someone removed");
      const what = element("span", "what");
      what.append(element("span", "meta", time), element("b", "", who), `${a.action}${a.detail ? ` ${a.detail}` : ""}`);
      item.append(what);
      if (a.place) item.append(element("span", "meta", a.place));
      return item;
    }));
    return [element("h2", "", day), list];
  }));
  if (!activity.length) $("done").replaceChildren(element("p", "meta", "Nothing yet."));
}

// a time, in seconds, as the day it fell on: today, yesterday, or its date
function dayOf(seconds) {
  const at = new Date(seconds * 1000), today = new Date();
  const days = Math.round((new Date(today.toDateString()) - new Date(at.toDateString())) / 86400000);
  return days === 0 ? "Today" : days === 1 ? "Yesterday"
    : at.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
}

// whether Leat may reach the internet, an admin's to choose, and what each choice means
function renderInternet() {
  $("internet").hidden = !me?.admin;
  $("reach").replaceChildren(...[[false, "On"], [true, "Off"]].map(([off, said]) => {
    const button = element("button", off === settings.offline ? "on" : "", said);
    button.onclick = () => off !== settings.offline && changeSettings({ offline: off });
    return button;
  }));
  $("reached").textContent = settings.offline
    ? "Leat answers from what it knows and your files alone: nothing it does reaches the internet."
    : "Leat may search the web and read pages to answer. What it searches for leaves the office; your files never do.";
}

// the box itself, an admin's to set: its office's name, which its screen shows beside Leat's, and
// whether the screen shows its address and the code a phone scans to open it
function renderBox() {
  $("box").hidden = !me?.admin;
  const office = $("office");
  if (document.activeElement !== office) office.value = settings.office ?? "";
  office.onchange = () => changeSettings({ office: office.value });
  $("showing").replaceChildren(...[[true, "Address shown"], [false, "Hidden"]].map(([shown, said]) => {
    const button = element("button", shown === settings.address ? "on" : "", said);
    button.onclick = () => shown !== settings.address && changeSettings({ address: shown });
    return button;
  }));
  $("shown").textContent = settings.address
    ? "The box's screen shows its address, and a code a phone scans to open Leat."
    : "The box's screen says Leat is ready, without its address: tell the office where it is yourself.";
  const zone = $("zone");
  const zones = Intl.supportedValuesOf("timeZone");
  if (settings.timezone && !zones.includes(settings.timezone)) zones.unshift(settings.timezone);
  zone.replaceChildren(...zones.map((z) => Object.assign(element("option", "", z.replaceAll("_", " ")), { value: z })));
  zone.value = settings.timezone ?? "";
  zone.onchange = () => changeSettings({ timezone: zone.value });
  if (!$("wifi").childElementCount) renderWifi();
  if (!$("power").childElementCount) renderPower();
  if (!$("software").childElementCount) renderSoftware();
}

// the box's software, an admin's to see: its version, and an update installed, which it restarts
// into at night, or now; or that it is looking for one, or when it last did, and a button to look
async function renderSoftware(asked = false) {
  let state;
  try {
    state = await (await fetch("/api/update")).json();
    if (state.error) throw new Error(state.error.message);
  } catch (error) {
    return $("software").replaceChildren(element("p", "about", error.message));
  }
  const said = element("p", "about");
  const parts = [element("p", "", `Leat ${state.version ?? "of no version"}`), said];
  if (state.ready) {
    said.textContent = `An update, ${state.ready}, is ready: the box restarts into it tonight, ` +
      "at about four, or now.";
    const restart = element("button", "quiet", "Restart now");
    restart.type = "button";
    restart.onclick = () => post("/api/power", { power: "restart" })
      .then(() => status("The box is restarting into its update…"))
      .catch((error) => status(error.message, true));
    parts.push(restart);
  } else {
    said.textContent = state.checking || asked ? "Looking for an update…"
      : state.checked ? `Up to date, as of ${new Date(state.checked * 1000).toLocaleString()}. ` +
        "The box looks every few hours, installs what it finds, and restarts into it at night."
      : "The box looks for updates every few hours, installs what it finds, and restarts into it " +
        "at night.";
    const check = element("button", "quiet", "Look for an update");
    check.type = "button";
    check.disabled = state.checking || asked;
    check.onclick = () => post("/api/update", {})
      .then(() => {
        renderSoftware(true);
        setTimeout(() => renderSoftware(), 30000); // what it found, once it has looked
      })
      .catch((error) => status(error.message, true));
    parts.push(check);
  }
  $("software").replaceChildren(...parts);
}

// the box restarted, or switched off, of an admin, once they confirm it; or reset, its state erased,
// once they type the office's name
function renderPower() {
  const buttons = [["restart", "Restart", "Restart the box? Leat is away for a minute or two."],
    ["off", "Switch off", "Switch the box off? Leat is away until someone presses its power button."]]
    .map(([power, label, asked]) => {
      const button = element("button", "quiet", label);
      button.type = "button";
      button.onclick = () => confirm(asked) && post("/api/power", { power })
        .then(() => status(power === "off" ? "The box is switching off." : "The box is restarting…"))
        .catch((error) => status(error.message, true));
      return button;
    });
  const reset = element("button", "danger", "Erase everything");
  reset.type = "button";
  reset.onclick = () => {
    const word = settings.office || "reset";
    const typed = prompt("Erase everything on this Leat? Everyone's chats, files and projects, its " +
      "people and devices, its settings and Wi-Fi networks are erased for good, and it starts again " +
      `at "Set up your Leat", its address the same. Type ${word} to go on.`);
    if (typed !== null) post("/api/reset", { confirm: typed })
      .then(() => status("The box is erasing everything, and restarting."))
      .catch((error) => status(error.message, true));
  };
  $("power").replaceChildren(...buttons, reset);
}

// the box's Wi-Fi, an admin's to set: the networks in reach, found as asked, each joined of its
// passphrase, the one the box is on forgotten; a cable stays its way out while it has one
function renderWifi(found = null, error = "") {
  const parts = [element("h3", "", "Wi-Fi")];
  if (!found) {
    const find = element("button", "quiet", "Find Wi-Fi networks");
    find.type = "button";
    find.onclick = () => scanWifi();
    parts.push(find);
  } else if (!found.radio) {
    parts.push(element("p", "about", "This box has no Wi-Fi: it is on its network by cable."));
  } else {
    const list = element("ul");
    for (const network of found.networks) {
      const row = element("li");
      const name = element("span", "", network.name);
      const about = [["", "▂", "▂▄", "▂▄▆", "▂▄▆█"][network.strength], network.secure ? "secured" : "open"];
      if (network.connected) about.push("connected");
      row.append(name, element("span", "meta", about.filter(Boolean).join(" · ")));
      if (network.connected) {
        const forget = element("button", "quiet", "Forget");
        forget.type = "button";
        forget.onclick = () => del(`/api/wifi/${encodeURIComponent(network.name)}`).then(() => scanWifi());
        row.append(forget);
      } else {
        row.onclick = () => joinWifi(row, network);
      }
      list.append(row);
    }
    if (!found.networks.length) list.append(element("li", "meta", "No Wi-Fi network is in reach."));
    const again = element("button", "quiet", "Look again");
    again.type = "button";
    again.onclick = () => scanWifi();
    parts.push(list, again);
  }
  if (error) parts.push(element("p", "error", error));
  $("wifi").replaceChildren(...parts);
}

// the Wi-Fi networks in reach, as the box finds them
async function scanWifi() {
  $("wifi").querySelector("button")?.setAttribute("disabled", "");
  try {
    const response = await fetch("/api/wifi");
    const found = await response.json();
    if (!response.ok) throw new Error(found.error.message);
    renderWifi(found);
  } catch (error) {
    renderWifi(null, error.message);
  }
}

// a Wi-Fi network joined, of its passphrase if it asks for one, the box on it a few seconds after
function joinWifi(row, network) {
  if (row.querySelector("form")) return;
  const form = element("form", "join");
  const passphrase = Object.assign(element("input"), { type: "password", placeholder: "Passphrase",
    autocomplete: "off", minLength: 8, maxLength: 63, required: true });
  const join = element("button", "", "Join");
  if (network.secure) form.append(passphrase);
  form.append(join);
  form.onsubmit = async (event) => {
    event.preventDefault();
    join.disabled = true;
    try {
      await post("/api/wifi", { name: network.name, passphrase: network.secure ? passphrase.value : null });
      renderWifi({ radio: true, networks: [] }, "");
      $("wifi").querySelector("ul").replaceChildren(element("li", "meta", `Joining ${network.name}…`));
      setTimeout(scanWifi, 6000);
    } catch (error) {
      renderWifi(null, error.message);
    }
  };
  row.append(form);
  (network.secure ? passphrase : join).focus();
}

// changes the box's settings, which every app is told of as they change
function changeSettings(changes) {
  post("/api/settings", changes, "PATCH").catch((e) => status(e.message, true));
}

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

// the workflows: the person's own, as chips below the composer of a new chat, and the shown
// project's, as its page lists them; each puts its request in the composer, to send as it is or
// change first
function renderWorkflows() {
  $("shortcuts").replaceChildren(...workflows.filter((w) => !w.project).map((w) => {
    const chip = element("span", "chip"), use = element("button", "use", w.name);
    use.title = w.prompt;
    use.onclick = () => compose(w.prompt);
    chip.append(use, forget(w));
    return chip;
  }));
  $("flows").replaceChildren(...workflows.filter((w) => w.project && w.project === viewing).map((w) => {
    const item = element("li"), use = element("button", "use");
    use.append(element("span", "", w.name), element("span", "meta", w.prompt));
    use.onclick = () => compose(w.prompt);
    item.append(use, ...scheduler(w), forget(w));
    return item;
  }));
}

// when a workflow runs by itself, as a chat of the person who set it: on request alone, as files
// arrive in its project, or every day, weekday, a day of the week or the month's first, at a time
// of the box's
const EVERY = [["", "On request"], ["files", "When files arrive"], ["daily", "Every day"],
  ["weekdays", "Every weekday"],
  ...["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    .map((day) => [`weekly ${day.slice(0, 3).toLowerCase()}`, `Every ${day}`]),
  ["monthly 1", "On the 1st of each month"]];

function scheduler(w) {
  const every = element("select"), at = element("input");
  const timed = w.schedule && w.schedule !== "files", words = (w.schedule ?? "").split(" ");
  const kind = timed ? words.slice(0, -1).join(" ") : w.schedule ?? "";
  const kinds = EVERY.some(([k]) => k === kind) ? EVERY : [...EVERY, [kind, kind]];
  every.replaceChildren(...kinds.map(([value, label]) => new Option(label, value)));
  every.value = kind;
  Object.assign(at, { type: "time", value: timed ? words.at(-1) : "08:00", hidden: !timed });
  every.title = "When it runs by itself, as a chat of yours";
  every.onchange = at.onchange = () => {
    at.hidden = !every.value || every.value === "files";
    const schedule = !every.value ? null : every.value === "files" ? "files" : `${every.value} ${at.value || "08:00"}`;
    post(`/api/workflows/${w.id}`, { schedule }, "PATCH").catch((error) => status(error.message, true));
  };
  return [every, at];
}

// a workflow's button that deletes it, once the user says so
function forget(w) {
  const button = element("button", "", "×");
  button.title = "Delete";
  button.onclick = () => confirm(`Delete the workflow “${w.name}”?`) && del(`/api/workflows/${w.id}`);
  return button;
}

// puts a request in the composer, to send or change
function compose(text) {
  const input = $("input");
  input.value = text;
  controls();
  input.focus();
  input.setSelectionRange(text.length, text.length);
}

// below a message the user sent, an action that saves it as a workflow, of a name they give it
// there: the project's, if the chat is in one, or their own
function saver(m) {
  const box = element("div", "actions"), save = element("button", "", "Save as workflow");
  save.onclick = () => {
    const form = element("form"), name = element("input"), ok = element("button", "", "Save");
    const cancel = element("button", "", "Cancel");
    Object.assign(name, { value: m.content.split(/\s+/).slice(0, 6).join(" "), maxLength: 80, placeholder: "The workflow's name" });
    cancel.type = "button";
    cancel.onclick = () => box.replaceChildren(save);
    form.onsubmit = async (event) => {
      event.preventDefault();
      if (!name.value.trim()) return;
      try {
        await post("/api/workflows", { name: name.value.trim(), prompt: m.content, project: shown?.project ?? undefined });
        box.replaceChildren(element("span", "done", shown?.project ? "Saved to this project's workflows" : "Saved to your workflows, below a new chat"));
      } catch (error) {
        status(error.message, true);
      }
    };
    form.append(name, ok, cancel);
    box.replaceChildren(form);
    name.select();
  };
  box.append(save);
  return box;
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
  $("discard").hidden = !mine && !me?.admin;
  renderSyncing(p);
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

// asks the user for files, or a `folder`, and hands them to `take`: {file, path}, each by its path
// in the folder chosen, as "Invoices/March/1.pdf"
function pick(take, folder = false) {
  const picker = element("input");
  Object.assign(picker, { type: "file", multiple: true, webkitdirectory: folder });
  picker.onchange = () => take([...picker.files].map((file) => ({ file, path: file.webkitRelativePath || file.name })));
  picker.click();
}

// the files dropped, those of the folders dropped too, each by its path in its folder
async function dropped(transfer) {
  const entries = [...transfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...transfer.files].map((file) => ({ file, path: file.name }));
  const found = [];
  const walk = async (entry) => {
    if (entry.isFile) return found.push({ file: await new Promise((done, failed) => entry.file(done, failed)), path: entry.fullPath.slice(1) });
    const reader = entry.createReader();
    for (let batch; (batch = await new Promise((done, failed) => reader.readEntries(done, failed))).length;) {
      for (const inner of batch) await walk(inner);
    }
  };
  for (const entry of entries) await walk(entry);
  return found;
}

// uploads a file to a project's files, or the person's own, in the folders of its path, a zip
// unpacked into a folder of its name; returns the name it got there, or null if it did not
async function upload({ file, path }, project) {
  try {
    const unpack = /\.zip$/i.test(path) ? "?unpack" : "";
    const response = await fetch(`/api${place(project)}/files/${encodeURIComponent(path)}${unpack}`, { method: "PUT", body: file });
    if (!response.ok) throw new Error((await response.json()).error.message);
    return (await response.json()).name;
  } catch (error) {
    status(`${file.name} was not uploaded: ${error.message}`, true);
    return null;
  }
}

// uploads files, UPLOADS at once, saying how many are done of many; returns the names they got
const UPLOADS = 4;
async function uploads(items, project) {
  const names = [], waiting = fresh(items, project);
  let done = 0;
  const say = () => items.length > 1 && status(done < items.length ? `Uploading ${done} of ${items.length} files…` : "");
  say();
  await Promise.all(Array.from({ length: Math.min(UPLOADS, items.length) }, async () => {
    for (let item; (item = waiting.shift());) {
      const name = await upload(item, project);
      if (name) names.push(name);
      done++;
      if ($("status").className !== "error") say();
    }
  }));
  return names;
}

// files to upload, each folder at their top named anew, "Invoices (2)", if the space has one of its
// name, so that a folder uploaded again is a folder of its own
function fresh(items, project) {
  const taken = new Set(filesOf(project).map((f) => f.name.split("/")[0])), named = new Map();
  return items.map(({ file, path }) => {
    if (!path.includes("/")) return { file, path };
    const [top, ...rest] = path.split("/");
    if (!named.has(top)) {
      let n = 1, name = top;
      while (taken.has(name)) name = `${top} (${++n})`;
      named.set(top, name);
    }
    return { file, path: [named.get(top), ...rest].join("/") };
  });
}

// attaches files and folders to the next message, once they are uploaded: a folder, or a zip
// unpacked into one, as a chip of its own
async function attach(items) {
  const groups = new Map(); // by the folder at the top of each's path, or by its own name
  for (const item of items) {
    const top = item.path.includes("/") ? `${item.path.split("/")[0]}/` : item.path;
    groups.set(top, [...(groups.get(top) ?? []), item]);
  }
  await Promise.all([...groups].map(async ([name, group]) => {
    const chip = { name, uploading: true, files: group.length };
    attached.push(chip);
    renderAttached();
    const names = await uploads(group, here());
    const folder = name.endsWith("/") || /\.zip$/i.test(name);
    chip.name = !names.length ? null : folder ? `${names[0].split("/")[0]}/` : names[0];
    chip.uploading = false;
    attached = attached.filter((a) => a.name);
    renderAttached();
  }));
}

function renderAttached() {
  $("attached").replaceChildren(...attached.map((a) => {
    const folder = a.name.endsWith("/");
    const named = folder ? `${a.name.slice(0, -1)}${a.files > 1 ? ` · ${a.files} files` : ""}` : a.name;
    const chip = element("span", "chip", a.uploading ? `${named}…` : named);
    if (folder) chip.prepend(icon("folder"));
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
  renderWorkflows();
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
  if (shown.messages[start].role === "user") turn.append(saver(shown.messages[start]));
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
  make_document: ["made a document", (n) => `made ${n} documents`],
  compare_documents: ["compared documents", (n) => `compared documents ${n} times`],
  fill_template: ["filled a template", (n) => `filled ${n} templates`],
  suggest_edits: ["suggested edits", (n) => `suggested edits ${n} times`],
  search_web: ["searched the web", (n) => `searched the web ${n} times`],
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
  $("model").disabled = !me?.admin || loading !== null || conversations.some((c) => c.running);
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
  search_web: (a) => [`Searching for “${a.query}”…`, `Searched for “${a.query}”`,
    `Couldn't search for “${a.query}”`],
  search_files: (a) => [`Searching the files for “${a.query}”…`, `Searched the files for “${a.query}”`,
    `Couldn't search the files for “${a.query}”`],
  fetch: (a, i) => [`Reading ${host(a.url)}…`, `Read ${i.title || host(i.url ?? a.url)}`,
    `Couldn't read ${host(a.url)}`],
  read: (a) => [`Reading ${named(a.path)}…`, `Read ${named(a.path)}`, `Couldn't read ${named(a.path)}`],
  write: (a) => [`Writing ${a.path}…`, `Wrote ${a.path}`, `Couldn't write ${a.path}`],
  make_document: (a, i) => [`Making ${a.name}…`, `Made ${i.files?.[0] ?? a.name}`, `Couldn't make ${a.name}`],
  compare_documents: (a, i) => [`Comparing ${a.original} and ${a.revised}…`,
    `Compared ${a.original} and ${a.revised}: ${i.changes ?? 0} ${i.changes === 1 ? "change" : "changes"}`,
    `Couldn't compare ${a.original} and ${a.revised}`],
  fill_template: (a, i) => [`Filling ${a.path}…`, `Filled ${a.path} as ${i.files?.[0] ?? a.name}`,
    `Couldn't fill ${a.path}`],
  suggest_edits: (a, i) => [`Suggesting edits to ${a.path}…`,
    `Suggested ${i.suggested ?? 0} ${i.suggested === 1 ? "edit" : "edits"} to ${a.path}`, `Couldn't suggest edits to ${a.path}`],
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
  if (m.name === "search_web" && results?.length) {
    return [cited(listed(results.map((r) => link(r.url, r.title)), true), results)];
  }
  if (m.name === "ask_files") { // what it asked of each file, and the spreadsheet of the answers
    const asked = element("p", "", `Asked of each file: ${args?.question ?? ""}`);
    const columns = m.info?.columns?.length > 1 ? [element("p", "meta", `Columns: ${m.info.columns.join(", ")}`)] : [];
    const gaps = [["No answer in", m.info?.unanswered], ["Not read", m.info?.unread]]
      .filter(([, names]) => names?.length).map(([said, names]) => element("p", "meta", `${said}: ${names.join(", ")}`));
    return [asked, ...columns, ...gaps, cards(m.info?.files ?? [])];
  }
  if (m.name === "search_files" && results?.length) {
    return [cited(listed(results.map((r) => sourceLink(r, r.title))), results)];
  }
  if (m.name === "fetch") { // and the question it was read for, if one
    const asked = m.info?.question ? [element("p", "meta", `Read for: ${m.info.question}`)] : [];
    return [link(url, title || url), ...asked];
  }
  if (m.name === "write" || m.name === "edit") return [fileLink(args.path)];
  if (m.name === "suggest_edits" || m.name === "make_document") return [cards(m.info?.files ?? [])]; // the document made
  if (m.name === "fill_template") { // the document made, and the fields it left as they were
    const left = m.info?.missing?.length ? [element("p", "meta", `Without values: ${m.info.missing.join(", ")}`)] : [];
    return [cards(m.info?.files ?? []), ...left];
  }
  if (m.info?.images) return [cards(m.info.images)]; // an image read, which the model saw
  if (m.name === "compare_documents") return [element("p", "changes", m.content)]; // each, whole
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

// a file's card, which opens or downloads it: an image's, the image; a folder's, its name
function card(name) {
  if (name.endsWith("/")) {
    const folder = element("span", "card", name.slice(0, -1));
    folder.prepend(icon("folder"));
    return folder;
  }
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
  if (n < 1000) return `${n} B`;
  if (n < 1e6) return `${Math.round(n / 1e3)} KB`;
  const [size, unit] = n < 1e9 ? [n / 1e6, "MB"] : n < 1e12 ? [n / 1e9, "GB"] : [n / 1e12, "TB"];
  return `${size.toFixed(1)} ${unit}`;
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
  search_web: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  search_files: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  fetch: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  read: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>',
  write: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  fill_template: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M8 13h8M8 17h5"/>',
  make_document: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M12 12v6M9 15h6"/>',
  compare_documents: '<path d="M8 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h3M16 3h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3M12 2v20"/>',
  suggest_edits: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  run: '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
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
