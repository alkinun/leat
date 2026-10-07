# leat agent: plan

A working plan for the agent side of leat, changed as we learn. Engine work goes on alongside it;
where the agent needs something of the engine, section 8 says so.

## 1. What we are building

**The leatbox**: a Strix Halo machine with 128 GB of unified memory that sits in a home, runs a
strong open model on leat, and runs **leat agent**, an assistant that every member of the household
reaches from their phone, laptop or messaging app. It chats, searches, researches, remembers, writes
documents, runs small jobs on a schedule, codes a little, and does it all on hardware in the house:
no account with us, no usage limits, nothing leaving the box unless you send it.

Positioning, in one line: **Grok Bot's always-on agent with a computer of its own, except the
computer is in your home, it answers only to you, and every line of it is open.**

Where it sits among what people use in October 2026:

| | What people like | What we take | What we do differently |
|---|---|---|---|
| ChatGPT, Grok | fast, good answers; live search; memory; voice | the baseline every answer is judged by | local, private, no limits, yours to instruct |
| Grok Bot (Aug 2026) | always-on agents with their own computer; jobs go on with the laptop shut; message it like a colleague; routines learned by watching; approvals for risky actions | the shape of the product | the computer is the box at home; no weekly limit, which reviewers' chief complaint was |
| Poke | no app: you text it, and it texts you first | messaging as a first-class surface; proactive tasks | runs locally; channels the user owns |
| Hermes Agent | memory, self-written skills, 20+ messaging platforms, cron, subagents | the feature checklist | a small core built for non-technical people and for this engine |
| pi | a tiny core, four tools, a short system prompt, everything else as extensions | the philosophy | Python, in leat's style, for general use rather than coding |
| OpenClaw | a personal agent on your own machine | the lesson | security is designed in from the first commit |

## 2. Who uses it and what they want

Not very technical people, a household of them. They will never open a terminal or edit a config
file. What they ask of it, and what that means for us:

| They want to | So leat agent needs |
|---|---|
| ask things and chat | a fast, good chat with a model tuned for it; a voice later |
| ask about today: news, prices, opening hours | web search and page reading, answers with their sources |
| research a topic properly | several readers in parallel and a report with sources |
| be known: "it remembers me" | memory they can see and edit, and recall of past chats |
| make documents: a CV, a letter, a spreadsheet, slides | a workspace on the box and skills for each format; files to download |
| hand it a file: a PDF, a photo | uploads; reading PDFs now, photos once the engine sees (section 8) |
| learn something | patient explanations, quizzes, a tutor's persona |
| roleplay, talk to a character | characters: a name, a personality, a greeting, their own chats |
| have things done for them | tasks: one-off and scheduled, delivered where they are |
| reach it from anywhere | the app on every device, and the messaging apps they already use |
| code a little | the same workspace, with Python and a shell in a sandbox |
| trust it | private by construction, nothing irreversible without asking |

## 3. Decision: build our own, not Hermes, not on pi

**Recommendation: write leat agent ourselves, in Python, in this repo, to the engine's standard.**
Take pi's philosophy, Hermes' and Grok Bot's feature lists, and neither's code.

Why not Hermes Agent:
- It is a developer's tool: CLI and TUI first, configured in files and terminals. The product we sell
  is the experience of someone who never sees either, so we would build the UX anyway, on top of a
  codebase we do not control.
- It is very large and moves daily: Python and Node, 40+ tools, seven terminal backends, 20+
  platforms, a paid model portal behind it, around 250k stars of contributors. We would either fork
  it and drift, or ship code we do not understand into people's homes.
- It treats the model as a remote API and its prompts are written for frontier models, not a ~30B
  mixture of experts on a home GPU.

Why not pi:
- It is TypeScript; the engine is Python. Two runtimes on the box for no gain.
- It is a coding harness. A personal agent on pi's SDK already exists: OpenClaw, which became 2026's
  first agent security crisis (a one-click RCE, malicious sites hijacking local agents over a
  WebSocket, and around 12% of its skills marketplace malicious). "Actually secure" is our pitch; we
  should not inherit that lineage.
- Its ideas are the right ones, and they are easy to carry over: a short system prompt, a few
  strong tools, skills as files read on demand, sessions as plain logs.

Why our own wins:
- **The engine is the moat, and only our own agent can use it fully.** Prompts that stay cached,
  subagents decoding in one batch, and latency measured end to end on the box (section 8).
- **We own every line that touches a family's data.** That is what makes "secure" a claim we can
  defend rather than hope for.
- **It is not that much code.** The engine, ~8,000 lines of Python, beats llama.cpp. A coherent
  agent core (loop, tools, storage, app, a channel or two) is of the same order.

What it costs: we write each integration ourselves. We do a few, well: the app first, then one
messaging platform, then more as people ask.

## 4. Decision: the agent runs on the box

**The box is the agent's computer. Phones and laptops are windows into it.**

- **Always on.** Scheduled tasks, messages arriving on Telegram at 3 am, a research job that takes
  twenty minutes: none of these can live on a laptop that sleeps. Grok Bot rents a cloud computer
  for this; we sell one.
- **One agent, one memory, every device.** The chat started on the phone continues on the laptop.
- **Private.** Conversations, memory and files stay on the box. Clients store nothing.
- **Nothing to install.** The app is a web page the box serves, installable on a phone's home
  screen; messaging apps reach the box with no port opened, the box connecting out to them.

The agent loop runs on the box, not in the browser. Today's chat app runs its tool loop in the page,
so closing the tab stops it. In leat agent a turn is a job on the box; the app only watches it.

Later, and optional: a small companion for a laptop that lends the agent one folder or the screen
when the user wants it to work on that machine. Not in the first versions.

## 5. Architecture

```
  phone · laptop · tablet               Telegram · Signal · email …
  (the leat app, in a browser)          (anywhere; the box connects out)
            │ home network                          │
            ▼                                       ▼
 ┌──────────────────────────── leatbox: Strix Halo, Arch ───────────────────────────┐
 │                                                                                  │
 │  leat agent    app and its API · channels · scheduler · the loop · tools         │
 │      │         memory and conversations in leat.db · files in workspace/         │
 │      │                                                                           │
 │      │ OpenAI chat API, on 127.0.0.1         sandbox (bubblewrap)   SearXNG      │
 │      ▼                                                                           │
 │  leat serve    the engine, on the GPU                                            │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

**Two processes, one boundary.** `leat serve` is the engine and stays an engine: it serves the
OpenAI chat API on 127.0.0.1 and nothing else. `leat agent` is everything a person touches. They
talk over HTTP: the agent never imports the engine, so loading a model or restarting the engine
never takes the agent down, and in development the agent runs on a laptop against `leat serve` on
the 3090. What the agent needs beyond the OpenAI API, the engine exposes as small extensions, as
`/v1/models/load` already is.

**The nouns.** Everything in the agent is one of these, and nothing else:

| Noun | What it is |
|---|---|
| User | a member of the household: their own conversations, memory and workspace |
| Conversation | an append-only log of messages in OpenAI's format, the engine's own; no translation layer |
| Turn | one job on the box: a user's message, then model replies and tool calls until a reply calls none; it streams events to whoever watches |
| Tool | a function the model calls: a name, a short description, arguments, and whether it acts outside the box |
| Skill | a folder with a `SKILL.md` (the agentskills.io format) and any scripts; listed by name in the prompt, read when needed |
| Memory | what the agent knows about a user: short facts, each a memory they can see and delete, and search over past conversations |
| Task | a prompt the agent runs later or on a schedule, delivering its answer to a channel |
| Channel | where a conversation happens: the app, Telegram, … |
| Character | a persona: a name, instructions, a greeting; the default one is leat |
| Approval | a question the agent asks before a tool acts outside the box or cannot be undone |

**Storage.** One SQLite file, `leat.db`, for users, conversations, memory and tasks; its full-text
index (FTS5) searches past conversations. Files the agent makes or is given live in each user's
`workspace/`. Both in one data directory, which is the whole state of the box: back it up and you
have everything.

**Code.** Same rules as the engine: the standard library first; any dependency pinned and justified;
small modules; docstrings and comments in the same voice; tests against a scripted fake model
server, fast and deterministic, plus real-model tests marked as the engine's are. The app stays a
no-build page of plain JavaScript and CSS, as today's.

Proposed layout, settled in step 1:

```
leat/agent/
  __init__.py
  server.py      the app's files, its API, and events streamed to it
  loop.py        a turn: model, tools, model, …, as events
  client.py      the OpenAI chat API, streamed, over urllib
  store.py       leat.db: users, conversations, memory, tasks
  context.py     the prompt a conversation sends, kept stable for the cache
  tools/         search.py, files.py, run.py, memory.py, tasks.py
  skills/        the skills we ship
  channels/      telegram.py, …
  sandbox.py     running code in bubblewrap, confined to a workspace
  app/           index.html, markdown.mjs, vendor/temml
tests/agent/
```

## 6. The context, designed for local inference

On the box, prompt processing is the slowest part of an agent: every step resends the conversation,
and Strix Halo has far less compute than the 3090. leat's prefix cache makes a step cost only its
new tokens, but only if the prompt before them is exactly what it was. So:

- **The prompt is append-only.** System prompt, tool list, skill index and the memory snapshot are
  fixed when a conversation starts. Messages are only ever appended. A step after a tool call
  prefills the tool's result and nothing else.
- **Nothing volatile at the top.** The date goes in at the start; a time that matters goes in the
  message it matters to. Memory saved mid-conversation applies from the next one, and the tool says
  so.
- **When the context fills, compact once**: a summary and the recent turns become a new prefix, which
  the following steps then extend.
- **Reasoning is kept** across a turn's steps, as Qwen3.5's template shows it again.
- **Tools suit a local model**: few of them, short descriptions, one string argument where possible,
  results trimmed and labelled with their source, errors returned as text for the model to retry.

## 7. Tools

A small set; skills do the rest by giving the model instructions and scripts for `run`.

| Tool | Does | Outside the box? |
|---|---|---|
| `search` | web search through SearXNG on the box: titles, links, snippets | reads only |
| `fetch` | a web page as text | reads only |
| `read` | a file in the workspace as text; PDFs too | no |
| `write`, `edit` | create or change a file in the workspace | no |
| `run` | Python or a shell command in the sandbox, in the workspace | no network unless a skill needs it |
| `remember` | change the user's profile, from the next conversation on | no |
| `recall` | search past conversations | no |
| `schedule` | create, list or cancel tasks | no |
| `delegate` | run sub-tasks in parallel, each in its own context (research) | as their tools |

Tools that act outside the box, as sending an email or a message to someone else, come with the
channels that need them, and always through an approval.

## 8. What the agent needs of the engine

The co-design that Hermes and pi cannot have. In rough order of how much users will feel it:

1. **Prefill speed on Strix Halo.** Agents are prefill-heavy: the first token of every step waits on
   it. The engine's first optimization target on the box.
2. **More cached conversations.** Four slots today. A household's chats, background tasks and
   subagents want many more: keep idle conversations' KV in the box's 128 GB, or on disk, and
   restore it when they continue.
3. **Parallel subagents for free.** Already there: up to 8 sequences decode in one batch, reading
   each weight once. Research fans out to it.
4. **Tool calls that always parse.** Constrained decoding of a call's JSON, so a local model never
   emits a malformed one.
5. **Long contexts** that decode fast at 32k tokens and beyond.
6. **More than text.** Photos (vision models), speech in and speech out. Consumers ask for photos
   early.
7. **More than one model loaded**, which 128 GB allows, should one model not do everything well.

## 9. Security

The pitch is "actually secure"; these hold from the first version:

1. **Nothing open to the internet.** The box serves its home network alone; messaging channels
   connect out. Remote access is a later, deliberate feature.
2. **Devices pair.** A phone or laptop pairs with the box once, by a code; a guest on the Wi-Fi or a
   smart plug gets nothing.
3. **Other sites' pages are refused**, as leat serve refuses their POSTs today: the attack that
   hijacked OpenClaw from a web page does not exist here.
4. **The agent cannot touch the box.** It runs as its own unprivileged user; `run` executes in a
   sandbox that sees one workspace.
5. **Content is not authority.** Web pages, emails and files are data. Anything that leaves the box
   or cannot be undone asks the user first, showing exactly what it will do.
6. **No skills marketplace.** Skills ship with leat, reviewed. A user's own stay on their box.
7. **Secrets never enter the context.** Tokens for Telegram and the like are the agent's, not the
   model's.
8. **No telemetry.** The prompts, memory and instructions are all visible to the user and theirs to
   change; nothing is hidden behind them. That is what "aligned to you" means in practice.

## 10. The app

Rebuilt from scratch, in today's design language: the colors, type, layout and the Markdown and math
rendering, which are well tested and stay. Mobile first, as most of these users live on their phones.

- **Chats**: the conversation, with the agent's work folded into short lines ("Searched for …",
  "Read 3 pages", "Ran a script"), each expandable; sources under the answer; files it made as cards
  to open or download; approvals as cards with Allow and Deny.
- **Memory**: what it knows about you, editable.
- **Tasks**: what it will do and when; past runs.
- **Files**: your workspace.
- **Settings**: model, your instructions to it, characters, connections (Telegram …), household and
  devices, and the box's state.

## 11. Steps

Each step ends with something you can use and test in the app; we agree on it before writing it and
you test it before the next.

| Step | Builds | You test |
|---|---|---|
| 1. The loop on the box | `leat agent`, the new app, conversations in leat.db, turns run on the box and streamed as events; `leat serve` becomes API-only | chat from laptop and phone; the same chats on both; reload or switch device mid-reply; stop; switch models |
| 2. Search | `search`, `fetch`, sources in answers, the agent's work shown in the chat; an eval set of real questions begins | questions about today; how the work and sources read |
| 3. Memory | the profile, `remember`, `recall`, the memory page | tell it things; a new chat knows them; edit the memory; "what did we say about …" |
| 4. Files and documents | uploads, the workspace, `run` in the sandbox, skills for Word, Excel, PDF and slides, the files page | "summarize this PDF", "make my CV as a Word file" |
| 5. Tasks | `schedule`, the scheduler, the tasks page, notifications | "every morning at 8, the weather and the headlines"; reminders |
| 6. Household | accounts, device pairing, approvals | a second member; pairing a phone; approving and denying |
| 7. Telegram | the first messaging channel; one ongoing conversation per chat; tasks delivered there | talking to the box from outside the home |
| 8. Research | `delegate`, parallel readers in one batch, long reports | "research …" and the report it writes |
| 9. Characters | personas and their chats | roleplay; a tutor |
| 10. Voice and photos | with the engine's work on them | talking to it; sending a photo |

Done: steps 1 to 5, and 7.

Alongside, the box track: first boot and setup, `leat.local` on the network, HTTPS on the home
network (a phone's microphone, notifications and home-screen install all need it), updates, and
later remote access.

## 12. As the proven agents do it

What ChatGPT, Claude, Hermes Agent and OpenClaw do in October 2026, and what leat agent takes of it,
after a real conversation filled the 3090's 16k context with a search's pages and the replies after
it were cut off, empty.

**The context.** Hermes counts the tokens the provider reports, and past 50% of the context first
clears old tool outputs over 200 characters ("[Old tool output cleared to save context space]"), then
has a model summarize the middle of the conversation into a structured summary (goal, preferences,
progress, decisions, files, next steps), keeping the system prompt and the recent tail whole, and
re-summarizes with the last summary when it fills again. OpenClaw has the model save its memories
just before. leat agent: the engine reports its context; the agent clears, then summarizes, at a
share of it, saving memories first; a reply cut off by the context is redone after, or said to be.

**Memory.** Three layers, which all of them have in some form:
- *Facts the user can see*: ChatGPT's saved memories, Claude's categorized entries (since July
  2026, updated live during a conversation), Hermes' bounded USER.md and MEMORY.md (some 1,300
  tokens, so that the model consolidates). Hermes' rules of what to save: preferences, facts about
  the user and their world, corrections; and what not: the trivial ("User asked about Python"),
  what a search finds again, raw data, the session's ephemera. It scans each entry for injected
  instructions and invisible characters before the system prompt carries it. leat agent: facts in
  categories, a bounded room, replace as well as add and remove, those rules, that scan.
- *A background pass*: ChatGPT's "dreaming" re-reads past conversations and rewrites the memory,
  freshness first ("going to Singapore in July" becomes "went"); Hermes reviews each turn in the
  background, as small models "often claim saves without executing them", as Qwen3.6 did here.
  leat agent: once a conversation is idle, the model reviews what is new in it, with the memory
  tools alone, and names the conversation.
- *Past conversations, searched*: Claude's conversation search and recent chats, Hermes'
  session_search over FTS5, which returns the messages themselves. leat agent: recall, by words
  and by time, giving the messages around each match.

**Search.** Hermes converts a page to markdown and gives it whole up to a budget (15,000
characters), or its start with the full text saved, to read on; Claude has code filter results
before they reach the context; all cite their sources inline, numbered, the app linking them.
leat agent: pages extracted to markdown, their boilerplate gone, in the sandbox; a budget, the
rest saved to read on; sources numbered across the conversation, cited as [1], linked in the app.

**Files.** Documents are read as markdown that keeps their headings and tables, and made by
skills, as Claude's; Hermes reads files by ranges. leat agent: PDF, Word, Excel and PowerPoint
converted to markdown in the sandbox, read by ranges.

## 13. What goes, what stays

Goes: the chat app's logic in `leat/app.html` (chats in the browser's storage, the tool loop in the
page, the tools-server setting), and `examples/tools.py`, whose search becomes the agent's `search`.

Stays: the design; `leat/markdown.mjs` and its tests, now `leat/agent/app/markdown.mjs`; Temml; `examples/searxng.yml`; the OpenAI
message format; the origin checks.

## 14. Decided

1. Our own agent, not Hermes, not on pi.
2. The agent is `leat/agent/`, run as `leat agent`: one package, one CLI.
3. `leat serve` serves the API alone. The agent's app and CLI are the only ways to use models at a
   high level: no plain chat page beside them. The codebase stays as lean as the engine's.
4. English first. Telegram is the first messaging channel.
5. Qwen3.6 35B A3B is the model the agent is tuned to now. On the Strix Halo the list grows, with
   bigger main models, chosen by measurements.
