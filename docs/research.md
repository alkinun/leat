# What the proven assistants do, and what leat agent takes of it

October 2026. Four studies, run side by side: how the consumer assistants keep memory (ChatGPT,
Claude, Gemini, Grok, Copilot, Perplexity); how the open agents and memory systems do (Hermes Agent,
OpenClaw, Letta, mem0, Zep, Honcho, Hindsight, and the benchmarks); how the best harnesses manage
context (Claude Code, Codex, Hermes, OpenClaw, Manus, Cursor, and the research on long contexts);
and what people use, love and leave (Grok and @grok, ChatGPT, Claude, Gemini, Alexa+, OpenClaw,
Hermes Agent, Dot, the companions, the self-hosted tools, families). The sources are linked; claims
from a company about itself are marked as such.

## 1. Memory

### What everyone converged on

Every assistant that lasted ended with the same three layers:

1. **A small profile, always in the prompt.** Claude's memory summary was about 1,000 tokens;
   Hermes Agent caps its two files at 2,200 and 1,375 characters; Claude Code loads the first 200
   lines of an index. ChatGPT's inferred profile is the outlier, some 4,000 dense tokens, and it is
   the one people find wrong: it recorded a trip to Turkey for a user who had only researched one
   ([Khemani](https://www.shloked.com/writing/chatgpt-memory-bitter-lesson)).
2. **The past conversations, searched when needed.** Claude's `conversation_search` and
   `recent_chats` (3 by default), Hermes' `session_search` over SQLite FTS5, OpenClaw's
   `memory_search` (BM25 and vectors), ChatGPT citing the past chat it drew on since January 2026.
3. **A background pass that writes and tidies the profile.** ChatGPT's "Dreaming" (June 2026,
   [OpenAI](https://openai.com/index/chatgpt-memory-dreaming)), Hermes' review every 10 user turns,
   Letta's sleep-time agent every 5, OpenClaw's nightly dreaming, Grok Build's `/dream`,
   Perplexity's overnight pass.

leat agent has all three already: a bounded profile of 3,000 characters, recall by words and by
time, and the review of idle conversations. What separates good memory from bad is in the details.

### What goes wrong, and how the best avoid it

- **Bloat and "memory full".** ChatGPT's list filled up until October 2025, when it began
  prioritizing by recency and frequency and moving the rest to the background
  ([TechRadar](https://www.techradar.com/ai-platforms-assistants/chatgpt/chatgpt-is-smarter-now-that-its-learned-to-forget-a-huge-memory-upgrade-is-coming)).
  Hermes refuses an add over the budget and answers with every entry and "retry as one batch that
  removes or shortens stale entries"; its header shows how full memory is,
  `[67% — 1,474/2,200 chars]`, so the model consolidates before it must. Perplexity reportedly
  recalls better with half as many memories. *Fewer, better facts beat more.*
- **Staleness.** "Going to Singapore in July" must become "went to Singapore in July 2026" (OpenAI's
  own Dreaming example). Gemini dates every statement and its source; Zep keeps when a fact held
  and retires a contradicted one rather than deleting it; Honcho turns relative dates into absolute
  ones. *Facts need dates, and plans need an end.*
- **Junk and wrong facts.** The worst errors come from taking what the model read for what the
  user said: research about Turkey becoming a trip. Hermes' rules: facts that apply to every
  session, declarative ("User prefers concise answers", not "Always answer concisely"), nothing
  stale within a week, no task progress, one fact in one place, no negative claims about tools.
  *A memory should rest on something the user said, never on a page or a tool's answer.*
- **Destructive background passes.** Hermes lets its unattended review only add; any change or
  removal waits for the user. OpenClaw rejects a consolidation that loses more than a quarter of
  the entries or passes the budget, and still promoted junk when its gates were bypassed (issues
  #89444, #112349). Claude replaced its rewritten daily summary with discrete entries in July 2026.
  *Small, checked operations; never a free rewrite; and an undo.*
- **Creepiness and over-use.** A divorce recalled for months; an old chat's sign appearing in an
  unrelated picture ([Willison](https://simonwillison.net/2025/May/21/chatgpt-new-memory/));
  memory feeding sycophancy, by OpenAI's own postmortem. Gemini's prompt forbids using memory
  unless asked. *Use a memory only where it changes the answer; the conversation overrides it.*
- **What is shown is not what is used.** ChatGPT's memory summary page is generated on demand and
  is not what the model reads, which its main reverse-engineer says erodes trust. leat agent's
  memory page is the very list the model reads: keep it so.
- **Injection.** A hidden instruction in a document wrote memories into Gemini and ChatGPT
  ([Embrace the Red](https://embracethered.com/blog/posts/2025/gemini-memory-persistence-prompt-injection/)).
  Hermes scans entries for it; OpenClaw labels recalled notes as untrusted.
- **Households.** OpenClaw shares one memory across everyone who messages an agent, and its own
  docs say true isolation needs an agent per person. Meta remembers only from one-to-one chats.
  Honcho models who observes whom, the right shape for a home.
- **Sensitive facts.** Claude leaves out health, religion, politics, ethnicity and gender identity
  unless the user opts in, and never stores IDs, criminal records, account numbers or immigration
  status ([Anthropic](https://support.claude.com/en/articles/11817273)).

### What the benchmarks really say

Elaborate systems mostly lose to plain ones. On LongMemEval with a Qwen3-32B reader, retrieving
single turns scored 66.0, whole sessions 62.0, the full context 55.2, A-Mem 53.2, mem0 50.6 and
LangMem 41.6 ([CoM, table 2](https://arxiv.org/html/2601.14287)); Letta's agent grepping files
scored 74.0 on LoCoMo, above mem0 ([Letta](https://www.letta.com/blog/benchmarking-ai-agent-memory)).
Knowledge graphs cost 10 to 20 model calls a turn and break on small models' structured output
(Graphiti's own README). What matters: a stronger reader, keeping the raw conversations, retrieving
small units, and handling time; updates and temporal questions are where every system is weakest.
The one strong result for small open models is Hindsight: 83.6% on LongMemEval with a 20B open model
against 39.0% for the full context ([paper](https://arxiv.org/abs/2512.12818)). LoCoMo's answer key
is about 6% wrong, so vendor numbers on it say little.

## 2. Context

### What the best do

- **The prompt is append-only between compactions**, and the cache's hit rate is "the single most
  important metric" ([Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)).
  Codex never rewrites an earlier message: changed instructions and the time arrive as new
  messages. Manus masks tools at decoding rather than removing them.
- **Clear before summarizing.** Claude Code clears old tool outputs first; a JetBrains study found
  masking old observations as good as summarizing at half the cost
  ([paper](https://arxiv.org/abs/2508.21433)); Anthropic's context editing alone gave +29% on a
  100-turn web eval, +39% with the memory tool
  ([Anthropic](https://www.anthropic.com/news/context-management)).
- **Compact rarely, in big steps**: Hermes triggers at 75% for windows under 512k, OpenClaw clears
  only when 50,000 characters can go, Anthropic's `clear_at_least` exists "to make the cache
  invalidation worthwhile".
- **Restorable placeholders**: drop a page, keep its address or file ("detail you can fetch again
  needs only a pointer", Claude Code's lean summary); Cursor writes long outputs to files the agent
  reads back.
- **The summary is asked of the warm conversation**, at its end, not of a cold transcript (Claude
  Code), and keeps the user's own words, errors, decisions and the next step quoted (Claude Code's
  nine sections, Codex keeps 20k tokens of recent user messages word for word); it is updated, not
  rewritten (Hermes).
- **Memory is saved before compacting**: OpenClaw's silent "memory flush" turn.
- **Readers keep the main context small**: subagents return 1–2k-token findings (Anthropic);
  Hermes' `delegate_task` children start fresh, cannot touch memory, and return a structured
  summary. Claude's multi-agent research beat a single agent by 90% at 15× the tokens.
- **Long contexts degrade**: all 18 models in Chroma's study get worse as input grows, focused
  ~300-token inputs beating ~113k ones ([Chroma](https://www.trychroma.com/research/context-rot)).
  For a model with 3B active parameters, a working budget of 32–48k is the sensible ceiling, whatever
  the window.

### What we found in leat, checked

- **Fixed now**: Qwen3.6's template draws the replies after the last user message with their
  `<think>` block and those before without. After any turn that called a tool, the next message
  therefore changed the prompt far back, and Qwen3.6, a hybrid model whose engine keeps one
  recurrent state per slot, read the whole conversation again from token 0. Rendered offline, a
  turn of one search: 870 of 1,616 tokens shared before, all of them now. leat agent now passes
  `preserve_thinking`, as Qwen recommends for agents.
- **The date is frozen in the system prompt** when a conversation starts. A Telegram chat is one
  conversation for weeks, and so are the conversations tasks run in: the model believes it is the
  day the conversation began.
- **The turn's last reply drops the tools**, which Qwen3.6 renders first: the whole prompt is read
  again, once, when a turn runs out of rounds.
- **Every compaction reads the whole prompt again** (one recurrent state per slot), and the
  summary is made by a cold prompt of its own, which also takes a slot.
- **Compaction comes too often**: after it, the prompt is the system prompt, the summary and a 25%
  tail, one or two pages from the 60% trigger again at 16k.
- **Placeholders are dead ends**: "call the tool again" rather than "read .web/….md".
- **The KV cache is cheap**: Qwen3.6 35B A3B takes about 20 KiB per token, 1.3 GiB at 64k, and a
  recurrent state about 63 MB. On the Strix Halo the window can be 64k or more; the limit is quality
  and prefill time, not memory.

## 3. Features

### What people use, love and leave

- **Use**: about 70% of ChatGPT's use is personal, and 80% is practical guidance, looking things
  up and writing (OpenAI's study). Families want a "family plan": 64% of US parents
  ([Qlik](https://www.qlik.com/us/news/company/press-room/press-releases/parents-want-ai-family-plans-as-bundles-drive-purchase)).
- **Love**: memory that notices (Dot recalled a photo from weeks before, and told its maker to
  drink less late at night); warmth (GPT-4o's retirement drew grief and petitions); the assistant
  living where people already are (OpenClaw went viral in WhatsApp and Telegram, people bought Mac
  minis to run it; 80% of @grok's calls are replies inside a thread already going,
  [study](https://arxiv.org/pdf/2605.19720)); daily briefings (the first thing OpenClaw users set
  up; Gemini's Daily Brief); pictures (ChatGPT's image launch, a million users in an hour).
- **Left**: ChatGPT Pulse, folded into scheduled tasks in June 2026, because people engaged with
  tasks they steer; ChatGPT's agent mode, from 4M weekly users to under 1M, as nobody knew what it
  was for; ChatGPT's group chats; Dot, Friend, Limitless, Pi.
- **Harm**: OpenClaw's skill market spread malware (1,184 malicious skills by February 2026) and
  30,000 instances sat exposed on the internet; Grok undressed real people, minors among them;
  companions flirted with children (Grok, Meta); heavy companion use goes with loneliness (MIT and
  OpenAI's study).

### Worth building, in order

1. **Household**: a profile for each person, private memory and chats, a shared family memory,
   device pairing. Every family feature rests on it.
2. **A daily brief each person steers**, and quiet check-ins that speak only when something needs
   it (OpenClaw's heartbeat): the scheduler is there.
3. **Photos in**: "what is this?", homework, receipts to a spreadsheet, a flyer to an event; needs a
   vision model in the engine.
4. **Voice**: Telegram voice notes and push-to-talk in the app first, a speaker later.
5. **Research**: readers in parallel, a report with sources; shares its readers with the context
   work below.
6. **Family logistics**: a forwarded school email or a photo of a flyer becomes events, reminders
   and a shared list.
7. **Children's profiles**: rules by age, no romance, quiet hours, parents set rules without reading
   chats.
8. **In the family's Telegram group**: "@leat is this true?", "sum this up", with sources.
9. **Watchers**: tell me when the price drops, the page changes, a storm warning comes.
10. **Sealed spaces** for health and money, whose memory stays in them (as ChatGPT Health's).
11. **WhatsApp and Signal**; **Home Assistant**, with timers and lights on plain paths, never the
    model; **skills that learn**, kept only once a parent approves; **local pictures** made, never of
    real people; **a tutor and characters**, pinned and honest.

Not to build: an open skill market; the box on the internet without pairing; sending or buying
without asking; an "agent mode" as a headline; romantic companions, or any companion for children;
flattery and engagement tricks; always-listening wearables; the model in place of a timer; changing
the model or persona silently.

## 4. The plan

### Memory v3: small, dated, checked, undoable, per person

- **Facts with dates.** Each memory keeps when it was made and last confirmed; a plan has the date
  it ends. The prompt shows them compactly ("Plans: [12] Ada's party on 17 Oct"), today's date is
  given, and plans past their end are turned into the past tense or forgotten by the review.
- **Evidence.** A memory the review adds must quote the user's own words, checked in code against
  the conversation's user messages; nothing comes of a page, a file or a tool's answer.
- **Nothing lost.** Forgetting and replacing keep the old text in a history, which the memory page
  can undo; the unattended review may add and date, and what it changes or forgets is undoable.
- **Full is a policy.** The prompt's header says how full memory is; past the budget, remember
  answers with the least recently confirmed memories and asks for a merge or a replacement.
- **Writing rules** as Hermes': facts about the user and their world, declarative, lasting, one
  place each, no task progress, nothing about tools.
- **A nightly consolidation**, while the box is idle: duplicates merged, plans past dated, the
  budget kept; refused in code if it loses more than a quarter of the entries.
- **Used sparingly**: "use a memory only where it changes the answer; what is said now overrides
  it".
- **Sensitive and secret**: health, religion, politics and the like only if the user asks; IDs,
  account numbers and passwords never, filtered in code.
- **Per person**, with household accounts: each person's profile, a shared household block, recall
  filtered by person, nothing personal from a group chat.
- **Recall by turn**, citing the conversation and its date, linked in the app; later local
  embeddings beside FTS5.
- **Our own memory eval**, LongMemEval's kind of cases from a household's life: updates, dates,
  "I don't know", and no leaks between people.

### Context v2: append-only, measured, compacted rarely

1. The date and time in each user message, as the model reads it, not in the system prompt.
2. The last round keeps its tools and is told to answer; calls it makes anyway are not run.
3. The cache's hit rate measured on every reply, shown in the reply's details, and in the eval.
4. Compaction in big steps: past ~80% of a working budget, down to ~35% at once; clearing first;
   placeholders that name the saved file or address; the summary asked of the warm conversation,
   updated rather than rewritten, keeping the user's words, decisions, errors and promises; memory
   flushed before.
5. In the engine: recurrent states kept at more points (after the system prompt and tools, at
   each user message) and slots evicted to memory, so that a change costs from where it is, and
   conversations, tasks and readers stop evicting each other.
6. Readers: a page read by a fresh call that answers the question in a few hundred tokens with its
   source, the page kept in a file; the same readers run in parallel for research.
7. On the Strix Halo, a 64k window and a 32–48k working budget.

### Order

1. Context v2's quick fixes (1–3): small, and they make every conversation faster and right about
   the date.
2. Memory v3 together with household accounts: both change what a memory belongs to, so they are
   designed once.
3. Readers and research.
4. The daily brief and check-ins, then photos and voice, with the engine's vision and speech work.
