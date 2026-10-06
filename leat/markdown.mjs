// Markdown as models write it. parse() reads text to a tree in JsonML, a string or
// [tag, attributes?, ...children], of the elements a reply can hold and no others: none of the
// HTML a model writes reaches the page. markdown() shows the tree in an element.

const shown = new WeakMap(); // each element's blocks, as JSON

// shows text in element, keeping the elements of the blocks that are as they were: as a reply
// streams, those before its last, and what is selected in them
export function markdown(element, text) {
  const trees = parse(text), before = shown.get(element) ?? [];
  const keys = trees.map((tree) => JSON.stringify(tree));
  keys.forEach((key, i) => {
    const old = element.children[i];
    if (old && key === before[i]) return;
    const [e] = render([trees[i]]);
    if (old) old.replaceWith(e);
    else element.append(e);
  });
  while (element.children.length > keys.length) element.lastElementChild.remove();
  shown.set(element, keys);
}

export function parse(text) {
  return blocks(text.split("\n"));
}

function render(nodes) {
  return nodes.map((node) => {
    if (typeof node === "string") return node;
    const [tag, ...children] = node;
    const e = document.createElement(tag);
    const attributes = children[0]?.constructor === Object ? children.shift() : {};
    if (tag === "a") Object.assign(attributes, { target: "_blank", rel: "noopener noreferrer" });
    for (const [name, value] of Object.entries(attributes)) e.setAttribute(name, value);
    e.append(...render(children));
    return e;
  });
}

function blocks(lines) {
  const out = [];
  for (let i = 0; i < lines.length; ) {
    if (!lines[i].trim()) i++;
    else i = (starts(lines, i) ?? paragraph)(lines, i, out);
  }
  return out;
}

const FENCE = /^( {0,3})(`{3,}|~{3,})\s*([^\s`]*)/;
const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const QUOTE = /^ {0,3}> ?/;
const ITEM = /^( {0,3})([-*+]|(\d{1,9})[.)])([ \t]+|$)/;
const DELIMITER = /^ {0,3}\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$/;
const TABLE = {
  test: (line, next = "") => line.includes("|") && DELIMITER.test(next)
    && cells(line).length === cells(next).length,
};

// each block but the paragraph: a test of whether a line, before the next, starts one, and what
// reads it to out from there, returning the line after it
const BLOCKS = [
  [FENCE, code], [HEADING, heading], [RULE, rule], [QUOTE, quote], [ITEM, list], [TABLE, table],
];

// what reads the block that starts at line i, if one but a paragraph does
function starts(lines, i) {
  return BLOCKS.find(([start]) => start.test(lines[i], lines[i + 1]))?.[1];
}

// a fenced code block, to its closing fence or, still streaming, to the end
function code(lines, i, out) {
  const [, indent, fence, language] = FENCE.exec(lines[i]);
  const close = new RegExp(`^ {0,3}${fence[0]}{${fence.length},}\\s*$`);
  const body = [];
  for (i++; i < lines.length && !close.test(lines[i]); i++) {
    body.push(lines[i].replace(new RegExp(`^ {0,${indent.length}}`), ""));
  }
  const attributes = language ? [{ class: `language-${language}` }] : [];
  out.push(["pre", ["code", ...attributes, body.join("\n")]]);
  return i + 1;
}

function heading(lines, i, out) {
  const [, hashes, text = ""] = HEADING.exec(lines[i]);
  out.push([`h${hashes.length}`, ...inline(text)]);
  return i + 1;
}

function rule(lines, i, out) {
  out.push(["hr"]);
  return i + 1;
}

// > a quote's lines, of blocks of their own
function quote(lines, i, out) {
  const start = i;
  while (i < lines.length && QUOTE.test(lines[i])) i++;
  out.push(["blockquote", ...blocks(lines.slice(start, i).map((l) => l.replace(QUOTE, "")))]);
  return i;
}

// a list's items, while their markers are of a kind. An item runs on to the lines indented past
// its marker's width, items indented past its marker, and a paragraph's lines run on lazily. A
// blank line between items, or between an item's blocks, makes the list loose: its items'
// paragraphs stay paragraphs.
function list(lines, i, out) {
  const [, , marker, start] = ITEM.exec(lines[i]);
  const items = [];
  let loose = false, gap = false;
  for (let item; i < lines.length && (item = ITEM.exec(lines[i])); ) {
    const [whole, indent, mark, , spaces] = item;
    if (mark.at(-1) !== marker.at(-1)) break;
    const width = spaces && spaces.length <= 4 ? whole.length : indent.length + mark.length + 1;
    const body = [lines[i].slice(whole.length)];
    for (i++; i < lines.length; i++) {
      const line = lines[i], indented = /^ */.exec(line)[0].length;
      if (!line.trim()) body.push("");
      else if (indented >= width || (indented > indent.length && ITEM.test(line))) {
        body.push(line.slice(Math.min(indented, width)));
      } else if (body.at(-1).trim() && !starts(lines, i)) body.push(line);
      else break;
    }
    loose ||= gap;
    for (gap = false; body.at(-1) === ""; gap = true) body.pop();
    loose ||= body.includes("");
    items.push(blocks(body));
  }
  const unwrap = (block) => (!loose && block[0] === "p" ? block.slice(1) : [block]);
  const attributes = start !== undefined && +start !== 1 ? [{ start: +start }] : [];
  const tag = start === undefined ? "ul" : "ol";
  out.push([tag, ...attributes, ...items.map((item) => ["li", ...item.flatMap(unwrap)])]);
  return i;
}

// | a table | its header |, a row of its columns' alignments, then rows to a blank line or a block
function table(lines, i, out) {
  const align = cells(lines[i + 1]).map((c) => ALIGN[c[0] + c.at(-1)]);
  const row = (line, tag) => {
    const texts = cells(line);
    return ["tr", ...align.map((a, j) => {
      const attributes = a ? [{ align: a }] : [];
      return [tag, ...attributes, ...inline(texts[j] ?? "")];
    })];
  };
  const head = row(lines[i], "th"), body = [];
  for (i += 2; i < lines.length && lines[i].trim() && !starts(lines, i); i++) {
    body.push(row(lines[i], "td"));
  }
  out.push(["table", ["thead", head], ...(body.length ? [["tbody", ...body]] : [])]);
  return i;
}

const ALIGN = { ":-": "left", "-:": "right", "::": "center" }; // by a delimiter's ends

// a row's cells, split at the pipes not escaped
function cells(line) {
  const row = line.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "");
  return row.split(/(?<!\\)\|/).map((cell) => cell.trim().replaceAll("\\|", "|"));
}

// lines to a blank one or one that starts another block
function paragraph(lines, i, out) {
  const start = i++;
  while (i < lines.length && lines[i].trim() && !starts(lines, i)) i++;
  out.push(["p", ...inline(lines.slice(start, i).map((l) => l.trim()).join("\n"))]);
  return i;
}

// a paragraph's text, its spans set apart
function inline(text) {
  const out = [];
  for (let i = 0; i < text.length; ) {
    const [node, end] = span(text, i);
    if (typeof node === "string" && typeof out.at(-1) === "string") out[out.length - 1] += node;
    else out.push(node);
    i = end;
  }
  return out;
}

// the span at i, or the text that stands for itself there, and where either ends
function span(text, i) {
  return SPANS[text[i]]?.(text, i) ?? [text[i], i + 1];
}

const SPANS = {
  "\\": escape, "`": codeSpan, "*": emphasis, _: emphasis, "~": emphasis,
  "[": link, "!": link, "<": autolink, h: url,
};

function escape(text, i) {
  return PUNCTUATION.test(text[i + 1] ?? "") ? [text[i + 1], i + 2] : null;
}

// `code`, between runs of as many backticks, its one space each side dropped if both have one
function codeSpan(text, i) {
  const run = /^`+/.exec(text.slice(i))[0];
  const close = new RegExp(`(?<!\`)${run}(?!\`)`, "g");
  close.lastIndex = i + run.length;
  const match = close.exec(text);
  if (!match) return [run, i + run.length];
  let content = text.slice(i + run.length, match.index).replaceAll("\n", " ");
  if (/^ .*[^ ].* $/.test(content)) content = content.slice(1, -1);
  return [["code", content], match.index + run.length];
}

// *em*, **strong**, ***both*** and ~~struck~~: a run of a delimiter opens before a non-space,
// and one as long closes after one, an _ neither within a word. A run of three that no run of
// three closes may open strong in em, or em in strong.
function emphasis(text, i) {
  const c = text[i], run = runAt(text, i);
  if (!opens(text, i, run)) return null;
  for (const n of c === "~" ? [2] : [3, 2, 1]) {
    const end = n <= run ? closer(text, i + n, c, n) : -1;
    if (end < 0) continue;
    const content = inline(text.slice(i + n, end));
    const tag = { 1: "em", 2: c === "~" ? "del" : "strong" }[n];
    return [tag ? [tag, ...content] : ["em", ["strong", ...content]], end + n];
  }
  return null;
}

// where the run of n c's that closes the span from j starts, past the spans within it
function closer(text, j, c, n) {
  const from = j;
  while (j < text.length) {
    if (text[j] === "\\") {
      j += 2;
    } else if (text[j] === "`") {
      j = codeSpan(text, j)[1];
    } else if (text[j] !== c) {
      j++;
    } else {
      const run = runAt(text, j), inner = Math.min(run, 3);
      if (j > from && run >= n && closes(text, j, run)) return j;
      const end = opens(text, j, run) ? closer(text, j + run, c, inner) : -1;
      j = end < 0 ? j + run : end + inner;
    }
  }
  return -1;
}

// [text](url "title"), and ![alt](url) as a link to the image: a link to anything but the web or
// mail stays text, so that none runs script
function link(text, i) {
  const open = text[i] === "!" ? i + 1 : i, close = bracket(text, open);
  if (text[open] !== "[" || close < 0) return null;
  const destination = DESTINATION.exec(text.slice(close + 1));
  const href = destination?.[1] ?? destination?.[2];
  if (!SAFE.test(href ?? "")) return null;
  const end = close + 1 + destination[0].length;
  return [["a", { href }, ...inline(text.slice(open + 1, close))], end];
}

// where the ] that closes the [ at i is, past the code spans and escapes within
function bracket(text, i) {
  for (let j = i, depth = 0; j < text.length; ) {
    if (text[j] === "\\") {
      j += 2;
    } else if (text[j] === "`") {
      j = codeSpan(text, j)[1];
    } else {
      depth += { "[": 1, "]": -1 }[text[j]] ?? 0;
      if (depth === 0) return j;
      j++;
    }
  }
  return -1;
}

// <https://example.com>
function autolink(text, i) {
  const match = /^<((?:https?|mailto):[^\s<>]+)>/i.exec(text.slice(i));
  return match && [["a", { href: match[1] }, match[1]], i + match[0].length];
}

// https://example.com bare, after a space or the start, without the punctuation that ends a
// sentence or wraps it
function url(text, i) {
  let href = /^https?:\/\/[^\s<]+/.exec(text.slice(i))?.[0];
  if (!href || /[^\s(*_~]/.test(text[i - 1] ?? " ")) return null;
  const unbalanced = () => href.split(")").length > href.split("(").length;
  while (/[?!.,:;*_~'"]$/.test(href) || (href.endsWith(")") && unbalanced())) {
    href = href.slice(0, -1);
  }
  return /^https?:\/\/./.test(href) ? [["a", { href }, href], i + href.length] : null;
}

// a link's (destination "title"), the destination in <> or with its parentheses balanced
const DESTINATION = /^\(\s*(?:<([^<>\n]*)>|((?:[^\s()]|\([^\s()]*\))+))(?:\s+"[^"]*")?\s*\)/;
const SAFE = /^(?:https?:\/\/|mailto:)/i;
const PUNCTUATION = /[!-/:-@[-`{-~]/; // ASCII's, which a backslash escapes
const WORD = /[\p{L}\p{N}]/u;

function runAt(text, i) {
  let j = i;
  while (text[j] === text[i]) j++;
  return j - i;
}

function opens(text, i, run) {
  return !/\s/.test(text[i + run] ?? " ") && !(text[i] === "_" && WORD.test(text[i - 1] ?? ""));
}

function closes(text, j, run) {
  return !/\s/.test(text[j - 1]) && !(text[j] === "_" && WORD.test(text[j + run] ?? ""));
}
