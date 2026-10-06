// Markdown as models write it. parse() reads text to a tree in JsonML, a string or
// [tag, attributes?, ...children], of the elements a reply can hold and no others: none of the
// HTML a model writes reaches the page. markdown() makes the tree elements.

export function markdown(text) {
  return render(parse(text));
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
    for (const [name, value] of Object.entries(attributes)) e.setAttribute(name, value);
    e.append(...render(children));
    return e;
  });
}

const FENCE = /^( {0,3})(`{3,}|~{3,})\s*([^\s`]*)/;
const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const BREAKS = [FENCE, HEADING, RULE]; // the blocks that break into a paragraph

function blocks(lines) {
  const out = [];
  for (let i = 0; i < lines.length; ) {
    const line = lines[i];
    let match;
    if (!line.trim()) {
      i++;
    } else if (FENCE.test(line)) {
      i = code(lines, i, out);
    } else if ((match = HEADING.exec(line))) {
      out.push([`h${match[1].length}`, ...inline(match[2] ?? "")]);
      i++;
    } else if (RULE.test(line)) {
      out.push(["hr"]);
      i++;
    } else { // a paragraph, to a blank line or a block that breaks in
      const start = i++;
      while (i < lines.length && lines[i].trim() && !BREAKS.some((b) => b.test(lines[i]))) i++;
      out.push(["p", ...inline(lines.slice(start, i).map((l) => l.trim()).join("\n"))]);
    }
  }
  return out;
}

// a fenced code block, to its closing fence or, still streaming, to the end: the line after it
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
  const c = text[i];
  if (c === "\\" && PUNCTUATION.test(text[i + 1] ?? "")) return [text[i + 1], i + 2];
  if (c === "`") return codeSpan(text, i);
  return ("*_~".includes(c) && emphasis(text, i)) || [c, i + 1];
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
