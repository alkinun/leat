// Markdown as models write it. parse() reads text to a tree in JsonML, a string or
// [tag, attributes?, ...children], of the elements a reply can hold and no others: none of the
// HTML a model writes reaches the page. markdown() shows the tree in an element, its math as
// MathML by Temml, which leat/vendor/temml holds.

import temml from "./vendor/temml/temml.mjs";

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
  closers.clear();
  brackets.clear();
  return blocks(text.split("\n"));
}

function render(nodes) {
  return nodes.map((node) => {
    if (typeof node === "string") return node;
    const [tag, ...children] = node;
    const attributes = children[0]?.constructor === Object ? children.shift() : {};
    if (tag === "math") return tex(children[0] ?? "", attributes.display === "block");
    const e = document.createElement(tag);
    if (tag === "a") Object.assign(attributes, { target: "_blank", rel: "noopener noreferrer" });
    for (const [name, value] of Object.entries(attributes)) e.setAttribute(name, value);
    e.append(...render(children));
    return e;
  });
}

// TeX as MathML, or what Temml cannot read, half streamed say, as code
function tex(source, display) {
  const e = document.createElement(display ? "div" : "span");
  e.className = "math";
  try {
    temml.render(source, e, { displayMode: display, throwOnError: true });
    return e;
  } catch {
    const code = ["code", ...(display ? [{ class: "language-tex" }] : []), source];
    return render([display ? ["pre", code] : code])[0];
  }
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
const MATH = /^ {0,3}(\$\$|\\\[)/;
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
  [FENCE, code], [MATH, displayMath], [HEADING, heading], [RULE, rule], [QUOTE, quote],
  [ITEM, list], [TABLE, table],
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
  out.push(["pre", ["code", ...attributes, ...highlight(body.join("\n"), language)]]);
  return i + 1;
}

// $$ math $$ or \[ math \], displayed, on a line or over lines to its close or, streaming, the end
function displayMath(lines, i, out) {
  const [opening, open] = MATH.exec(lines[i]);
  const text = lines.slice(i).join("\n").slice(opening.length), end = text.indexOf(CLOSE[open]);
  out.push(["math", { display: "block" }, (end < 0 ? text : text.slice(0, end)).trim()]);
  if (end < 0) return lines.length;
  const rest = text.slice(end + 2).split("\n", 1)[0]; // what follows the close on its line
  if (rest.trim()) out.push(["p", ...inline(rest.trim())]);
  return i + text.slice(0, end).split("\n").length;
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
  "\\": escape, "`": codeSpan, $: math, "*": emphasis, _: emphasis, "~": emphasis,
  "[": link, "!": link, "<": autolink, h: url,
};

// where the escape, code span or math at j ends, none of which another span reaches into, or -1
function atom(text, j) {
  return { "\\": escape, "`": codeSpan, $: math }[text[j]]?.(text, j)?.[1] ?? -1;
}

// \* as *, and \(math\) and \[math\]
function escape(text, i) {
  const next = text[i + 1] ?? "";
  return (/[([]/.test(next) && math(text, i)) || (PUNCTUATION.test(next) ? [next, i + 2] : null);
}

// $math$ and \(math\), and $$math$$ and \[math\] displayed. A lone $ opens after no letter or
// digit and before a non-space, and the next closes it after a non-space and before no digit, or
// none does: $5 to $10, and US$5, stay text.
function math(text, i) {
  const open = text.startsWith("$$", i) ? "$$" : text.slice(i, text[i] === "$" ? i + 1 : i + 2);
  const close = CLOSE[open], from = i + open.length, lone = open === "$";
  if (lone && (/\s/.test(text[from] ?? " ") || WORD.test(text[i - 1] ?? ""))) return null;
  for (let end = text.indexOf(close, from + 1); end >= 0; end = text.indexOf(close, end + 1)) {
    if (lone && text[end - 1] === "\\") continue; // \$, a dollar
    if (lone && (/\s/.test(text[end - 1]) || /\d/.test(text[end + 1] ?? ""))) return null;
    const display = lone || open === "\\(" ? [] : [{ display: "block" }];
    return [["math", ...display, text.slice(from, end).trim()], end + close.length];
  }
  return null;
}

const CLOSE = { $: "$", $$: "$$", "\\(": "\\)", "\\[": "\\]" }; // each math's, by its opening

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

// where the run of n c's that closes the span from j starts, past the spans within it, or -1:
// searched once for each text, j, c and n, as the spans that never close would each search the
// rest of the text again for each before them, and on a stack of searches rather than by
// recursion, as such spans may nest thousands deep
function closer(text, j, c, n) {
  const memo = closers.get(text) ?? closers.set(text, new Map()).get(text);
  if (memo.has(`${c}${n}:${j}`)) return memo.get(`${c}${n}:${j}`);
  const searches = [{ from: j, j, n, run: 0 }];
  for (;;) {
    const s = searches.at(-1), found = search(text, s, c, memo);
    if (found === undefined) { // it waits on the span that opens at s.j, searched first
      searches.push({ from: s.j + s.run, j: s.j + s.run, n: Math.min(s.run, 3), run: 0 });
      continue;
    }
    memo.set(`${c}${s.n}:${s.from}`, found);
    searches.pop();
    if (!searches.length) return found;
    const outer = searches.at(-1); // past the span within it, or its opening run if none closes
    outer.j = found < 0 ? outer.j + outer.run : found + Math.min(outer.run, 3);
  }
}

const closers = new Map(); // closer()'s answers by text, of the text parse() last took

// advances a search of closer()'s to the close of its span, or -1 where none closes it, past the
// spans within whose closes are known; undefined where it must wait on one that opens at s.j
function search(text, s, c, memo) {
  while (s.j < text.length) {
    const end = atom(text, s.j);
    if (end >= 0) {
      s.j = end;
    } else if (text[s.j] !== c) {
      s.j++;
    } else {
      const run = (s.run = runAt(text, s.j)), inner = Math.min(run, 3);
      if (s.j > s.from && run >= s.n && closes(text, s.j, run)) return s.j;
      if (opens(text, s.j, run) && !memo.has(`${c}${inner}:${s.j + run}`)) return undefined;
      const end = opens(text, s.j, run) ? memo.get(`${c}${inner}:${s.j + run}`) : -1;
      s.j = end < 0 ? s.j + run : end + inner;
    }
  }
  return -1;
}

// [text](url "title"), and ![alt](url) as a link to the image: a link to anything but the web or
// mail stays text, so that none runs script
function link(text, i) {
  const open = text[i] === "!" ? i + 1 : i;
  if (text[open] !== "[") return null;
  const close = bracket(text, open);
  if (close < 0) return null;
  const destination = DESTINATION.exec(text.slice(close + 1));
  const href = destination?.[1] ?? destination?.[2];
  if (!SAFE.test(href ?? "")) return null;
  const end = close + 1 + destination[0].length;
  return [["a", { href }, ...inline(text.slice(open + 1, close))], end];
}

// where the ] that closes the [ at i is, past the escapes, code spans and math within, or -1:
// each [ a search passes is matched too, and kept, so that no text is searched twice
function bracket(text, i) {
  const memo = brackets.get(text) ?? brackets.set(text, new Map()).get(text);
  if (memo.has(i)) return memo.get(i);
  const open = [];
  for (let j = i; j < text.length; ) {
    const end = atom(text, j);
    if (end >= 0) {
      j = end;
      continue;
    }
    if (text[j] === "[") {
      open.push(j);
    } else if (text[j] === "]" && open.length) {
      memo.set(open.pop(), j);
      if (!open.length) return j;
    }
    j++;
  }
  for (const k of open) memo.set(k, -1);
  return -1;
}

const brackets = new Map(); // bracket()'s answers by text, of the text parse() last took

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

// code, its comments, strings, numbers and keywords set apart as the languages models write most
// have them, the rest as it is
function highlight(code, language) {
  const grammar = GRAMMARS.get(language.toLowerCase());
  if (!grammar || !code) return code ? [code] : [];
  const out = [];
  let end = 0;
  for (const match of code.matchAll(grammar.tokens)) {
    const [token, comment, string, number, word] = match;
    const kind = (comment && "comment") || (string && "string") || (number && "number")
      || (grammar.keywords.has(grammar.caseless ? word.toLowerCase() : word) && "keyword");
    if (!kind) continue;
    if (match.index > end) out.push(code.slice(end, match.index));
    out.push(["span", { class: kind }, token]);
    end = match.index + token.length;
  }
  if (end < code.length) out.push(code.slice(end));
  return out;
}

// a language's tokens, each kind a group of one expression, and its keywords
function grammar(comments, strings, keywords, caseless = false) {
  const number = String.raw`\b(?:0x[\da-f]+|\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?)\b`;
  const word = String.raw`(?<![\w$])[a-z_$][\w$]*`;
  const tokens = `(${comments || "(?!)"})|(${strings.join("|")})|(${number})|(${word})`;
  return { tokens: new RegExp(tokens, "gi"), keywords: new Set(keywords.split(" ")), caseless };
}

const HASH = String.raw`(?<!\S)#.*`; // after a space, not in $# or a URL's #
const SLASHES = String.raw`\/\/.*|\/\*[\s\S]*?(?:\*\/|$)`;
const DOUBLE = String.raw`"(?:\\.|[^"\\\n])*"`;
const SINGLE = String.raw`'(?:\\.|[^'\\\n])*'`;
const CHAR = String.raw`'(?:\\[^'\n]+|[^'\\\n])'`; // not a Rust lifetime's '
const TEMPLATE = String.raw`\x60(?:\\[\s\S]|[^\x60\\])*\x60`;
const TRIPLE = String.raw`"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)`;

// each grammar, by the names a code block's language has for it
const GRAMMARS = new Map(Object.entries({ // a Map, which holds no constructor or __proto__
  "py python python3": grammar(HASH, [TRIPLE, DOUBLE, SINGLE], "False None True and as assert "
    + "async await break case class continue def del elif else except finally for from global if "
    + "import in is lambda match nonlocal not or pass raise return self try while with yield"),
  "js javascript jsx mjs ts typescript tsx": grammar(SLASHES, [DOUBLE, SINGLE, TEMPLATE], "as "
    + "async await break case catch class const continue default delete do else enum export "
    + "extends false finally for from function if implements import in instanceof interface let "
    + "new null of private protected public readonly return static super switch this throw true "
    + "try type typeof undefined var void while yield"),
  "sh bash shell zsh console": grammar(HASH, [DOUBLE, "'[^']*'"], "case do done elif else esac "
    + "export fi for function if in local return then until while"),
  "c h cpp c++ cc hpp cs csharp java kotlin kt go rust rs swift zig": grammar(SLASHES,
    [DOUBLE, CHAR], "abstract as async auto await bool break byte case catch char chan class "
    + "const continue crate default defer delete do double dyn else enum extends extern false "
    + "final float fn for func go goto if impl implements import in int interface let long loop "
    + "map match mod move mut namespace new nil null nullptr override package private protected "
    + "pub public range return select self Self short signed sizeof static struct super switch "
    + "template this throw trait true try type typedef typename union unsafe unsigned use using "
    + "var virtual void volatile where while"),
  sql: grammar("--.*", [SINGLE, DOUBLE], "all alter and as asc between by case create default "
    + "delete desc distinct drop else end exists foreign from group having in index inner insert "
    + "into is join key left like limit not null offset on or order outer primary references "
    + "right select set table then union update values view when where with", true),
  "json jsonc": grammar("", [DOUBLE], "true false null"),
  "yaml yml toml ini": grammar(HASH, [DOUBLE, SINGLE], "true false null yes no"),
  "css scss": grammar(String.raw`\/\*[\s\S]*?(?:\*\/|$)`, [DOUBLE, SINGLE], ""),
}).flatMap(([names, g]) => names.split(" ").map((name) => [name, g])));
