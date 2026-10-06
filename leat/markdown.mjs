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

function blocks(lines) {
  const out = [];
  for (let i = 0; i < lines.length; ) {
    if (!lines[i].trim()) {
      i++;
    } else if (FENCE.test(lines[i])) {
      i = code(lines, i, out);
    } else { // a paragraph, to a blank line or a block that breaks in
      const start = i++;
      while (i < lines.length && lines[i].trim() && !FENCE.test(lines[i])) i++;
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

// a paragraph's text with its code spans and bold set apart
function inline(text) {
  const out = [];
  let plain = "";
  for (let i = 0; i < text.length; ) {
    const span = text[i] === "`" ? codeSpan(text, i) : text.startsWith("**", i) && strong(text, i);
    if (span) {
      if (plain) out.push(plain);
      plain = "";
      out.push(span.node);
      i = span.end;
    } else {
      plain += text[i++];
    }
  }
  if (plain) out.push(plain);
  return out;
}

// `code`, between runs of as many backticks, its one space each side dropped if both have one
function codeSpan(text, i) {
  const run = /^`+/.exec(text.slice(i))[0];
  const close = new RegExp(`(?<!\`)${run}(?!\`)`, "g");
  close.lastIndex = i + run.length;
  const match = close.exec(text);
  if (!match) return null;
  let content = text.slice(i + run.length, match.index).replaceAll("\n", " ");
  if (/^ .*[^ ].* $/.test(content)) content = content.slice(1, -1);
  return { node: ["code", content], end: match.index + run.length };
}

function strong(text, i) {
  const end = text.indexOf("**", i + 2);
  if (end <= i + 2) return null;
  return { node: ["strong", ...inline(text.slice(i + 2, end))], end: end + 2 };
}
