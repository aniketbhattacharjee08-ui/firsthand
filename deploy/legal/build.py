"""Render deploy/legal/*.md to standalone HTML pages in the site's look.

    .venv/bin/python deploy/legal/build.py            # -> deploy/legal/html/
    .venv/bin/python deploy/legal/build.py --out web/legal   # at merge time

Stdlib only: the markdown here uses headings, paragraphs, bold, italics,
links and lists, nothing else. The CSS is self-contained and copies the
values from web/styles.css (Berkeley Blue surfaces, Inter, California Gold) so the pages
match without depending on the app's stylesheet or class names.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

CSS = """
:root{--g0:#010133;--g1:#031540;--g2:#071d4f;--line:rgba(159,209,255,.14);--line-2:rgba(159,209,255,.26);
--fg:#f3f4f8;--fg-2:#c3cae0;--fg-3:#96a0c2;--accent:#FDB515;--accent-deep:#FC9313;
--sans:"Inter",ui-sans-serif,-apple-system,"Segoe UI","Helvetica Neue",Arial,sans-serif;--r:4px}
html{background:var(--g0);color:var(--fg);font:16px/1.6 var(--sans);-webkit-text-size-adjust:100%}
body{margin:0;padding:48px 20px 80px}
main{max-width:66ch;margin:0 auto}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(253,181,21,.42)}
a:hover{color:var(--accent-deep)} a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:var(--r)}
nav{font-size:.875rem;color:var(--fg-3);display:flex;gap:16px;margin-bottom:32px}
nav a{border:0;color:var(--fg-2)} nav a[aria-current]{color:var(--fg)}
h1{font-size:1.375rem;font-weight:500;margin:0 0 8px}
h2{font-size:1.125rem;font-weight:500;margin:32px 0 8px;padding-top:16px;border-top:1px solid var(--line)}
p,li{color:var(--fg-2)} strong{color:var(--fg);font-weight:500} em{color:var(--fg-3)}
p.draft{font-size:.8125rem;color:var(--fg-3);border:1px solid var(--line-2);border-radius:var(--r);padding:8px 12px;background:var(--g1)}
ul{padding-left:20px} li{margin:4px 0}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);font-size:.8125rem;color:var(--fg-3)}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | {product}</title>
<meta name="color-scheme" content="dark">
<meta name="robots" content="index,follow">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap">
<style>{css}</style>
</head>
<body>
<main>
<nav aria-label="Legal pages"><a href="/">{product}</a>{nav}</nav>
{body}
<footer>Last updated {updated}.</footer>
</main>
</body>
</html>
"""

_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"_(.+?)_"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
]


def inline(text: str) -> str:
    out = html.escape(text, quote=False)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def render_markdown(md: str) -> str:
    blocks = re.split(r"\n\s*\n", md.strip())
    parts = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        first = lines[0]
        if first.startswith("# "):
            parts.append("<h1>%s</h1>" % inline(first[2:]))
            continue
        if first.startswith("## "):
            parts.append("<h2>%s</h2>" % inline(first[3:]))
            continue
        if all(l.lstrip().startswith(("- ", "* ")) for l in lines):
            items = "".join("<li>%s</li>" % inline(l.lstrip()[2:]) for l in lines)
            parts.append("<ul>%s</ul>" % items)
            continue
        text = " ".join(l.strip() for l in lines)
        if text.startswith("_") and text.endswith("_"):
            parts.append('<p class="draft">%s</p>' % inline(text[1:-1]))
            continue
        parts.append("<p>%s</p>" % inline(text))
    return "\n".join(parts)


def build(out_dir: Path, product: str, updated: str) -> list:
    pages = [("terms", "Terms of Service"), ("privacy", "Privacy Policy")]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for slug, title in pages:
        md = (HERE / ("%s.md" % slug)).read_text(encoding="utf-8")
        nav = "".join(
            '<a href="/legal/%s.html"%s>%s</a>' % (s, ' aria-current="page"' if s == slug else "", t)
            for s, t in pages
        )
        page = TEMPLATE.format(title=title, product=product, css=CSS.strip(), nav=nav, body=render_markdown(md), updated=updated)
        target = out_dir / ("%s.html" % slug)
        target.write_text(page, encoding="utf-8")
        written.append(target)
    return written


def main() -> int:
    import datetime as dt

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "html"))
    p.add_argument("--product", default="Vervly")
    p.add_argument("--updated", default=dt.date.today().isoformat())
    args = p.parse_args()
    for path in build(Path(args.out), args.product, args.updated):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
