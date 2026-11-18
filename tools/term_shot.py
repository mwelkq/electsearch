"""
Знімок виводу команди у вигляді термінала VS Code (для рисунків курсової).

  python tools/term_shot.py screenshots/01_bm25.png -- python search_bm25.py -k 3 "де подивитись розклад пар"

Команда справді виконується в папці проєкту; її реальний вивід (stdout+stderr,
без рядків завантаження моделі) рендериться в HTML і фотографується Edge у
headless-режимі.
"""
import html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PROMPT = r"PS C:\Users\1\Desktop\унік\курсова робота> "
NOISE = re.compile(r"Loading weights|HF_TOKEN|unauthenticated requests|^\s*$(?!\n)")


def run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, env=env)
    out = (p.stdout + p.stderr).decode("utf-8", "replace").replace("\r", "")
    lines = [l for l in out.split("\n") if not re.search(r"Loading weights|HF_TOKEN|unauthenticated", l)]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def render(png, shown_cmd, lines, width=1180, max_lines=None):
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]
    body = html.escape("\n".join(lines))
    cmd = html.escape(shown_cmd)
    page = f"""<!doctype html><meta charset="utf-8"><style>
html,body{{margin:0;background:#181818}}
.panel{{font:14px/1.45 'Cascadia Mono',Consolas,monospace;color:#cccccc;background:#181818}}
.tabs{{display:flex;gap:22px;padding:7px 18px 0;border-bottom:1px solid #2b2b2b;font:11px 'Segoe UI',sans-serif;letter-spacing:.04em;color:#9d9d9d}}
.tabs span{{padding-bottom:6px}} .tabs .on{{color:#e7e7e7;border-bottom:1px solid #0078d4}}
.right{{margin-left:auto;color:#9d9d9d}}
pre{{margin:0;padding:10px 18px 16px;white-space:pre-wrap;word-break:break-word;font:inherit}}
.p{{color:#cccccc}} .c{{color:#e5e510}}
</style><div class="panel"><div class="tabs"><span>PROBLEMS</span><span>OUTPUT</span><span>DEBUG CONSOLE</span><span class="on">TERMINAL</span><span>PORTS</span><span class="right">powershell</span></div>
<pre><span class="p">{html.escape(PROMPT)}</span><span class="c">{cmd}</span>
{body}
<span class="p">{html.escape(PROMPT)}</span></pre></div>"""
    n = len(lines) + 3 + sum(len(l) // 128 for l in lines)
    height = 46 + int(n * 20.3) + 20
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(page)
    png = Path(png).resolve()
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={width},{height}", f"--screenshot={png}",
                    Path(f.name).as_uri()], capture_output=True, timeout=60)
    os.unlink(f.name)
    print(f"{png}  ({len(lines)} рядків)")


if __name__ == "__main__":
    a = sys.argv[1:]
    png, cmd = a[0], a[a.index("--") + 1:]
    shown = " ".join(c if " " not in c else f'"{c}"' for c in cmd)
    max_lines = int(os.environ.get("MAX_LINES", "0")) or None
    render(png, shown, run(cmd), max_lines=max_lines)
