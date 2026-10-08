"""Shows the executed Colab notebook (cells + saved outputs) inside the app, without needing Jupyter.

The notebook runs on a Colab GPU; once it has been run and saved back into notebooks/, its outputs
(tables, figures, the embedded video, printed numbers) are part of the .ipynb file and are shown here as they are.
"""
import json
import os
import re
from html import escape

from markupsafe import Markup

try:
    import markdown as _md
except ImportError:  # the page still works, markdown cells just show as plain text
    _md = None


def _text(v):
    return "".join(v) if isinstance(v, list) else (v or "")


def _markdown(src):
    if _md is None:
        return Markup(f"<pre>{escape(src)}</pre>")
    return Markup(_md.markdown(src, extensions=["tables", "fenced_code"]))


def _output(o):
    kind = o.get("output_type")
    if kind == "stream":
        return {"kind": "text", "html": Markup(f"<pre>{escape(_text(o.get('text')))}</pre>")}
    if kind == "error":
        tb = "\n".join(o.get("traceback", []))
        tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
        return {"kind": "error", "html": Markup(f"<pre>{escape(o.get('ename', ''))}: {escape(o.get('evalue', ''))}\n{escape(tb)}</pre>")}
    data = o.get("data", {})
    if "image/png" in data:
        return {"kind": "image", "html": Markup(f'<img src="data:image/png;base64,{_text(data["image/png"]).strip()}">')}
    if "image/jpeg" in data:
        return {"kind": "image", "html": Markup(f'<img src="data:image/jpeg;base64,{_text(data["image/jpeg"]).strip()}">')}
    if "text/html" in data:
        # our own notebook: pandas tables and the embedded <video>
        return {"kind": "html", "html": Markup(_text(data["text/html"]))}
    if "text/plain" in data:
        return {"kind": "text", "html": Markup(f"<pre>{escape(_text(data['text/plain']))}</pre>")}
    return None


def render(path):
    """Returns (cells, executed). executed is False when the notebook has no saved outputs yet."""
    if not os.path.exists(path):
        return [], False
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    cells, executed = [], False
    for c in nb.get("cells", []):
        src = _text(c.get("source"))
        if c.get("cell_type") == "markdown":
            cells.append({"type": "markdown", "html": _markdown(src)})
        elif c.get("cell_type") == "code":
            outs = [x for x in (_output(o) for o in c.get("outputs", [])) if x]
            executed = executed or bool(outs)
            cells.append({"type": "code", "source": src, "n": c.get("execution_count"), "outputs": outs})
    return cells, executed
