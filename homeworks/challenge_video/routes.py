import json
import os
import re
import uuid
from html import escape

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from .. import samples
from . import judge, llm, metrics, notebook, pipeline, planner, real_eval

bp = Blueprint("challenge", __name__)

PRESETS = [
    ("Mia and Max (park -> kitchen)",
     "Mia, a girl with a red jacket and black hair, walks her brown dog Max in the park. She sits on a bench next "
     "to a yellow ball. Max picks up the ball. It starts to rain and Mia opens a blue umbrella. They go home to "
     "the kitchen. Mia puts the ball on the table and Max sleeps under the table."),
    ("Bolt the robot (kitchen, night)",
     "A robot named Bolt stands in the kitchen next to a red teapot. Leo, a boy with a green hoodie, enters the "
     "kitchen. Leo picks up the teapot and puts it on the table. At night Bolt reads a book next to the table. "
     "Leo changes into a blue sweater and sits on a chair next to Bolt."),
    ("Ana and Snow (street -> beach -> street)",
     "Ana, a woman with a yellow dress and brown hair, walks on the street next to a red bicycle. A white cat "
     "named Snow sits under a green tree. Ana picks up Snow and walks to the beach. At sunset Ana sits on a bench "
     "with Snow. Ana returns to the street and stands next to the bicycle."),
]


def _dirs():
    static_dir = current_app.static_folder
    upload_dir = os.path.join(static_dir, "challenge", "uploads")
    output_dir = os.path.join(static_dir, "challenge", "outputs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return upload_dir, output_dir


def _ctx(**kw):
    return dict(llm_name=llm.describe(), judge_available=judge.available(), presets=PRESETS,
                axes=metrics.AXES, real_axes=real_eval.REAL_AXES, **kw)


@bp.route("/")
def overview():
    return render_template("challenge/overview.html", active_page="overview", **_ctx())


@bp.route("/method")
def method():
    return render_template("challenge/method.html", active_page="method", **_ctx())


NOTEBOOK = "notebooks/challenge1_colab.ipynb"
COLAB_URL = "https://colab.research.google.com/github/soonebabu/CV_HomeworkModule/blob/main/" + NOTEBOOK


@bp.route("/notebook")
def notebook_view():
    # root_path is the folder app.py lives in, i.e. the repo root
    cells, executed = notebook.render(os.path.join(current_app.root_path, NOTEBOOK))
    return render_template("challenge/notebook.html", active_page="notebook", cells=cells, executed=executed,
                           colab_url=COLAB_URL, nb_path=NOTEBOOK, **_ctx())


@bp.route("/prototype", methods=["GET", "POST"])
def prototype_view():
    if request.method == "GET":
        return render_template("challenge/prototype.html", active_page="prototype", result=None,
                               form={"story": PRESETS[0][1], "frames": 24, "backend": "auto"}, **_ctx())

    _, output_dir = _dirs()
    story = request.form.get("story", "").strip()
    plan_json = request.form.get("plan_json", "").strip()
    backend = request.form.get("backend", "auto")
    use_judge = request.form.get("use_judge") == "on"
    try:
        frames = max(12, min(36, int(request.form.get("frames", 24))))
    except ValueError:
        frames = 24
    form = {"story": story, "frames": frames, "backend": backend, "plan_json": plan_json, "use_judge": use_judge}
    if not story and not plan_json:
        flash("Write a story (or paste a plan) first.")
        return render_template("challenge/prototype.html", active_page="prototype", result=None, form=form, **_ctx())

    run_id = uuid.uuid4().hex[:10]
    try:
        out = pipeline.run(story, output_dir, run_id, plan_json=plan_json, frames_per_scene=frames,
                           planner_backend=backend, use_judge=use_judge)
    except (ValueError, json.JSONDecodeError) as e:
        flash(f"Could not run the prototype: {e}")
        return render_template("challenge/prototype.html", active_page="prototype", result=None, form=form, **_ctx())

    urls = {k: url_for("static", filename=f"challenge/outputs/{v}") for k, v in out["files"].items()}
    ev = out["evals"]
    result = dict(out, urls=urls, judge_rows=judge.as_rows(out["judge"]))

    sample_token = samples.stage(
        "challenge", "prototype", "Prototype: baseline vs. Scene-Graph Orchestrator",
        images=[
            (os.path.join(output_dir, out["files"]["video_compare"]),
             "Same story, same generator. Left: every scene prompted on its own. Right: scenes conditioned on "
             "the persistent state graph.", True),
            (os.path.join(output_dir, out["files"]["storyboard"]),
             "Middle frame of every scene, baseline (top) vs. orchestrator (bottom).", True),
            (os.path.join(output_dir, out["files"]["chart"]), "Evaluation on the six axes (0 = worst, 1 = best).",
             True),
            (os.path.join(output_dir, out["files"]["graph"]),
             "Stage 3 - the persistent world graph after every scene, with what changed.", True),
            (os.path.join(output_dir, out["files"]["layout"]),
             "Stage 4a - layout the control adapter hands the generator (boxes persist per location; arrows = "
             "motion).", True),
        ],
        params=[("Story", (story[:160] + "...") if len(story) > 160 else story or "(pasted plan)"),
                ("Planner", out["plan_info"]["planner"]), ("Scenes", str(len(out["plan"]["scenes"]))),
                ("Frames per scene", f"{frames} @ {pipeline.FPS} fps")],
        metrics=[("Overall - orchestrator", f"{ev['orchestrated']['overall']:.2f}"),
                 ("Overall - baseline", f"{ev['baseline']['overall']:.2f}"),
                 ("Objects vanishing (baseline)", str(len(ev["baseline"]["details"]["vanish_events"]))),
                 ("Objects vanishing (orchestrator)", str(len(ev["orchestrated"]["details"]["vanish_events"])))],
        table_html=_axes_table(ev, out["judge"]),
        table_title="Scores per axis",
        note="Only the conditioning differs between the two videos; the generator is the same. The state graph "
             "keeps identities, objects and layout fixed, so drift and disappearing objects drop sharply while "
             "per-scene prompt adherence stays the same.",
    )
    return render_template("challenge/prototype.html", active_page="prototype", result=result, form=form,
                           sample_token=sample_token, **_ctx())


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def _axes_table(ev, judged):
    rows = []
    jrows = dict(judge.as_rows(judged)) if judged else {}
    for (key, label), jkey in zip(metrics.AXES, judge.AXES):
        o, b = ev["orchestrated"]["scores"][key], ev["baseline"]["scores"][key]
        delta = "-" if o is None or b is None else f"{o - b:+.2f}"
        j = jrows.get(jkey.replace("_", " ").capitalize(), {})
        jtxt = f"{j.get('orchestrated', '-')} / {j.get('baseline', '-')}" if j else ""
        rows.append(f"<tr><td>{escape(label)}</td><td>{_fmt(b)}</td><td>{_fmt(o)}</td><td>{delta}</td>"
                    + (f"<td>{escape(jtxt)}</td>" if jrows else "") + "</tr>")
    o, b = ev["orchestrated"]["overall"], ev["baseline"]["overall"]
    rows.append(f"<tr><td><strong>Overall</strong></td><td>{_fmt(b)}</td><td>{_fmt(o)}</td><td>{o - b:+.2f}</td>"
                + ("<td></td>" if jrows else "") + "</tr>")
    head = "<tr><th>Axis</th><th>Baseline</th><th>Orchestrator</th><th>Change</th>" + \
           ("<th>Video-LLM judge (orch / base, 1-5)</th>" if jrows else "") + "</tr>"
    return f'<table class="result-table">{head}{"".join(rows)}</table>'


@bp.route("/evaluate", methods=["GET", "POST"])
def evaluate_view():
    if request.method == "GET":
        return render_template("challenge/evaluate.html", active_page="evaluate", result=None, **_ctx())

    upload_dir, output_dir = _dirs()
    run_id = uuid.uuid4().hex[:10]
    paths = {}
    for arm in ("baseline", "orchestrated"):
        files = [f for f in request.files.getlist(arm) if f and f.filename]
        # scene order follows the file names (scene1.mp4, scene2.mp4, ...)
        files.sort(key=lambda f: _natural(f.filename))
        paths[arm] = []
        for i, f in enumerate(files[:planner.MAX_SCENES]):
            ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
            p = os.path.join(upload_dir, f"{run_id}_{arm}_{i + 1:02d}{ext}")
            f.save(p)
            paths[arm].append(p)
    if not paths["baseline"] and not paths["orchestrated"]:
        flash("Upload at least one set of clips.")
        return redirect(url_for("challenge.evaluate_view"))
    if any(0 < len(p) < 2 for p in paths.values()):
        flash("Each set needs at least two clips (two scenes), or nothing can be compared across scenes.")
        return redirect(url_for("challenge.evaluate_view"))

    plan = None
    plan_src = request.files.get("plan_file")
    plan_text = request.form.get("plan_json", "").strip()
    if plan_src and plan_src.filename:
        plan_text = plan_src.read().decode("utf-8", "replace")
    if plan_text:
        try:
            plan = planner.normalize_plan(plan_text)
        except (ValueError, json.JSONDecodeError) as e:
            flash(f"Ignored the plan - it is not valid JSON ({e}).")
    use_judge = request.form.get("use_judge") == "on"
    label = request.form.get("label", "").strip()[:80] or "uploaded clips"

    try:
        out = real_eval.run({k: v for k, v in paths.items() if v}, output_dir, run_id, plan, use_judge)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("challenge.evaluate_view"))

    urls = {k: url_for("static", filename=f"challenge/outputs/{v}") for k, v in out["files"].items()}
    result = dict(out, urls=urls, label=label, plan=plan, judge_rows=judge.as_rows(out["judge"]))
    res = out["results"]
    metric_rows = []
    for key, lbl in real_eval.REAL_AXES:
        for arm in out["arms"]:
            v = res[arm]["scores"][key]
            metric_rows.append((f"{lbl} - {real_eval.ARM_NAMES[arm].lower()}", _fmt(v)))
    sample_token = samples.stage(
        "challenge", "evaluate", f"Real clips: {label}",
        images=[(os.path.join(output_dir, out["files"]["sheet"]), "Middle frame of every uploaded clip.", True),
                (os.path.join(output_dir, out["files"]["chart"]), "Model-free consistency measurements.", True)],
        params=[("Clips", ", ".join(f"{real_eval.ARM_NAMES[a]}: {len(paths[a])}" for a in out["arms"])),
                ("Plan supplied", "yes" if plan else "no")],
        metrics=metric_rows[:8],
        note="Measured directly on the uploaded videos (optical-flow warping error, colour histograms, ORB "
             "feature matching), no ground truth needed.",
    )
    return render_template("challenge/evaluate.html", active_page="evaluate", result=result,
                           sample_token=sample_token, **_ctx())


def _natural(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]
