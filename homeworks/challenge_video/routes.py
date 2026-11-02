"""Challenge 1: shows the saved outputs in challenge1_results/. Nothing is generated here."""
import json
import os

from flask import Blueprint, render_template, send_from_directory

bp = Blueprint("challenge", __name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "challenge1_results")
SCENES = range(1, 7)
SCORE_LABELS = [
    ("character", "Character consistency"), ("object", "Object persistence"), ("spatial", "Spatial consistency"),
    ("temporal", "Temporal consistency"), ("prompt", "Prompt adherence"), ("narrative", "Narrative flow"),
]
JSON_FILES = ["results.json", "plan.json", "prompts.json", "offline_report.json", "offline_plan.json",
              "offline_prompts.json"]


def _load(name):
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _score_rows(baseline, orchestrated):
    rows = [(label, baseline["scores"][key], orchestrated["scores"][key]) for key, label in SCORE_LABELS]
    return rows + [("Overall", baseline["overall"], orchestrated["overall"])]


@bp.route("/files/<path:name>")
def result_file(name):
    return send_from_directory(RESULTS_DIR, name)


@bp.route("/")
def overview():
    results = _load("results.json")
    scores, free = results["scores"], results["model_free"]
    model_free = [
        ("Temporal", free["baseline"]["scores"]["temporal"], free["orchestrated"]["scores"]["temporal"]),
        ("Cut", free["baseline"]["scores"]["cut"], free["orchestrated"]["scores"]["cut"]),
        ("Appearance", free["baseline"]["scores"]["appearance"], free["orchestrated"]["scores"]["appearance"]),
        ("Structure", free["baseline"]["scores"]["structure"], free["orchestrated"]["scores"]["structure"]),
        ("Warp error", free["baseline"]["warp_error"], free["orchestrated"]["warp_error"]),
        ("Warp error at cuts", free["baseline"]["cut_error"], free["orchestrated"]["cut_error"]),
    ]
    return render_template("challenge/overview.html", active_page="overview", plan=results["plan"],
                           score_rows=_score_rows(scores["baseline"], scores["orchestrated"]), model_free=model_free,
                           scores=scores, judge=results["judge"])


@bp.route("/scenes")
def scenes():
    prompts = _load("prompts.json")
    scene_list = [{"index": i, "text": scene["text"], "baseline": prompts["baseline"][i - 1],
                   "orchestrated": prompts["orchestrated"][i - 1]}
                  for i, scene in zip(SCENES, _load("plan.json")["scenes"])]
    return render_template("challenge/scenes.html", active_page="scenes", scenes=scene_list)


@bp.route("/offline")
def offline():
    report = _load("offline_report.json")
    evaluation = report["evaluation"]
    numbers = [("Objects shown", "objects_shown"), ("Objects expected", "objects_expected"),
               ("Entity continuity", "entity_continuity"), ("Background stability", "background_stability"),
               ("Warp error in clip", "warp_error_in_clip"), ("Warp error at cuts", "warp_error_at_cuts")]
    details = [(label, evaluation["baseline"]["details"][key], evaluation["orchestrated"]["details"][key])
               for label, key in numbers]
    prompts = _load("offline_prompts.json")
    return render_template("challenge/offline.html", active_page="offline", details=details,
                           score_rows=_score_rows(evaluation["baseline"], evaluation["orchestrated"]),
                           effort=report["prompt_effort"],
                           prompts=list(zip(prompts["baseline"], prompts["orchestrated"])))


@bp.route("/data")
def data():
    files = [(name, json.dumps(_load(name), indent=2)) for name in JSON_FILES]
    return render_template("challenge/data.html", active_page="data", files=files)
