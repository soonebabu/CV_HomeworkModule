"""Video-LLM judge: a multimodal LLM watches sampled frames and scores the same six axes (1-5).

Used on both arms of a prototype run and on uploaded real clips. It only runs when an LLM key is configured
(see llm.py); otherwise callers just show the pixel metrics.
"""
import json

from . import llm

AXES = ["character_consistency", "object_persistence", "spatial_consistency", "temporal_consistency",
        "prompt_adherence", "narrative_flow"]

SYSTEM = """You are a strict evaluator of long-form generated video. You see frames sampled from consecutive
scenes of one video (labelled with scene and position) and the story it should tell. Score each axis from 1
(very poor) to 5 (perfect):
- character_consistency: same faces, hair, clothing, body for each character across scenes (unless the story
  says it changes)
- object_persistence: objects stay present and keep their look until the story removes them
- spatial_consistency: the same place keeps its layout; things don't jump around between cuts
- temporal_consistency: motion within and across cuts is smooth and plausible, no flicker
- prompt_adherence: each scene shows what its own text asks for
- narrative_flow: the scenes together follow the story, carrying over earlier events (carried objects,
  locations, weather)
Return ONLY JSON: {"scores": {axis: int}, "issues": [short strings naming the scene], "summary": str}"""


def available():
    return llm.provider() is not None


def _sample(frames_by_scene, per_scene=3, max_total=24):
    per_scene = max(1, min(per_scene, max_total // max(1, len(frames_by_scene))))
    frames, labels = [], []
    for k, frames_k in enumerate(frames_by_scene):
        if not frames_k:
            continue
        idx = sorted({int(round(i * (len(frames_k) - 1) / max(1, per_scene - 1))) for i in range(per_scene)})
        for j, i in enumerate(idx):
            frames.append(frames_k[i])
            pos = "start" if i == 0 else "end" if i == len(frames_k) - 1 else "middle"
            labels.append(f"Scene {k + 1}, {pos}:")
    return frames, labels


def _story_text(plan, snapshots):
    if plan is None:
        return "No plan was supplied; judge consistency between consecutive clips."
    lines = [f"Title: {plan.get('title', '')}"]
    for k, sc in enumerate(plan["scenes"]):
        line = f"Scene {k + 1}: {sc['text']}"
        if snapshots:
            snap = snapshots[k]
            who = ", ".join(f"{e['name']} ({', '.join(f'{a}={v}' for a, v in e['attributes'].items() if a != 'size')})"
                            for e in snap["entities"].values())
            line += f"\n  expected world: location={snap['env']['location']}, weather={snap['env']['weather']}; {who}"
        lines.append(line)
    return "\n".join(lines)


def judge_frames(frames_by_scene, plan=None, snapshots=None):
    frames, labels = _sample(frames_by_scene)
    out = llm.complete_json(SYSTEM, "Story and expected state:\n" + _story_text(plan, snapshots) +
                            "\n\nScore the frames above.", frames, labels)
    scores = {}
    for a in AXES:
        try:
            scores[a] = max(1, min(5, int(round(float(out.get("scores", {}).get(a))))))
        except (TypeError, ValueError):
            scores[a] = None
    return {"scores": scores, "issues": [str(i) for i in (out.get("issues") or [])][:10],
            "summary": str(out.get("summary") or "")}


def judge_runs(frames_by_arm, plan=None, snapshots=None):
    """{arm: [frames per scene]} -> {"provider", "arms": {arm: result}, "error"}"""
    if not available():
        return {"provider": None, "arms": {}, "error": "No LLM key configured (set GEMINI_API_KEY or "
                                                       "ANTHROPIC_API_KEY to enable the Video-LLM judge)."}
    res = {"provider": llm.describe(), "arms": {}, "error": None}
    for arm, frames in frames_by_arm.items():
        try:
            res["arms"][arm] = judge_frames(frames, plan, snapshots)
        except llm.LLMError as e:
            res["error"] = f"{arm}: {e}"
    return res


def as_rows(judged):
    """Rows for a template table: (axis label, {arm: score})."""
    if not judged or not judged.get("arms"):
        return []
    return [(a.replace("_", " ").capitalize(), {arm: r["scores"].get(a) for arm, r in judged["arms"].items()})
            for a in AXES]


def dumps(judged):
    return json.dumps(judged, indent=2)
