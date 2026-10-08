"""The LLM / Video-LLM behind the planner (stage 2) and the evaluation judge.

Two providers are supported and picked from the environment, so the app keeps working with no key at all:
  GEMINI_API_KEY (or GOOGLE_API_KEY)  -> Gemini over REST (model: GEMINI_MODEL, default gemini-2.5-flash)
  ANTHROPIC_API_KEY                   -> Claude through the official `anthropic` SDK (model: CLAUDE_MODEL)
CHALLENGE_LLM=gemini|claude|none forces one. With neither available, callers fall back to the rule-based
planner and the judge is skipped.
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request

import cv2

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_S = 120


class LLMError(RuntimeError):
    pass


def provider():
    forced = os.environ.get("CHALLENGE_LLM", "").strip().lower()
    if forced in ("gemini", "claude", "none"):
        return None if forced == "none" else forced
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def describe():
    p = provider()
    if p == "gemini":
        return f"Gemini ({os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')})"
    if p == "claude":
        return f"Claude ({os.environ.get('CLAUDE_MODEL', 'claude-opus-5')})"
    return None


def encode_frame(frame, max_dim=384, quality=80):
    """BGR frame -> base64 JPEG, small enough that a few dozen fit in one request."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise LLMError("could not encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def complete_json(system, text, frames=(), frame_labels=None):
    """Sends text (+ optional BGR frames) and returns the parsed JSON object the model answers with."""
    p = provider()
    if p is None:
        raise LLMError("no LLM configured - set GEMINI_API_KEY or ANTHROPIC_API_KEY")
    images = [encode_frame(f) for f in frames]
    labels = frame_labels or [f"frame {i + 1}" for i in range(len(images))]
    raw = _gemini(system, text, images, labels) if p == "gemini" else _claude(system, text, images, labels)
    return parse_json(raw)


def parse_json(raw):
    """Pulls the first JSON object out of a reply, tolerating ```json fences or a stray sentence."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fenced:
        raw = fenced.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("the model did not return JSON")
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"the model returned invalid JSON ({e})") from e


def _gemini(system, text, images, labels):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    parts = []
    for label, data in zip(labels, images):
        parts.append({"text": label})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": data}})
    parts.append({"text": text})
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }
    req = urllib.request.Request(
        GEMINI_URL.format(model=model), data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"Gemini returned HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach Gemini: {e.reason}") from e
    try:
        return "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        raise LLMError(f"unexpected Gemini response: {str(out)[:300]}") from e


def _claude(system, text, images, labels):
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("pip install anthropic to use Claude") from e
    content = []
    for label, data in zip(labels, images):
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    content.append({"type": "text", "text": text})

    client = anthropic.Anthropic(timeout=TIMEOUT_S)
    try:
        # server-side fallbacks keep the request alive if the primary model declines it
        response = client.beta.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-opus-5"),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Claude rejected the API key") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Claude rate limit hit - try again in a minute") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude returned HTTP {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("could not reach the Claude API") from e
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined this request")
    return "".join(b.text for b in response.content if b.type == "text")
