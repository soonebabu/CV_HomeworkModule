"""Stage 2 - Narrative Planner: story prompt -> structured scene plan.

The plan is plain JSON so it can come from an LLM, from the rule-based fallback below, or be pasted/edited by
hand on the web page. Every source goes through normalize_plan(), so the rest of the pipeline sees one shape:

{
  "title": "...",
  "entities": [{"id": "mia", "name": "Mia", "type": "character"|"object", "kind": "girl",
                "attributes": {"hair_color": "black", "clothing": "jacket", "clothing_color": "red"}}],
  "scenes": [{"index": 1, "text": "<the sentence(s) this scene comes from>",
              "location": "park", "weather": "clear"|"rain"|"snow"|null, "time_of_day": "day"|"sunset"|"night"|null,
              "present": ["mia", "max"],                       # entities the scene text refers to
              "actions": [{"actor": "max", "verb": "pick_up", "target": "ball"}],
              "relations": [{"subject": "ball", "relation": "on", "object": "table"}],
              "attribute_changes": [{"entity": "mia", "attribute": "clothing_color", "value": "blue"}],
              "exits": []}]
}
"""
import json
import re

from . import llm

MAX_SCENES = 8

CHARACTER_KINDS = {
    "girl": "f", "woman": "f", "lady": "f", "grandma": "f", "mother": "f", "mom": "f", "princess": "f",
    "boy": "m", "man": "m", "grandpa": "m", "father": "m", "dad": "m", "king": "m", "chef": "m",
    "child": None, "kid": None, "person": None, "student": None, "farmer": None, "astronaut": None,
    "robot": None, "dog": None, "puppy": None, "cat": None, "kitten": None, "fox": None, "rabbit": None,
    "bear": None, "horse": None,
}
HUMAN_KINDS = {"girl", "woman", "lady", "grandma", "mother", "mom", "princess", "boy", "man", "grandpa",
               "father", "dad", "king", "chef", "child", "kid", "person", "student", "farmer", "astronaut"}
ANIMAL_KINDS = {"dog", "puppy", "cat", "kitten", "fox", "rabbit", "bear", "horse"}
OBJECT_KINDS = {
    "ball", "bench", "umbrella", "table", "cup", "mug", "book", "lamp", "chair", "box", "gift", "car",
    "bicycle", "bike", "tree", "kite", "bag", "backpack", "guitar", "vase", "hat", "plant", "sofa", "bed",
    "clock", "lantern", "basket", "flower", "apple", "phone", "laptop", "bowl", "bottle", "toy", "cake",
    "suitcase", "map", "key", "candle", "teapot", "pot", "balloon",
}
ALIASES = {"bike": "bicycle", "mug": "cup", "backpack": "bag", "puppy": "dog", "kitten": "cat", "gift": "box"}
LOCATIONS = {
    "park": "park", "garden": "park", "meadow": "park", "playground": "park", "kitchen": "kitchen",
    "beach": "beach", "shore": "beach", "street": "street", "city": "street", "road": "street", "town": "street",
    "forest": "forest", "woods": "forest", "jungle": "forest", "room": "room", "bedroom": "room",
    "house": "room", "home": "room", "apartment": "room", "library": "room", "office": "office",
    "classroom": "office", "cafe": "cafe", "café": "cafe", "restaurant": "cafe", "bakery": "cafe",
}
KNOWN_LOCATIONS = ("park", "kitchen", "beach", "street", "forest", "room", "office", "cafe")
COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray",
          "grey", "golden", "silver", "blonde"}
CLOTHING = {"jacket", "coat", "shirt", "dress", "sweater", "hoodie", "scarf", "t-shirt", "uniform", "raincoat",
            "suit", "jumper", "cape", "overalls"}
ARTICLES = {"a", "an", "the", "her", "his", "their", "its", "my", "our", "some", "this", "that"}
SIZE_WORDS = {"big": "large", "large": "large", "huge": "large", "small": "small", "little": "small",
              "tiny": "small", "tall": "tall", "short": "short", "old": "old", "young": "young"}

VERBS = {
    "walk": "walk", "walks": "walk", "walked": "walk", "walking": "walk", "stroll": "walk", "strolls": "walk",
    "run": "run", "runs": "run", "ran": "run", "running": "run", "chase": "run", "chases": "run",
    "sit": "sit", "sits": "sit", "sat": "sit", "sitting": "sit",
    "stand": "stand", "stands": "stand", "stood": "stand",
    "jump": "jump", "jumps": "jump", "jumped": "jump", "hop": "jump", "hops": "jump",
    "dance": "dance", "dances": "dance", "wave": "wave", "waves": "wave",
    "pick": "pick_up", "picks": "pick_up", "picked": "pick_up", "grab": "pick_up", "grabs": "pick_up",
    "take": "pick_up", "takes": "pick_up", "took": "pick_up", "fetch": "pick_up", "fetches": "pick_up",
    "carry": "hold", "carries": "hold", "hold": "hold", "holds": "hold", "holding": "hold",
    "put": "put", "puts": "put", "place": "put", "places": "put", "set": "put", "sets": "put",
    "drop": "put", "drops": "put", "leave": "leave", "leaves": "leave", "left": "leave",
    "open": "open", "opens": "open", "opened": "open", "close": "close", "closes": "close", "closed": "close",
    "read": "read", "reads": "read", "eat": "eat", "eats": "eat", "drink": "drink", "drinks": "drink",
    "play": "play", "plays": "play", "sleep": "sleep", "sleeps": "sleep", "look": "look", "looks": "look",
    "go": "go", "goes": "go", "went": "go", "return": "go", "returns": "go", "enter": "go", "enters": "go",
    "arrive": "go", "arrives": "go", "head": "go", "heads": "go", "come": "go", "comes": "go",
    "wear": "wear", "wears": "wear", "change": "wear", "changes": "wear",
}
RELATION_PHRASES = [  # longest first so "in front of" wins over "of"
    (("to", "the", "left", "of"), "left_of"), (("to", "the", "right", "of"), "right_of"),
    (("in", "front", "of"), "in_front_of"), (("on", "top", "of"), "on"), (("next", "to"), "next_to"),
    (("left", "of"), "left_of"), (("right", "of"), "right_of"), (("beside",), "next_to"), (("near",), "near"),
    (("under",), "under"), (("beneath",), "under"), (("behind",), "behind"), (("on",), "on"), (("onto",), "on"),
]
PHRASAL_ON = {"put", "puts", "turn", "turns", "turned", "try", "tries", "switch", "switches", "get", "gets",
              "keep", "keeps", "carry", "carries", "goes", "go", "move", "moves"}
RELATIONS = {"on", "next_to", "near", "under", "behind", "in_front_of", "left_of", "right_of"}
ACTIONS = set(VERBS.values())
STOP_CAPS = {"the", "a", "an", "she", "he", "they", "it", "her", "his", "then", "later", "suddenly", "finally",
             "after", "when", "while", "at", "in", "on", "next", "meanwhile", "soon", "now", "there", "one",
             "back", "inside", "outside", "together", "and", "but", "so", "as", "once", "that", "this"}

SYSTEM_PROMPT = """You are the narrative planner of a long-form video generation system.
Turn the user's story into a JSON scene plan. One scene per story beat (at most %d scenes).
Return ONLY a JSON object with this shape:
{"title": str,
 "entities": [{"id": lowercase_id, "name": str, "type": "character" or "object", "kind": one word,
               "attributes": {...}}],
 "scenes": [{"index": 1-based int, "text": the story sentence(s) for this scene,
             "location": one of %s,
             "weather": "clear" | "rain" | "snow" | null, "time_of_day": "day" | "sunset" | "night" | null,
             "present": [entity ids the scene refers to, including pronouns resolved],
             "actions": [{"actor": id, "verb": one of %s, "target": id or null}],
             "relations": [{"subject": id, "relation": one of %s, "object": id}],
             "attribute_changes": [{"entity": id, "attribute": str, "value": str}],
             "exits": [ids that leave the scene]}]}
Rules:
- Character kinds should be one of: %s. Object kinds should be one of: %s (pick the closest).
- Character attributes: hair_color, clothing, clothing_color, size. Animal attributes: color, size.
  Object attributes: color, size. Fill in a specific value for every attribute even if the story does not
  say it - the plan must pin down appearance once so it never drifts.
- Use attribute_changes only when the story explicitly changes something (e.g. changes into a blue coat).
- Use the verb "go" with the new location for moving between places. Carried objects use pick_up/hold/put."""


def plan(story, backend="auto"):
    """Returns (plan, info). info says which planner ran and why it might have fallen back."""
    story = (story or "").strip()
    if not story:
        raise ValueError("Write a story first.")
    want_llm = backend in ("auto", "llm") and llm.provider() is not None
    if want_llm:
        try:
            raw = llm.complete_json(_system_prompt(), f"Story:\n{story}")
            p = normalize_plan(raw, story)
            if p["scenes"]:
                p["source"] = llm.describe()
                return p, {"planner": llm.describe(), "fallback_reason": None}
            reason = "the LLM returned no scenes"
        except llm.LLMError as e:
            reason = str(e)
    else:
        reason = None if backend == "rule" else "no LLM key configured"
    p = normalize_plan(rule_plan(story), story)
    p["source"] = "rule-based planner"
    return p, {"planner": "rule-based planner (offline)", "fallback_reason": reason}


def _system_prompt():
    return SYSTEM_PROMPT % (MAX_SCENES, list(KNOWN_LOCATIONS), sorted(ACTIONS), sorted(RELATIONS),
                            ", ".join(sorted(CHARACTER_KINDS)), ", ".join(sorted(OBJECT_KINDS)))


# ---------------------------------------------------------------------------------------------------------
# rule-based fallback: a small, deterministic parser for simple English stories

def _sentences(story):
    parts = re.split(r"(?<=[.!?])\s+|\n+", story.strip())
    return [p.strip() for p in parts if len(p.strip().split()) >= 2][:MAX_SCENES]


def _tokens(sentence):
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*|[.,;!?]", sentence)


def _kind_of(word):
    w = word.lower()
    if w in CHARACTER_KINDS or w in OBJECT_KINDS:
        return ALIASES.get(w, w)
    if w.endswith("s") and (w[:-1] in OBJECT_KINDS or w[:-1] in CHARACTER_KINDS):
        return ALIASES.get(w[:-1], w[:-1])
    return None


def _gender(kind):
    return CHARACTER_KINDS.get(kind)


def _is_name(tok, i):
    return tok[:1].isupper() and tok.lower() not in STOP_CAPS and _kind_of(tok) is None \
        and tok.lower() not in LOCATIONS and tok.lower() not in VERBS and tok.lower() not in COLORS


class _Story:
    def __init__(self):
        self.entities = {}     # id -> entity dict
        self.by_name = {}      # lowercase name -> id
        self.order = []

    def add(self, kind, name=None):
        is_char = kind in CHARACTER_KINDS
        if name:
            eid = re.sub(r"[^a-z0-9]", "", name.lower()) or kind
        else:
            # one entity per unnamed kind - "the ball" later in the story is the same ball
            for e in self.entities.values():
                if e["kind"] == kind and (not e["name"] or e["name"].lower() == kind):
                    return e["id"]
            eid = kind
        base, n = eid, 2
        while eid in self.entities and self.entities[eid]["kind"] != kind:
            eid, n = f"{base}{n}", n + 1
        if eid not in self.entities:
            self.entities[eid] = {"id": eid, "name": name or kind,
                                  "type": "character" if is_char else "object", "kind": kind,
                                  "attributes": {}}
            self.order.append(eid)
            if name:
                self.by_name[name.lower()] = eid
        return eid

    def find_kind(self, kind):
        ids = [e["id"] for e in self.entities.values() if e["kind"] == kind]
        return ids[0] if ids else None


def _scan_appearance(toks, start, stop):
    """Collects 'red jacket', 'black hair', 'tall' style descriptors in toks[start:stop]."""
    attrs = {}
    for j in range(start, stop):
        w = toks[j].lower()
        nxt = toks[j + 1].lower() if j + 1 < len(toks) else ""
        if w in COLORS and nxt in CLOTHING:
            attrs["clothing"], attrs["clothing_color"] = nxt, _color(w)
        elif w in COLORS and nxt == "hair":
            attrs["hair_color"] = _color(w)
    return attrs


def _color(w):
    return {"grey": "gray", "golden": "yellow", "blonde": "yellow", "silver": "gray"}.get(w, w)


def _clause_end(toks, i):
    """End of the descriptive clause after a noun: next verb, sentence break or new character."""
    j = i + 1
    while j < len(toks):
        w = toks[j].lower()
        if w in ".;!?" or (w in VERBS and w not in ("set", "left", "close", "head")):
            break
        if j > i + 1 and _kind_of(toks[j]) in CHARACTER_KINDS:
            break
        j += 1
    return j


def rule_plan(story):
    st = _Story()
    sentences = _sentences(story)
    intro_sentence = {}

    # pass 1: who and what exists, and what they look like when first introduced
    for si, sent in enumerate(sentences):
        toks = _tokens(sent)
        for i, tok in enumerate(toks):
            kind = _kind_of(tok)
            if kind is None:
                continue
            name = None
            if kind in CHARACTER_KINDS:
                # "dog Max", "dog named Max", "Mia, a girl", "Mia the girl"
                if i + 1 < len(toks) and _is_name(toks[i + 1], i + 1):
                    name = toks[i + 1]
                elif i + 2 < len(toks) and toks[i + 1].lower() in ("named", "called") and toks[i + 2][:1].isupper():
                    name = toks[i + 2]
                else:
                    j = i - 1
                    while j >= 0 and (toks[j].lower() in ARTICLES or toks[j].lower() in COLORS
                                      or toks[j].lower() in SIZE_WORDS or toks[j] == ","):
                        j -= 1
                    if j >= 0 and j < i - 1 and _is_name(toks[j], j):
                        name = toks[j]
            if name and name.lower() in st.by_name:
                eid = st.by_name[name.lower()]
            elif not name and kind in CHARACTER_KINDS and st.find_kind(kind):
                eid = st.find_kind(kind)
            else:
                eid = st.add(kind, name)
            ent = st.entities[eid]
            if eid in intro_sentence and intro_sentence[eid] != si:
                continue
            intro_sentence.setdefault(eid, si)
            attrs = ent["attributes"]
            for j in range(max(0, i - 3), i):
                w = toks[j].lower()
                if w in COLORS:
                    attrs.setdefault("color", _color(w))
                if w in SIZE_WORDS:
                    attrs.setdefault("size", SIZE_WORDS[w])
            if kind in HUMAN_KINDS:
                attrs.pop("color", None)
                for k, v in _scan_appearance(toks, i + 1, _clause_end(toks, i)).items():
                    attrs.setdefault(k, v)

    scenes = []
    last_char = {"f": None, "m": None, None: None}
    last_obj = None
    seen_chars, gone = [], set()
    for si, sent in enumerate(sentences):
        toks = _tokens(sent)
        # names are not weather: a cat called Snow must not make it snow
        low = [t.lower() for t in toks]
        wlow = [w for t, w in zip(toks, low) if not (t[:1].isupper() and w in st.by_name)]
        weatherish = any(w in ("rain", "rains", "raining", "snow", "snows", "snowing") for w in wlow)

        mentions = []  # (token index, entity id)
        for i, tok in enumerate(toks):
            w = low[i]
            eid = None
            if w in st.by_name:
                eid = st.by_name[w]
            elif _kind_of(tok) is not None:
                kind = _kind_of(tok)
                named_next = i + 1 < len(toks) and low[i + 1] in st.by_name
                named_prev = i > 1 and low[i - 2] in st.by_name and toks[i - 1] in (",", "the", "a")
                if named_next or named_prev:
                    continue  # "dog Max" - the name token carries the mention
                eid = st.find_kind(kind)
            elif w in ("she", "her", "herself"):
                eid = last_char["f"] or last_char[None]
            elif w in ("he", "him", "his", "himself"):
                eid = last_char["m"] or last_char[None]
            elif w in ("they", "them", "their", "both"):
                for cid in seen_chars:
                    if cid not in gone:
                        mentions.append((i, cid))
                continue
            elif w == "it" and not weatherish:
                eid = last_obj
            if eid is None:
                continue
            mentions.append((i, eid))
            ent = st.entities[eid]
            if ent["type"] == "character":
                last_char[_gender(ent["kind"])] = eid
                last_char[None] = eid
            else:
                last_obj = eid

        present = []
        for _, eid in mentions:
            if eid not in present:
                present.append(eid)

        # the last place named wins: "from the park to the kitchen", "home to the kitchen"
        location = None
        for i, w in enumerate(low):
            if w in LOCATIONS:
                location = LOCATIONS[w]

        weather = None
        if any(w in ("rain", "rains", "raining", "storm") for w in wlow):
            weather = "clear" if any(w in ("stops", "stopped", "ends") for w in wlow) else "rain"
        elif any(w in ("snow", "snows", "snowing") for w in wlow):
            weather = "snow"
        elif any(w in ("sunny", "sunshine", "sun") for w in wlow):
            weather = "clear"
        time_of_day = None
        if any(w in ("night", "midnight", "dark", "evening") for w in low):
            time_of_day = "night"
        elif any(w in ("sunset", "dusk") for w in low):
            time_of_day = "sunset"
        elif any(w in ("morning", "noon", "daytime") for w in low):
            time_of_day = "day"

        def char_before(pos):
            for i, eid in reversed(mentions):
                if i < pos and st.entities[eid]["type"] == "character":
                    return eid
            for i, eid in mentions:
                if st.entities[eid]["type"] == "character":
                    return eid
            return None

        def mention_after(pos, limit=6, prefer_object=True):
            cands = [(i, eid) for i, eid in mentions if pos < i <= pos + limit]
            if prefer_object:
                objs = [c for c in cands if st.entities[c[1]]["type"] == "object"]
                if objs:
                    return objs[0][1]
            return cands[0][1] if cands else None

        actions, exits, verb_pos = [], [], []
        for i, w in enumerate(low):
            verb = VERBS.get(w)
            if verb is None or (w in ("left", "right") and i + 1 < len(low) and low[i + 1] == "of"):
                continue
            if w in ("set", "close", "head", "left") and i > 0 and low[i - 1] in ARTICLES:
                continue  # nouns/adjectives that happen to be verb forms
            actor = char_before(i)
            if verb_pos:
                # "Ana picks up Snow and walks": a verb joined by "and" keeps the previous subject, unless a new
                # character (one that wasn't just the previous verb's target) shows up in between
                pi, pactor, _ = verb_pos[-1]
                ptarget = actions[-1]["target"] if actions else None
                between = [eid for p, eid in mentions if pi < p < i and eid not in (ptarget, pactor)]
                if "and" in low[pi:i] and not between:
                    actor = pactor
            if actor is None:
                continue
            target = mention_after(i)
            if verb == "leave":
                if target and st.entities[target]["type"] == "object":
                    verb = "put"
                else:
                    exits.append(actor)
                    continue
            if verb == "wear":
                target = None
            verb_pos.append((i, actor, verb))
            actions.append({"actor": actor, "verb": verb, "target": target if target != actor else None})

        relations, rel_objects = [], set()
        i = 0
        while i < len(low):
            matched = None
            for phrase, rel in RELATION_PHRASES:
                if tuple(low[i:i + len(phrase)]) == phrase:
                    matched = (phrase, rel)
                    break
            if not matched:
                i += 1
                continue
            phrase, rel = matched
            # "puts on a hat", "turns on the lamp" - phrasal verbs, not spatial relations
            if phrase == ("on",) and i > 0 and low[i - 1] in PHRASAL_ON:
                i += 1
                continue
            obj = mention_after(i + len(phrase) - 1, limit=4, prefer_object=False)
            if obj is None:
                i += len(phrase)
                continue
            prev_verb = max((v for v in verb_pos if v[0] < i), default=None, key=lambda v: v[0])
            start = prev_verb[0] if prev_verb else -1
            cands = [eid for p, eid in mentions if start < p < i and eid not in rel_objects and eid != obj]
            if cands:
                subj = cands[-1]
            elif prev_verb:
                subj = prev_verb[1]
            else:
                before = [eid for p, eid in mentions if p < i and eid not in rel_objects and eid != obj]
                subj = before[-1] if before else None
            if subj and subj != obj:
                relations.append({"subject": subj, "relation": rel, "object": obj})
                rel_objects.add(obj)
            i += len(phrase)

        changes = []
        for i in range(len(toks) - 1):
            if low[i] in COLORS and (low[i + 1] in CLOTHING or low[i + 1] == "hair"):
                owner = char_before(i)
                if owner is None or intro_sentence.get(owner) == si:
                    continue
                attr = "hair_color" if low[i + 1] == "hair" else "clothing_color"
                changes.append({"entity": owner, "attribute": attr, "value": _color(low[i])})
                if attr == "clothing_color":
                    changes.append({"entity": owner, "attribute": "clothing", "value": low[i + 1]})

        scenes.append({"index": si + 1, "text": sent, "location": location, "weather": weather,
                       "time_of_day": time_of_day, "present": present, "actions": actions,
                       "relations": relations, "attribute_changes": changes, "exits": exits})
        for e in present:
            if st.entities[e]["type"] == "character" and e not in seen_chars:
                seen_chars.append(e)
        gone.update(exits)

    title = " ".join(story.split()[:7]).rstrip(".,;") + ("..." if len(story.split()) > 7 else "")
    return {"title": title, "entities": [st.entities[e] for e in st.order], "scenes": scenes}


# ---------------------------------------------------------------------------------------------------------

def normalize_plan(raw, story=""):
    """Validates a plan from any source and fixes what can be fixed (unknown ids, locations, verbs)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("The plan must be a JSON object.")

    entities, seen = [], set()
    for e in raw.get("entities", []) or []:
        if not isinstance(e, dict):
            continue
        eid = re.sub(r"[^a-z0-9_]", "", str(e.get("id") or e.get("name") or "").lower())
        if not eid or eid in seen:
            continue
        seen.add(eid)
        kind = str(e.get("kind") or "").lower().strip() or eid
        kind = _kind_of(kind) or kind
        etype = e.get("type")
        if etype not in ("character", "object"):
            etype = "character" if kind in CHARACTER_KINDS else "object"
        attrs = {str(k): str(v).lower() for k, v in (e.get("attributes") or {}).items() if v not in (None, "")}
        for k in ("color", "clothing_color", "hair_color"):
            if k in attrs:
                attrs[k] = _color(attrs[k])
        entities.append({"id": eid, "name": str(e.get("name") or eid), "type": etype, "kind": kind,
                         "attributes": attrs})
    ids = {e["id"] for e in entities}

    def ref(x):
        x = re.sub(r"[^a-z0-9_]", "", str(x or "").lower())
        return x if x in ids else None

    scenes = []
    for k, s in enumerate((raw.get("scenes") or [])[:MAX_SCENES]):
        if not isinstance(s, dict):
            continue
        loc = str(s.get("location") or "").lower().strip()
        loc = loc if loc in KNOWN_LOCATIONS else LOCATIONS.get(loc) or next(
            (LOCATIONS[w] for w in re.findall(r"[a-z]+", loc) if w in LOCATIONS), None)
        weather = s.get("weather") if s.get("weather") in ("clear", "rain", "snow") else None
        tod = s.get("time_of_day") if s.get("time_of_day") in ("day", "sunset", "night") else None
        present = []
        for x in s.get("present") or []:
            r = ref(x)
            if r and r not in present:
                present.append(r)
        actions = []
        for a in s.get("actions") or []:
            if not isinstance(a, dict):
                continue
            actor, verb = ref(a.get("actor")), str(a.get("verb") or "").lower().replace(" ", "_")
            verb = verb if verb in ACTIONS else VERBS.get(verb.split("_")[0], None)
            if actor and verb:
                actions.append({"actor": actor, "verb": verb, "target": ref(a.get("target"))})
                for r in (actor, ref(a.get("target"))):
                    if r and r not in present:
                        present.append(r)
        relations = []
        for r in s.get("relations") or []:
            if not isinstance(r, dict):
                continue
            subj, obj = ref(r.get("subject")), ref(r.get("object"))
            rel = str(r.get("relation") or "").lower().replace(" ", "_")
            rel = {"beside": "next_to", "on_top_of": "on", "nearby": "near", "below": "under"}.get(rel, rel)
            if subj and obj and subj != obj and rel in RELATIONS:
                relations.append({"subject": subj, "relation": rel, "object": obj})
                for x in (subj, obj):
                    if x not in present:
                        present.append(x)
        changes = []
        for c in s.get("attribute_changes") or []:
            if isinstance(c, dict) and ref(c.get("entity")) and c.get("attribute") and c.get("value"):
                changes.append({"entity": ref(c.get("entity")), "attribute": str(c["attribute"]),
                                "value": _color(str(c["value"]).lower())})
        exits = [r for r in (ref(x) for x in s.get("exits") or []) if r]
        scenes.append({"index": k + 1, "text": str(s.get("text") or "").strip(), "location": loc,
                       "weather": weather, "time_of_day": tod, "present": present, "actions": actions,
                       "relations": relations, "attribute_changes": changes, "exits": exits})

    # the first scene needs somewhere to happen
    if scenes and not scenes[0]["location"]:
        scenes[0]["location"] = "park"
    return {"title": str(raw.get("title") or " ".join(story.split()[:7]) or "Untitled story"),
            "entities": entities, "scenes": scenes, "source": raw.get("source", "")}
