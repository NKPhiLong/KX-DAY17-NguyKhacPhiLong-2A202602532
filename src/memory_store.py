from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: ~4 characters per token, 0 for empty text."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------

PROFILE_TITLE = "# User Profile"
FACTS_HEADER = "## Facts"
_FACT_LINE = re.compile(r"^- (?P<key>[a-z_]+): (?P<value>.*)$")

# Keys whose new value replaces the old one (corrections overwrite) vs. keys that
# accumulate (preferences added across turns are merged, not replaced).
SINGLE_VALUE_KEYS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")
SET_VALUE_KEYS = ("interests", "response_style")
PROFILE_KEYS = SINGLE_VALUE_KEYS + SET_VALUE_KEYS
SET_SEPARATOR = ", "


def _slugify_user_id(user_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (user_id or "").strip()).strip("_")
    return slug or "anonymous"


def default_profile(user_id: str) -> str:
    return f"{PROFILE_TITLE}: {user_id}\n\n{FACTS_HEADER}\n"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`, one file per user at `<root>/<user>/User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        root = Path(self.root_dir).resolve()
        path = (root / _slugify_user_id(user_id) / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Refusing to build a profile path outside {root}: {user_id!r}")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text or search_text == replacement:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse `- key: value` lines from User.md."""

        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE.match(line.strip())
            if match:
                result[match.group("key")] = match.group("value").strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one fact. Replacement goes through `edit_text()` so the old
        line is overwritten in place and the stale value never coexists with the new one."""

        value = value.strip()
        if not value:
            return False
        content = self.read_text(user_id) or default_profile(user_id)
        new_line = f"- {key}: {value}"
        for line in content.splitlines():
            match = _FACT_LINE.match(line.strip())
            if match and match.group("key") == key:
                if match.group("value").strip() == value:
                    return False
                if not self.path_for(user_id).exists():
                    self.write_text(user_id, content)
                return self.edit_text(user_id, line, new_line)
        if FACTS_HEADER not in content:
            content = content.rstrip("\n") + f"\n\n{FACTS_HEADER}\n"
        self.write_text(user_id, content.rstrip("\n") + "\n" + new_line + "\n")
        return True


def _split_set(value: str) -> list[str]:
    return [item.strip() for item in value.split(SET_SEPARATOR.strip()) if item.strip()]


def _merged_value(key: str, current: str | None, new: str) -> str:
    if key not in SET_VALUE_KEYS or not current:
        return new
    merged = _split_set(current)
    for item in _split_set(new):
        if item.lower() not in {m.lower() for m in merged}:
            merged.append(item)
    return SET_SEPARATOR.join(merged)


def merge_profile_updates(facts: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """In-memory version of `apply_profile_updates()` (no disk): same replace/merge rules."""

    merged = dict(facts)
    for key, value in updates.items():
        merged[key] = _merged_value(key, merged.get(key), value)
    return merged


def apply_profile_updates(store: UserProfileStore, user_id: str, updates: dict[str, str]) -> dict[str, str]:
    """Persist extracted facts. Single-value keys are replaced (correction), set keys are
    merged with what is already stored. Returns the keys that actually changed."""

    current = store.facts(user_id)
    changed: dict[str, str] = {}
    for key, value in updates.items():
        value = _merged_value(key, current.get(key), value)
        if store.upsert_fact(user_id, key, value):
            changed[key] = value
    return changed


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

# Facts below this confidence are not written to User.md.
CONFIDENCE_THRESHOLD = 0.6

KNOWN_CITIES = (
    "Đà Nẵng", "Huế", "Hà Nội", "Hồ Chí Minh", "TP.HCM", "Sài Gòn", "Hải Phòng", "Cần Thơ",
    "Nha Trang", "Đà Lạt", "Hội An", "Quảng Nam", "Quảng Ngãi", "Vinh", "Biên Hòa", "Vũng Tàu",
)
_CITY_PATTERN = "|".join(re.escape(c) for c in sorted(KNOWN_CITIES, key=len, reverse=True))

_LOCATION_EXPLICIT = re.compile(rf"nơi ở(?: hiện tại)?(?: của mình)? (?:là |vẫn là )?({_CITY_PATTERN})")
_LOCATION_PREP = re.compile(rf"(?:\bsống ở|\bở|\btại|\bsang) ({_CITY_PATTERN})")
_PROFESSION = re.compile(
    r"\b([A-Za-z][A-Za-z0-9-]*\s+(?:engineer|developer|scientist|manager|designer|analyst|researcher))\b"
)
_PROFESSION_VERB = re.compile(r"(?:\blàm|\blà|\bsang|\bnghề(?: nghiệp)?(?: hiện tại)?)\s+(?:vẫn là\s+)?$")
_NAME = re.compile(r"(?:(?:mình|tôi|em)\s+tên(?:\s+là)?|tên\s+(?:mình|tôi|em)\s+là|(?:^|:)\s*tên)\s+(.+)", re.I)
_DRINK_EXPLICIT = re.compile(r"đồ uống (?:yêu thích|ưa thích|ruột)(?: của mình)? là (.+)", re.I)
_DRINK_HABIT = re.compile(r"(?:vẫn|hay|thường|chỉ)\s+uống (.+)", re.I)
_FOOD_EXPLICIT = re.compile(r"món(?: ăn)? (?:yêu thích|ưa thích|ruột)(?: của mình)? là (.+)", re.I)
_PET = re.compile(r"nuôi (?:một |1 |hai |2 )?(?:bé |con |chú |em )?(\w+)(?: tên (\w+))?", re.I)

INTEREST_VOCAB = (
    "Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "evaluation", "LangChain", "LangGraph",
    "memory architecture", "benchmark memory",
)
_INTEREST_GATE = re.compile(r"\b(?:thích|quan tâm|học thêm|đam mê)\b", re.I)

# Style tags: (key, pattern, canonical label). Synonyms map to one label so User.md stays deduplicated.
_STYLE_GATE = re.compile(r"trả lời|giải thích|style|trình bày", re.I)
_STYLE_TAGS = (
    ("short", re.compile(r"ngắn gọn|trả lời ngắn|câu trả lời ngắn|lan man|trả lời gọn|câu trả lời gọn", re.I), "ngắn gọn"),
    ("clear", re.compile(r"rõ ý", re.I), "rõ ý"),
    ("structured", re.compile(r"có cấu trúc", re.I), "có cấu trúc"),
    ("examples", re.compile(r"ví dụ thực (?:tế|chiến)", re.I), "có ví dụ thực tế"),
    ("numbers", re.compile(r"ví dụ số liệu", re.I), "có ví dụ số liệu"),
    ("tradeoff", re.compile(r"trade-off", re.I), "so sánh trade-off"),
)
_BULLET_COUNT = re.compile(r"(\d+)\s*bullet", re.I)

_QUESTION_WORDS = re.compile(r"\b(?:là gì|là ai|ở đâu|nghề gì|tên gì|con gì|thế nào|như thế nào|bao nhiêu)\b", re.I)
_REJECT_VALUES = {"gì", "ai", "đâu", "nào", "không"}

# Clause-level cues that lower (or zero) confidence.
_NEGATION = re.compile(r"\b(?:không còn|không phải|chứ không|không làm|không ở|đừng|chẳng phải)\b", re.I)
_PAST = re.compile(r"\b(?:lúc đầu|ban đầu|trước đó|trước đây|hồi trước|đã từng|thông tin cũ|ví dụ cũ|nghề cũ)\b", re.I)
_NOISE = re.compile(r"\b(?:đùa|chỉ là|đi họp|ra họp|bay ra|giả sử|chẳng hạn|ví dụ như)\b", re.I)
_HYPOTHETICAL = re.compile(r"^\s*(?:nếu|giả sử|hay là|lỡ như)\b", re.I)
_CURRENT = re.compile(r"\b(?:giờ|bây giờ|hiện tại|hiện|đang|vẫn|thực ra|đính chính|chuyển sang|cập nhật|từ tuần này)\b", re.I)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r",\s*|\s+(?:nhưng|chứ|dù|mặc dù|song)\s+", re.I)


@dataclass
class FactCandidate:
    key: str
    value: str
    confidence: float
    clause: str


def _clean_value(value: str) -> str:
    value = re.split(r"\s+(?:như cũ|mỗi ngày|nhé|nha|nữa)\b", value.strip(), maxsplit=1)[0]
    return value.strip(" .,!?;:\"'")


def _leading_capitalized(text: str) -> str:
    words = []
    for word in text.split():
        word = word.strip(".,!?;:\"'()")
        if not word or not word[0].isupper():
            break
        words.append(word)
    return " ".join(words)


def _is_question(sentence: str) -> bool:
    stripped = sentence.strip()
    return stripped.endswith("?") or bool(_QUESTION_WORDS.search(stripped))


def _score(base: float, clause: str, sentence: str, identity: bool) -> float:
    """Confidence model: pattern strength, boosted by 'current/correction' cues,
    penalised by negation, past tense, noise and hypotheticals."""

    if _NEGATION.search(clause):
        return 0.05
    score = base
    if _CURRENT.search(clause):
        score += 0.1
    if _PAST.search(clause):
        score -= 0.5
    if _NOISE.search(clause):
        score -= 0.5
    if identity and (_HYPOTHETICAL.search(sentence) or _HYPOTHETICAL.search(clause)):
        score -= 0.4
    return round(max(0.0, min(1.0, score)), 2)


def _identity_candidates(clause: str, sentence: str) -> list[FactCandidate]:
    found: list[FactCandidate] = []

    def add(key: str, value: str, base: float) -> None:
        value = _clean_value(value)
        if value and value.lower() not in _REJECT_VALUES:
            found.append(FactCandidate(key, value, _score(base, clause, sentence, identity=True), clause))

    name_match = _NAME.search(clause)
    if name_match:
        add("name", _leading_capitalized(name_match.group(1)), 0.9)

    explicit = _LOCATION_EXPLICIT.search(clause)
    if explicit:
        add("location", explicit.group(1), 0.9)
    else:
        for match in _LOCATION_PREP.finditer(clause):
            add("location", match.group(1), 0.75)

    for match in _PROFESSION.finditer(clause):
        has_verb = bool(_PROFESSION_VERB.search(clause[: match.start()]))
        add("profession", match.group(1), 0.8 if has_verb else 0.5)

    drink = _DRINK_EXPLICIT.search(clause)
    if drink:
        add("favorite_drink", drink.group(1), 0.95)
    else:
        habit = _DRINK_HABIT.search(clause)
        if habit:
            add("favorite_drink", habit.group(1), 0.6)

    food = _FOOD_EXPLICIT.search(clause)
    if food:
        add("favorite_food", food.group(1), 0.95)

    pet = _PET.search(clause)
    if pet:
        species, pet_name = pet.group(1), pet.group(2)
        add("pet", f"{species} tên {pet_name}" if pet_name else species, 0.9)

    return found


def _preference_candidates(sentence: str) -> list[FactCandidate]:
    found: list[FactCandidate] = []
    if _NEGATION.search(sentence) and not _STYLE_GATE.search(sentence):
        return found

    if _INTEREST_GATE.search(sentence):
        interests = [term for term in INTEREST_VOCAB if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", sentence)]
        if interests:
            found.append(FactCandidate("interests", SET_SEPARATOR.join(interests), 0.8, sentence))

    if _STYLE_GATE.search(sentence):
        tags: list[str] = []
        bullet = _BULLET_COUNT.search(sentence)
        if bullet:
            tags.append(f"{bullet.group(1)} bullet")
        elif re.search(r"bullet", sentence, re.I):
            tags.append("có bullet")
        for _key, pattern, label in _STYLE_TAGS:
            if pattern.search(sentence):
                tags.append(label)
        if tags:
            found.append(FactCandidate("response_style", SET_SEPARATOR.join(tags), 0.85, sentence))
    return found


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Every fact the message seems to state, with a confidence score (for inspection/tests)."""

    candidates: list[FactCandidate] = []
    for sentence in _SENTENCE_SPLIT.split(message or ""):
        sentence = sentence.strip()
        if not sentence or _is_question(sentence):
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            if clause and clause.strip():
                candidates.extend(_identity_candidates(clause.strip(), sentence))
        candidates.extend(_preference_candidates(sentence))
    return candidates


def extract_profile_updates(message: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    - Question-only sentences are skipped.
    - Each clause is scored; only facts with confidence >= threshold are kept.
    - For single-value keys the last confident mention wins (later clauses correct earlier ones).
    - Set-valued keys (interests, response_style) are merged across sentences.
    """

    updates: dict[str, str] = {}
    for candidate in extract_profile_candidates(message):
        if candidate.confidence < threshold:
            continue
        if candidate.key in SET_VALUE_KEYS and candidate.key in updates:
            merged = _split_set(updates[candidate.key])
            merged += [item for item in _split_set(candidate.value) if item not in merged]
            updates[candidate.key] = SET_SEPARATOR.join(merged)
        else:
            updates[candidate.key] = candidate.value
    return updates


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------

_SUMMARY_LINE_CHARS = 140
_FACT_HINT = re.compile(
    r"tên|ở |nghề|engineer|thích|style|trả lời|bullet|nuôi|yêu thích|đính chính|correction|cập nhật", re.I
)


def _summarize_one(content: str) -> str:
    """Pick the most memory-relevant sentence of a message and truncate it."""

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(content) if s.strip()]
    if not sentences:
        return ""
    best = next((s for s in sentences if _FACT_HINT.search(s)), sentences[0])
    if len(best) > _SUMMARY_LINE_CHARS:
        best = best[: _SUMMARY_LINE_CHARS - 1].rstrip() + "…"
    return best


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages: one short line per user message (assistant
    turns are skipped — in offline mode they are templated and carry no new facts).
    Messages with role `summary` are earlier summary lines and are carried forward.
    Only the newest `max_items` lines are kept, so the summary has a bounded size."""

    lines: list[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "summary":
            lines.append(content)
        elif role == "user":
            line = _summarize_one(content)
            if line:
                lines.append(f"- user: {line}")
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - Keep the newest `keep_messages` messages verbatim.
    - When summary + messages exceed `threshold_tokens`, fold older messages into the summary.
    - Count compactions per thread for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    summary_max_items: int = 6
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.context_tokens(thread_id) > self.threshold_tokens:
            self._compact(thread_id)

    def _compact(self, thread_id: str) -> bool:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]
        if len(messages) <= self.keep_messages:
            return False
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        previous = [{"role": "summary", "content": line} for line in str(thread["summary"]).splitlines() if line]
        thread["summary"] = summarize_messages(previous + older, max_items=self.summary_max_items)
        thread["messages"] = recent
        thread["compactions"] = int(thread["compactions"]) + 1
        return True

    def context_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        return estimate_tokens(str(thread["summary"])) + sum(
            estimate_tokens(m["content"]) for m in thread["messages"]
        )

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {
            "messages": [dict(m) for m in thread["messages"]],
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"]) if thread_id in self.state else 0


# ---------------------------------------------------------------------------
# Deterministic offline responses (shared by both agents; they differ only in
# which facts they can see)
# ---------------------------------------------------------------------------

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "interests": "Mối quan tâm",
    "response_style": "Style trả lời",
}

# Keywords that show which fact a question asks about (checked in order of FACT_LABELS).
_TOPIC_PATTERNS = {
    "name": re.compile(r"\btên\b|là ai", re.I),
    "location": re.compile(r"ở đâu|nơi ở|còn ở|đang ở|sống ở", re.I),
    "profession": re.compile(r"nghề", re.I),
    "favorite_drink": re.compile(r"đồ uống|uống gì", re.I),
    "favorite_food": re.compile(r"món ăn|món gì|ăn gì", re.I),
    "pet": re.compile(r"nuôi|con gì|thú cưng", re.I),
    "interests": re.compile(r"quan tâm|sở thích", re.I),
    "response_style": re.compile(r"style|kiểu trả lời|trả lời (?:như thế nào|ra sao)|cách trả lời", re.I),
}
_RECALL_REQUEST = re.compile(
    r"^\s*(?:nhắc lại|tóm tắt|cho mình biết|kể lại)"
    r"|(?:nhắc lại|tóm tắt|kể lại)\s+(?:giúp|cho)\s+mình"
    r"|\?|\b(?:là gì|là ai|ở đâu)\s*[?.]?\s*$",
    re.I,
)


def requested_topics(message: str) -> list[str]:
    """Fact keys a message asks for; empty when the message is not a recall request."""

    text = message or ""
    if not (_RECALL_REQUEST.search(text) or _QUESTION_WORDS.search(text)):
        return []
    return [key for key, pattern in _TOPIC_PATTERNS.items() if pattern.search(text)]


def compose_offline_reply(message: str, facts: dict[str, str]) -> str:
    """Answer recall questions from `facts`; acknowledge everything else briefly.

    Unknown facts are reported as unknown — the agent never guesses.
    """

    topics = requested_topics(message)
    if not topics:
        return "Đã ghi nhận."

    use_bullets = "bullet" in facts.get("response_style", "").lower()
    parts = []
    for key in topics:
        value = facts.get(key)
        label = FACT_LABELS[key]
        parts.append(f"{label}: {value}" if value else f"{label}: mình chưa có thông tin này")
    if use_bullets:
        return "\n".join(f"- {part}" for part in parts)
    return "; ".join(parts) + "."
