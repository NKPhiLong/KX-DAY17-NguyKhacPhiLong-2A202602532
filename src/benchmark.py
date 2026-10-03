from __future__ import annotations

import dataclasses
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import estimate_tokens

COLUMNS = (
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
)

# Phrases that mean "I don't know" — honest, but not a useful answer.
_UNKNOWN = re.compile(r"chưa có thông tin|không biết|không nhớ", re.I)
_CONCISE_TOKEN_LIMIT = 80


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and validate a benchmark dataset."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of conversations.")
    for conv in data:
        missing = {"id", "user_id", "turns", "recall_questions"} - conv.keys()
        if missing:
            raise ValueError(f"{path}: conversation {conv.get('id')!r} is missing {sorted(missing)}.")
        for question in conv["recall_questions"]:
            if not {"question", "expected_contains"} <= question.keys():
                raise ValueError(f"{path}: malformed recall question in {conv['id']!r}.")
    return data


def _hits(answer: str, expected: list[str]) -> int:
    lowered = (answer or "").lower()
    return sum(1 for item in expected if item.lower() in lowered)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    if not expected:
        return 1.0
    hits = _hits(answer, expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1], identical for both agents.

    - 0.6 x fact coverage (share of expected facts present)
    - 0.2 if the answer commits to an answer instead of saying it doesn't know
    - 0.2 if the answer is concise (<= 80 estimated tokens)
    """

    coverage = _hits(answer, expected) / len(expected) if expected else 1.0
    committed = 0.0 if _UNKNOWN.search(answer or "") else 1.0
    concise = 1.0 if 0 < estimate_tokens(answer) <= _CONCISE_TOKEN_LIMIT else 0.0
    return round(0.6 * coverage + 0.2 * committed + 0.2 * concise, 4)


def recall_thread_id(conversation_id: str) -> str:
    """Fresh thread used for recall questions — same naming for both agents."""

    return f"{conversation_id}::recall"


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over the conversations, in dataset order.

    1. Feed every turn into the conversation's own thread.
    2. Ask its recall questions in a fresh thread (`<id>::recall`).
    3. Token columns sum every thread the agent touched (chat + recall).
    4. Memory growth = User.md size after - before, per user; compactions summed per thread.
    """

    file_size = getattr(agent, "memory_file_size", None)
    users = list(dict.fromkeys(conv["user_id"] for conv in conversations))
    size_before = {user: file_size(user) if file_size else 0 for user in users}

    threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    for conv in conversations:
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], conv["id"], turn)
        threads.append(conv["id"])

        recall_thread = recall_thread_id(conv["id"])
        for question in conv["recall_questions"]:
            answer = agent.reply(conv["user_id"], recall_thread, question["question"])["response"]
            recall_scores.append(recall_points(answer, question["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, question["expected_contains"]))
        threads.append(recall_thread)

    growth = sum((file_size(user) if file_size else 0) - size_before[user] for user in users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=sum(agent.compaction_count(t) for t in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        (
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.0%}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            str(row.compactions),
        )
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github", disable_numparse=True, colalign=("left",) + ("right",) * 6)
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(cells) + " |" for cells in table]
        return "\n".join(lines)


def _isolated_config(config: LabConfig, suite: str) -> LabConfig:
    """Each suite writes to its own fresh state dir so reruns are reproducible."""

    state_dir = config.state_dir / "benchmark" / suite
    shutil.rmtree(state_dir, ignore_errors=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(config, state_dir=state_dir)


def run_suite(suite: str, conversations: list[dict[str, Any]], config: LabConfig) -> list[BenchmarkRow]:
    suite_config = _isolated_config(config, suite)
    return [
        run_agent_benchmark("Baseline", BaselineAgent(suite_config), conversations, suite_config),
        run_agent_benchmark("Advanced", AdvancedAgent(suite_config), conversations, suite_config),
    ]


def _comparison(rows: list[BenchmarkRow]) -> str:
    base, adv = rows
    ratio = adv.prompt_tokens_processed / base.prompt_tokens_processed if base.prompt_tokens_processed else 0.0
    change = (ratio - 1) * 100
    direction = "more" if change >= 0 else "fewer"
    return (
        f"Advanced vs Baseline: {abs(change):.0f}% {direction} prompt tokens processed, "
        f"recall {base.recall_score:.0%} -> {adv.recall_score:.0%}."
    )


def main() -> None:
    """Run the Standard and Long-Context Stress benchmarks for Baseline vs Advanced."""

    config = load_config(Path(__file__).resolve().parent.parent)
    mode = "live" if config.live else "offline (deterministic)"
    print(f"Mode: {mode} | compact threshold: {config.compact_threshold_tokens} tokens, "
          f"keep {config.compact_keep_messages} messages\n")

    suites = (
        ("standard", "Standard Benchmark", config.data_dir / "conversations.json"),
        ("stress", "Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    for suite, title, path in suites:
        conversations = load_conversations(path)
        n_turns = sum(len(c["turns"]) for c in conversations)
        n_questions = sum(len(c["recall_questions"]) for c in conversations)
        rows = run_suite(suite, conversations, config)
        print(f"## {title}")
        print(f"{path.name}: {len(conversations)} conversations, {n_turns} turns, {n_questions} recall questions\n")
        print(format_rows(rows))
        print(f"\n{_comparison(rows)}\n")


if __name__ == "__main__":
    main()
