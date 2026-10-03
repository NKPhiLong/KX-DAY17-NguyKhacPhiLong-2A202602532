from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

LONG_TURN = (
    "Mình kể thêm một đoạn dài về tin tức để thread phình ra: Artemis III, X-59, El Nino và "
    "kế hoạch điện sạch đều là ví dụ về readiness, externality, uncertainty và efficiency. "
    "Đoạn này cố tình dài để prompt của baseline tăng dần theo từng lượt trong cùng một thread. "
)


def make_config(tmp_path: Path) -> LabConfig:
    """Isolated config: state in tmp_path, tiny compact threshold, offline stub model."""

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=80,  # small so compaction kicks in early
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(make_config(tmp_path).state_dir / "profiles")

    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0

    path = store.write_text("dungct", "# User Profile: dungct\n\n## Facts\n- location: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert path.is_relative_to(tmp_path)
    assert "- location: Đà Nẵng" in store.read_text("dungct")
    size_before = store.file_size("dungct")
    assert size_before > 0

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert store.facts("dungct")["location"] == "Huế"
    assert "Đà Nẵng" not in store.read_text("dungct")
    # Editing text that is not there reports no change and leaves the file intact.
    assert store.edit_text("dungct", "Hà Nội", "Sài Gòn") is False
    assert store.facts("dungct")["location"] == "Huế"

    # A hostile user id must not escape the profiles directory.
    assert store.path_for("../../etc/passwd").is_relative_to(store.root_dir.resolve())


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)

    memory.append("t1", "user", "Mình tên là DũngCT.")
    assert memory.compaction_count("t1") == 0  # short thread: nothing to compact

    for _ in range(6):
        memory.append("t1", "user", LONG_TURN)
        memory.append("t1", "assistant", "Đã ghi nhận.")

    context = memory.context("t1")
    assert memory.compaction_count("t1") > 0
    assert context["summary"], "older messages must be folded into a summary"
    assert len(context["messages"]) <= config.compact_keep_messages + 1
    assert memory.compaction_count("other-thread") == 0

    # Agent-level: the advanced agent actually reports compactions on a long thread.
    agent = AdvancedAgent(config, force_offline=True)
    for _ in range(6):
        agent.reply("dungct", "long", LONG_TURN)
    assert agent.compaction_count("long") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    for message in ("Chào bạn, mình tên là DũngCT.", "Mình đang làm MLOps engineer."):
        advanced.reply("dungct", "thread-1", message)
        baseline.reply("dungct", "thread-1", message)

    # Same thread: both agents have short-term memory.
    assert "DũngCT" in baseline.reply("dungct", "thread-1", "Mình tên gì?")["response"]

    # New thread: only the agent with persistent memory remembers.
    advanced_answer = advanced.reply("dungct", "thread-2", "Mình tên gì và làm nghề gì?")["response"]
    baseline_answer = baseline.reply("dungct", "thread-2", "Mình tên gì và làm nghề gì?")["response"]
    assert "DũngCT" in advanced_answer and "MLOps engineer" in advanced_answer
    assert "DũngCT" not in baseline_answer and "MLOps engineer" not in baseline_answer

    # Persistent memory survives a brand-new agent instance (process restart).
    restarted = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in restarted.reply("dungct", "thread-3", "Mình tên gì?")["response"]
    assert restarted.memory_file_size("dungct") > 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    for _ in range(12):
        advanced.reply("dungct", "long", LONG_TURN)
        baseline.reply("dungct", "long", LONG_TURN)

    assert advanced.compaction_count("long") > 0
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")

    # Per-turn context stays bounded for advanced but keeps growing for baseline.
    adv_last = advanced.reply("dungct", "long", LONG_TURN)["prompt_tokens"]
    base_last = baseline.reply("dungct", "long", LONG_TURN)["prompt_tokens"]
    assert adv_last < base_last / 2


# --- Beyond the happy path ---------------------------------------------------


def test_correction_keeps_only_latest_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    agent.reply("u", "t2", "Mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    agent.reply("u", "t3", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    profile = agent.profile_store.read_text("u")
    assert "location: Huế" in profile and "Đà Nẵng" not in profile
    assert "profession: MLOps engineer" in profile and "backend" not in profile
    answer = agent.reply("u", "t4", "Hiện tại mình ở đâu và làm nghề gì?")["response"]
    assert "Huế" in answer and "MLOps engineer" in answer and "backend" not in answer


def test_noise_and_questions_are_not_saved() -> None:
    noisy = (
        "Có lúc mình đùa rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa. "
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại."
    )
    assert extract_profile_updates(noisy) == {}
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}
    assert extract_profile_updates("Lúc đầu mình nói hiện ở Huế, nhưng thực ra mình đang ở Đà Nẵng.") == {
        "location": "Đà Nẵng"
    }


def test_stress_dataset_recall_and_correction(tmp_path: Path) -> None:
    conv = json.loads((REPO_ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))[0]
    # Same compact settings as the benchmark defaults.
    config = dataclasses.replace(make_config(tmp_path), compact_threshold_tokens=600, compact_keep_messages=4)
    agent = AdvancedAgent(config, force_offline=True)
    for turn in conv["turns"]:
        agent.reply(conv["user_id"], conv["id"], turn)

    assert agent.compaction_count(conv["id"]) > 0
    facts = agent.profile_store.facts(conv["user_id"])
    assert facts["location"] == "Đà Nẵng"
    assert facts["profession"] == "MLOps engineer"
    for question in conv["recall_questions"]:
        answer = agent.reply(conv["user_id"], "fresh", question["question"])["response"]
        for expected in question["expected_contains"]:
            assert expected in answer, (question["question"], expected, answer)


def test_benchmark_is_deterministic(tmp_path: Path) -> None:
    from benchmark import load_conversations, run_agent_benchmark

    conversations = load_conversations(REPO_ROOT / "data" / "conversations.json")[:3]
    rows = []
    for run in range(2):
        config = make_config(tmp_path / f"run{run}")
        rows.append(run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config))
    assert rows[0] == rows[1]
