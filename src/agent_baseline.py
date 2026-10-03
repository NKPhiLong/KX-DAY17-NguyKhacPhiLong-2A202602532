from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import compose_offline_reply, estimate_tokens, extract_profile_updates, merge_profile_updates
from model_provider import build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Bạn chỉ biết những gì người dùng đã nói trong cuộc trò chuyện hiện tại; "
    "nếu không có thông tin thì nói rõ là chưa biết, không đoán."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    # Facts seen in *this thread only* — short-term memory, never persisted.
    facts: dict[str, str] = field(default_factory=dict)


class BaselineAgent:
    """Agent A: within-session memory only.

    - Sessions are keyed by `thread_id` (not `user_id`), so a new thread starts empty.
    - No `User.md`, no compact memory: the whole thread history is the prompt every turn.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if not force_offline and self.config.live and self.config.model.is_live_ready():
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response plus token accounting for this turn.

        `user_id` is accepted for interface parity with the advanced agent but deliberately
        unused: the baseline has no per-user memory.
        """

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline turn.

        The prompt is the full thread history (including the new message), so the per-turn
        prompt cost grows with thread length. Both counters accumulate turn by turn.
        """

        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        session.facts = merge_profile_updates(session.facts, extract_profile_updates(message))

        prompt_tokens = estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        response = compose_offline_reply(message, session.facts)
        output_tokens = estimate_tokens(response)

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += output_tokens
        return {
            "response": response,
            "mode": "offline",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "compactions": 0,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        last = result["messages"][-1]
        response = last.content if isinstance(last.content, str) else str(last.content)
        usage = getattr(last, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or sum(estimate_tokens(m["content"]) for m in session.messages)
        output_tokens = usage.get("output_tokens") or estimate_tokens(response)

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += output_tokens
        return {
            "response": response,
            "mode": "live",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: `create_agent` + `InMemorySaver` (thread-scoped checkpoints only).

        Returns None when LangChain/LangGraph or the provider SDK is unavailable, so the
        caller falls back to the offline path.
        """

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            return create_agent(
                model=build_chat_model(self.config.model),
                tools=[],
                system_prompt=BASELINE_SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
        except Exception:
            return None
