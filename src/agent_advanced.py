from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    PROFILE_KEYS,
    CompactMemoryManager,
    UserProfileStore,
    apply_profile_updates,
    compose_offline_reply,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. Hồ sơ người dùng (User.md) là nguồn sự thật cho các fact ổn định; "
    "luôn dùng giá trị mới nhất, bỏ qua thông tin cũ hoặc câu đùa. Trả lời theo style người dùng đã chọn."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. within-session memory  -> recent messages kept verbatim in `CompactMemoryManager`
    2. persistent `User.md`   -> stable facts, survives across threads
    3. compact memory         -> older messages folded into a bounded summary

    `User.md` and compact memory are independent paths: stable facts never depend on
    whether the conversation summary happened to keep them.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline and self.config.live and self.config.model.is_live_ready():
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live mode (real model) and the deterministic offline path."""

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _record(self, thread_id: str, prompt_tokens: int, output_tokens: int) -> None:
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path (one turn)."""

        # 1-2. Stable facts -> User.md (corrections overwrite the old line via edit_text()).
        updates = extract_profile_updates(message)
        changed = apply_profile_updates(self.profile_store, user_id, updates)

        # 3. Conversation -> compact memory (may compact here).
        self.compact_memory.append(thread_id, "user", message)

        # 4. Context carried into this turn: User.md + summary + recent messages.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 5. Answer from persisted memory.
        response = self._offline_response(user_id, thread_id, message)
        output_tokens = estimate_tokens(response)

        # 6. Store the reply and update counters.
        self.compact_memory.append(thread_id, "assistant", response)
        self._record(thread_id, prompt_tokens, output_tokens)
        return {
            "response": response,
            "mode": "offline",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """System prompt + User.md + compact summary + recent kept messages."""

        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(context["summary"]))
            + sum(estimate_tokens(m["content"]) for m in context["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer built from `User.md` facts (latest value per key)."""

        return compose_offline_reply(message, self.profile_store.facts(user_id))

    # ------------------------------------------------------------------
    # Live mode (optional extension)
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # The deterministic extractor still runs as a safety net; the model can also edit
        # User.md itself through the `save_user_fact` tool.
        changed = apply_profile_updates(self.profile_store, user_id, extract_profile_updates(message))
        # Mirror the thread into compact memory so `Compactions` uses the same trigger rule
        # as the summarization middleware configured below.
        self.compact_memory.append(thread_id, "user", message)

        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        last = result["messages"][-1]
        response = last.content if isinstance(last.content, str) else str(last.content)
        usage = getattr(last, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens") or self._estimate_prompt_context_tokens(user_id, thread_id)
        output_tokens = usage.get("output_tokens") or estimate_tokens(response)

        self.compact_memory.append(thread_id, "assistant", response)
        self._record(thread_id, prompt_tokens, output_tokens)
        return {
            "response": response,
            "mode": "live",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Live agent with tools and compact middleware.

        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - `read_user_memory` / `save_user_fact` tools over User.md
        - dynamic prompt that injects the current profile
        - `SummarizationMiddleware` using the same threshold / keep settings as offline

        Returns None when dependencies are missing so the caller falls back to offline.
        """

        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
            from langchain.tools import ToolRuntime, tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        store = self.profile_store

        @tool
        def read_user_memory(runtime: ToolRuntime[AgentContext]) -> str:
            """Read the current user's persistent profile (User.md)."""

            return store.read_text(runtime.context.user_id) or "(User.md chưa có nội dung)"

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Save or correct one stable user fact in User.md.

            key must be one of: name, location, profession, favorite_drink, favorite_food, pet,
            interests, response_style. Only call this for facts the user states about themselves
            as current truth — never for questions, jokes, or past/hypothetical values.
            """

            if key not in PROFILE_KEYS:
                return f"Bỏ qua: key không hợp lệ ({key})."
            changed = apply_profile_updates(store, runtime.context.user_id, {key: value})
            return f"Đã lưu {key}." if changed else f"{key} không đổi."

        @dynamic_prompt
        def profile_prompt(request: ModelRequest) -> str:
            profile = store.read_text(request.runtime.context.user_id) or "(chưa có)"
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n<user_profile>\n{profile}\n</user_profile>"

        try:
            model = build_chat_model(self.config.model)
            return create_agent(
                model=model,
                tools=[read_user_memory, save_user_fact],
                middleware=[
                    profile_prompt,
                    SummarizationMiddleware(
                        model=model,
                        trigger=("tokens", self.config.compact_threshold_tokens),
                        keep=("messages", self.config.compact_keep_messages),
                    ),
                ],
                context_schema=AgentContext,
                checkpointer=InMemorySaver(),
            )
        except Exception:
            return None
