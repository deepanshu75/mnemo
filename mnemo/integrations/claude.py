from __future__ import annotations

from mnemo.core.memory import MemoryStore


class ClaudeMemory(MemoryStore):
    def build_messages(
        self,
        *,
        query: str,
        user_message: str,
        top_k: int = 5,
    ) -> list[dict[str, str]]:
        context_block = self.context(query=query, top_k=top_k)
        system_text = (
            "Use relevant long-term memory context when useful.\n\n"
            f"{context_block}"
        )
        return [
            {"role": "user", "content": f"{system_text}\n\nUser: {user_message}"},
        ]
