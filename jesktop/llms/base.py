from typing import Generator, List, Protocol

from jesktop.llms.schemas import LLMMessage


class LLMChat(Protocol):
    def chat(self, messages: List[LLMMessage]) -> LLMMessage: ...

    def chat_stream(self, messages: List[LLMMessage]) -> Generator[LLMMessage, None, None]:
        """Stream chat completions, yielding only new content chunks."""
        ...
