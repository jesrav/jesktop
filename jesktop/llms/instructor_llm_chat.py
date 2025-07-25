from typing import Generator, List

from instructor import Instructor

from jesktop.llms.schemas import AssistantResponse, LLMMessage


class InstructorLLMChat:
    def __init__(self, instructor: Instructor) -> None:
        self.instructor = instructor

    def chat(self, messages: List[LLMMessage]) -> LLMMessage:
        response = self.instructor.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[m.model_dump() for m in messages],  # type: ignore
            response_model=AssistantResponse,
        )
        return LLMMessage(role="assistant", content=response.answer)

    def chat_stream(self, messages: List[LLMMessage]) -> Generator[LLMMessage, None, None]:
        """Stream chat completions, yielding only new content chunks."""
        responses = self.instructor.chat.completions.create_partial(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[m.model_dump() for m in messages],  # type: ignore
            response_model=AssistantResponse,
            stream=True,
        )

        for response in responses:
            yield LLMMessage(role="assistant", content=response.answer)
