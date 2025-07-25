from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class NoteReference(BaseModel):
    """A reference to a note that was used to answer the question"""

    note_id: str = Field(..., description="The ID of the referenced note")
    title: str = Field(..., description="The title of the referenced note")
    link: str = Field(..., description="The markdown link in format [title](/note/id)")


class RelevantNote(BaseModel):
    """Detailed information from a relevant note"""

    text: str = Field(
        ...,
        description=(
            "Detailed information from the relevant note. Include a summary of the information and actual quotes from the note. "
            "Quotes should be in markdown blockquotes `>`"
        ),
    )
    note_reference: NoteReference = Field(..., description="Reference to the relevant note")

    @property
    def answer(self) -> str:
        if not self.text:
            return ""
        lines = []
        if self.note_reference:
            lines.append(f"### [{self.note_reference.title}](/note/{self.note_reference.note_id})")
        else:
            lines.append("## Link to note")
        lines.append(self.text)
        return "\n".join(lines)


class AssistantResponse(BaseModel):
    """Structured response from the assistant following the prompt template"""

    summary: str = Field(
        ..., description="A detailed summary of the information that was found in the notes."
    )
    relevant_notes: List[RelevantNote] = Field(
        ..., description="Detailed information from the most relevant notes"
    )
    additional_context: Optional[str] = Field(
        description="Related information or connections between notes"
    )
    no_information: bool = Field(
        description="True if no relevant information was found in the notes"
    )

    @property
    def answer(self) -> str:
        """
        The answer string that aligns with the promt template
        """

        if self.no_information:
            return "I don't have any information about that in your notes."
        lines = ["## Summary"]
        if self.summary:
            lines.append(self.summary)
        if self.relevant_notes is not None and len(self.relevant_notes) > 0:
            lines.append("## Details from your notes:")
            for note in self.relevant_notes:
                lines.append(note.answer)

        if self.additional_context:
            lines.append("## Additional Context")
            lines.append(self.additional_context)

        return "\n".join(lines)
