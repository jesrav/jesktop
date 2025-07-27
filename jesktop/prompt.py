from typing import List

from jesktop.domain.note import Chunk
from jesktop.embedders.base import Embedder
from jesktop.vector_dbs.base import VectorDB

PROMPT_TEMPLATE = """Answer the question based on the context from your notes below.

Relevant notes:
{context}

Question: {question}

Answer: """


def get_context(relevant_notes: List[Chunk]) -> str:
    context = ""
    for note in relevant_notes:
        context += f"Note ID: {note.note_id}\nTitle: {note.title}\nContent: {note.text}\n\n"
    return context


def get_prompt(
    *,
    input_texts: List[str],
    embedder: Embedder,
    vector_db: VectorDB,
    closest: int,
) -> str:
    input_vector = embedder.embed("\n".join(input_texts))
    closest_chunks = vector_db.get_closest_chunks(input_vector, closest=closest)
    context = get_context(relevant_notes=closest_chunks)
    prompt = PROMPT_TEMPLATE.format(question=input_texts[-1], context=context)
    return prompt
