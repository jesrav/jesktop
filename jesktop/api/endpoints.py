from pathlib import Path
from typing import Generator
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from jesktop.api.auth import verify_session
from jesktop.config import settings
from jesktop.embedders.base import Embedder
from jesktop.image_store import ImageStore
from jesktop.llms.base import LLMChat
from jesktop.llms.schemas import LLMMessage
from jesktop.prompt import get_prompt
from jesktop.vector_dbs.base import VectorDB


def stream_response(
    answer_generator: Generator[LLMMessage, None, None],
) -> Generator[str, None, None]:
    """Format LLM messages as SSE events.

    Each message is formatted as an SSE event with 'data: ' prefix for each line.
    Handles multiline content and adds appropriate newlines for SSE format.
    """
    try:
        for answer in answer_generator:
            content = answer.content
            lines = content.split("\n")
            data = "\n".join(f"data: {line}" for line in lines)
            yield f"{data}\n\n"

        yield "event: done\ndata:\n\n"
    except Exception as e:
        logger.error(f"Error in stream: {str(e)}")
        yield f"event: error\ndata: {str(e)}\n\n"


def _create_chat_endpoint(embedder: Embedder, vector_db: VectorDB, chatbot: LLMChat):
    """Create the chat endpoint handler."""

    async def chat(
        message: str,
        request: Request,  # noqa: ARG001
        _: str = Depends(verify_session),
    ) -> StreamingResponse:
        if not message:
            return StreamingResponse(
                iter(["event: error\ndata: No message provided\n\n"]),
                media_type="text/event-stream",
            )

        try:
            messages = [{"role": "user", "content": message}]
            messages[0]["content"] = get_prompt(
                input_texts=[messages[0]["content"]],
                embedder=embedder,
                vector_db=vector_db,
                closest=settings.rag_closest_chunks,
            )

            system_message = [LLMMessage(role="system", content=settings.system_message)]
            messages = system_message + [LLMMessage.model_validate(m) for m in messages]
            answer_generator = chatbot.chat_stream(messages=messages)

            return StreamingResponse(
                stream_response(answer_generator),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return StreamingResponse(
                iter([f"event: error\ndata: {str(e)}\n\n"]),
                media_type="text/event-stream",
            )

    return chat


def _create_notes_search_endpoint(vector_db: VectorDB):
    """Create the notes search endpoint handler."""

    async def search_notes_by_title(
        title: str,
        request: Request,  # noqa: ARG001
        _: str = Depends(verify_session),
    ):
        """Search for a note by title for wikilink resolution."""
        try:
            note = vector_db.find_note_by_title(title)
            if note:
                note_stem = Path(note.path).stem
                return {
                    "note_id": note.id,
                    "title": note.title or note_stem,
                    "exists": True,
                    "url": f"/note/{note.id}",
                }
            else:
                return {
                    "note_id": None,
                    "title": title,
                    "exists": False,
                    "url": None,
                }
        except Exception as e:
            logger.error(f"Error searching for note by title '{title}': {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return search_notes_by_title


def _create_image_endpoint(image_store: ImageStore):
    """Create the image endpoint handler."""

    async def get_image(
        note_id: str,
        path: str,
        request: Request,  # noqa: ARG001
        _: str = Depends(verify_session),
    ):
        try:
            decoded_path = unquote(path)

            image_id = image_store.get_image_id_by_path(note_id, decoded_path)
            if not image_id:
                logger.warning(f"Image not found for note {note_id} and path {decoded_path}")
                raise HTTPException(status_code=404, detail="Image not found")

            image = image_store.get_image(image_id)
            try:
                return Response(
                    content=image.content,
                    media_type=image.mime_type,
                    headers={
                        "Cache-Control": "public, max-age=31536000",
                        "ETag": f'"{image_id}"',
                    },
                )
            except Exception as e:
                logger.error(f"Error processing image content: {e}")
                raise HTTPException(status_code=500, detail="Error processing image") from e
        except KeyError as err:
            logger.error("Failed to retrieve image")
            raise HTTPException(status_code=404, detail="Image not found") from err
        except Exception as e:
            logger.error(f"Error serving image: {e}")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return get_image


def get_endpoints_router(
    *,
    vector_db: VectorDB,
    embedder: Embedder,
    chatbot: LLMChat,
    image_store: ImageStore,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health_check():
        return {"status": "healthy"}

    router.get("/chat")(_create_chat_endpoint(embedder, vector_db, chatbot))
    router.get("/api/notes/search")(_create_notes_search_endpoint(vector_db))
    router.get("/api/images/{note_id}/{path:path}")(_create_image_endpoint(image_store))

    return router
