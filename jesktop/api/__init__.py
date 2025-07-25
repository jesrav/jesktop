from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from jesktop.api.endpoints import get_endpoints_router
from jesktop.api.views import get_views_router
from jesktop.config import settings
from jesktop.embedders.base import Embedder
from jesktop.image_store import ImageStore
from jesktop.llms.base import LLMChat
from jesktop.vector_dbs.base import VectorDB


def create_app(
    *,
    vector_db: VectorDB,
    embedder: Embedder,
    chatbot: LLMChat,
    image_store: ImageStore,
) -> FastAPI:
    """Create FastAPI app."""
    app = FastAPI()

    # Add session middleware first
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(
        router=get_endpoints_router(
            vector_db=vector_db, embedder=embedder, chatbot=chatbot, image_store=image_store
        )
    )
    app.include_router(router=get_views_router(vector_db=vector_db))

    return app
