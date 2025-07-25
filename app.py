import sys

import instructor
from anthropic import Anthropic
from loguru import logger

from jesktop.api import create_app
from jesktop.config import settings
from jesktop.embedders.voyage_embedder import VoyageEmbedder
from jesktop.image_store.local import LocalImageStore
from jesktop.llms.instructor_llm_chat import InstructorLLMChat
from jesktop.vector_dbs.local_db import LocalVectorDB

logger.configure(handlers=[{"sink": sys.stderr, "level": settings.log_level}])

logger.info("Initializing Claude chatbot with Instructor and Voyage AI embeddings")
# Create instructor client with Anthropic Claude
anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
instructor_client = instructor.from_anthropic(
    anthropic_client, mode=instructor.Mode.ANTHROPIC_TOOLS
)

vector_db = LocalVectorDB.load(settings.local_vector_db_path)
image_store = LocalImageStore.load(settings.local_image_store_path)
embedder = VoyageEmbedder(api_key=settings.voyage_ai_api_key)
chatbot = InstructorLLMChat(instructor_client)
app = create_app(
    vector_db=vector_db,
    embedder=embedder,
    chatbot=chatbot,
    image_store=image_store,
)
