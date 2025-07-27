from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Basic auth settings
    auth_username: str
    auth_password: str

    # Session settings
    session_secret: str = "your-super-secret-session-key-change-in-production"

    # Web server settings
    static_dir: Path = Path("jesktop/web/static")

    # Database settings
    local_vector_db_path: str = "data/vector.json"
    local_image_store_path: str = "data/images.json"

    # LLM settings
    anthropic_api_key: str
    voyage_ai_api_key: str
    system_message: str = """You are a helpful assistant that helps users explore and understand their personal notes. Structure your responses clearly using proper spacing and Markdown formatting.

Use proper Markdown formatting:
- Bold for emphasis using **text**
- Code blocks with ```language
- Lists with - or numbers
- Quote blocks with >

Keep responses clear and well-organized, and always link to the relevant notes when discussing their content.
"""
    rag_closest_chunks: int = 10
    log_level: str = "INFO"  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL


settings = Settings()  # type: ignore
