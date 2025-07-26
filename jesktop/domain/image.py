"""Image domain models."""

import base64
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, PlainSerializer


class Image(BaseModel):
    """Represents an image stored in the database.

    Attributes:
        id: The ID of the image.
        note_id: The ID of the note the image belongs to.
        content: The image content, encoded as base64.
        mime_type: The MIME type of the image.
        relative_path: The path of the image relative to the note.
        absolute_path: The absolute path of the image.
    """

    id: str
    note_id: str
    content: Annotated[
        bytes,
        BeforeValidator(lambda x: base64.b64decode(x) if isinstance(x, str) else x),
        PlainSerializer(lambda x: base64.b64encode(x).decode(), return_type=str),
    ]
    mime_type: str
    relative_path: str
    absolute_path: str
