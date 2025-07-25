import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from jesktop.config import settings

security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:  # noqa: B008
    """Verify basic auth credentials."""
    is_correct_username = secrets.compare_digest(credentials.username, settings.auth_username)
    is_correct_password = secrets.compare_digest(credentials.password, settings.auth_password)

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def verify_basic_auth(credentials: HTTPBasicCredentials) -> bool:
    """Verify basic auth credentials without raising exceptions."""
    is_correct_username = secrets.compare_digest(credentials.username, settings.auth_username)
    is_correct_password = secrets.compare_digest(credentials.password, settings.auth_password)
    return is_correct_username and is_correct_password


def verify_session(request: Request) -> str:
    """Verify session-based authentication."""
    if not request.session.get("authenticated"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return request.session.get("username", "")


def is_authenticated(request: Request) -> bool:
    """Check if user is authenticated via session."""
    return bool(request.session.get("authenticated"))
