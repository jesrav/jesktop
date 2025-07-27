import base64

from fastapi.testclient import TestClient


def login_user(
    client: TestClient, username: str = "admin", password: str = "password"
) -> TestClient:
    """Helper function to log in a user and return a client with session cookies."""
    response = client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=True
    )
    # Should either be 200 (followed redirect to home) or 302 (redirect response)
    assert response.status_code in [200, 302], (
        f"Unexpected status: {response.status_code}, content: {response.text}"
    )
    return client


def test_chat_endpoint_streams_response(test_client: TestClient) -> None:
    """Test that chat endpoint streams responses properly."""
    login_user(test_client)
    with test_client.stream("GET", "/chat?message=test") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Collect all messages
        messages = []
        current_message = []
        for chunk in response.iter_lines():
            if chunk != "":
                current_message.append(chunk)
            elif current_message != []:  # Empty line marks end of message
                messages.append("\n".join(current_message))
                current_message = []

        # Expected messages with proper line prefixes
        expected_messages = [
            "data: Summary\ndata: The notes are about emojis.",
            "data: Details\ndata: In your note titled\ndata: The banana emoji I found ...",
            "data: Additional Context\ndata: The notes also refer to\ndata: resources for emojis",
            "event: done\ndata:",
        ]

        assert messages == expected_messages


def test_chat_endpoint_empty_message(test_client: TestClient) -> None:
    """Test that chat endpoint handles empty messages."""
    login_user(test_client)
    response = test_client.get("/chat?message=")
    assert response.status_code == 200

    messages = []
    current_message = []
    for chunk in response.iter_lines():
        if chunk:
            current_message.append(chunk)
        elif current_message:
            messages.append("\n".join(current_message))
            current_message = []

    assert len(messages) == 1
    assert messages[0] == "event: error\ndata: No message provided"


def test_note_endpoint_returns_note(test_client: TestClient) -> None:
    """Test that note endpoint returns correct note."""
    login_user(test_client)
    response = test_client.get("/note/note1")
    assert response.status_code == 200


def test_note_endpoint_not_found(test_client: TestClient) -> None:
    """Test that note endpoint handles missing notes."""
    login_user(test_client)
    response = test_client.get("/note/nonexistent")
    assert response.status_code == 404
    assert "Note not found" in response.text


def test_image_endpoint_returns_image(test_client: TestClient) -> None:
    """Test that image endpoint returns correct image."""
    login_user(test_client)
    response = test_client.get("/api/images/note1/test.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # The endpoint returns base64 encoded content
    assert base64.b64decode(response.content) == b"fake image data"
    assert response.headers["cache-control"] == "public, max-age=31536000"


def test_unauthorized_access_without_credentials(test_client: TestClient) -> None:
    """Test that endpoints require authentication."""
    # Test chat endpoint - still returns 401 for API endpoints
    response = test_client.get("/chat?message=test")
    assert response.status_code == 401

    # Test note endpoint - now redirects to login
    response = test_client.get("/note/note1", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login?next=/note/note1"

    # Test home page - now redirects to login
    response = test_client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"

    # Test image endpoint - images still require authentication (401 for API)
    response = test_client.get("/api/images/note1/test.png")
    assert response.status_code == 401


def test_unauthorized_access_with_wrong_credentials(test_client: TestClient) -> None:
    """Test that endpoints reject incorrect credentials."""
    # Try to login with wrong credentials
    response = test_client.post("/login", data={"username": "wrong", "password": "credentials"})
    assert response.status_code == 400  # Bad request for wrong credentials

    # Test endpoints without proper session
    response = test_client.get("/chat?message=test")
    assert response.status_code == 401

    # Test note endpoint - redirects to login when not authenticated
    response = test_client.get("/note/note1", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login?next=/note/note1"

    response = test_client.get("/api/images/note1/test.png")
    assert response.status_code == 401


def test_chat_endpoint_error_handling(test_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    """Test that chat endpoint handles errors properly."""

    def raise_error(*args, **kwargs):  # noqa: ARG001
        raise ValueError("Test error")

    # Patch the chat method to raise an error
    monkeypatch.setattr("tests.fakes.FakeLLMChat.chat_stream", raise_error)

    login_user(test_client)
    with test_client.stream("GET", "/chat?message=test") as response:
        assert response.status_code == 200
        messages = []
        current_message = []
        for chunk in response.iter_lines():
            if chunk:
                current_message.append(chunk)
            elif current_message:
                messages.append("\n".join(current_message))
                current_message = []

        assert len(messages) == 1
        assert messages[0] == "event: error\ndata: Test error"


def test_notes_search_endpoint_existing_note(test_client: TestClient) -> None:
    """Test notes search endpoint returns existing note."""
    login_user(test_client)

    # Test searching for existing note by title
    response = test_client.get("/api/notes/search?title=Test Note 1")
    assert response.status_code == 200

    data = response.json()
    assert data["exists"] is True
    assert data["note_id"] == "note1"
    assert data["title"] == "Test Note 1"
    assert data["url"] == "/note/note1"


def test_notes_search_endpoint_nonexistent_note(test_client: TestClient) -> None:
    """Test notes search endpoint returns correct response for missing note."""
    login_user(test_client)

    # Test searching for nonexistent note
    response = test_client.get("/api/notes/search?title=Nonexistent Note")
    assert response.status_code == 200

    data = response.json()
    assert data["exists"] is False
    assert data["note_id"] is None
    assert data["title"] == "Nonexistent Note"
    assert data["url"] is None


def test_notes_search_endpoint_case_insensitive(test_client: TestClient) -> None:
    """Test notes search endpoint is case insensitive."""
    login_user(test_client)

    # Test case insensitive search
    response = test_client.get("/api/notes/search?title=test note 1")
    assert response.status_code == 200

    data = response.json()
    assert data["exists"] is True
    assert data["note_id"] == "note1"


def test_notes_search_endpoint_missing_title(test_client: TestClient) -> None:
    """Test notes search endpoint handles missing title parameter."""
    login_user(test_client)

    # Test without title parameter
    response = test_client.get("/api/notes/search")
    assert response.status_code == 422  # Validation error for missing required parameter


def test_notes_search_endpoint_unauthorized(test_client: TestClient) -> None:
    """Test notes search endpoint requires authentication."""
    # Test without authentication
    response = test_client.get("/api/notes/search?title=Test Note 1")
    assert response.status_code == 401
