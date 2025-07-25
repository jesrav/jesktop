"""Endpoints returning HTML pages and static assets"""

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from jesktop.api.auth import is_authenticated, verify_basic_auth
from jesktop.config import settings
from jesktop.vector_dbs.base import VectorDB


def get_views_router(vector_db: VectorDB) -> APIRouter:  # noqa: C901
    router = APIRouter()

    router.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    templates = Jinja2Templates(directory="jesktop/web/templates")

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, error: str = None):
        # If already authenticated, redirect to home
        if is_authenticated(request):
            return RedirectResponse("/", status_code=302)

        return templates.TemplateResponse("login.html", {"request": request, "error": error})

    @router.post("/login")
    async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        from fastapi.security import HTTPBasicCredentials

        credentials = HTTPBasicCredentials(username=username, password=password)

        if verify_basic_auth(credentials):
            request.session["authenticated"] = True
            request.session["username"] = username

            # Redirect to the page they were trying to access, or home
            next_url = request.query_params.get("next", "/")
            return RedirectResponse(next_url, status_code=302)
        else:
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": "Invalid username or password",
                    "username": username,
                },
                status_code=400,
            )

    @router.post("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=302)

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        # If not authenticated, redirect to login
        if not is_authenticated(request):
            return RedirectResponse("/login", status_code=302)

        return templates.TemplateResponse("index.html", {"request": request})

    @router.get("/note/{note_id}")
    async def note(request: Request, note_id: str):
        # If not authenticated, redirect to login with next parameter
        if not is_authenticated(request):
            return RedirectResponse(f"/login?next={request.url.path}", status_code=302)

        note = vector_db.get_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="Note not found")
        return templates.TemplateResponse(
            "note.html",
            {
                "request": request,
                "note": {**note.model_dump(), "content": note.content},
            },
        )

    @router.get("/static/assets/{path:path}")
    async def assets(path: str) -> FileResponse:
        return FileResponse(settings.static_dir / "assets" / path)

    return router
