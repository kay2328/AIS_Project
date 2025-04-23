# index.py

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Index page."""

    base_url = request.base_url

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": "message"
        },
    )
