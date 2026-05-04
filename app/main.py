import os
import sys

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.routes import router

app = FastAPI(title="ISM CyberRAG", description="Query the Australian ISM using natural language")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline(request: Request):
    return templates.TemplateResponse(request, "pipeline.html")

@app.get("/evaluations", response_class=HTMLResponse)
async def eval_dashboard(request: Request):
    return templates.TemplateResponse(request, "evaluations.html")
