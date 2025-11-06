from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import api_router

app = FastAPI(title="RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

static_dir = Path(__file__).parent.parent / "public"
if static_dir.exists():
    app.mount("/app", StaticFiles(directory=str(static_dir), html=True), name="static")

