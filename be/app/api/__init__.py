from fastapi import APIRouter
from . import chat, upload, history, sources

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(upload.router)
api_router.include_router(history.router)
api_router.include_router(sources.router)

