"""HTTP API for SigLIP2 image and text embeddings."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .runtime import EmbeddingService, InputError
from .settings import Settings


MAX_ITEMS = 32


class TextEmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_ITEMS)


class EmbeddingResponse(BaseModel):
    dimension: int
    vectors: list[list[float]]


class EmbeddingSpaceResponse(BaseModel):
    space_id: str
    dimension: int
    modalities: tuple[str, str] = ("image", "text")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.service = await asyncio.to_thread(EmbeddingService, settings)
    yield


app = FastAPI(title="SigLIP2 Embed Service", lifespan=lifespan)


@app.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    return {"status": "ok", "backend": settings.backend}


@app.get("/v1/embedding-space", response_model=EmbeddingSpaceResponse)
def embedding_space(request: Request) -> EmbeddingSpaceResponse:
    settings: Settings = request.app.state.settings
    service: EmbeddingService = request.app.state.service
    vector = service.embed_texts(["probe"])
    return EmbeddingSpaceResponse(
        space_id=settings.embedding_space_id,
        dimension=int(vector.shape[1]),
    )


@app.post("/v1/embed/images", response_model=EmbeddingResponse)
async def embed_images(
    request: Request,
    files: list[UploadFile] = File(...),
) -> EmbeddingResponse:
    settings: Settings = request.app.state.settings
    payloads = await _read_uploads(files, settings)
    try:
        vectors = await asyncio.to_thread(request.app.state.service.embed_images, payloads)
    except InputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return EmbeddingResponse(
        dimension=int(vectors.shape[1]),
        vectors=vectors.tolist(),
    )


@app.post("/v1/embed/texts", response_model=EmbeddingResponse)
async def embed_texts(
    body: TextEmbeddingRequest,
    request: Request,
) -> EmbeddingResponse:
    vectors = await asyncio.to_thread(request.app.state.service.embed_texts, body.texts)
    return EmbeddingResponse(
        dimension=int(vectors.shape[1]),
        vectors=vectors.tolist(),
    )


async def _read_uploads(files: list[UploadFile], settings: Settings) -> list[bytes]:
    if not 1 <= len(files) <= MAX_ITEMS:
        raise HTTPException(
            status_code=422,
            detail=f"files must contain between 1 and {MAX_ITEMS} items",
        )

    payloads: list[bytes] = []
    total_size = 0
    for file in files:
        payload = await file.read(settings.max_image_bytes + 1)
        if len(payload) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="an image exceeds MAX_IMAGE_BYTES")
        total_size += len(payload)
        if total_size > settings.max_request_bytes:
            raise HTTPException(status_code=413, detail="request exceeds MAX_REQUEST_BYTES")
        payloads.append(payload)
    return payloads
