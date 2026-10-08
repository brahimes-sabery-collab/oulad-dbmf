"""Authenticated cold inference. Start with uvicorn dbmf.api:create_app --factory."""
from contextlib import asynccontextmanager
import hmac
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .artifacts import Recommender

class Profile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    gender: str = Field(max_length=100)
    age_band: str = Field(max_length=100)
    highest_education: str = Field(max_length=100)

class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile: Profile
    k: int = Field(default=5, ge=1, le=100)
    eligible: list[str] | None = Field(default=None, max_length=1000)
    excluded: list[str] = Field(default_factory=list, max_length=1000)

def create_app():
    @asynccontextmanager
    async def lifespan(app):
        token = os.environ.get('DBMF_API_TOKEN', '')
        if len(token) < 32:
            raise RuntimeError('DBMF_API_TOKEN must contain at least 32 characters')
        app.state.token = token
        app.state.model = Recommender(os.environ['DBMF_MODEL_DIR'])
        yield

    app = FastAPI(title='Course recommendation', lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get('/health')
    def health():
        return {'status': 'ready'}

    @app.post('/recommend')
    def recommend(request: Request, authorization: str = Header(default='')):
        supplied = authorization.removeprefix('Bearer ')
        if not authorization.startswith('Bearer ') or not hmac.compare_digest(supplied.encode(), app.state.token.encode()):
            raise HTTPException(status_code=401, detail='Unauthorized')
        try:
            return app.state.model.recommend(request.profile.model_dump(), request.k, request.eligible, request.excluded)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return app
