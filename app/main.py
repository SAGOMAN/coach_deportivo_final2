# -*- coding: utf-8 -*-
"""
FastAPI: HTML/JS + predicción sentadillas.
Arranque: uvicorn app.main:app --reload --port 2000
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.templating import Jinja2Templates

from model_service import FEATURE_NAMES, predict

_N_FEATURES = len(FEATURE_NAMES)

# Raíz del repo (padre de la carpeta app/)
_BASE = Path(__file__).resolve().parent.parent
app = FastAPI(title="Sentadillas")
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))


class PredictBody(BaseModel):
    features: list[float] = Field(
        ..., min_length=_N_FEATURES, max_length=_N_FEATURES
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"feature_names": FEATURE_NAMES},
    )


@app.get("/api/feature_names")
async def api_feature_names():
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}


@app.post("/api/predict")
async def api_predict(body: PredictBody):
    etiqueta, ok = predict(body.features)
    return {"etiqueta": etiqueta, "ok": ok}
