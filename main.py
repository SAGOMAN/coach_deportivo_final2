# -*- coding: utf-8 -*-
"""Compatibilidad: la app vive en app.main.

  uvicorn app.main:app --reload --port 2000
  uvicorn main:app --reload --port 2000
"""
from app.main import app

__all__ = ["app"]
