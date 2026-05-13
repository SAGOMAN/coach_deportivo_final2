# -*- coding: utf-8 -*-
"""
Mismo pipeline que contexto_analisis/modelosupervisado2.py sobre df_processed.csv:
solo las columnas listadas allí + label_encoded, test_size=0.3, RF igual.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent
_CSV = _ROOT / "contexto_analisis" / "df_processed.csv"

# Debe coincidir con contexto_analisis/modelosupervisado2.py (orden = orden de X)
COLUMNAS_DESEADAS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "trunk_inclination",
    "knee_angle_mean",
    "label_encoded",
]

_df = pd.read_csv(_CSV)
_faltan = set(COLUMNAS_DESEADAS) - set(_df.columns)
if _faltan:
    raise ValueError(f"df_processed.csv no tiene columnas: {_faltan}")
_df = _df[COLUMNAS_DESEADAS]

_X = _df.drop("label_encoded", axis=1)
_y = _df["label_encoded"]
FEATURE_NAMES: list[str] = list(_X.columns)

_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.3, stratify=_y, random_state=42
)
_scaler = StandardScaler()
_X_train_s = _scaler.fit_transform(_X_train)
_X_test_s = _scaler.transform(_X_test)
_rf = RandomForestClassifier(n_estimators=250, criterion="entropy", random_state=42)
_rf.fit(_X_train_s, _y_train)


def predict(features: list[float]) -> tuple[str, bool]:
    """Devuelve (etiqueta, ok) con ok=True si Correcto (clase 0)."""
    arr = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Se esperaban {len(FEATURE_NAMES)} valores, llegaron {arr.shape[1]}"
        )
    df = pd.DataFrame(arr, columns=FEATURE_NAMES)
    scaled = _scaler.transform(df)
    pred = int(_rf.predict(scaled)[0])
    etiqueta = "Correcto" if pred == 0 else "Incorrecto"
    return etiqueta, pred == 0
