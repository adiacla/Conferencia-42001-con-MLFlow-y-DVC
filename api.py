"""
api.py - Servicio de inferencia con registro automático (Art. 12), explicabilidad (Art. 13),
supervisión humana (Art. 14) y transparencia hacia la persona afectada (Art. 86).

Ejecutar: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import json
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import gov

warnings.filterwarnings("ignore")

NOMBRES_LEGIBLES = {
    "edad": "Edad", "ingresos_anuales_eur": "Ingresos anuales",
    "deuda_actual_eur": "Deuda actual", "historial_pagos_meses": "Antigüedad del historial de pagos",
    "impagos_previos": "Impagos previos", "zona_postal_vulnerable": "Zona postal",
    "genero_Masculino": "Género",
}
AVISO_IA = ("Esta decisión ha sido apoyada por un sistema de inteligencia artificial. "
            "Tiene derecho a recibir una explicación y a solicitar revisión por una persona.")


class Solicitud(BaseModel):
    id_solicitante: str | None = Field(None, max_length=64)
    edad: int = Field(..., ge=18, le=100)
    genero: Literal["Femenino", "Masculino"]
    ingresos_anuales_eur: float = Field(..., gt=0, le=10_000_000)
    deuda_actual_eur: float = Field(..., ge=0, le=10_000_000)
    historial_pagos_meses: int = Field(..., ge=0, le=900)
    impagos_previos: int = Field(..., ge=0, le=50)
    zona_postal_vulnerable: Literal[0, 1]


class SolicitudRevision(BaseModel):
    motivo: str = Field(..., min_length=3, max_length=500)


class Estado:
    pass


STATE = Estado()


def _log_dir() -> Path:
    d = Path(gov.env("LOG_DIR", str(gov.ROOT / "logs")))
    d.mkdir(parents=True, exist_ok=True)
    return d


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_path = Path(gov.env("MODEL_PATH", str(gov.ROOT / "models/credit_model.joblib")))
    run_json = gov.ROOT / "reports/mlflow_run.json"
    run_meta = json.loads(run_json.read_text(encoding="utf-8"))
    if gov.sha256_file(model_path) != run_meta["model_sha256"]:
        raise RuntimeError("El hash del modelo no coincide con el registrado. Despliegue bloqueado.")
    bundle = joblib.load(model_path)
    STATE.base, STATE.mitigator = bundle["base"], bundle["mitigator"]
    STATE.meta = {**bundle["meta"], "model_sha256": run_meta["model_sha256"]}
    STATE.explainer = shap.TreeExplainer(STATE.base)
    STATE.params = gov.load_params()
    STATE.band = tuple(STATE.params["governance"]["human_review_band"])
    STATE.secret = gov.env("PSEUDONYM_SECRET") or "dev-only-secret"
    yield


app = FastAPI(title="CreditScore-EU API", version=gov.API_VERSION, lifespan=lifespan)


def _auth(x_api_key: str | None):
    esperada = gov.env("API_KEY")
    if esperada and x_api_key != esperada:
        raise HTTPException(status_code=401, detail="API key inválida")


def _write_local(record: dict, fname: str = "predictions.jsonl"):
    with open(_log_dir() / fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _forward_to_n8n(record: dict):
    url = gov.env("N8N_WEBHOOK_URL")
    if not url:
        return
    for intento in range(3):
        try:
            r = requests.post(url, json=record, timeout=5,
                              headers={"x-api-key": gov.env("N8N_API_KEY", "")})
            if r.status_code < 300:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5 * 2 ** intento)
    _write_local(record, "failed_forward.jsonl")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": STATE.meta["model_version"]}


@app.get("/model-info")
def model_info(x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    m = STATE.meta
    return {"model_name": m["model_name"], "model_version": m["model_version"],
            "mlflow_run_id": m["mlflow_run_id"], "dataset_sha256": m["dataset_sha256"],
            "model_sha256": m["model_sha256"], "trained_at": m["trained_at"],
            "risk_level": gov.RISK_LEVEL, "human_review_band": list(STATE.band),
            "intended_use": "Apoyo a la decisión de un analista humano en crédito al consumo."}


@app.post("/predict")
def predict(s: Solicitud, bg: BackgroundTasks, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    t0 = time.perf_counter()
    pid = gov.new_prediction_id()
    seed = int(uuid.UUID(pid).int % (2**31 - 1))

    inputs = s.model_dump(exclude={"id_solicitante"})
    df = pd.DataFrame([inputs])
    X = gov.build_features(df)
    prob = float(STATE.base.predict_proba(X)[0][1])
    decision = int(STATE.mitigator.predict(
        X, sensitive_features=df[STATE.params["governance"]["protected_attribute"]],
        random_state=seed)[0])
    revision = gov.needs_human_review(prob, decision, STATE.band)

    sv = STATE.explainer.shap_values(X)
    if isinstance(sv, list):
        vals = np.asarray(sv[1])[0]
    else:
        vals = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    top = np.argsort(np.abs(vals))[::-1][:3]
    factores = [{"feature": X.columns[i], "etiqueta": NOMBRES_LEGIBLES[X.columns[i]],
                 "impacto": round(float(vals[i]), 4),
                 "efecto": "favorece la aprobación" if vals[i] > 0 else "perjudica la aprobación"}
                for i in top]

    pseudo = gov.pseudonymize(s.id_solicitante or pid, STATE.secret)
    latency = int((time.perf_counter() - t0) * 1000)
    record = gov.build_log_record(
        prediction_id=pid, model_meta=STATE.meta, applicant_pseudo_id=pseudo, inputs=inputs,
        decision=decision, probability=prob, decision_seed=seed, top_factors=factores,
        requires_human_review=revision, latency_ms=latency)

    _write_local(record)
    bg.add_task(_forward_to_n8n, record)

    return {"prediction_id": pid, "decision": record["decision"],
            "probability": record["probability"], "requires_human_review": revision,
            "top_factors": factores, "model_version": record["model_version"],
            "notice": AVISO_IA, "latency_ms": latency}


@app.post("/predictions/{prediction_id}/review-request")
def solicitar_revision(prediction_id: str, body: SolicitudRevision, bg: BackgroundTasks,
                       x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    try:
        uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="prediction_id inválido")
    evento = {"event_type": "human_review_request", "prediction_id": prediction_id,
              "reason": body.motivo, "api_version": gov.API_VERSION,
              "occurred_at": pd.Timestamp.now(tz="UTC").isoformat()}
    evento["payload_sha256"] = gov.canonical_sha256(evento)
    _write_local(evento, "review_requests.jsonl")
    bg.add_task(_forward_to_n8n, evento)
    return {"status": "solicitud registrada", "prediction_id": prediction_id}
