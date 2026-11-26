"""
gov.py - Utilidades compartidas de gobernanza (ISO/IEC 42001 + EU AI Act).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "CreditScore-EU"
RISK_LEVEL = "ALTO_RIESGO_ANEXO_III_5b"
AI_ACT_ARTICLES = "9,10,11,12,13,14,15,17,72"
ISO42001_CONTROLS = (
    "6.1.2,8.2,8.4,9.1,9.2,A.5.2,A.6.2.4,A.6.2.6,A.6.2.7,A.6.2.8,"
    "A.7.4,A.7.5,A.8.2,A.9.2"
)
API_VERSION = "1.0.0"

FEATURE_COLUMNS = [
    "edad",
    "ingresos_anuales_eur",
    "deuda_actual_eur",
    "historial_pagos_meses",
    "impagos_previos",
    "zona_postal_vulnerable",
    "genero_Masculino",
]

REQUIRED_LOG_FIELDS = [
    "event_type", "prediction_id", "occurred_at", "model_name", "model_version",
    "mlflow_run_id", "dataset_sha256", "git_commit", "applicant_pseudo_id",
    "inputs", "decision", "probability", "decision_seed", "top_factors",
    "requires_human_review", "latency_ms", "api_version", "payload_sha256",
]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Dataset sintético de crédito con sesgo de género INTENCIONADO (didáctico)."""
    rng = np.random.RandomState(seed)
    edad = rng.randint(21, 68, size=n)
    genero = rng.choice(["Femenino", "Masculino"], size=n, p=[0.48, 0.52])
    ingresos = np.round(rng.normal(38000, 14000, size=n).clip(14000, 95000), 2)
    deuda = np.round(rng.uniform(0, 25000, size=n), 2)
    historial = rng.randint(0, 120, size=n)
    impagos = rng.choice([0, 1, 2, 3], size=n, p=[0.65, 0.20, 0.10, 0.05])
    zona = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    ratio = deuda / ingresos
    score = (ingresos / 10000) * 0.85 - ratio * 3.6 + (historial / 24) * 0.65 - impagos * 1.45
    sesgo = np.where(genero == "Femenino", -2.35, 0.0)
    prob = 1 / (1 + np.exp(-(score + sesgo + rng.normal(0, 0.45, size=n) - 0.5)))
    return pd.DataFrame({
        "id_solicitante": [f"SOL-{1000 + i}" for i in range(n)],
        "edad": edad, "genero": genero, "ingresos_anuales_eur": ingresos,
        "deuda_actual_eur": deuda, "historial_pagos_meses": historial,
        "impagos_previos": impagos, "zona_postal_vulnerable": zona,
        "credito_aprobado": (prob >= 0.5).astype(int),
    })


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quality_checks(df: pd.DataFrame) -> list[dict]:
    """Contratos de calidad de datos (Art. 10.2 y 10.3 / ISO 42001 A.7.4)."""
    def chk(name, passed, detail):
        return {"name": name, "passed": bool(passed), "detail": detail}

    g = df["genero"].value_counts(normalize=True)
    return [
        chk("sin_nulos", df.isnull().sum().sum() == 0, f"nulos={int(df.isnull().sum().sum())}"),
        chk("sin_duplicados_id", df["id_solicitante"].is_unique, "id_solicitante único"),
        chk("mayores_de_edad", (df["edad"] >= 18).all(), f"edad_min={int(df['edad'].min())}"),
        chk("ingresos_positivos", (df["ingresos_anuales_eur"] > 0).all(), "ingresos > 0"),
        chk("deuda_no_negativa", (df["deuda_actual_eur"] >= 0).all(), "deuda >= 0"),
        chk("etiqueta_binaria", set(df["credito_aprobado"].unique()) <= {0, 1}, "etiqueta ∈ {0,1}"),
        chk("representatividad_genero", g.min() >= 0.30,
            f"proporción mínima por género={g.min():.2f} (umbral 0.30)"),
        chk("tamano_minimo", len(df) >= 500, f"filas={len(df)} (umbral 500)"),
    ]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "edad": df["edad"], "ingresos_anuales_eur": df["ingresos_anuales_eur"],
        "deuda_actual_eur": df["deuda_actual_eur"],
        "historial_pagos_meses": df["historial_pagos_meses"],
        "impagos_previos": df["impagos_previos"],
        "zona_postal_vulnerable": df["zona_postal_vulnerable"],
        "genero_Masculino": (df["genero"] == "Masculino").astype(int),
    })
    return out[FEATURE_COLUMNS]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "taller-v1.0"


def canonical_sha256(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pseudonymize(value: str, secret: str) -> str:
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:32]


def needs_human_review(prob: float, decision: int, band: tuple[float, float]) -> bool:
    return (band[0] <= prob <= band[1]) or decision == 0


def build_log_record(*, prediction_id: str, model_meta: dict, applicant_pseudo_id: str,
                     inputs: dict, decision: int, probability: float, decision_seed: int,
                     top_factors: list[dict], requires_human_review: bool,
                     latency_ms: int) -> dict:
    record = {
        "event_type": "prediction",
        "prediction_id": prediction_id,
        "occurred_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": model_meta.get("model_name", MODEL_NAME),
        "model_version": str(model_meta.get("model_version", "desconocida")),
        "mlflow_run_id": model_meta.get("mlflow_run_id", "desconocido"),
        "dataset_sha256": model_meta.get("dataset_sha256", "desconocido"),
        "git_commit": model_meta.get("git_commit", "sin-git"),
        "applicant_pseudo_id": applicant_pseudo_id,
        "inputs": inputs,
        "decision": "APROBADO" if decision == 1 else "DENEGADO",
        "probability": round(float(probability), 4),
        "decision_seed": int(decision_seed),
        "top_factors": top_factors,
        "requires_human_review": bool(requires_human_review),
        "latency_ms": int(latency_ms),
        "api_version": API_VERSION,
    }
    record["payload_sha256"] = canonical_sha256(record)
    return record


def verify_record_hash(record: dict) -> bool:
    body = {k: v for k, v in record.items() if k != "payload_sha256"}
    return canonical_sha256(body) == record.get("payload_sha256")


def new_prediction_id() -> str:
    return str(uuid.uuid4())


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


SECTORES_ANEXO_III = {
    "credito_solvencia", "seleccion_personal_rrhh", "educacion_admision",
    "salud_dispositivos", "infraestructura_critica", "justicia_policia", "migracion_fronteras",
}


def clasificar_riesgo_eu_ai_act(caso_uso: dict) -> dict:
    if (caso_uso.get("puntuacion_social_ciudadana") or caso_uso.get("manipulacion_subliminal")
            or caso_uso.get("identificacion_biometrica_remota_tiempo_real_publico")):
        return {"nivel": "RIESGO INACEPTABLE (PROHIBIDO - Art. 5)",
                "accion": "Despliegue ilegal en la UE. Detener proyecto.",
                "requisitos_tecnicos": ["Ninguno: práctica prohibida."]}
    if caso_uso.get("sector") in SECTORES_ANEXO_III or caso_uso.get("componente_seguridad_producto_ue"):
        return {"nivel": "ALTO RIESGO (Art. 6 / Anexo III)",
                "accion": "Permitido sujeto a evaluación de conformidad y controles estrictos.",
                "requisitos_tecnicos": [
                    "Art. 9: Sistema de gestión de riesgos continuo",
                    "Art. 10: Gobernanza de datos y auditoría de sesgos (Fairlearn/AIF360)",
                    "Art. 11-12: Documentación técnica (Anexo IV) y registro automático de logs",
                    "Art. 13: Transparencia y explicabilidad (SHAP/LIME)",
                    "Art. 14: Supervisión humana efectiva (Human-in-the-loop)",
                    "Art. 15: Precisión, solidez y ciberseguridad"]}
    if caso_uso.get("interaccion_chatbot") or caso_uso.get("genera_contenido_sintetico_deepfake"):
        return {"nivel": "RIESGO LIMITADO (Obligación de Transparencia - Art. 50)",
                "accion": "Permitido informando al usuario y marcando contenido sintético.",
                "requisitos_tecnicos": ["Aviso claro de que se interactúa con una IA",
                                        "Marcado legible por máquina (C2PA / marcas de agua)"]}
    return {"nivel": "RIESGO MÍNIMO O NULO",
            "accion": "Uso libre sin obligaciones adicionales (se recomiendan códigos de conducta).",
            "requisitos_tecnicos": ["Buenas prácticas estándar de ingeniería de software."]}
