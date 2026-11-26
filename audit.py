"""
audit.py - Etapa 3 CRISP-ML(Q): AUDITORÍA DE CUMPLIMIENTO PREVIA AL DESPLIEGUE.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import requests
import yaml
from mlflow.tracking import MlflowClient

import gov

BLOQUEANTE, INFO = "BLOQUEANTE", "INFO"
RESULTADOS: list[dict] = []


def check(cid, nombre, ok, detalle, iso, ai_act, severidad=BLOQUEANTE):
    RESULTADOS.append({"id": cid, "control": nombre, "resultado": "PASA" if ok else "FALLA",
                       "severidad": severidad, "detalle": detalle,
                       "iso42001": iso, "ai_act": ai_act})


def cargar(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(promote: bool) -> int:
    RESULTADOS.clear()
    p = gov.load_params()
    g = p["governance"]
    metrics = cargar("metrics.json") or {}
    card = cargar("data/data_card.json") or {}
    run = cargar("reports/mlflow_run.json") or {}
    csv = gov.ROOT / p["data"]["path"]

    # C01 Linaje
    lock = Path("dvc.lock")
    if lock.exists() and csv.exists():
        stages = yaml.safe_load(lock.read_text())["stages"]
        outs = {o["path"]: o["md5"] for o in stages["prepare"]["outs"]}
        esperado = outs.get(p["data"]["path"])
        real = gov.md5_file(csv)
        check("C01", "Datos versionados e íntegros (DVC)", esperado == real,
              f"md5 lock={str(esperado)[:10]}… actual={real[:10]}…",
              "A.7.5 Procedencia de datos", "Art. 10, 11, 12")
    else:
        check("C01", "Datos versionados e íntegros (DVC)", False,
              "Falta dvc.lock o el CSV: ejecute `python prepare_data.py`.", "A.7.5", "Art. 10")

    # C02 Calidad de datos
    check("C02", "Contratos de calidad de datos superados", card.get("todos_los_controles_ok", False),
          f"{sum(c['passed'] for c in card.get('controles_calidad', []))}/"
          f"{len(card.get('controles_calidad', []))} controles OK",
          "A.7.4 Calidad de datos", "Art. 10.2 y 10.3")

    # C03 Consistencia modelo <-> datos
    sha_real = gov.sha256_file(csv) if csv.exists() else "NA"
    check("C03", "El modelo se entrenó con el dataset actual",
          run.get("dataset_sha256") == sha_real == card.get("sha256"),
          f"sha256 modelo={str(run.get('dataset_sha256'))[:10]}… dataset={sha_real[:10]}…",
          "A.6.2.7 Documentación técnica", "Art. 11, 12")

    # C04..C07 Equidad y rendimiento
    dp = metrics.get("dp_diff_mitigated", 9)
    check("C04", "Paridad demográfica tras mitigación <= umbral", dp <= g["max_dp_difference"],
          f"DP={dp:.3f} (antes {metrics.get('dp_diff_base', 0):.3f}); umbral {g['max_dp_difference']}",
          "A.5.2 Evaluación de impacto; 8.4", "Art. 10.2.f")
    acc = metrics.get("accuracy_mitigated", 0)
    check("C05", "Exactitud >= mínimo declarado", acc >= g["min_accuracy"],
          f"exactitud={acc:.3f}; mínimo {g['min_accuracy']}", "A.6.2.4 Verificación y validación",
          "Art. 15")
    drop = metrics.get("accuracy_drop_mitigation", 9)
    check("C06", "Coste de equidad aceptable", drop <= g["max_accuracy_drop_mitigation"],
          f"caída de exactitud={drop:.3f}; máximo {g['max_accuracy_drop_mitigation']}",
          "8.3 Tratamiento del riesgo", "Art. 9")
    check("C07", "Equalized odds (informativo, no bloquea)",
          metrics.get("eo_diff_mitigated", 9) <= 0.25,
          f"EO={metrics.get('eo_diff_mitigated', 0):.3f}. Criterio alternativo de equidad.",
          "9.1 Seguimiento", "Art. 10.2.f", INFO)

    check("C17", "Decisiones cambiadas por el mitigador (informativo)",
          metrics.get("override_rate", 1) <= 0.25,
          f"{metrics.get('override_rate', 0):.1%} de decisiones difieren del modelo base",
          "9.1 Seguimiento", "Art. 14, 15", INFO)

    # C08..C09 MLflow
    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    client = MlflowClient()
    name, ver = p["mlflow"]["registered_model"], str(run.get("model_version"))
    try:
        mv = client.get_model_version(name, ver)
        req = ["compliance.ai_act.risk_level", "compliance.iso42001.controls",
               "compliance.dataset_sha256", "compliance.git_commit", "compliance.model_sha256"]
        faltan = [t for t in req if t not in mv.tags]
        sig = mlflow.models.get_model_info(f"models:/{name}/{ver}").signature is not None
        check("C08", "Modelo registrado con tags de cumplimiento y firma",
              not faltan and sig and mv.run_id == run.get("mlflow_run_id"),
              f"faltan tags={faltan}; firma={'sí' if sig else 'no'}; run={mv.run_id[:8]}…",
              "A.6.2.7; A.6.2.8", "Art. 11, 12, 17")
        sha_modelo = gov.sha256_file("models/credit_model.joblib")
        check("C09", "Integridad del artefacto del modelo",
              sha_modelo == mv.tags.get("compliance.model_sha256") == run.get("model_sha256"),
              f"sha256 archivo={sha_modelo[:10]}… registro={mv.tags.get('compliance.model_sha256', '')[:10]}…",
              "A.6.2.5 Despliegue", "Art. 15 (ciberseguridad)")
    except Exception as e:
        mv = None
        check("C08", "Modelo registrado con tags de cumplimiento y firma", False, f"Error: {e}",
              "A.6.2.7", "Art. 11")

    # C10..C11 Documentación y explicabilidad
    card_md = Path("models/model_card.md")
    secciones = ["Propósito previsto", "Datos de entrenamiento", "Rendimiento y equidad",
                 "Explicabilidad", "Supervisión humana", "Registro de eventos", "Riesgos residuales"]
    txt = card_md.read_text(encoding="utf-8") if card_md.exists() else ""
    falta = [s for s in secciones if s not in txt]
    check("C10", "Model card completa (Anexo IV simplificado)", not falta,
          f"secciones faltantes={falta}", "A.6.2.7; A.8.2", "Art. 11, Anexo IV")
    check("C11", "Evidencia de explicabilidad global (SHAP)", Path("reports/shap_global.png").exists(),
          "reports/shap_global.png", "A.8.2 Información a usuarios", "Art. 13")

    # C12 Clasificación de riesgo
    r = gov.clasificar_riesgo_eu_ai_act({"sector": "credito_solvencia"})
    check("C12", "Clasificación de riesgo documentada y coherente",
          r["nivel"].startswith("ALTO RIESGO") and gov.RISK_LEVEL.startswith("ALTO"),
          r["nivel"], "6.1.2 Evaluación de riesgos de IA", "Art. 6, Anexo III 5.b")

    # C13 Retención de logs
    check("C13", "Retención de logs >= 6 meses", g["log_retention_days"] >= 183,
          f"{g['log_retention_days']} días", "A.6.2.8 Registro de eventos", "Art. 19, 26.6")

    # C14..C16 Prueba funcional end-to-end
    prev_log_dir = os.environ.get("LOG_DIR")
    try:
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["LOG_DIR"] = tmp
            import api
            casos = [
                {"edad": 45, "genero": "Masculino", "ingresos_anuales_eur": 80000,
                 "deuda_actual_eur": 2000, "historial_pagos_meses": 100, "impagos_previos": 0,
                 "zona_postal_vulnerable": 0},
                {"edad": 30, "genero": "Femenino", "ingresos_anuales_eur": 16000,
                 "deuda_actual_eur": 20000, "historial_pagos_meses": 6, "impagos_previos": 3,
                 "zona_postal_vulnerable": 1},
                {"edad": 30, "genero": "Masculino", "ingresos_anuales_eur": 16000,
                 "deuda_actual_eur": 20000, "historial_pagos_meses": 6, "impagos_previos": 3,
                 "zona_postal_vulnerable": 1},
                {"edad": 38, "genero": "Femenino", "ingresos_anuales_eur": 38000,
                 "deuda_actual_eur": 12000, "historial_pagos_meses": 40, "impagos_previos": 1,
                 "zona_postal_vulnerable": 0},
            ]
            with TestClient(api.app) as c:
                resp = [c.post("/predict", json=x) for x in casos]
                invalida = c.post("/predict", json={**casos[0], "edad": 17})
            lineas = [json.loads(l) for l in open(Path(tmp) / "predictions.jsonl", encoding="utf-8")]
        campos_ok = all(all(k in rec for k in gov.REQUIRED_LOG_FIELDS) for rec in lineas)
        hash_ok = all(gov.verify_record_hash(rec) for rec in lineas)
        def esperado(r):
            d, pr = int(r["decision"] == "APROBADO"), r["probability"]
            return (0.40 <= pr <= 0.60) or d == 0
        hubo_denegacion = any(r["decision"] == "DENEGADO" for r in lineas)
        regla_ok = hubo_denegacion and all(r["requires_human_review"] == esperado(r) for r in lineas)
        check("C14", "Logging automático completo e íntegro por predicción",
              len(lineas) == len(casos) and campos_ok and hash_ok and all(r.status_code == 200 for r in resp),
              f"{len(lineas)}/{len(casos)} registros; campos={campos_ok}; hash={hash_ok}",
              "A.6.2.8 Registro de eventos", "Art. 12")
        check("C15", "Supervisión humana: denegaciones y zona gris escalan", regla_ok,
              ", ".join(f"{r['decision']}(p={r['probability']})/rev={r['requires_human_review']}" for r in lineas),
              "A.9.2 Uso responsable", "Art. 14")
        check("C16", "Validación de entradas (rechaza edad < 18)", invalida.status_code == 422,
              f"HTTP {invalida.status_code}", "A.6.2.4", "Art. 15")
    except Exception as e:
        check("C14", "Logging automático completo e íntegro por predicción", False, f"Error: {e}",
              "A.6.2.8", "Art. 12")
    finally:
        if prev_log_dir is None:
            os.environ.pop("LOG_DIR", None)
        else:
            os.environ["LOG_DIR"] = prev_log_dir

    bloqueantes = [r for r in RESULTADOS if r["severidad"] == BLOQUEANTE]
    aprobado = all(r["resultado"] == "PASA" for r in bloqueantes)
    informe = {"modelo": gov.MODEL_NAME, "version": run.get("model_version"),
               "run_id": run.get("mlflow_run_id"), "git_commit": gov.git_commit(),
               "fecha_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
               "aprobado": aprobado, "controles": RESULTADOS}
    informe["report_sha256"] = gov.canonical_sha256(informe)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/audit_report.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    md = [f"# Informe de auditoría - {informe['modelo']} v{informe['version']}",
          f"\n**Resultado: {'APROBADO' if aprobado else 'RECHAZADO'}** · {informe['fecha_utc']} · "
          f"commit `{informe['git_commit']}`\n",
          "| ID | Control | Resultado | Tipo | Detalle | ISO 42001 | AI Act |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['control']} | {r['resultado']} | {r['severidad']} | {r['detalle']} | "
           f"{r['iso42001']} | {r['ai_act']} |" for r in RESULTADOS]
    Path("reports/audit_report.md").write_text("\n".join(md), encoding="utf-8")

    for r in RESULTADOS:
        print(f"[{r['resultado']:5s}] {r['id']} {r['control']} -> {r['detalle']}")
    print("\n>>> AUDITORÍA:", "APROBADA" if aprobado else "RECHAZADA")

    if mv is not None:
        estado = "PASSED" if aprobado else "FAILED"
        client.set_model_version_tag(name, ver, "compliance.status", estado)
        client.set_model_version_tag(name, ver, "compliance.audit_report_sha256", informe["report_sha256"])
        with mlflow.start_run(run_id=run["mlflow_run_id"]):
            mlflow.set_tag("compliance.status", estado)
            mlflow.log_artifact("reports/audit_report.json", artifact_path="governance")
            mlflow.log_artifact("reports/audit_report.md", artifact_path="governance")
        if aprobado and promote:
            client.set_registered_model_alias(name, "production", ver)
            print(f"Alias `production` -> {name} v{ver}")
        elif promote:
            print("PROMOCIÓN BLOQUEADA: la auditoría no pasó. El alias `production` no se modifica.")

    return 0 if aprobado else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    sys.exit(main(ap.parse_args().promote))
