"""
app_streamlit.py - Interfaz Web Unificada para el Taller MLOps CRISP-ML(Q) & EU AI Act.

Ejecutar: streamlit run app_streamlit.py
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

import gov

st.set_page_config(
    page_title="FlexiScore-UNAB | Gobernanza MLOps (CRISP-ML(Q))",
    page_icon="🛡️",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"


def api_disponible() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def predecir_directo_o_api(payload: dict) -> dict:
    """Usa la API FastAPI si está levantada; si no, ejecuta el bundle validando SHA-256."""
    if api_disponible():
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        r.raise_for_status()
        res = r.json()
        res["modo_ejecucion"] = "API FastAPI en Producción (http://localhost:8000)"
        return res

    # Fallback local con idéntica gobernanza (por si solo se lanzó Streamlit)
    import time, uuid
    import numpy as np
    import shap

    model_path = Path("models/credit_model.joblib")
    run_meta = json.loads(Path("reports/mlflow_run.json").read_text(encoding="utf-8"))
    if gov.sha256_file(model_path) != run_meta["model_sha256"]:
        raise RuntimeError("¡ALERTA DE SEGURIDAD! El hash SHA-256 del modelo no coincide.")

    bundle = joblib.load(model_path)
    base, mitigator = bundle["base"], bundle["mitigator"]
    params = gov.load_params()
    band = tuple(params["governance"]["human_review_band"])

    t0 = time.perf_counter()
    pid = gov.new_prediction_id()
    seed = int(uuid.UUID(pid).int % (2**31 - 1))
    inputs = {k: v for k, v in payload.items() if k != "id_solicitante"}
    df = pd.DataFrame([inputs])
    X = gov.build_features(df)
    prob = float(base.predict_proba(X)[0][1])
    decision = int(mitigator.predict(X, sensitive_features=df[params["governance"]["protected_attribute"]], random_state=seed)[0])
    revision = gov.needs_human_review(prob, decision, band)

    explainer = shap.TreeExplainer(base)
    sv = explainer.shap_values(X)
    vals = np.asarray(sv[1])[0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
    nombres = {
        "edad": "Edad", "ingresos_anuales_eur": "Ingresos anuales",
        "deuda_actual_eur": "Deuda actual", "historial_pagos_meses": "Antigüedad historial pagos",
        "impagos_previos": "Impagos previos", "zona_postal_vulnerable": "Zona postal",
        "genero_Masculino": "Género",
    }
    top = np.argsort(np.abs(vals))[::-1][:3]
    factores = [{"feature": X.columns[i], "etiqueta": nombres[X.columns[i]],
                 "impacto": round(float(vals[i]), 4),
                 "efecto": "favorece la aprobación" if vals[i] > 0 else "perjudica la aprobación"} for i in top]

    pseudo = gov.pseudonymize(payload.get("id_solicitante") or pid, "dev-only-secret")
    latency = int((time.perf_counter() - t0) * 1000)
    record = gov.build_log_record(
        prediction_id=pid, model_meta={**bundle["meta"], "model_sha256": run_meta["model_sha256"]},
        applicant_pseudo_id=pseudo, inputs=inputs, decision=decision, probability=prob,
        decision_seed=seed, top_factors=factores, requires_human_review=revision, latency_ms=latency
    )
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "predictions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "prediction_id": pid,
        "decision": record["decision"],
        "probability": record["probability"],
        "requires_human_review": revision,
        "top_factors": factores,
        "model_version": record["model_version"],
        "notice": "Esta decisión ha sido apoyada por un sistema de IA. Tiene derecho a recibir una explicación y solicitar revisión humana (Art. 86 EU AI Act).",
        "latency_ms": latency,
        "modo_ejecucion": "Motor Local Gobernado (SHA-256 Verificado)"
    }


# --- HEADER ---
st.title("🛡️ CreditScore-EU — Sistema de IA de Alto Riesgo Gobernado")
st.caption("Marco Metodológico: **CRISP-ML(Q)** | Cumplimiento: **Reglamento (UE) 2024/1689 (EU AI Act)** & **ISO/IEC 42001**")

with st.expander("📘 Guía Rápida de Uso de la Plataforma (Taller MLOps & Gobernanza)", expanded=False):
    st.markdown("""
    **¿Cómo explorar este sistema de IA de Alto Riesgo (CRISP-ML(Q) + EU AI Act + ISO/IEC 42001)?**
    1. **Producción (Pestaña 1):** Ingresa los datos de un solicitante y pulsa **Evaluar Solicitud**. Observa la decisión, la explicación local (**SHAP - Art. 13**), el aviso de transparencia (**Art. 86**) y si se activa la **Revisión Humana Obligatoria (Art. 14)**.
    2. **Auditoría Pre-Despliegue (Pestaña 2):** Ejecuta los **17 controles automatizados** (Quality Gate) que verifican equidad, exactitud, firma criptográfica `SHA-256` e integridad de datos antes de permitir el paso a producción.
    3. **Datos, EDA y Model Card (Pestaña 3):** Inspecciona el análisis exploratorio (sesgo histórico), los 8 contratos de calidad de datos, la *Model Card* (Anexo IV) y el historial de versiones en **Git + DVC**.
    4. **Monitoreo en Vivo (Pestaña 4):** Audita el registro inmutable de predicciones (`predictions.jsonl`), verifica la seudonimización (RGPD) y comprueba la integridad de los hashes `SHA-256`.
    """)
    
# Sidebar con estado del sistema y control humano
with st.sidebar:
    st.header("Estado de Gobernanza")
    modelo_ok = Path("models/credit_model.joblib").exists() and Path("reports/mlflow_run.json").exists()
    if modelo_ok:
        meta = json.loads(Path("reports/mlflow_run.json").read_text(encoding="utf-8"))
        st.success(f"Modelo Activo: **{meta['model_name']} v{meta['model_version']}**")
        st.code(f"SHA-256 Modelo:\n{meta['model_sha256'][:24]}...", language="text")
        st.code(f"SHA-256 Datos:\n{meta['dataset_sha256'][:24]}...", language="text")
    else:
        st.warning("Modelo no entrenado aún. Ejecuta `prepare_data.py` y `train.py` en VS Code.")

    if api_disponible():
        st.info("🟢 API FastAPI conectada (:8000)")
    else:
        st.warning("🟡 API FastAPI apagada (usando motor local verificado)")

    st.divider()
    st.markdown("**¿Quién controla el sistema?**")
    st.markdown(
        "- **Nivel Técnico:** Puerta de Auditoría (`audit.py`) + Hash SHA-256."
        "- **Nivel Operativo:** Analista de Riesgos (Supervisión Humana Art. 14)."
        "- **Nivel Ciudadano:** Derecho a explicación y apelación (Art. 86)."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Producción: Predicción y Supervisión Humana",
    "2️⃣ Auditoría Pre-Despliegue (Quality Gate)",
    "3️⃣ Datos, EDA y Model Card",
    "4️⃣ Monitoreo en Vivo (Logs Inmutables)"
])

# ==============================================================================
# TAB 1: INFERENCIA Y SUPERVISIÓN HUMANA
# ==============================================================================
with tab1:
    st.subheader("Simulador de Evaluación Crediticia con Explicabilidad (Art. 13 y 14)")
    if not modelo_ok:
        st.error("Primero debes entrenar el modelo ejecutando `python prepare_data.py` y `python train.py`.")
    else:
        col_form, col_res = st.columns([1, 1.2], gap="large")
        with col_form:
            st.markdown("#### Datos del Solicitante")
            id_sol = st.text_input("ID Solicitante (se seudonimizará según RGPD)", value="CLI-2026-089")
            c1, c2 = st.columns(2)
            edad = c1.number_input("Edad", min_value=18, max_value=100, value=34)
            genero = c2.selectbox("Género (Atributo protegido)", ["Femenino", "Masculino"])
            ingresos = c1.number_input("Ingresos anuales (€)", min_value=1000.0, max_value=500000.0, value=32000.0, step=1000.0)
            deuda = c2.number_input("Deuda actual (€)", min_value=0.0, max_value=500000.0, value=14000.0, step=500.0)
            historial = c1.slider("Historial de pagos (meses)", 0, 120, 24)
            impagos = c2.selectbox("Impagos previos", [0, 1, 2, 3], index=1)
            zona = st.selectbox("Zona postal vulnerable", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Sí (1)")

            evaluar = st.button("Evaluar Solicitud con IA Gobernada", type="primary", width='stretch')

        with col_res:
            st.markdown("#### Resultado, Explicabilidad y Control Humano")
            if evaluar:
                payload = {
                    "id_solicitante": id_sol,
                    "edad": int(edad),
                    "genero": genero,
                    "ingresos_anuales_eur": float(ingresos),
                    "deuda_actual_eur": float(deuda),
                    "historial_pagos_meses": int(historial),
                    "impagos_previos": int(impagos),
                    "zona_postal_vulnerable": int(zona),
                }
                res = predecir_directo_o_api(payload)
                st.session_state["ultimo_resultado"] = res

            if "ultimo_resultado" in st.session_state:
                res = st.session_state["ultimo_resultado"]
                st.caption(f"Modo: {res['modo_ejecucion']} | Latencia: {res['latency_ms']} ms | ID: `{res['prediction_id']}`")

                m1, m2, m3 = st.columns(3)
                if res["decision"] == "APROBADO":
                    m1.success(f"**DECISIÓN:** {res['decision']}")
                else:
                    m1.error(f"**DECISIÓN:** {res['decision']}")
                m2.metric("Probabilidad Base", f"{res['probability']:.1%}")

                if res["requires_human_review"]:
                    m3.warning("⚠️ **REVISIÓN HUMANA OBLIGATORIA**")
                else:
                    m3.info("✅ Flujo Estándar")

                if res["requires_human_review"]:
                    st.warning(
                        "**Control Art. 14 EU AI Act activado:** Esta solicitud se ha escalado a la bandeja de un "
                        "analista humano porque la decisión es desfavorable o cae en la banda de incertidumbre (40%-60%)."
                    )

                st.markdown("**Explicabilidad Local — Top 3 Factores SHAP (Art. 13):**")
                df_shap = pd.DataFrame(res["top_factors"])
                fig, ax = plt.subplots(figsize=(6, 2.3))
                colors = ["#2E8B57" if x > 0 else "#D9534F" for x in df_shap["impacto"]]
                ax.barh(df_shap["etiqueta"][::-1], df_shap["impacto"][::-1], color=colors[::-1])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Impacto SHAP en la probabilidad de aprobación")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.info(f"📢 **Transparencia al Ciudadano (Art. 86):** {res['notice']}")

                # Formulario de apelación / revisión humana
                with st.expander("🧑‍⚖️ Canal de Apelación / Solicitud de Revisión Humana (Art. 86)"):
                    motivo = st.text_input("Motivo de la revisión:", value="Solicito revisión manual de mi capacidad de pago actual.")
                    if st.button("Registrar Solicitud de Revisión"):
                        if api_disponible():
                            r_rev = requests.post(f"{API_URL}/predictions/{res['prediction_id']}/review-request", json={"motivo": motivo})
                            st.success(f"Registrado vía API: {r_rev.json()}")
                        else:
                            ev = {
                                "event_type": "human_review_request",
                                "prediction_id": res["prediction_id"],
                                "reason": motivo,
                                "occurred_at": pd.Timestamp.now(tz="UTC").isoformat(),
                            }
                            ev["payload_sha256"] = gov.canonical_sha256(ev)
                            Path("logs").mkdir(exist_ok=True)
                            with open("logs/review_requests.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                            st.success("Solicitud de revisión humana guardada con hash de integridad en `logs/review_requests.jsonl`.")

# ==============================================================================
# TAB 2: AUDITORÍA PRE-DESPLIEGUE (QUALITY GATE)
# ==============================================================================
with tab2:
    st.subheader("Auditoría Automatizada Pre-Despliegue (ISO/IEC 42001 & EU AI Act)")
    st.markdown(
        "Esta puerta de calidad (**Quality Gate**) verifica 17 controles técnicos antes de permitir que el modelo "
        "obtenga el alias `production` en MLflow."
    )

    c_btn1, c_btn2 = st.columns([1, 1])
    run_audit = c_btn1.button("🔍 Ejecutar Auditoría de Cumplimiento", type="primary", width='stretch')
    run_promote = c_btn2.button("🚀 Auditar y Promover a Producción (--promote)", width='stretch')

    if run_audit or run_promote:
        cmd = [sys.executable, "audit.py"] + (["--promote"] if run_promote else [])
        with st.spinner("Ejecutando 17 controles de gobernanza y pruebas end-to-end..."):
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            st.success("✅ AUDITORÍA APROBADA: Todos los controles bloqueantes han pasado.")
        else:
            st.error("❌ AUDITORÍA RECHAZADA: Se bloquea el paso a producción.")
            if proc.stderr:
                st.code(proc.stderr)

    rep_path = Path("reports/audit_report.json")
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estado Global", "APROBADO" if rep["aprobado"] else "RECHAZADO")
        k2.metric("Versión Modelo", f"v{rep['version']}")
        pasados = sum(1 for c in rep["controles"] if c["resultado"] == "PASA")
        k3.metric("Controles Superados", f"{pasados} / {len(rep['controles'])}")
        k4.metric("Commit Git", rep["git_commit"])

        df_ctrl = pd.DataFrame(rep["controles"])
        st.dataframe(
            df_ctrl[["id", "control", "resultado", "severidad", "detalle", "iso42001", "ai_act"]], width='stretch' ,
            hide_index=True,
        )
    else:
        st.info("Pulsa uno de los botones superiores para ejecutar la auditoría y ver el informe.")

# ==============================================================================
# TAB 3: DATOS, EDA Y MODEL CARD
# ==============================================================================
with tab3:
    st.subheader("Exploración de Datos (EDA), Calidad (Art. 10) y Model Card (Art. 11)")
    card_path = Path("data/data_card.json")
    if card_path.exists():
        dcard = json.loads(card_path.read_text(encoding="utf-8"))
        st.markdown("#### 1. Resultados del EDA y Detección Temprana de Sesgo")
        eda = dcard.get("eda", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Registros", eda.get("n_filas", 0))
        c2.metric("Brecha Histórica Género", f"{eda.get('brecha_aprobacion_historica', 0):.1%}")
        c3.metric("Contratos de Calidad", "8 / 8 PASS" if dcard.get("todos_los_controles_ok") else "FALLA")

        if Path("reports/eda_resumen.png").exists():
            st.image("reports/eda_resumen.png", caption="Análisis Exploratorio de Datos (EDA): Sesgo histórico y distribución de ingresos")

        with st.expander("Ver detalle de los 8 Contratos de Calidad de Datos (data_card.json)"):
            st.dataframe(pd.DataFrame(dcard["controles_calidad"]), width='stretch', hide_index=True)

    st.divider()
    col_mc, col_sh = st.columns([1.3, 1])
    with col_mc:
        st.markdown("#### 2. Model Card Generada Automáticamente (Anexo IV)")
        if Path("models/model_card.md").exists():
            st.markdown(Path("models/model_card.md").read_text(encoding="utf-8"))

    with col_sh:
            st.markdown("#### 3. Explicabilidad Global (SHAP)")
            if Path("reports/shap_global.png").exists():
                st.image("reports/shap_global.png", caption="Importancia global de variables (SHAP)")

        # --- PEGAR DESDE AQUÍ (4 espacios de sangría, alineado con 'with col_sh:') ---
    st.divider()
    st.markdown("#### 4. Historial de Versiones de Datos Auditadas (Git + DVC)")
    try:
        log_dvc = subprocess.check_output(
            ["git", "log", "-n", "5", "--pretty=format:%h | %ad | %d %s", "--date=short", "dvc.lock"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        st.code(log_dvc if log_dvc else "Sin historial de commits en dvc.lock aún.", language="text")
    except Exception:
        st.info("Inicializa Git y DVC (`git init && dvc init`) para visualizar el historial de versiones de datos.")
# ==============================================================================
# TAB 4: MONITOREO EN VIVO (LOGS)
# ==============================================================================
with tab4:
    st.subheader("Trazabilidad y Monitoreo Continuo en Producción (Art. 12)")
    log_file = Path("logs/predictions.jsonl")
    if log_file.exists():
        registros = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if registros:
            df_logs = pd.DataFrame(registros)
            integridad = all(gov.verify_record_hash(r) for r in registros)
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicciones Registradas", len(df_logs))
            m2.metric("Escaladas a Revisión Humana", int(df_logs["requires_human_review"].sum()))
            m3.metric("Integridad Criptográfica Logs", "✅ ÍNTEGROS (SHA-256)" if integridad else "❌ ALTERADOS")

            cols_mostrar = ["occurred_at", "prediction_id", "applicant_pseudo_id", "decision", "probability", "requires_human_review", "model_version"]
            st.dataframe(df_logs[cols_mostrar].iloc[::-1], width='stretch', hide_index=True)

            with st.expander("Ver último registro JSON completo (evidencia de auditoría Art. 12)"):
                st.json(registros[-1])
    else:
        st.info("Aún no hay predicciones registradas. Ve a la Pestaña 1 y evalúa una solicitud.")

st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.88rem; padding: 10px 0 20px 0;">
        <strong>Aviso Académico:</strong> Este sistema constituye un ejercicio académico experimental desarrollado con fines formativos y de investigación en ingeniería de MLOps, gobernanza de IA (ISO/IEC 42001) y cumplimiento regulatorio (EU AI Act). Los datos y evaluaciones son simulados y no representan asesoría ni decisiones financieras reales.<br>
        <strong>&copy; 2026 Todos los derechos reservados &mdash; Elaborado por Alfredo Díaz.</strong>
    </div>
    """,
    unsafe_allow_html=True,
)