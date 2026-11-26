"""
train.py - Etapa 2 CRISP-ML(Q): Modelado, Mitigación de Sesgos, Explicabilidad y MLflow.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                               equalized_odds_difference, false_negative_rate,
                               selection_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import gov

warnings.filterwarnings("ignore")


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def write_model_card(path, meta, m, p):
    txt = f"""# Model Card - {meta['model_name']} v{meta['model_version']}

> Documentación técnica simplificada (Art. 11 y Anexo IV AI Act; ISO/IEC 42001 A.6.2.7 y A.8.2).

## 1. Identificación
| Campo | Valor |
|---|---|
| Modelo | {meta['model_name']} |
| Versión (MLflow) | {meta['model_version']} |
| Run de MLflow | `{meta['mlflow_run_id']}` |
| Commit de git | `{meta['git_commit']}` |
| SHA-256 del dataset | `{meta['dataset_sha256']}` |
| Clasificación de riesgo | {gov.RISK_LEVEL} (Anexo III, punto 5.b: evaluación de solvencia) |
| Fecha de entrenamiento (UTC) | {meta['trained_at']} |

## 2. Propósito previsto y limitaciones de uso
- **Uso previsto:** apoyar (no sustituir) la decisión de un analista humano sobre solicitudes de crédito al consumo.
- **Uso NO previsto:** decisiones totalmente automatizadas, scoring de empleo/seguros, personas menores de 18 años.
- **Datos:** dataset SINTÉTICO con sesgo de género intencionado para fines didácticos.

## 3. Datos de entrenamiento y EDA (Art. 10)
- Filas: {m['n_rows']} · Test: {p['train']['test_size']:.0%}
- Atributo protegido: `{p['governance']['protected_attribute']}`
- Variable proxy potencial: `zona_postal_vulnerable`.

## 4. Rendimiento y equidad (Art. 15 y 10.2.f)
| Métrica | Modelo base | Tras mitigación |
|---|---|---|
| Exactitud | {m['accuracy_base']:.3f} | {m['accuracy_mitigated']:.3f} |
| Dif. paridad demográfica | {m['dp_diff_base']:.3f} | {m['dp_diff_mitigated']:.3f} |
| Dif. equalized odds | {m['eo_diff_base']:.3f} | {m['eo_diff_mitigated']:.3f} |

Criterio bloqueante: DP <= {p['governance']['max_dp_difference']} y exactitud >= {p['governance']['min_accuracy']}.

## 5. Explicabilidad (Art. 13 y 86)
SHAP TreeExplainer. Gráfico global en `reports/shap_global.png`; la API devuelve los 3 factores principales por decisión.

## 6. Supervisión humana (Art. 14)
Toda decisión DENEGADA o con probabilidad en {p['governance']['human_review_band']} se marca `requires_human_review=true`.

## 7. Registro de eventos (Art. 12)
Cada predicción se registra con ID, versión de modelo, hash de datos, entradas, salida, factores, latencia y hash de integridad. Retención mínima: {p['governance']['log_retention_days']} días.

## 8. Riesgos residuales conocidos
- `ThresholdOptimizer` requiere el atributo protegido en inferencia: evaluar licitud con el DPO.
- Dataset sintético con fines demostrativos.
"""
    Path(path).write_text(txt, encoding="utf-8")


def main():
    print("=== ETAPA 2 CRISP-ML(Q): ENTRENAMIENTO, EQUIDAD Y REGISTRO MLFLOW ===")
    p = gov.load_params()
    seed = p["seed"]
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    csv = gov.ROOT / p["data"]["path"]
    if not csv.exists():
        raise FileNotFoundError(f"No existe {csv}. Ejecuta primero: python prepare_data.py")

    df = pd.read_csv(csv)
    data_sha = gov.sha256_file(csv)
    X = gov.build_features(df)
    y = df["credito_aprobado"]
    sf = df[p["governance"]["protected_attribute"]]

    X_tr, X_te, y_tr, y_te, sf_tr, sf_te = train_test_split(
        X, y, sf, test_size=p["train"]["test_size"], random_state=seed)

    # 1. Modelo base
    base = RandomForestClassifier(
        n_estimators=p["train"]["n_estimators"], max_depth=p["train"]["max_depth"],
        random_state=seed)
    base.fit(X_tr, y_tr)
    pred_base = base.predict(X_te)

    # 2. Mitigación de sesgo
    mit = ThresholdOptimizer(estimator=base, constraints=p["train"]["mitigation_constraint"],
                             predict_method="predict_proba", prefit=True)
    mit.fit(X_tr, y_tr, sensitive_features=sf_tr)
    pred_mit = mit.predict(X_te, sensitive_features=sf_te, random_state=seed)

    # 3. Métricas
    frame = lambda pred: MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate,
                 "false_negative_rate": false_negative_rate},
        y_true=y_te, y_pred=pred, sensitive_features=sf_te)
    mf_base, mf_mit = frame(pred_base), frame(pred_mit)

    metrics = {
        "n_rows": int(len(df)),
        "accuracy_base": float(accuracy_score(y_te, pred_base)),
        "accuracy_mitigated": float(accuracy_score(y_te, pred_mit)),
        "dp_diff_base": float(demographic_parity_difference(y_te, pred_base, sensitive_features=sf_te)),
        "dp_diff_mitigated": float(demographic_parity_difference(y_te, pred_mit, sensitive_features=sf_te)),
        "eo_diff_base": float(equalized_odds_difference(y_te, pred_base, sensitive_features=sf_te)),
        "eo_diff_mitigated": float(equalized_odds_difference(y_te, pred_mit, sensitive_features=sf_te)),
        "override_rate": float(np.mean(pred_base != pred_mit)),
    }
    metrics["accuracy_drop_mitigation"] = metrics["accuracy_base"] - metrics["accuracy_mitigated"]
    for grupo, fila in mf_mit.by_group.iterrows():
        metrics[f"selection_rate_mitigated_{grupo}"] = float(fila["selection_rate"])
    for grupo, fila in mf_base.by_group.iterrows():
        metrics[f"selection_rate_base_{grupo}"] = float(fila["selection_rate"])

    # 4. SHAP global
    explainer = shap.TreeExplainer(base)
    sv = explainer(X_te)
    vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
    plt.figure()
    shap.summary_plot(vals, X_te, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_global.png", dpi=130)
    plt.close()
    mf_base.by_group.round(4).to_csv("reports/fairness_by_group_base.csv")
    mf_mit.by_group.round(4).to_csv("reports/fairness_by_group_mitigated.csv")

    # 5. Registro en MLflow
    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment"])
    client = MlflowClient()
    reg_name = p["mlflow"]["registered_model"]

    with mlflow.start_run(run_name="train-mitigated") as run:
        mlflow.log_params(dict(flatten({k: p[k] for k in ("seed", "train", "governance")})))
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, float) and "_" in k})

        try:
            ds = mlflow.data.from_pandas(df, source=str(csv), name="datos_credito",
                                         targets="credito_aprobado")
            mlflow.log_input(ds, context="training")
        except Exception:
            pass

        sig = infer_signature(X_te, base.predict_proba(X_te))
        kwargs = dict(sk_model=base, signature=sig, input_example=X_te.head(3),
                      registered_model_name=reg_name,
                      serialization_format="cloudpickle")
        try:
            mlflow.sklearn.log_model(name="model", **kwargs)
        except TypeError:
            mlflow.sklearn.log_model(artifact_path="model", **kwargs)

        versions = client.search_model_versions(f"name='{reg_name}'")
        version = max((int(v.version) for v in versions if v.run_id == run.info.run_id),
                      default=max(int(v.version) for v in versions))

        meta = {
            "model_name": gov.MODEL_NAME, "model_version": version,
            "mlflow_run_id": run.info.run_id, "dataset_sha256": data_sha,
            "git_commit": gov.git_commit(),
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "feature_columns": gov.FEATURE_COLUMNS,
        }

        tags = {
            "compliance.ai_act.risk_level": gov.RISK_LEVEL,
            "compliance.ai_act.articles": gov.AI_ACT_ARTICLES,
            "compliance.iso42001.controls": gov.ISO42001_CONTROLS,
            "compliance.dataset_sha256": data_sha,
            "compliance.git_commit": meta["git_commit"],
            "compliance.human_oversight": "true",
            "compliance.status": "PENDING_AUDIT",
        }
        mlflow.set_tags(tags)
        for k, v in tags.items():
            client.set_model_version_tag(reg_name, str(version), k, v)
        client.set_registered_model_alias(reg_name, "staging", str(version))

        joblib.dump({"base": base, "mitigator": mit, "meta": meta}, "models/credit_model.joblib")
        meta["model_sha256"] = gov.sha256_file("models/credit_model.joblib")
        Path("models/credit_model.sha256").write_text(meta["model_sha256"], encoding="utf-8")
        client.set_model_version_tag(reg_name, str(version), "compliance.model_sha256",
                                     meta["model_sha256"])
        write_model_card("models/model_card.md", meta, metrics, p)

        with open("reports/mlflow_run.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: round(v, 4) if isinstance(v, float) else v
                       for k, v in metrics.items()}, f, indent=2)

        for art in ("models/credit_model.joblib", "models/credit_model.sha256", "models/model_card.md",
                    "data/data_card.json", "params.yaml", "reports/shap_global.png",
                    "reports/eda_resumen.png", "reports/fairness_by_group_base.csv",
                    "reports/fairness_by_group_mitigated.csv"):
            if Path(art).exists():
                mlflow.log_artifact(art, artifact_path="governance")

    print(f"[MLflow] Run ID: {run.info.run_id} | Modelo: {reg_name} v{version} (alias: staging)")
    print(f"[Métricas] Exactitud Base: {metrics['accuracy_base']:.3f} -> Mitigada: {metrics['accuracy_mitigated']:.3f}")
    print(f"[Equidad]  Paridad Demográfica Base: {metrics['dp_diff_base']:.3f} -> Mitigada: {metrics['dp_diff_mitigated']:.3f}")


if __name__ == "__main__":
    main()
