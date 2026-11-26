"""
prepare_data.py - Etapa 1 CRISP-ML(Q): Preparación, EDA y Contratos de Calidad de Datos.

Genera el dataset, ejecuta Análisis Exploratorio de Datos (EDA) orientado a riesgos y sesgos
(Art. 10 EU AI Act / ISO 42001 A.7.4), guarda gráficos en reports/ y genera data/data_card.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import gov


def run_eda(df: pd.DataFrame, out_dir: Path) -> dict:
    """Ejecuta EDA estadístico y de equidad previo al entrenamiento."""
    num_cols = ["edad", "ingresos_anuales_eur", "deuda_actual_eur", "historial_pagos_meses", "impagos_previos"]
    desc = df[num_cols].describe().round(2).to_dict()

    # Tasa de aprobación histórica por género (detección temprana de sesgo en datos)
    tasa_genero = df.groupby("genero")["credito_aprobado"].agg(["count", "mean"]).round(4)
    tasa_dict = {
        idx: {"total": int(row["count"]), "tasa_aprobacion": float(row["mean"])}
        for idx, row in tasa_genero.iterrows()
    }
    brecha_historica = round(
        abs(tasa_dict["Masculino"]["tasa_aprobacion"] - tasa_dict["Femenino"]["tasa_aprobacion"]), 4
    )

    # Gráfico EDA combinado: Distribución de aprobaciones por género + Ingresos vs Deuda
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Subplot 1: Sesgo histórico en datos crudos
    generos = list(tasa_dict.keys())
    tasas = [tasa_dict[g]["tasa_aprobacion"] * 100 for g in generos]
    colors = ["#E05A47", "#2B5B84"]
    bars = axes[0].bar(generos, tasas, color=colors, width=0.5)
    axes[0].set_title("EDA: Tasa de Aprobación Histórica por Género", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("% Crédito Aprobado en Dataset")
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars, tasas):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{val:.1f}%", ha="center", fontweight="bold")

    # Subplot 2: Distribución de Ingresos por Estado de Aprobación
    aprob = df[df["credito_aprobado"] == 1]["ingresos_anuales_eur"]
    deneg = df[df["credito_aprobado"] == 0]["ingresos_anuales_eur"]
    axes[1].hist([aprob, deneg], bins=18, label=["Aprobado (1)", "Denegado (0)"], color=["#2E8B57", "#D9534F"], alpha=0.8)
    axes[1].set_title("EDA: Distribución de Ingresos Anuales (EUR)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Ingresos Anuales (€)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].legend()

    plt.tight_layout()
    eda_plot_path = out_dir / "eda_resumen.png"
    plt.savefig(eda_plot_path, dpi=130)
    plt.close()

    return {
        "n_filas": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "balance_clase_objetivo": df["credito_aprobado"].value_counts(normalize=True).round(4).to_dict(),
        "distribucion_por_genero": tasa_dict,
        "brecha_aprobacion_historica": brecha_historica,
        "alerta_sesgo_datos": bool(brecha_historica > 0.10),
        "estadisticas_numericas": desc,
        "grafico_eda": str(eda_plot_path),
    }


def main():
    p = gov.load_params()
    csv_path = gov.ROOT / p["data"]["path"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = gov.ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=== ETAPA 1 CRISP-ML(Q): PREPARACIÓN, EDA Y VALIDACIÓN DE DATOS ===")
    df = gov.generate_dataset(n=p["data"]["n_rows"], seed=p["seed"])
    df.to_csv(csv_path, index=False)

    # 1. Ejecutar EDA
    eda_summary = run_eda(df, reports_dir)
    print(f"[EDA] Dataset generado: {eda_summary['n_filas']} registros, {eda_summary['n_columnas']} variables.")
    print(f"[EDA] Tasa aprobación histórica -> Masculino: {eda_summary['distribucion_por_genero']['Masculino']['tasa_aprobacion']:.1%} | "
          f"Femenino: {eda_summary['distribucion_por_genero']['Femenino']['tasa_aprobacion']:.1%}")
    if eda_summary["alerta_sesgo_datos"]:
        print(f"[EDA ALERTA Art. 10] Brecha histórica detectada en datos crudos: {eda_summary['brecha_aprobacion_historica']:.1%}")

    # 2. Contratos de Calidad de Datos
    controles = gov.quality_checks(df)
    todos_ok = all(c["passed"] for c in controles)
    for c in controles:
        estado = "PASS" if c["passed"] else "FAIL"
        print(f"  [{estado}] {c['name']}: {c['detail']}")

    # 3. Huellas criptográficas y Data Card
    sha = gov.sha256_file(csv_path)
    md5 = gov.md5_file(csv_path)
    data_card = {
        "dataset_name": "datos_credito_eu_ai_act",
        "path": p["data"]["path"],
        "sha256": sha,
        "md5": md5,
        "todos_los_controles_ok": todos_ok,
        "controles_calidad": controles,
        "eda": eda_summary,
    }
    card_path = csv_path.parent / "data_card.json"
    card_path.write_text(json.dumps(data_card, indent=2, ensure_ascii=False), encoding="utf-8")

    # Registro ligero de linaje local (compatible con DVC o ejecución directa)
    dvc_lock_content = f"""schema: '2.0'
stages:
  prepare:
    cmd: python prepare_data.py
    outs:
    - path: {p['data']['path']}
      md5: {md5}
"""
# En prepare_data.py, reemplaza las líneas que escriben dvc.lock por solo esto:
    card_path = csv_path.parent / "data_card.json"
    card_path.write_text(json.dumps(data_card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Data Card guardada en {card_path} (SHA-256: {sha[:12]}...)")

if __name__ == "__main__":
    main()
