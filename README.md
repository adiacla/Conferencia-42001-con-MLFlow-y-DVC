# Taller Práctico: MLOps con CRISP-ML(Q) y Cumplimiento EU AI Act (CreditScore-EU)

## Estructura del Proyecto
- `params.yaml`: Configuración central de entrenamiento, equidad y gobernanza.
- `gov.py`: Reglas compartidas (hashes SHA-256, seudonimización RGPD, clasificación de riesgo).
- `prepare_data.py`: **Etapa 1** — Generación de datos, EDA con gráficos y validación de 8 contratos de calidad (`data/data_card.json`).
- `train.py`: **Etapa 2** — Entrenamiento, mitigación de sesgos (`Fairlearn`), explicabilidad (`SHAP`), generación de `model_card.md` y registro en **MLflow**.
- `audit.py`: **Etapa 3** — Auditoría automatizada de 17 controles (ISO/IEC 42001 + EU AI Act) antes de promover a producción.
- `api.py`: Servicio de inferencia **FastAPI** con verificación de integridad al arranque y logging inmutable.
- `app_streamlit.py`: Interfaz web visual para demostrar despliegue, predicción, supervisión humana y auditoría.

## Comandos Rápidos para el Taller (en VS Code)

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preparar entorno inicial (antes de empezar el taller o durante la demo de entrenamiento):**
   ```bash
   python prepare_data.py
   python train.py
   ```

3. **Levantar la interfaz web de Streamlit (para mostrar despliegue, predicción y auditoría):**
   ```bash
   streamlit run app_streamlit.py
   ```

4. *(Opcional)* **Levantar la API FastAPI y MLflow UI en terminales paralelas:**
   ```bash
   uvicorn api:app --host 127.0.0.1 --port 8000
   mlflow ui --port 5000
   ```
