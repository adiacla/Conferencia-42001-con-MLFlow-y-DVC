# Informe de auditoría - CreditScore-EU v1

**Resultado: APROBADO** · 2026-11-26T20:25:15+00:00 · commit `608c916`

| ID | Control | Resultado | Tipo | Detalle | ISO 42001 | AI Act |
|---|---|---|---|---|---|---|
| C01 | Datos versionados e íntegros (DVC) | PASA | BLOQUEANTE | md5 lock=9d5194bd54… actual=9d5194bd54… | A.7.5 Procedencia de datos | Art. 10, 11, 12 |
| C02 | Contratos de calidad de datos superados | PASA | BLOQUEANTE | 8/8 controles OK | A.7.4 Calidad de datos | Art. 10.2 y 10.3 |
| C03 | El modelo se entrenó con el dataset actual | PASA | BLOQUEANTE | sha256 modelo=5a893a1495… dataset=5a893a1495… | A.6.2.7 Documentación técnica | Art. 11, 12 |
| C04 | Paridad demográfica tras mitigación <= umbral | PASA | BLOQUEANTE | DP=0.081 (antes 0.342); umbral 0.12 | A.5.2 Evaluación de impacto; 8.4 | Art. 10.2.f |
| C05 | Exactitud >= mínimo declarado | PASA | BLOQUEANTE | exactitud=0.793; mínimo 0.78 | A.6.2.4 Verificación y validación | Art. 15 |
| C06 | Coste de equidad aceptable | PASA | BLOQUEANTE | caída de exactitud=0.093; máximo 0.1 | 8.3 Tratamiento del riesgo | Art. 9 |
| C07 | Equalized odds (informativo, no bloquea) | FALLA | INFO | EO=0.545. Criterio alternativo de equidad. | 9.1 Seguimiento | Art. 10.2.f |
| C17 | Decisiones cambiadas por el mitigador (informativo) | PASA | INFO | 18.7% de decisiones difieren del modelo base | 9.1 Seguimiento | Art. 14, 15 |
| C08 | Modelo registrado con tags de cumplimiento y firma | PASA | BLOQUEANTE | faltan tags=[]; firma=sí; run=431a0df9… | A.6.2.7; A.6.2.8 | Art. 11, 12, 17 |
| C09 | Integridad del artefacto del modelo | PASA | BLOQUEANTE | sha256 archivo=86255b2536… registro=86255b2536… | A.6.2.5 Despliegue | Art. 15 (ciberseguridad) |
| C10 | Model card completa (Anexo IV simplificado) | PASA | BLOQUEANTE | secciones faltantes=[] | A.6.2.7; A.8.2 | Art. 11, Anexo IV |
| C11 | Evidencia de explicabilidad global (SHAP) | PASA | BLOQUEANTE | reports/shap_global.png | A.8.2 Información a usuarios | Art. 13 |
| C12 | Clasificación de riesgo documentada y coherente | PASA | BLOQUEANTE | ALTO RIESGO (Art. 6 / Anexo III) | 6.1.2 Evaluación de riesgos de IA | Art. 6, Anexo III 5.b |
| C13 | Retención de logs >= 6 meses | PASA | BLOQUEANTE | 183 días | A.6.2.8 Registro de eventos | Art. 19, 26.6 |
| C14 | Logging automático completo e íntegro por predicción | PASA | BLOQUEANTE | 4/4 registros; campos=True; hash=True | A.6.2.8 Registro de eventos | Art. 12 |
| C15 | Supervisión humana: denegaciones y zona gris escalan | PASA | BLOQUEANTE | APROBADO(p=0.9806)/rev=False, APROBADO(p=0.0344)/rev=False, DENEGADO(p=0.1118)/rev=True, APROBADO(p=0.4618)/rev=True | A.9.2 Uso responsable | Art. 14 |
| C16 | Validación de entradas (rechaza edad < 18) | PASA | BLOQUEANTE | HTTP 422 | A.6.2.4 | Art. 15 |