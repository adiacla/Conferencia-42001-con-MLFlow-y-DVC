# Model Card - CreditScore-EU v1

> Documentación técnica simplificada (Art. 11 y Anexo IV AI Act; ISO/IEC 42001 A.6.2.7 y A.8.2).

## 1. Identificación
| Campo | Valor |
|---|---|
| Modelo | CreditScore-EU |
| Versión (MLflow) | 1 |
| Run de MLflow | `431a0df98dfd4151998db5f1e6ba7308` |
| Commit de git | `3148e68` |
| SHA-256 del dataset | `5a893a14955f20debf310f1c35698ece3087c0d824210274851fb9fbaac7f22b` |
| Clasificación de riesgo | ALTO_RIESGO_ANEXO_III_5b (Anexo III, punto 5.b: evaluación de solvencia) |
| Fecha de entrenamiento (UTC) | 2026-11-26T19:10:31+00:00 |

## 2. Propósito previsto y limitaciones de uso
- **Uso previsto:** apoyar (no sustituir) la decisión de un analista humano sobre solicitudes de crédito al consumo.
- **Uso NO previsto:** decisiones totalmente automatizadas, scoring de empleo/seguros, personas menores de 18 años.
- **Datos:** dataset SINTÉTICO con sesgo de género intencionado para fines didácticos.

## 3. Datos de entrenamiento y EDA (Art. 10)
- Filas: 600 · Test: 25%
- Atributo protegido: `genero`
- Variable proxy potencial: `zona_postal_vulnerable`.

## 4. Rendimiento y equidad (Art. 15 y 10.2.f)
| Métrica | Modelo base | Tras mitigación |
|---|---|---|
| Exactitud | 0.887 | 0.793 |
| Dif. paridad demográfica | 0.342 | 0.080 |
| Dif. equalized odds | 0.446 | 0.545 |

Criterio bloqueante: DP <= 0.12 y exactitud >= 0.8.

## 5. Explicabilidad (Art. 13 y 86)
SHAP TreeExplainer. Gráfico global en `reports/shap_global.png`; la API devuelve los 3 factores principales por decisión.

## 6. Supervisión humana (Art. 14)
Toda decisión DENEGADA o con probabilidad en [0.4, 0.6] se marca `requires_human_review=true`.

## 7. Registro de eventos (Art. 12)
Cada predicción se registra con ID, versión de modelo, hash de datos, entradas, salida, factores, latencia y hash de integridad. Retención mínima: 183 días.

## 8. Riesgos residuales conocidos
- `ThresholdOptimizer` requiere el atributo protegido en inferencia: evaluar licitud con el DPO.
- Dataset sintético con fines demostrativos.
