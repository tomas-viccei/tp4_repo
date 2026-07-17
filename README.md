# 🛢️ Pozos de Hidrocarburos de Argentina — Predicción y Clasificación

Aplicación web del **Trabajo Práctico 4** (proyecto integrador) de *Inteligencia
Artificial y Aprendizaje Automático I* — Licenciatura en Ciencia de Datos, UCA, 2026.

**Autores:** Andrisani, Facundo · Feser, Ignacio · Lauria, Francisco · Viccei, Tomás

🔗 **App desplegada:** `<URL de Streamlit Community Cloud — completar tras el despliegue>`

## ¿Qué hace?

Despliega los dos mejores modelos del proyecto, entrenados sobre **872.186 registros
mensuales** de producción de pozos declarados ante la Secretaría de Energía (2025):

| Modelo | Tarea | Métricas (test) |
|---|---|---|
| **XGBoost Regressor** (TP2) | Estimar la producción mensual de petróleo (`prod_pet`, m³/mes) | R² = 0,45 · MAE ≈ 105 m³/mes |
| **XGBoost Classifier** (TP3) | Predecir el estado operativo (Activo / Inactivo) | F1 pond. = 0,88 · AUC = 0,94 |

Ambos usan solo variables **estructurales y geográficas** del pozo (profundidad, mes,
sistema de extracción, tipo de pozo, cuenca, provincia, tipo de recurso), sin requerir
datos de producción como entrada. La partición de entrenamiento se agrupó por pozo para
evitar *data leakage* de panel.

**Funcionalidades:** predicción individual con validación de entradas y visualización
del resultado en contexto (distribución histórica / probabilidades por clase),
**carga masiva por CSV** con descarga de resultados, sección de **explicabilidad**
(importancia de variables) y **log de predicciones** de la sesión.

## Estructura del repositorio

```
├── app.py                              # Aplicación Streamlit (archivo único)
├── modelo_prod_pet_xgboost.joblib      # Modelo de regresión serializado (TP2)
├── modelo_estado_binario_clf.joblib    # Modelo de clasificación serializado (TP3)
├── requirements.txt                    # Dependencias con versiones fijadas
├── notebooks/                          # Notebooks TP1–TP3 (reproducibles)
└── README.md
```

> ⚠️ `scikit-learn` está fijado en **1.6.1** porque es la versión con la que se
> serializaron los pipelines; otra versión puede fallar al deserializar.

## Ejecución local

```bash
git clone <URL-del-repo>
cd <repo>
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La app abre en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud (gratuito)

1. Subir este repositorio a GitHub (público).
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
3. **New app** → elegir el repo, rama `main`, archivo `app.py` → **Deploy**.
4. Copiar la URL pública generada y pegarla arriba en este README.

Los `.joblib` (~11 MB en total) están dentro del límite de GitHub y de Streamlit Cloud;
no se requiere Git LFS.

## Datos

Fuente: [Producción de pozos de gas y petróleo 2025 — Datos Abiertos, Secretaría de
Energía de la Nación](http://datos.energia.gob.ar). El preprocesamiento (limpieza,
tratamiento de nulos y outliers, ingeniería de características) está documentado en el
notebook del TP1.

## Licencia y alcance

Proyecto con fines académicos. Las predicciones son estimaciones estadísticas de orden
de magnitud y no reemplazan estudios de ingeniería de reservorios.
