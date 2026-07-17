# -*- coding: utf-8 -*-
"""
TP4 — Despliegue en Producción | IA y Aprendizaje Automático I — UCA 2026
Aplicación web que despliega los dos mejores modelos del proyecto integrador:
  • Regresión (TP2): XGBoost → producción mensual de petróleo (prod_pet, m³/mes)
  • Clasificación (TP3): XGBoost → estado operativo del pozo (Activo / Inactivo)
Autores: Andrisani, Feser, Lauria, Viccei.
Ejecución local:  streamlit run app.py
"""
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN Y CONSTANTES DEL DOMINIO
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Pozos de Hidrocarburos — Predicción y Clasificación",
                   page_icon="🛢️", layout="wide")

FEATURES = ['profundidad', 'mes', 'tipoextraccion', 'tipopozo',
            'cuenca', 'provincia', 'tipo_de_recurso']

TIPOS_EXTRACCION = ['Surgencia Natural', 'Bombeo Mecánico', 'Electrosumergible',
                    'Cavidad Progresiva', 'Plunger Lift', 'Gas Lift', 'Jet Pump',
                    'Bombeo Hidráulico', 'Pistoneo (Swabbing)',
                    'Sin Sistema de Extracción', 'Otros Tipos de Extracción']
TIPOS_POZO = ['Petrolífero', 'Gasífero', 'Inyección de Agua', 'Inyección de Gas',
              'Acuífero', 'Sumidero', 'Monitoreo de almacenamiento',
              'Bidireccional de almacenamiento', 'Otro tipo']
TIPOS_RECURSO = ['CONVENCIONAL', 'NO CONVENCIONAL', 'NO DISCRIMINADO', 'SIN RESERVORIO']

# Combinaciones cuenca → provincias observadas en el dataset (validación de coherencia)
CUENCA_PROVINCIAS = {
    'NEUQUINA': ['Neuquén', 'Rio Negro', 'Mendoza', 'La Pampa'],
    'GOLFO SAN JORGE': ['Chubut', 'Santa Cruz'],
    'AUSTRAL': ['Santa Cruz', 'Tierra del Fuego', 'Estado Nacional'],
    'CUYANA': ['Mendoza'],
    'NOROESTE': ['Salta', 'Jujuy', 'Formosa'],
    'NORESTE': ['Jujuy'],
    'CAÑADON ASFALTO': ['Chubut'],
    'ÑIRIHUAU': ['Chubut'],
}
PROF_MIN, PROF_MAX, PROF_MEDIANA = 1.0, 8687.0, 1776.0

# Estadísticas de referencia precomputadas sobre el dataset del TP1 (872.186 registros)
# para brindar contexto sin necesidad de cargar el CSV completo.
REF = {
    "cuantiles_prod": {"p10": 4.8, "p25": 18.3, "p50": 42.8, "p75": 96.7,
                       "p90": 243.9, "p95": 612.1, "p99": 2800.4},
    "hist_log1p": {
        "counts": [1856, 2478, 4021, 6317, 8968, 11486, 13411, 14830, 15645, 15862,
                   15497, 14831, 13993, 12889, 11797, 10651, 9528, 8397, 7360, 6423,
                   5605, 4842, 4149, 3505, 2937, 2416, 1971, 1571, 1229, 940,
                   706, 517, 366, 250, 163, 101, 59, 31, 14, 8],
        "edges_max": 10.19},  # np.log1p(26593) ≈ 10.19 — solo para el eje
    "mediana_por_cuenca": {"NOROESTE": 117.8, "CUYANA": 67.8, "NEUQUINA": 42.6,
                           "GOLFO SAN JORGE": 42.3, "AUSTRAL": 25.6},
    "tasa_activo_por_extraccion": {
        "Plunger Lift": 0.773, "Surgencia Natural": 0.697, "Cavidad Progresiva": 0.678,
        "Electrosumergible": 0.629, "Gas Lift": 0.616, "Bombeo Mecánico": 0.560,
        "Jet Pump": 0.520, "Otros Tipos de Extracción": 0.209,
        "Pistoneo (Swabbing)": 0.163, "Bombeo Hidráulico": 0.010,
        "Sin Sistema de Extracción": 0.003},
    "tasa_activo_global": 0.339,
    "importancias_regresion": {"tipoextraccion": 0.358, "tipopozo": 0.322,
                               "provincia": 0.208, "tipo_de_recurso": 0.038,
                               "cuenca": 0.037, "profundidad": 0.027, "mes": 0.010},
    "importancias_clasificacion": {"tipoextraccion": 0.689, "tipopozo": 0.261,
                                   "tipo_de_recurso": 0.018, "provincia": 0.017,
                                   "cuenca": 0.012, "mes": 0.003, "profundidad": 0.001},
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE MODELOS (cacheada) Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando modelos entrenados…")
def cargar_modelos():
    reg = joblib.load('modelo_prod_pet_xgboost.joblib')       # TransformedTargetRegressor
    clf = joblib.load('modelo_estado_binario_clf.joblib')     # Pipeline clasificación
    
    # FIX: Restaurar base_score perdido por bug de serialización de XGBoost >= 2.0
    try:
        xgb_reg = reg.regressor_.named_steps['modelo']
        booster = xgb_reg.get_booster()
        # El promedio de log1p(prod_pet) es aprox 3.178 (se pierde al usar joblib.dump en pipeline)
        booster.set_param({'base_score': 3.178})
    except Exception as e:
        print("Error al restaurar base_score:", e)
        
    return reg, clf

def validar_entrada(fila: dict) -> list[str]:
    """Devuelve una lista de advertencias/errores de validación de la entrada."""
    avisos = []
    prof = fila['profundidad']
    if not (PROF_MIN <= prof <= PROF_MAX):
        avisos.append(f"⛔ La profundidad debe estar entre {PROF_MIN:.0f} y {PROF_MAX:.0f} m "
                      f"(rango observado en el dataset).")
    if fila['provincia'] not in CUENCA_PROVINCIAS.get(fila['cuenca'], []):
        avisos.append(f"⚠️ La combinación cuenca **{fila['cuenca']}** + provincia "
                      f"**{fila['provincia']}** no se observa en el dataset: la predicción "
                      f"se realizará por extrapolación y es menos confiable.")
    if fila['tipoextraccion'] == 'Sin Sistema de Extracción' and fila['tipopozo'] == 'Petrolífero':
        avisos.append("ℹ️ Un pozo petrolífero sin sistema de extracción suele estar inactivo; "
                      "la producción esperada será muy baja.")
    return avisos

def fila_df(prof, mes, extr, tpozo, cuenca, prov, rec) -> pd.DataFrame:
    return pd.DataFrame([{'profundidad': float(prof), 'mes': int(mes),
                          'tipoextraccion': extr, 'tipopozo': tpozo, 'cuenca': cuenca,
                          'provincia': prov, 'tipo_de_recurso': rec}])[FEATURES]

def registrar_log(tipo: str, entrada: dict, salida: str):
    """Log de predicciones en memoria de sesión (extensión: monitoreo básico)."""
    if 'log' not in st.session_state:
        st.session_state['log'] = []
    st.session_state['log'].append(
        {'timestamp': datetime.now().isoformat(timespec='seconds'),
         'modelo': tipo, **entrada, 'resultado': salida})

def grafico_importancias(imps: dict, titulo: str) -> go.Figure:
    keys = list(imps.keys())[::-1]
    fig = go.Figure(go.Bar(x=[imps[k] for k in keys], y=keys, orientation='h',
                           marker_color='#31a354'))
    fig.update_layout(title=titulo, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title='Importancia relativa')
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCABEZADO Y BARRA LATERAL
# ══════════════════════════════════════════════════════════════════════════════
st.title("🛢️ Pozos de Hidrocarburos de Argentina — Predicción y Clasificación")
st.markdown(
    "Aplicación del proyecto integrador de **IA y Aprendizaje Automático I (UCA, 2026)**. "
    "Despliega los dos mejores modelos entrenados sobre **872.186 registros mensuales** "
    "de producción declarados ante la Secretaría de Energía (2025): un **XGBoost de "
    "regresión** que estima la producción mensual de petróleo y un **XGBoost de "
    "clasificación** que predice el estado operativo del pozo, ambos a partir de "
    "características técnicas y geográficas.")

with st.sidebar:
    st.header("ℹ️ Acerca del proyecto")
    st.markdown(
        "- **Datos:** Producción de pozos de gas y petróleo 2025 — Datos Abiertos, "
        "Secretaría de Energía.\n"
        "- **Modelo de regresión (TP2):** XGBoost, R² = 0,45, MAE ≈ 105 m³/mes (test).\n"
        "- **Modelo de clasificación (TP3):** XGBoost, F1 pond. = 0,88, AUC = 0,94 (test).\n"
        "- Partición agrupada por pozo (sin *leakage* de panel).\n\n"
        "**Autores:** Andrisani · Feser · Lauria · Viccei")
    st.divider()
    st.caption("Los modelos usan solo variables estructurales del pozo; no requieren "
               "datos de producción como entrada.")

reg_model, clf_model = cargar_modelos()

tab_reg, tab_clf, tab_lote, tab_info = st.tabs(
    ["📈 Predicción de producción", "🔎 Clasificación de estado",
     "📂 Carga masiva (CSV)", "📊 Explicabilidad y contexto"])

# ══════════════════════════════════════════════════════════════════════════════
# 4. PESTAÑA 1 — REGRESIÓN: producción mensual de petróleo
# ══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.subheader("Estimar la producción mensual de petróleo de un pozo")
    st.caption("Modelo XGBoost entrenado sobre pozos productores (264.580 registros). "
               "La estimación corresponde a un mes calendario, en metros cúbicos.")

    c1, c2, c3 = st.columns(3)
    with c1:
        cuenca_r = st.selectbox("Cuenca", list(CUENCA_PROVINCIAS.keys()), key='cu_r')
        prov_r = st.selectbox("Provincia", CUENCA_PROVINCIAS[cuenca_r], key='pr_r')
        rec_r = st.selectbox("Tipo de recurso", TIPOS_RECURSO, key='re_r')
    with c2:
        tpozo_r = st.selectbox("Tipo de pozo", TIPOS_POZO, key='tp_r',
                               help="El modelo se entrenó principalmente sobre pozos "
                                    "petrolíferos y gasíferos productores.")
        extr_r = st.selectbox("Sistema de extracción", TIPOS_EXTRACCION, key='ex_r')
    with c3:
        prof_r = st.number_input("Profundidad (m)", min_value=PROF_MIN, max_value=PROF_MAX,
                                 value=PROF_MEDIANA, step=50.0, key='pf_r',
                                 help=f"Rango válido: {PROF_MIN:.0f}–{PROF_MAX:.0f} m "
                                      f"(mediana del dataset: {PROF_MEDIANA:.0f} m).")
        mes_r = st.select_slider("Mes del año", options=list(range(1, 13)), value=6, key='ms_r')

    if st.button("Calcular producción esperada", type="primary", key='btn_r'):
        fila = fila_df(prof_r, mes_r, extr_r, tpozo_r, cuenca_r, prov_r, rec_r)
        avisos = validar_entrada(fila.iloc[0].to_dict())
        errores = [a for a in avisos if a.startswith("⛔")]
        for a in avisos:
            (st.error if a.startswith("⛔") else st.warning if a.startswith("⚠️") else st.info)(a)

        if not errores:
            pred = float(reg_model.predict(fila)[0])
            pred = max(pred, 0.0)
            q = REF['cuantiles_prod']
            pct = sum(v <= pred for v in q.values()) / len(q)  # posición aproximada

            m1, m2, m3 = st.columns(3)
            m1.metric("Producción estimada", f"{pred:,.1f} m³/mes")
            m2.metric("≈ barriles/día", f"{pred*6.2898/30:,.1f} bbl/d")
            comparado = ("por encima de la mediana" if pred >= q['p50']
                         else "por debajo de la mediana")
            m3.metric("Mediana nacional", f"{q['p50']:.1f} m³/mes", delta=comparado,
                      delta_color="off")
            st.caption("La estimación es de orden de magnitud: el MAE del modelo en test "
                       "es ≈ 105 m³/mes y el error crece en pozos de muy alta producción.")

            # Contexto: ubicación del valor predicho en la distribución histórica
            counts = REF['hist_log1p']['counts']
            edges = np.linspace(0, REF['hist_log1p']['edges_max'], len(counts) + 1)
            centros = (edges[:-1] + edges[1:]) / 2
            fig = go.Figure(go.Bar(x=centros, y=counts, marker_color='#9ecae1',
                                   name='Pozos productores'))
            fig.add_vline(x=float(np.log1p(pred)), line_color='#e6550d', line_width=3,
                          annotation_text=f"Predicción: {pred:,.0f} m³",
                          annotation_position="top right")
            for et, v in [('P25', q['p25']), ('Mediana', q['p50']), ('P75', q['p75']),
                          ('P95', q['p95'])]:
                fig.add_vline(x=float(np.log1p(v)), line_dash='dot', line_color='gray',
                              annotation_text=et, annotation_position="bottom")
            fig.update_layout(
                title='¿Dónde cae la predicción en la distribución histórica? '
                      '(eje en escala log1p)',
                xaxis_title='log1p(producción mensual de petróleo, m³)',
                yaxis_title='Cantidad de registros', height=380,
                margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            registrar_log('regresión', fila.iloc[0].to_dict(), f"{pred:.1f} m3/mes")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PESTAÑA 2 — CLASIFICACIÓN: estado operativo del pozo
# ══════════════════════════════════════════════════════════════════════════════
with tab_clf:
    st.subheader("Clasificar el estado operativo de un pozo (Activo / Inactivo)")
    st.caption("Modelo XGBoost entrenado sobre el parque completo (872.186 registros, "
               "33,9 % activos). «Activo» = extracción efectiva en el mes.")

    c1, c2, c3 = st.columns(3)
    with c1:
        cuenca_c = st.selectbox("Cuenca", list(CUENCA_PROVINCIAS.keys()), key='cu_c')
        prov_c = st.selectbox("Provincia", CUENCA_PROVINCIAS[cuenca_c], key='pr_c')
        rec_c = st.selectbox("Tipo de recurso", TIPOS_RECURSO, key='re_c')
    with c2:
        tpozo_c = st.selectbox("Tipo de pozo", TIPOS_POZO, key='tp_c')
        extr_c = st.selectbox("Sistema de extracción", TIPOS_EXTRACCION, key='ex_c')
    with c3:
        prof_c = st.number_input("Profundidad (m)", min_value=PROF_MIN, max_value=PROF_MAX,
                                 value=PROF_MEDIANA, step=50.0, key='pf_c')
        mes_c = st.select_slider("Mes del año", options=list(range(1, 13)), value=6, key='ms_c')

    if st.button("Clasificar estado", type="primary", key='btn_c'):
        fila = fila_df(prof_c, mes_c, extr_c, tpozo_c, cuenca_c, prov_c, rec_c)
        avisos = validar_entrada(fila.iloc[0].to_dict())
        errores = [a for a in avisos if a.startswith("⛔")]
        for a in avisos:
            (st.error if a.startswith("⛔") else st.warning if a.startswith("⚠️") else st.info)(a)

        if not errores:
            proba = float(clf_model.predict_proba(fila)[0, 1])
            etiqueta = "ACTIVO 🟢" if proba >= 0.5 else "INACTIVO 🔴"

            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Estado predicho", etiqueta)
                st.metric("Probabilidad de estar activo", f"{proba*100:.1f} %")
                base = REF['tasa_activo_por_extraccion'].get(extr_c,
                                                             REF['tasa_activo_global'])
                st.caption(f"Referencia: el {base*100:.1f} % de los registros con sistema "
                           f"«{extr_c}» está activo (promedio nacional: "
                           f"{REF['tasa_activo_global']*100:.1f} %).")
            with m2:
                # Contexto: probabilidades por clase (requisito de la consigna)
                fig = go.Figure(go.Bar(
                    x=[1 - proba, proba], y=['Inactivo', 'Activo'], orientation='h',
                    marker_color=['#de2d26', '#31a354'],
                    text=[f"{(1-proba)*100:.1f} %", f"{proba*100:.1f} %"],
                    textposition='auto'))
                fig.add_vline(x=0.5, line_dash='dot', line_color='gray',
                              annotation_text='umbral 0,5')
                fig.update_layout(title='Probabilidades por clase',
                                  xaxis=dict(range=[0, 1], title='Probabilidad'),
                                  height=260, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.caption("Interpretación operativa: el modelo prioriza el *recall* de la "
                       "clase Activo (0,99 en test) — casi nunca deja pasar un pozo "
                       "productivo —, a costa de marcar como activos algunos inactivos "
                       "(precisión de Activo ≈ 0,74).")
            registrar_log('clasificación', fila.iloc[0].to_dict(),
                          f"{etiqueta} ({proba:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PESTAÑA 3 — CARGA MASIVA (extensión opcional de la consigna)
# ══════════════════════════════════════════════════════════════════════════════
with tab_lote:
    st.subheader("Predicciones en lote a partir de un CSV")
    st.markdown(f"El archivo debe contener las columnas: `{'`, `'.join(FEATURES)}`.")
    ejemplo = pd.DataFrame([
        {'profundidad': 2900, 'mes': 6, 'tipoextraccion': 'Surgencia Natural',
         'tipopozo': 'Petrolífero', 'cuenca': 'NEUQUINA', 'provincia': 'Neuquén',
         'tipo_de_recurso': 'NO CONVENCIONAL'},
        {'profundidad': 1500, 'mes': 6, 'tipoextraccion': 'Bombeo Mecánico',
         'tipopozo': 'Petrolífero', 'cuenca': 'GOLFO SAN JORGE', 'provincia': 'Chubut',
         'tipo_de_recurso': 'CONVENCIONAL'}])
    st.download_button("⬇️ Descargar CSV de ejemplo",
                       ejemplo.to_csv(index=False).encode('utf-8'),
                       "ejemplo_pozos.csv", "text/csv")

    archivo = st.file_uploader("Subir CSV de pozos", type=['csv'])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
            faltan = [c for c in FEATURES if c not in lote.columns]
            if faltan:
                st.error(f"Faltan columnas requeridas: {faltan}")
            else:
                lote = lote.copy()
                lote['profundidad'] = pd.to_numeric(lote['profundidad'], errors='coerce')
                lote['mes'] = pd.to_numeric(lote['mes'], errors='coerce').astype('Int64')
                invalidas = (lote['profundidad'].isna() | lote['mes'].isna()
                             | ~lote['profundidad'].between(PROF_MIN, PROF_MAX)
                             | ~lote['mes'].between(1, 12))
                if invalidas.any():
                    st.warning(f"Se descartaron {int(invalidas.sum())} filas con valores "
                               "fuera de rango o no numéricos.")
                    lote = lote[~invalidas]
                if len(lote) == 0:
                    st.error("No quedaron filas válidas para predecir.")
                elif len(lote) > 50000:
                    st.error("Máximo 50.000 filas por carga.")
                else:
                    X = lote[FEATURES].astype({'mes': int})
                    lote['prod_pet_estimado_m3'] = np.maximum(
                        reg_model.predict(X), 0).round(1)
                    lote['proba_activo'] = clf_model.predict_proba(X)[:, 1].round(3)
                    lote['estado_predicho'] = np.where(lote['proba_activo'] >= 0.5,
                                                       'Activo', 'Inactivo')
                    st.success(f"Predicciones generadas para {len(lote):,} pozos.")
                    st.dataframe(lote.head(200), use_container_width=True)
                    st.download_button("⬇️ Descargar resultados",
                                       lote.to_csv(index=False).encode('utf-8'),
                                       "predicciones_pozos.csv", "text/csv",
                                       type="primary")
        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PESTAÑA 4 — EXPLICABILIDAD, CONTEXTO Y LOG (extensiones opcionales)
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("¿En qué se fijan los modelos?")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_importancias(REF['importancias_regresion'],
                        'Regresión — importancia de variables'),
                        use_container_width=True)
    with c2:
        st.plotly_chart(grafico_importancias(REF['importancias_clasificacion'],
                        'Clasificación — importancia de variables'),
                        use_container_width=True)
    st.markdown(
        "En ambos modelos dominan las variables **estructurales**: el sistema de "
        "extracción y el tipo de pozo concentran la señal, seguidos por la ubicación. "
        "La profundidad y el mes aportan poco. Esto coincide con el dominio: el estado y "
        "el rendimiento de un pozo dependen del equipamiento instalado y de la cuenca en "
        "la que opera, no de una relación lineal con variables numéricas.")

    st.divider()
    st.subheader("Referencias del parque nacional (dataset 2025)")
    c1, c2 = st.columns(2)
    with c1:
        med = REF['mediana_por_cuenca']
        fig = go.Figure(go.Bar(x=list(med.values()), y=list(med.keys()),
                               orientation='h', marker_color='#3182bd'))
        fig.update_layout(title='Mediana de producción por cuenca (pozos productores)',
                          xaxis_title='m³/mes', height=320,
                          margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        act = REF['tasa_activo_por_extraccion']
        fig = go.Figure(go.Bar(x=[v * 100 for v in act.values()], y=list(act.keys()),
                               orientation='h', marker_color='#31a354'))
        fig.add_vline(x=REF['tasa_activo_global'] * 100, line_dash='dot',
                      line_color='gray', annotation_text='promedio nacional')
        fig.update_layout(title='% de registros activos por sistema de extracción',
                          xaxis_title='% activo', height=320,
                          margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Registro de predicciones de esta sesión")
    if st.session_state.get('log'):
        log_df = pd.DataFrame(st.session_state['log'])
        st.dataframe(log_df, use_container_width=True)
        st.download_button("⬇️ Descargar log (CSV)",
                           log_df.to_csv(index=False).encode('utf-8'),
                           "log_predicciones.csv", "text/csv")
    else:
        st.caption("Aún no se realizaron predicciones en esta sesión.")

st.divider()
st.caption("⚠️ Uso académico. Las predicciones son estimaciones estadísticas de orden de "
           "magnitud y no reemplazan estudios de ingeniería de reservorios. "
           "Fuente de datos: Datos Abiertos — Secretaría de Energía de la Nación.")
