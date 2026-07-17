# -*- coding: utf-8 -*-
"""
EstimAR — Prediccion de pozos de hidrocarburos de Argentina
TP4 | IA y Aprendizaje Automatico I — UCA 2026
Autores: Andrisani, Feser, Lauria, Viccei.
"""
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EstimAR", page_icon="E", layout="wide")

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .brand-block {padding: 16px 0 12px 0; margin-bottom: 4px;}
    .brand-block .brand {font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; color: #E0E0E0; line-height: 1;}
    .brand-block .brand span {color: #6EA8FE;}
    .brand-block .tagline {color: #9CA3AF; font-size: 0.8rem; margin-top: 6px; line-height: 1.4;}
    .disclaimer {border-top: 1px solid #2A2D34; padding-top: 12px; margin-top: 32px; color: #6B7280; font-size: 0.78rem;}
    .status-badge {display: inline-block; padding: 6px 18px; border-radius: 4px; font-weight: 600; font-size: 1.1rem; margin-bottom: 8px;}
    .status-activo {background-color: rgba(34,197,94,0.15); border: 1px solid #22C55E; color: #22C55E;}
    .status-inactivo {background-color: rgba(239,68,68,0.15); border: 1px solid #EF4444; color: #EF4444;}
    .sidebar-footer {color: #6B7280; font-size: 0.75rem; line-height: 1.4;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap: 10px;}
    [data-testid="stMetric"] {
        background-color: #1A1D24;
        border: 1px solid #2A2D34;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetric"] label {
        color: #9CA3AF;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        .brand-block .brand {font-size: 2rem;}
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── CONSTANTES DE DOMINIO ───────────────────────────────────────────────────
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

CUENCA_PROVINCIAS = {
    'NEUQUINA': ['Neuquén', 'Rio Negro', 'Mendoza', 'La Pampa'],
    'GOLFO SAN JORGE': ['Chubut', 'Santa Cruz'],
    'AUSTRAL': ['Santa Cruz', 'Tierra del Fuego', 'Estado Nacional'],
    'CUYANA': ['Mendoza'],
    'NOROESTE': ['Salta', 'Jujuy', 'Formosa'],
    'NORESTE': ['Jujuy'],
    'CAÑADÓN ASFALTO': ['Chubut'],
    'ÑIRIHUAU': ['Chubut'],
}
PROF_MIN, PROF_MAX, PROF_MEDIANA = 1.0, 8687.0, 1776.0

REF = {
    "cuantiles_prod": {"p10": 4.8, "p25": 18.3, "p50": 42.8, "p75": 96.7,
                       "p90": 243.9, "p95": 612.1, "p99": 2800.4},
    "hist_log1p": {
        "counts": [1856, 2478, 4021, 6317, 8968, 11486, 13411, 14830, 15645, 15862,
                   15497, 14831, 13993, 12889, 11797, 10651, 9528, 8397, 7360, 6423,
                   5605, 4842, 4149, 3505, 2937, 2416, 1971, 1571, 1229, 940,
                   706, 517, 366, 250, 163, 101, 59, 31, 14, 8],
        "edges_max": 10.19},
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

NAV_OPTIONS = ["Prediccion individual", "Carga masiva", "Sobre el proyecto"]

# ─── PALETA Y LAYOUT DE CHARTS ──────────────────────────────────────────────
PALETTE = {
    'primary': '#6EA8FE',
    'positive': '#22C55E',
    'negative': '#EF4444',
    'neutral': '#9CA3AF',
    'highlight': '#F59E0B',
}

PLOTLY_LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E0E0E0', size=12),
    height=340,
    margin=dict(l=10, r=10, t=45, b=10),
    title_font=dict(size=14, color='#E0E0E0'),
)

# ─── CARGA DE MODELOS ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos...")
def cargar_modelos():
    reg = joblib.load('modelo_prod_pet_xgboost.joblib')
    clf = joblib.load('modelo_estado_binario_clf.joblib')
    try:
        xgb_reg = reg.regressor_.named_steps['modelo']
        booster = xgb_reg.get_booster()
        booster.set_param({'base_score': 3.178})
    except Exception:
        pass
    return reg, clf

# ─── UTILIDADES ──────────────────────────────────────────────────────────────
def validar_entrada(fila: dict) -> list[dict]:
    avisos = []
    prof = fila['profundidad']
    if not (PROF_MIN <= prof <= PROF_MAX):
        avisos.append({
            'level': 'error',
            'msg': f"La profundidad debe estar entre {PROF_MIN:.0f} y {PROF_MAX:.0f} m "
                   f"(rango observado en el dataset)."
        })
    if fila['provincia'] not in CUENCA_PROVINCIAS.get(fila['cuenca'], []):
        avisos.append({
            'level': 'warning',
            'msg': f"La combinacion cuenca **{fila['cuenca']}** + provincia "
                   f"**{fila['provincia']}** no se observa en el dataset: la prediccion "
                   f"se realizara por extrapolacion y es menos confiable."
        })
    if (fila['tipoextraccion'] == 'Sin Sistema de Extracción'
            and fila['tipopozo'] == 'Petrolífero'):
        avisos.append({
            'level': 'info',
            'msg': "Un pozo petrolifero sin sistema de extraccion suele estar inactivo; "
                   "la produccion esperada sera muy baja."
        })
    return avisos


def fila_df(prof, mes, extr, tpozo, cuenca, prov, rec) -> pd.DataFrame:
    return pd.DataFrame([{'profundidad': float(prof), 'mes': int(mes),
                          'tipoextraccion': extr, 'tipopozo': tpozo, 'cuenca': cuenca,
                          'provincia': prov, 'tipo_de_recurso': rec}])[FEATURES]


def registrar_log(tipo: str, entrada: dict, salida: str):
    if 'log' not in st.session_state:
        st.session_state['log'] = []
    st.session_state['log'].append(
        {'timestamp': datetime.now().isoformat(timespec='seconds'),
         'modelo': tipo, **entrada, 'resultado': salida})


def grafico_importancias(imps: dict, titulo: str) -> go.Figure:
    keys = list(imps.keys())[::-1]
    fig = go.Figure(go.Bar(x=[imps[k] for k in keys], y=keys, orientation='h',
                           marker_color=PALETTE['primary']))
    fig.update_layout(**PLOTLY_LAYOUT, title=titulo, xaxis_title='Importancia relativa')
    return fig


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand-block">'
        '<div class="brand">Estim<span>AR</span></div>'
        '<div class="tagline">Estimador de produccion y estado '
        'operativo de pozos de hidrocarburos</div>'
        '</div>',
        unsafe_allow_html=True)
    st.divider()
    pagina = st.radio("Navegacion", NAV_OPTIONS, label_visibility="collapsed")
    st.divider()
    st.markdown(
        '<p class="sidebar-footer">'
        'Modelos XGBoost entrenados sobre 872.186 registros '
        'de la Secretaria de Energia (Cap. IV).<br><br>'
        'TP4 -- IA y Aprendizaje Automatico I -- UCA 2026.<br>'
        'Andrisani, Feser, Lauria, Viccei.</p>',
        unsafe_allow_html=True)

reg_model, clf_model = cargar_modelos()

# ─── PREDICCION INDIVIDUAL ──────────────────────────────────────────────────
if pagina == "Prediccion individual":
    st.subheader("Estimar produccion y estado operativo")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            cuenca = st.selectbox("Cuenca", list(CUENCA_PROVINCIAS.keys()), key='cuenca')
            prov = st.selectbox("Provincia", CUENCA_PROVINCIAS[cuenca], key='prov')
            rec = st.selectbox("Tipo de recurso", TIPOS_RECURSO, key='rec')
            prof = st.number_input("Profundidad (m)", min_value=PROF_MIN,
                                   max_value=PROF_MAX, value=PROF_MEDIANA,
                                   step=50.0, key='prof',
                                   help=f"Rango: {PROF_MIN:.0f} -- {PROF_MAX:.0f} m")
        with c2:
            tpozo = st.selectbox("Tipo de pozo", TIPOS_POZO, key='tpozo',
                                 help="Modelo entrenado principalmente sobre pozos "
                                      "petroliferos y gasiferos.")
            extr = st.selectbox("Sistema de extraccion", TIPOS_EXTRACCION, key='extr')
            mes = st.segmented_control("Mes del año", options=list(range(1, 13)),
                                       default=6, key='mes')

    if st.button("Estimar", type="primary", key='btn_estimar'):
        fila = fila_df(prof, mes, extr, tpozo, cuenca, prov, rec)
        avisos = validar_entrada(fila.iloc[0].to_dict())
        errores = [a for a in avisos if a['level'] == 'error']
        for a in avisos:
            getattr(st, a['level'])(a['msg'])

        if not errores:
            with st.spinner("Ejecutando modelos..."):
                pred = max(float(reg_model.predict(fila)[0]), 0.0)
                proba = float(clf_model.predict_proba(fila)[0, 1])

            st.session_state['last_prediction'] = {
                'pred': pred, 'proba': proba,
                'input': fila.iloc[0].to_dict(), 'extr': extr,
            }
            registrar_log('regresion', fila.iloc[0].to_dict(), f"{pred:.1f} m3/mes")
            estado_txt = 'ACTIVO' if proba >= 0.5 else 'INACTIVO'
            registrar_log('clasificacion', fila.iloc[0].to_dict(),
                          f"{estado_txt} ({proba:.3f})")
        else:
            st.session_state.pop('last_prediction', None)

    if st.session_state.get('last_prediction'):
        r = st.session_state['last_prediction']
        pred = r['pred']
        proba = r['proba']
        q = REF['cuantiles_prod']

        st.markdown("---")

        with st.container(border=True):
            st.markdown("**Produccion estimada**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Produccion mensual", f"{pred:,.1f} m3/mes")
            m2.metric("Barriles/dia", f"{pred * 6.2898 / 30:,.1f} bbl/d")
            m3.metric("Mediana nacional", f"{q['p50']:.1f} m3/mes")
            st.caption("MAE del modelo en test: ~105 m3/mes. El error crece en pozos "
                       "de muy alta produccion.")

            counts = REF['hist_log1p']['counts']
            edges = np.linspace(0, REF['hist_log1p']['edges_max'], len(counts) + 1)
            centros = (edges[:-1] + edges[1:]) / 2
            fig = go.Figure(go.Bar(x=centros, y=counts,
                                   marker_color=PALETTE['primary'],
                                   opacity=0.6, name='Pozos productores'))
            fig.add_vline(x=float(np.log1p(pred)),
                          line_color=PALETTE['highlight'], line_width=3,
                          annotation_text=f"Prediccion: {pred:,.0f} m3",
                          annotation_position="top right",
                          annotation_font_color=PALETTE['highlight'])
            for et, v in [('P25', q['p25']), ('Mediana', q['p50']),
                          ('P75', q['p75']), ('P95', q['p95'])]:
                fig.add_vline(x=float(np.log1p(v)), line_dash='dot',
                              line_color=PALETTE['neutral'],
                              annotation_text=et, annotation_position="bottom",
                              annotation_font_color=PALETTE['neutral'])
            fig.update_layout(**PLOTLY_LAYOUT,
                              title='Prediccion en la distribucion historica (escala log1p)',
                              xaxis_title='log1p(produccion, m3/mes)',
                              yaxis_title='Registros')
            st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            st.markdown("**Estado operativo**")
            col_info, col_chart = st.columns([1, 2])
            with col_info:
                if proba >= 0.5:
                    st.markdown('<div class="status-badge status-activo">ACTIVO</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-badge status-inactivo">INACTIVO</div>',
                                unsafe_allow_html=True)
                st.metric("Probabilidad activo", f"{proba * 100:.1f}%")
                base = REF['tasa_activo_por_extraccion'].get(
                    r['extr'], REF['tasa_activo_global'])
                st.caption(
                    f"Referencia: {base * 100:.1f}% de registros con {r['extr']} "
                    f"estan activos (promedio nacional: "
                    f"{REF['tasa_activo_global'] * 100:.1f}%).")
            with col_chart:
                fig = go.Figure(go.Bar(
                    x=[1 - proba, proba],
                    y=['Inactivo', 'Activo'], orientation='h',
                    marker_color=[PALETTE['negative'], PALETTE['positive']],
                    text=[f"{(1 - proba) * 100:.1f}%", f"{proba * 100:.1f}%"],
                    textposition='auto',
                    textfont=dict(color='#E0E0E0')))
                fig.add_vline(x=0.5, line_dash='dot', line_color=PALETTE['neutral'],
                              annotation_text='umbral 0,5',
                              annotation_font_color=PALETTE['neutral'])
                fig.update_layout(**PLOTLY_LAYOUT,
                                  title='Probabilidades por clase',
                                  xaxis=dict(range=[0, 1], title='Probabilidad'))
                st.plotly_chart(fig, use_container_width=True)
            st.caption("Recall de Activo: 0,99 en test. Precision de Activo: ~0,74.")

# ─── CARGA MASIVA ────────────────────────────────────────────────────────────
elif pagina == "Carga masiva":
    st.subheader("Predicciones en lote")
    st.markdown(f"El archivo debe contener las columnas: `{'`, `'.join(FEATURES)}`.")

    ejemplo = pd.DataFrame([
        {'profundidad': 2900, 'mes': 6, 'tipoextraccion': 'Surgencia Natural',
         'tipopozo': 'Petrolífero', 'cuenca': 'NEUQUINA', 'provincia': 'Neuquén',
         'tipo_de_recurso': 'NO CONVENCIONAL'},
        {'profundidad': 1500, 'mes': 6, 'tipoextraccion': 'Bombeo Mecánico',
         'tipopozo': 'Petrolífero', 'cuenca': 'GOLFO SAN JORGE', 'provincia': 'Chubut',
         'tipo_de_recurso': 'CONVENCIONAL'}])
    st.download_button("Descargar CSV de ejemplo",
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
                               "fuera de rango o no numericos.")
                    lote = lote[~invalidas]
                if len(lote) == 0:
                    st.error("No quedaron filas validas para predecir.")
                elif len(lote) > 50000:
                    st.error("Maximo 50.000 filas por carga.")
                else:
                    with st.spinner(f"Procesando {len(lote):,} registros..."):
                        X = lote[FEATURES].astype({'mes': int})
                        lote['prod_pet_estimado_m3'] = np.maximum(
                            reg_model.predict(X), 0).round(1)
                        lote['proba_activo'] = clf_model.predict_proba(X)[:, 1].round(3)
                        lote['estado_predicho'] = np.where(
                            lote['proba_activo'] >= 0.5, 'Activo', 'Inactivo')
                    st.success(f"Predicciones generadas para {len(lote):,} pozos.")
                    st.dataframe(lote.head(200), use_container_width=True)
                    if len(lote) > 200:
                        st.caption(f"Mostrando 200 de {len(lote):,} filas.")
                    st.download_button("Descargar resultados",
                                       lote.to_csv(index=False).encode('utf-8'),
                                       "predicciones_pozos.csv", "text/csv",
                                       type="primary")
        except Exception as e:
            st.error(f"No se pudo procesar el archivo: {e}")

# ─── SOBRE EL PROYECTO ──────────────────────────────────────────────────────
elif pagina == "Sobre el proyecto":
    st.subheader("Sobre el proyecto")
    st.caption("Proyecto integrador TP1--TP4 sobre datos de produccion de pozos "
               "de gas y petroleo de la Secretaria de Energia de la Nacion.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Registros procesados", "872.186")
    m2.metric("R2 del regresor (test)", "0.459")
    m3.metric("F1 ponderado del clasificador", "0.880")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Modelo de produccion (TP2)**")
        st.markdown(
            "- **Objetivo:** estimar `prod_pet` (m3/mes) de un pozo productivo.\n"
            "- **Algoritmo:** XGBoost con transformacion `log1p` del objetivo.\n"
            "- **Desempeno:** R2 = 0.459, RMSE ~ 434, MAE ~ 106 m3/mes.\n"
            "- **Variables:** solo atributos estructurales; se excluye la "
            "co-produccion para evitar *data leakage*."
        )
    with c2:
        st.markdown("**Modelo de estado (TP3)**")
        st.markdown(
            "- **Objetivo:** clasificar `estado_binario` (Activo / Inactivo).\n"
            "- **Algoritmo:** XGBoost con particion estratificada y agrupada por pozo.\n"
            "- **Desempeno:** F1 ponderado = 0.880, AUC = 0.944, recall Activo = 0.985.\n"
            "- **Balance de clases:** 33.9% Activo / 66.1% Inactivo."
        )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_importancias(REF['importancias_regresion'],
                        'Importancia de variables -- regresor'),
                        use_container_width=True)
    with c2:
        st.plotly_chart(grafico_importancias(REF['importancias_clasificacion'],
                        'Importancia de variables -- clasificador'),
                        use_container_width=True)

    st.caption("En ambos modelos, el sistema de extraccion y el tipo de pozo "
               "concentran la senal. La profundidad y el mes aportan poco.")

    st.divider()
    st.markdown("**Referencias del parque nacional (2025)**")
    c1, c2 = st.columns(2)
    with c1:
        med = REF['mediana_por_cuenca']
        fig = go.Figure(go.Bar(x=list(med.values()), y=list(med.keys()),
                               orientation='h', marker_color=PALETTE['primary']))
        fig.update_layout(**PLOTLY_LAYOUT,
                          title='Mediana de produccion por cuenca',
                          xaxis_title='m3/mes')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        act = REF['tasa_activo_por_extraccion']
        fig = go.Figure(go.Bar(x=[v * 100 for v in act.values()], y=list(act.keys()),
                               orientation='h', marker_color=PALETTE['positive']))
        fig.add_vline(x=REF['tasa_activo_global'] * 100, line_dash='dot',
                      line_color=PALETTE['neutral'],
                      annotation_text='promedio nacional',
                      annotation_font_color=PALETTE['neutral'])
        fig.update_layout(**PLOTLY_LAYOUT,
                          title='% activos por sistema de extraccion',
                          xaxis_title='% activo')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Log de predicciones (sesion)**")
    if st.session_state.get('log'):
        log_df = pd.DataFrame(st.session_state['log'])
        st.dataframe(log_df, use_container_width=True)
        st.download_button("Descargar log",
                           log_df.to_csv(index=False).encode('utf-8'),
                           "log_predicciones.csv", "text/csv")
    else:
        st.caption("Sin predicciones aun.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<p class="disclaimer">EstimAR -- Uso academico. Las predicciones son '
            'estimaciones estadisticas y no reemplazan estudios de ingenieria de '
            'reservorios. Fuente: Secretaria de Energia de la Nacion.</p>',
            unsafe_allow_html=True)
