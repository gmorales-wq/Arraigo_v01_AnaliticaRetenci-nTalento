"""Interfaz Streamlit de la app de riesgo de rotación voluntaria.

Este archivo solo contiene presentación. La lógica está en data.py,
scoring.py, recommendations.py y fairness.py.

Ejecución local:  streamlit run app.py
"""

from __future__ import annotations

import logging
import traceback

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import data
import fairness as fa
import model as ml
import recommendations as rec
import scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

# Tiempo máximo que los resultados en caché permanecen en memoria del servidor.
CACHE_TTL_SECONDS = 1800

st.set_page_config(layout="wide", page_title=cfg.APP_TITLE, page_icon=cfg.APP_ICON)

# CSS estático: solo estilo, sin interpolar ningún dato cargado por el usuario.
_CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px;}
h1 {font-weight: 700; letter-spacing: -0.01em;}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
}
[data-testid="stMetricLabel"] p {font-size: 0.85rem; font-weight: 600; color: #475467;}
[data-testid="stMetricValue"] {font-size: 1.7rem; color: #101828;}
.aviso-uso {
    font-size: 0.86rem; color: #344054; background: #F2F4F7;
    border-left: 3px solid #98A2B3; padding: 0.55rem 0.8rem; border-radius: 6px; margin: 0.2rem 0 1rem 0;
}
.contexto {font-size: 0.9rem; color: #475467; margin-bottom: 0.4rem;}
[data-testid="stSidebar"] {border-right: 1px solid #E4E7EC;}
div[data-testid="stExpander"] details {border-radius: 10px;}
</style>
"""

_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Source Sans Pro, sans-serif", size=13, color="#344054"),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
)
_TRAMO_ORDER = {scoring.COL_TRAMO: list(cfg.TRAMOS)}


# ---------------------------------------------------------------------------
# Estado de la sesión
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Inicializa en la sesión la configuración editable si aún no existe."""
    if "scoring_cfg" not in st.session_state:
        st.session_state.scoring_cfg = cfg.ScoringConfig()
    st.session_state.setdefault("k_min", cfg.K_MIN_DEFAULT)
    st.session_state.setdefault("max_mb", cfg.MAX_UPLOAD_MB_DEFAULT)


def config_from_key(key: tuple) -> cfg.ScoringConfig:
    """Reconstruye un ScoringConfig a partir de su representación hasheable.

    Args:
        key: Tupla devuelta por ``ScoringConfig.as_key``.

    Returns:
        Configuración equivalente.
    """
    weights, zero_at, max_at, umbral_medio, umbral_alto = key
    return cfg.ScoringConfig(
        weights=dict(weights), zero_at=dict(zero_at), max_at=dict(max_at),
        umbral_medio=umbral_medio, umbral_alto=umbral_alto,
    )


# ---------------------------------------------------------------------------
# Funciones en caché (solo transformaciones de datos)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=6, ttl=CACHE_TTL_SECONDS)
def cached_mock(n: int, seed: int, include_sensitive: bool) -> pd.DataFrame:
    """Genera y valida datos ficticios (en caché)."""
    return data.prepare_mock(n, seed, include_sensitive)


@st.cache_data(show_spinner=False, max_entries=3, ttl=CACHE_TTL_SECONDS)
def cached_load(content: bytes, filename: str, max_mb: float) -> tuple[pd.DataFrame | None, data.ValidationReport]:
    """Lee y valida un archivo cargado (en caché)."""
    return data.load_and_validate(content, filename, max_mb)


@st.cache_data(show_spinner=False, max_entries=6, ttl=CACHE_TTL_SECONDS)
def cached_score(df: pd.DataFrame, config_key: tuple) -> pd.DataFrame:
    """Calcula el score de todas las personas (en caché)."""
    return scoring.score_dataframe(df, config_from_key(config_key))


@st.cache_data(show_spinner=False, max_entries=6, ttl=CACHE_TTL_SECONDS)
def cached_plan(scored: pd.DataFrame) -> pd.DataFrame:
    """Construye el plan de acción priorizado (en caché)."""
    return rec.build_action_plan(scored)


@st.cache_data(show_spinner=False, max_entries=6, ttl=CACHE_TTL_SECONDS)
def cached_model(scored: pd.DataFrame, feature_keys: tuple[str, ...]) -> ml.ModelResult:
    """Entrena el modelo predictivo y puntúa a todas las personas (en caché)."""
    return ml.train_model(scored, list(feature_keys))


def clear_session_and_cache() -> None:
    """Vacía la caché de datos del servidor y el estado de la sesión."""
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
def template_csv() -> bytes:
    """Devuelve una plantilla CSV con las columnas obligatorias y 3 filas ficticias."""
    sample = data.generate_mock_data(n=10, seed=1).head(3)
    columns = [spec.name for spec in cfg.SCHEMA if spec.required or spec.name == cfg.COL_BAJA]
    return sample[columns].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def sidebar_data_source() -> tuple[pd.DataFrame | None, data.ValidationReport | None, bool]:
    """Dibuja la selección de origen de datos.

    Returns:
        Tupla (DataFrame validado o None, informe de validación o None,
        indicador de si los datos son ficticios).
    """
    st.sidebar.header("Datos")
    source = st.sidebar.radio(
        "Origen de los datos",
        ("Datos ficticios", "Archivo propio (CSV o XLSX)"),
        key="origen",
        help="Los datos ficticios se generan en memoria con una semilla fija.",
    )

    if source == "Datos ficticios":
        n = st.sidebar.number_input("Número de empleados", min_value=50, max_value=5000, value=500, step=50, key="mock_n")
        seed = st.sidebar.number_input("Semilla", min_value=0, max_value=99999, value=42, step=1, key="mock_seed")
        sensitive = st.sidebar.checkbox(
            "Añadir Edad y Género ficticios",
            value=False,
            key="mock_sensitive",
            help="Solo para probar la auditoría de equidad. Nunca intervienen en el score.",
        )
        with st.spinner("Generando datos ficticios…"):
            return cached_mock(int(n), int(seed), bool(sensitive)), None, True

    st.sidebar.download_button(
        "Descargar plantilla CSV",
        data=template_csv(),
        file_name="plantilla_rotacion.csv",
        mime="text/csv",
        help="Columnas esperadas con tres filas ficticias de ejemplo (separador «;»).",
    )
    max_mb = int(st.session_state.max_mb)
    uploaded = st.sidebar.file_uploader(
        f"Archivo (máx. {max_mb} MB)", type=["csv", "xlsx"], key="uploader",
        help="Los identificadores directos (nombre, email, DNI…) se eliminan al cargar.",
    )
    if uploaded is None:
        return None, None, False
    with st.spinner("Leyendo y validando el archivo…"):
        df, report = cached_load(uploaded.getvalue(), uploaded.name, float(max_mb))
    return df, report, False


def sidebar_filters(scored: pd.DataFrame) -> pd.DataFrame:
    """Dibuja los filtros globales y devuelve el subconjunto filtrado.

    Args:
        scored: DataFrame puntuado completo.

    Returns:
        DataFrame filtrado (una selección vacía equivale a «todos»).
    """
    st.sidebar.header("Filtros")
    st.sidebar.caption("Sin selección = todos.")
    filtered = scored
    filters = (
        (cfg.COL_DEPARTAMENTO, "Departamento"),
        (cfg.COL_NIVEL, "Nivel"),
        (scoring.COL_TRAMO, "Tramo de riesgo"),
        (cfg.COL_MODALIDAD, "Modalidad"),
    )
    for column, label in filters:
        if column not in scored.columns:
            continue
        if column == scoring.COL_TRAMO:
            options = [t for t in cfg.TRAMOS if t in set(scored[column].dropna())]
        elif column == cfg.COL_NIVEL:
            options = [v for v in cfg.NIVELES if v in set(scored[column].dropna())]
        else:
            options = sorted(scored[column].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(label, options, key=f"filtro_{column}")
        if selected:
            filtered = filtered[filtered[column].astype("string").isin(selected).fillna(False)]
    return filtered


def sidebar_privacy() -> None:
    """Dibuja el bloque de limpieza de caché y datos de sesión."""
    st.sidebar.divider()
    st.sidebar.caption(
        f"Los resultados intermedios se guardan en la memoria del servidor un máximo de "
        f"{CACHE_TTL_SECONDS // 60} minutos."
    )
    if st.sidebar.button("Borrar datos de la sesión y caché", type="secondary", width="stretch"):
        clear_session_and_cache()
        st.rerun()


# ---------------------------------------------------------------------------
# Bloques comunes
# ---------------------------------------------------------------------------
def show_validation_report(report: data.ValidationReport) -> None:
    """Muestra el informe de validación de un archivo cargado."""
    errors = sum(1 for i in report.issues if i.severity == "error")
    warnings = sum(1 for i in report.issues if i.severity == "warning")
    if errors:
        st.error(f"El archivo no se puede usar: {errors} error(es) bloqueante(s). Revisa el detalle.")
    elif warnings:
        st.warning(f"Archivo cargado con {warnings} advertencia(s). Las celdas no válidas se tratan como vacías.")
    else:
        st.success(f"Archivo validado: {report.n_rows_out} filas sin incidencias.")
    if report.issues:
        with st.expander("Detalle de la validación", expanded=bool(errors)):
            st.dataframe(report.to_frame(), hide_index=True, width="stretch")


def show_schema() -> None:
    """Muestra el esquema esperado de columnas."""
    schema = pd.DataFrame(
        [
            {
                "Columna": spec.name,
                "Obligatoria": "Sí" if spec.required else "No",
                "Tipo": spec.kind,
                "Rango / valores": (
                    ", ".join(spec.allowed) if spec.allowed
                    else (f"{spec.min_value:g}–{spec.max_value:g}" if spec.min_value is not None and spec.max_value is not None else "")
                ),
                "Descripción": spec.description,
            }
            for spec in cfg.SCHEMA
        ]
    )
    st.dataframe(schema, hide_index=True, width="stretch")


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Formatea una proporción (0–1) como porcentaje con coma decimal."""
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * value:.{decimals}f} %".replace(".", ",")


def fmt_num(value: float | None, decimals: int = 1) -> str:
    """Formatea un número con coma decimal."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}".replace(".", ",")


def active_factor_keys(config: cfg.ScoringConfig) -> list[str]:
    """Devuelve las claves de los factores con peso mayor que cero."""
    return [f.key for f in cfg.FACTORS if config.weights.get(f.key, 0) > 0]


# ---------------------------------------------------------------------------
# Pestaña 1: dashboard
# ---------------------------------------------------------------------------
def tab_dashboard(view: pd.DataFrame, config: cfg.ScoringConfig, is_mock: bool) -> None:
    """Dibuja el dashboard ejecutivo."""
    n = len(view)
    alto = view[scoring.COL_TRAMO] == cfg.TRAMOS[2]
    key_talent = view[scoring.COL_TALENTO_CLAVE]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Personas analizadas", f"{n:,}".replace(",", "."))
    c2.metric("En riesgo alto", f"{int(alto.sum())}", help=f"{fmt_pct(alto.mean())} de las personas analizadas")
    c3.metric("Score medio de riesgo", fmt_num(view[scoring.COL_SCORE].mean()), help="Escala 0–100")
    e24 = pd.to_numeric(view[cfg.COL_EVAL_2024], errors="coerce").mean()
    e25 = pd.to_numeric(view[cfg.COL_EVAL_2025], errors="coerce").mean()
    delta = None if pd.isna(e24) or pd.isna(e25) else f"{e25 - e24:+.2f} vs. 2024".replace(".", ",")
    c4.metric("Evaluación media 2025", fmt_num(e25, 2), delta=delta, help="Único KPI con periodo real de comparación.")

    c5, c6, c7, c8 = st.columns(4)
    if cfg.COL_BAJA in view.columns and view[cfg.COL_BAJA].notna().any():
        baja = view[cfg.COL_BAJA].astype("boolean")
        c5.metric(
            "Bajas voluntarias en el conjunto",
            fmt_pct(baja.mean()),
            help="Proporción de personas con Baja_Voluntaria = Sí en los datos cargados. Solo equivale a una tasa anual si el archivo cubre exactamente 12 meses.",
        )
        baja_talento = baja[key_talent]
        c6.metric(
            "Bajas de talento clave",
            fmt_pct(baja_talento.mean()) if len(baja_talento) else "—",
            help="Proporción de bajas entre personas con evaluación 2025 ≥ 4 o potencial alto.",
        )
    else:
        c5.metric("Bajas voluntarias en el conjunto", "No disponible", help="El archivo no incluye la columna Baja_Voluntaria.")
        c6.metric("Bajas de talento clave", "No disponible")
    c7.metric("Talento clave en riesgo alto", f"{int((alto & key_talent).sum())}")
    c8.metric("En riesgo medio", f"{int((view[scoring.COL_TRAMO] == cfg.TRAMOS[1]).sum())}", help=fmt_pct((view[scoring.COL_TRAMO] == cfg.TRAMOS[1]).mean()))

    st.markdown(f'<p class="contexto"><b>Objetivo de referencia:</b> {cfg.OBJETIVO_NEGOCIO}</p>', unsafe_allow_html=True)

    if n == 0:
        st.info("No hay personas con los filtros actuales.")
        return

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            view, x=scoring.COL_SCORE, color=scoring.COL_TRAMO, nbins=30,
            color_discrete_map=cfg.COLORES_TRAMO, category_orders=_TRAMO_ORDER,
            labels={scoring.COL_SCORE: "Score de riesgo (0–100)", "count": "Personas"},
            title="Distribución del score",
        )
        fig.update_layout(**_PLOTLY_LAYOUT, yaxis_title="Personas", bargap=0.05)
        st.plotly_chart(fig, width="stretch")
    with right:
        by_dept = (
            view.groupby([cfg.COL_DEPARTAMENTO, scoring.COL_TRAMO], observed=True).size().rename("Personas").reset_index()
        )
        by_dept["%"] = 100 * by_dept["Personas"] / by_dept.groupby(cfg.COL_DEPARTAMENTO)["Personas"].transform("sum")
        order = (
            by_dept[by_dept[scoring.COL_TRAMO] == cfg.TRAMOS[2]].sort_values("%", ascending=False)[cfg.COL_DEPARTAMENTO].tolist()
        )
        order += [d for d in by_dept[cfg.COL_DEPARTAMENTO].unique() if d not in order]
        fig = px.bar(
            by_dept, y=cfg.COL_DEPARTAMENTO, x="%", color=scoring.COL_TRAMO, orientation="h",
            color_discrete_map=cfg.COLORES_TRAMO,
            category_orders={**_TRAMO_ORDER, cfg.COL_DEPARTAMENTO: order},
            hover_data={"Personas": True, "%": ":.1f"},
            title="Distribución por tramo y departamento (%)",
        )
        fig.update_layout(**_PLOTLY_LAYOUT, xaxis_title="% del departamento", yaxis_title=None, barmode="stack")
        st.plotly_chart(fig, width="stretch")

    fig = px.scatter(
        view, x=cfg.COL_EVAL_2025, y=scoring.COL_SCORE, color=scoring.COL_TRAMO,
        color_discrete_map=cfg.COLORES_TRAMO, category_orders=_TRAMO_ORDER,
        hover_data={cfg.COL_ID: True, cfg.COL_DEPARTAMENTO: True, cfg.COL_POTENCIAL: True},
        labels={cfg.COL_EVAL_2025: "Evaluación de desempeño 2025", scoring.COL_SCORE: "Score de riesgo"},
        title="Matriz desempeño × riesgo",
        opacity=0.75,
    )
    fig.add_vline(x=cfg.UMBRAL_ALTO_DESEMPENO, line_dash="dot", line_color="#98A2B3",
                  annotation_text="Alto desempeño", annotation_position="top")
    fig.add_hline(y=config.umbral_alto, line_dash="dot", line_color="#C0392B",
                  annotation_text="Umbral riesgo alto", annotation_position="right")
    fig.update_layout(**_PLOTLY_LAYOUT, height=440)
    st.plotly_chart(fig, width="stretch")
    st.caption("Cuadrante superior derecho: alto desempeño y riesgo alto. Es el colectivo que se prioriza en el plan de acción.")

    result = scoring.evaluate_against_outcome(view)
    if result is not None:
        with st.expander("Validación del score frente a bajas reales", expanded=False):
            if is_mock:
                st.warning(
                    "Con datos ficticios esta validación es circular: las bajas se generaron con los mismos factores "
                    "que usa el score. Un buen resultado aquí no demuestra nada sobre datos reales."
                )
            v1, v2, v3 = st.columns(3)
            v1.metric("AUC", fmt_num(result.auc, 3), help="0,5 = azar; 1 = separación perfecta entre bajas y no bajas.")
            v2.metric("Precisión del tramo alto", fmt_pct(result.precision_alto), help="De las personas en tramo alto, qué proporción causó baja.")
            v3.metric("Cobertura del tramo alto", fmt_pct(result.recall_alto), help="De todas las bajas, qué proporción estaba en tramo alto.")
            st.dataframe(result.confusion, width="stretch")
            for message in result.warnings:
                st.caption(f"⚠️ {message}")


# ---------------------------------------------------------------------------
# Pestaña 2: diagnóstico individual
# ---------------------------------------------------------------------------
def tab_individual(view: pd.DataFrame, scored_all: pd.DataFrame, config: cfg.ScoringConfig) -> None:
    """Dibuja el diagnóstico individual."""
    if view.empty:
        st.info("No hay personas con los filtros actuales.")
        return

    table = view.sort_values(scoring.COL_SCORE, ascending=False)[
        [cfg.COL_ID, cfg.COL_DEPARTAMENTO, cfg.COL_PUESTO, cfg.COL_NIVEL, scoring.COL_SCORE,
         scoring.COL_TRAMO, scoring.COL_TALENTO_CLAVE, scoring.COL_FACTORES_DISP]
    ].reset_index(drop=True)

    st.caption("Haz clic en una fila para ver su diagnóstico, o elige un ID en el selector.")
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=300,
        on_select="rerun",
        selection_mode="single-row",
        key="tabla_individual",
        column_config={
            scoring.COL_SCORE: st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            scoring.COL_TRAMO: st.column_config.TextColumn("Tramo"),
            scoring.COL_TALENTO_CLAVE: st.column_config.CheckboxColumn("Talento clave"),
            scoring.COL_FACTORES_DISP: st.column_config.TextColumn("Factores con dato"),
        },
    )
    ids = table[cfg.COL_ID].tolist()
    selected_rows = event.selection.rows if event is not None else []
    if selected_rows and selected_rows[0] < len(ids):
        person_id = ids[selected_rows[0]]
        st.caption("Mostrando la fila seleccionada en la tabla. Quita la selección para usar el selector.")
    else:
        person_id = st.selectbox("Persona (ordenadas de mayor a menor riesgo)", ids, key="selector_persona")

    row = view[view[cfg.COL_ID] == person_id].iloc[0]
    dept = row[cfg.COL_DEPARTAMENTO]
    peers = scored_all[scored_all[cfg.COL_DEPARTAMENTO] == dept]

    st.subheader(f"Diagnóstico de {person_id}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score de riesgo", fmt_num(row[scoring.COL_SCORE]))
    m2.metric("Tramo", str(row[scoring.COL_TRAMO]) if pd.notna(row[scoring.COL_TRAMO]) else "—")
    m3.metric("Talento clave", "Sí" if row[scoring.COL_TALENTO_CLAVE] else "No")
    m4.metric("Factores con dato", row[scoring.COL_FACTORES_DISP], help="Si faltan factores, el score se apoya en menos información.")
    if ml.COL_PROBA in row.index and pd.notna(row[ml.COL_PROBA]):
        st.caption(
            f"Probabilidad estimada por el modelo predictivo: **{fmt_pct(row[ml.COL_PROBA])}** "
            "(detalle en la pestaña «Modelo predictivo»)."
        )

    keys = active_factor_keys(config)
    contrib = scoring.contributions_long(row)
    contrib = contrib[contrib["Clave"].isin(keys)].sort_values("Puntos")

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            contrib, x="Puntos", y="Factor", orientation="h",
            title="Contribución de cada factor al score",
            text=contrib["Puntos"].map(lambda v: fmt_num(v)),
            color_discrete_sequence=["#1F4E79"],
        )
        fig.update_layout(**_PLOTLY_LAYOUT, xaxis_title="Puntos de score", yaxis_title=None, showlegend=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Las contribuciones suman el score total (descomposición aditiva).")
    with right:
        labels = [cfg.FACTORS_BY_KEY[k].label for k in keys]
        person_values = [float(row[f"{scoring.INTENS_PREFIX}{k}"]) if pd.notna(row[f"{scoring.INTENS_PREFIX}{k}"]) else 0.0 for k in keys]
        peer_values = [float(peers[f"{scoring.INTENS_PREFIX}{k}"].mean()) for k in keys]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=peer_values + peer_values[:1], theta=labels + labels[:1], name=f"Media de {dept}",
                                      line=dict(color="#98A2B3"), fill="toself", fillcolor="rgba(152,162,179,0.15)"))
        fig.add_trace(go.Scatterpolar(r=person_values + person_values[:1], theta=labels + labels[:1], name=person_id,
                                      line=dict(color="#C0392B"), fill="toself", fillcolor="rgba(192,57,43,0.12)"))
        fig.update_layout(**_PLOTLY_LAYOUT, title="Intensidad de riesgo por factor (0 = sin riesgo, 1 = máximo)",
                          polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0, 0.5, 1])))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Comparación con la media de todo el departamento ({len(peers)} personas, sin aplicar filtros).")

    variables = [
        cfg.COL_EVAL_2024, cfg.COL_EVAL_2025, cfg.COL_POTENCIAL, cfg.COL_COMPETENCIAS, cfg.COL_ANTIGUEDAD,
        cfg.COL_MESES_PROMO, cfg.COL_COMPA, cfg.COL_CLIMA, cfg.COL_HORAS_EXTRA, cfg.COL_CAMBIOS_RESP,
        cfg.COL_FORMACION, cfg.COL_AUSENCIAS, cfg.COL_MODALIDAD,
    ]
    detail = []
    for variable in variables:
        value = row.get(variable)
        numeric_peers = pd.to_numeric(peers[variable], errors="coerce") if variable in peers else pd.Series(dtype=float)
        median = numeric_peers.median() if numeric_peers.notna().any() else None
        detail.append(
            {
                "Variable": variable,
                "Valor": "—" if pd.isna(value) else str(value),
                f"Mediana {dept}": "—" if median is None or pd.isna(median) else fmt_num(median, 2),
            }
        )
    logit_cols = [c for c in row.index if str(c).startswith(ml.LOGIT_PREFIX)]
    if logit_cols and pd.notna(row.get(ml.COL_PROBA)):
        with st.expander("Explicación del modelo predictivo para esta persona"):
            explain = pd.DataFrame(
                {
                    "Variable": [cfg.FACTORS_BY_KEY[c.removeprefix(ml.LOGIT_PREFIX)].source for c in logit_cols],
                    "Aporte al logit": [float(row[c]) for c in logit_cols],
                }
            ).sort_values("Aporte al logit")
            fig = px.bar(explain, x="Aporte al logit", y="Variable", orientation="h",
                         color=explain["Aporte al logit"] > 0,
                         color_discrete_map={True: "#C0392B", False: "#2E8B57"})
            fig.update_layout(**_PLOTLY_LAYOUT, showlegend=False, yaxis_title=None,
                              title="Positivo: empuja la probabilidad hacia arriba respecto a una persona media")
            st.plotly_chart(fig, width="stretch")

    with st.expander("Datos de la persona frente a su departamento"):
        st.dataframe(pd.DataFrame(detail), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Pestaña 3: plan de acción
# ---------------------------------------------------------------------------
def tab_action_plan(view: pd.DataFrame) -> None:
    """Dibuja el plan de acción priorizado."""
    if view.empty:
        st.info("No hay personas con los filtros actuales.")
        return

    with st.spinner("Preparando el plan de acción…"):
        plan = cached_plan(view)
    if ml.COL_PROBA in view.columns:
        plan = plan.merge(view[[cfg.COL_ID, ml.COL_PROBA]], on=cfg.COL_ID, how="left")

    counts = plan["Prioridad"].value_counts()
    cols = st.columns(5)
    for level, col in zip(range(1, 6), cols):
        col.metric(rec.PRIORITY_LABELS[level].split(" · ")[0], int(counts.get(level, 0)),
                   help=rec.PRIORITY_LABELS[level])
    st.caption(" · ".join(rec.PRIORITY_LABELS[i] for i in range(1, 6)))

    only_actionable = st.toggle("Mostrar solo prioridades P1–P4", value=True, key="solo_accionables")
    shown = plan[plan["Prioridad"] <= 4] if only_actionable else plan
    st.dataframe(
        shown.drop(columns=["Prioridad"]),
        hide_index=True,
        width="stretch",
        height=360,
        column_config={
            "Prioridad_Texto": st.column_config.TextColumn("Prioridad"),
            scoring.COL_SCORE: st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            scoring.COL_TRAMO: st.column_config.TextColumn("Tramo"),
            ml.COL_PROBA: st.column_config.NumberColumn("Prob. modelo", format="percent"),
        },
    )

    export = rec.sanitize_for_csv(shown).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        "Exportar plan a CSV",
        data=export,
        file_name="plan_de_accion_rotacion.csv",
        mime="text/csv",
        help="Separador «;» y coma decimal, para abrirlo directamente en Excel en español.",
    )
    st.caption("El CSV contiene datos seudonimizados de personas: guárdalo con acceso restringido y elimínalo cuando deje de ser necesario.")

    st.divider()
    actionable = plan[plan["Prioridad"] <= 4][cfg.COL_ID].tolist()
    if not actionable:
        st.info("Nadie en prioridades P1–P4 con los filtros actuales.")
        return
    person_id = st.selectbox("Ver recomendaciones de", actionable, key="plan_persona")
    row = view[view[cfg.COL_ID] == person_id].iloc[0]
    recs = rec.recommendations_for(row)
    if not recs:
        st.info("Ningún factor supera el mínimo para considerarse dominante: seguimiento ordinario.")
        return
    cols = st.columns(len(recs))
    for col, (factor_label, rule) in zip(cols, recs):
        with col.container(border=True):
            st.markdown(f"**{rule.titulo}**")
            st.caption(f"Factor: {factor_label}")
            st.markdown("\n".join(f"- {accion}" for accion in rule.acciones))
            st.caption(f"Responsable: {rule.responsable} · Plazo: {rule.plazo}")


# ---------------------------------------------------------------------------
# Pestaña: modelo predictivo
# ---------------------------------------------------------------------------
def tab_model(view: pd.DataFrame, result: ml.ModelResult, is_mock: bool) -> None:
    """Dibuja la pestaña del modelo de aprendizaje automático."""
    st.markdown(
        "Una **regresión logística** aprende del histórico de bajas (`Baja_Voluntaria`) cuánto pesa cada variable "
        "y estima una probabilidad de baja. Complementa al score de reglas: las reglas son transparentes y "
        "funcionan sin histórico; el modelo ajusta los pesos a los datos de tu organización."
    )
    if not result.trained:
        st.info(f"Modelo no disponible. {result.message}")
        st.caption(
            "Para activarlo, carga un archivo con la columna Baja_Voluntaria rellena para el histórico "
            f"(al menos {ml.MIN_POSITIVOS_MODELO} bajas y {ml.MIN_NEGATIVOS_MODELO} permanencias). "
            "Las filas con Baja_Voluntaria vacía se tratan como plantilla actual y el modelo las puntúa."
        )
        return

    st.success(result.message)
    if is_mock:
        st.warning(
            "Con datos ficticios la comparación es circular: las bajas se generaron con reglas parecidas a las "
            "del score. Sirve para probar la herramienta, no para decidir qué enfoque funciona mejor."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC del modelo (validación cruzada)", fmt_num(result.auc_model_cv, 3),
              help="0,5 = azar; 1 = separación perfecta. Calculada con predicciones fuera de muestra.")
    m2.metric("AUC del score de reglas", fmt_num(result.auc_rules, 3), help="Mismas personas, sin entrenamiento.")
    m3.metric("Personas en el entrenamiento", f"{result.n_train}", help=f"{result.n_positives} bajas voluntarias")
    m4.metric("Plantilla actual puntuada", f"{result.n_scored_unlabeled}",
              help="Filas con Baja_Voluntaria vacía, puntuadas por el modelo.")

    left, right = st.columns(2)
    coefs = ml.coefficient_table(result)
    with left:
        chart = coefs.sort_values("Coeficiente")
        fig = px.bar(chart, x="Coeficiente", y="Variable", orientation="h",
                     color=chart["Coeficiente"] > 0, color_discrete_map={True: "#C0392B", False: "#2E8B57"},
                     title="Qué ha aprendido el modelo (coeficientes estandarizados)")
        fig.update_layout(**_PLOTLY_LAYOUT, showlegend=False, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Rojo: a más valor, más riesgo. Verde: a más valor, menos riesgo. Efecto de subir una desviación típica.")
    with right:
        if ml.COL_PROBA in view.columns and view[ml.COL_PROBA].notna().any():
            fig = px.scatter(
                view, x=scoring.COL_SCORE, y=ml.COL_PROBA, color=scoring.COL_TRAMO,
                color_discrete_map=cfg.COLORES_TRAMO, category_orders=_TRAMO_ORDER, opacity=0.7,
                hover_data={cfg.COL_ID: True}, title="Score de reglas frente a probabilidad del modelo",
                labels={scoring.COL_SCORE: "Score de reglas", ml.COL_PROBA: "Probabilidad del modelo"},
            )
            fig.update_layout(**_PLOTLY_LAYOUT, yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.caption("Los puntos alejados de la diagonal son personas en las que ambos enfoques discrepan: merecen una revisión humana.")

    with st.expander("Tabla de coeficientes"):
        st.dataframe(coefs, hide_index=True, width="stretch")

    if ml.COL_PROBA in view.columns:
        top = view.sort_values(ml.COL_PROBA, ascending=False).head(20)[
            [cfg.COL_ID, cfg.COL_DEPARTAMENTO, ml.COL_PROBA, scoring.COL_SCORE, scoring.COL_TRAMO]
        ]
        st.markdown("**20 personas con mayor probabilidad estimada** (con los filtros actuales)")
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            ml.COL_PROBA: st.column_config.ProgressColumn("Probabilidad", min_value=0, max_value=1, format="percent"),
            scoring.COL_SCORE: st.column_config.NumberColumn("Score reglas", format="%.1f"),
        })

    for message in result.warnings:
        st.caption(f"⚠️ {message}")
    st.caption(
        "⚖️ Al usar aprendizaje automático para evaluar a personas trabajadoras, la herramienta podría considerarse "
        "un sistema de IA de alto riesgo según el Reglamento europeo de IA (Anexo III, empleo). Ver README."
    )


# ---------------------------------------------------------------------------
# Pestaña 4: auditoría de equidad
# ---------------------------------------------------------------------------
def tab_fairness(view: pd.DataFrame, attributes: list[str], config: cfg.ScoringConfig) -> None:
    """Dibuja la auditoría de equidad agregada."""
    k = int(st.session_state.k_min)
    st.markdown(
        "Edad y Género **no intervienen en el score**. Esta pestaña comprueba si, aun así, el score se distribuye "
        "de forma distinta entre grupos. Excluir una variable no elimina el sesgo si otras actúan como proxy: "
        "por ejemplo, la antigüedad suele estar correlacionada con la edad."
    )
    st.caption(
        f"Resultados agregados. Se suprimen los grupos con menos de {k} personas y, si solo se suprime uno, "
        "también el siguiente más pequeño, para que no pueda deducirse por diferencia con los totales. "
        "Se aplican los filtros de la barra lateral."
    )

    attribute = st.radio("Atributo", attributes, horizontal=True, key="atributo_equidad")
    table = fa.fairness_table(view, attribute, k)
    st.dataframe(table.drop(columns=["Suprimido"]), hide_index=True, width="stretch")
    visible = table[~table["Suprimido"]]
    if table["Suprimido"].any():
        st.caption("Para recuperar grupos suprimidos, quita filtros o amplía la población analizada.")

    if len(visible) >= 2:
        chart = visible.assign(**{"% riesgo alto": pd.to_numeric(visible["% riesgo alto"])})
        fig = px.bar(chart, x="Grupo", y="% riesgo alto", title="Proporción en riesgo alto por grupo",
                     color_discrete_sequence=["#1F4E79"], text="% riesgo alto")
        fig.update_layout(**_PLOTLY_LAYOUT, yaxis_title="% en riesgo alto", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "La columna «Ratio vs. menor tasa» compara cada grupo con el de menor proporción en riesgo alto. "
            "Como referencia orientativa, en el contexto estadounidense de procesos de selección se usa la regla "
            "de los cuatro quintos (ratios fuera de 0,8–1,25 merecen revisión). No es un criterio legal en España "
            "ni está pensada para este uso: tómala solo como señal para investigar."
        )
    else:
        st.info("Hacen falta al menos dos grupos visibles para comparar.")

    if ml.COL_PROBA in view.columns and view[ml.COL_PROBA].notna().any():
        st.subheader("Probabilidad del modelo predictivo por grupo")
        st.dataframe(fa.probability_by_group(view, attribute, k, ml.COL_PROBA), hide_index=True, width="stretch")
        st.caption("Un modelo entrenado con el histórico puede reproducir sesgos del pasado a través de variables proxy.")

    st.subheader("Posibles variables proxy")
    proxy = fa.proxy_analysis(view, attribute, k, active_factor_keys(config))
    if proxy.empty:
        st.info("No hay datos suficientes para este análisis con los filtros actuales.")
    else:
        st.dataframe(proxy, hide_index=True, width="stretch")
        if attribute == cfg.COL_EDAD:
            st.caption("Correlación de Spearman entre la edad y la intensidad de cada factor. Valores alejados de 0 señalan posibles proxies.")
        else:
            st.caption("Diferencia entre el grupo con mayor y menor intensidad media de cada factor (escala 0–1).")


# ---------------------------------------------------------------------------
# Pestaña 5: configuración
# ---------------------------------------------------------------------------
_WIDGET_PREFIX = "cfgw_"


def _init_config_widgets() -> None:
    """Carga en los widgets de configuración los valores vigentes de la sesión."""
    current: cfg.ScoringConfig = st.session_state.scoring_cfg
    for factor in cfg.FACTORS:
        st.session_state.setdefault(f"{_WIDGET_PREFIX}w_{factor.key}", int(current.weights[factor.key]))
        st.session_state.setdefault(f"{_WIDGET_PREFIX}z_{factor.key}", float(current.zero_at[factor.key]))
        st.session_state.setdefault(f"{_WIDGET_PREFIX}m_{factor.key}", float(current.max_at[factor.key]))
    st.session_state.setdefault(f"{_WIDGET_PREFIX}tramos", (float(current.umbral_medio), float(current.umbral_alto)))
    st.session_state.setdefault(f"{_WIDGET_PREFIX}k", int(st.session_state.k_min))
    st.session_state.setdefault(f"{_WIDGET_PREFIX}mb", int(st.session_state.max_mb))


def _reset_config() -> None:
    """Restaura la configuración por defecto y vacía los widgets asociados."""
    st.session_state.scoring_cfg = cfg.ScoringConfig()
    st.session_state.k_min = cfg.K_MIN_DEFAULT
    st.session_state.max_mb = cfg.MAX_UPLOAD_MB_DEFAULT
    for key in [k for k in st.session_state.keys() if str(k).startswith(_WIDGET_PREFIX)]:
        del st.session_state[key]


def tab_config() -> None:
    """Dibuja la pestaña de configuración del algoritmo."""
    _init_config_widgets()
    st.markdown(
        "Los pesos y umbrales por defecto son **supuestos razonables para una demostración**, no valores "
        "derivados de evidencia empírica. Revísalos con tu equipo y, si tienes histórico de bajas, contrástalos "
        "con él en el bloque de validación del dashboard."
    )

    with st.form("form_config", border=True):
        st.subheader("Factores de riesgo")
        st.caption("Peso 0 = factor desactivado. La intensidad crece linealmente del umbral «sin riesgo» al de «riesgo máximo».")
        for factor in cfg.FACTORS:
            with st.expander(f"{factor.label} · peso {st.session_state[f'{_WIDGET_PREFIX}w_{factor.key}']}"):
                st.caption(f"Variable: {factor.source}")
                st.caption(factor.rationale)
                c1, c2, c3 = st.columns(3)
                c1.slider("Peso", 0, 30, step=1, key=f"{_WIDGET_PREFIX}w_{factor.key}")
                c2.number_input("Umbral sin riesgo", step=0.1, format="%.2f", key=f"{_WIDGET_PREFIX}z_{factor.key}")
                c3.number_input("Umbral de riesgo máximo", step=0.1, format="%.2f", key=f"{_WIDGET_PREFIX}m_{factor.key}")

        st.subheader("Tramos de riesgo")
        st.slider("Umbrales medio y alto (score)", 0.0, 100.0, step=1.0, key=f"{_WIDGET_PREFIX}tramos",
                  help="Por debajo del primero: bajo. Entre ambos: medio. Desde el segundo: alto.")

        st.subheader("Privacidad y límites")
        c1, c2 = st.columns(2)
        c1.number_input("k mínimo por grupo (auditoría de equidad)", min_value=cfg.K_MIN_LOWER_BOUND, max_value=100,
                        step=1, key=f"{_WIDGET_PREFIX}k")
        c2.number_input("Tamaño máximo de archivo (MB)", min_value=1, max_value=cfg.MAX_UPLOAD_MB_SERVER, step=1,
                        key=f"{_WIDGET_PREFIX}mb",
                        help=f"No puede superar el límite del servidor ({cfg.MAX_UPLOAD_MB_SERVER} MB).")

        submitted = st.form_submit_button("Aplicar cambios", type="primary")

    if submitted:
        medio, alto = st.session_state[f"{_WIDGET_PREFIX}tramos"]
        candidate = cfg.ScoringConfig(
            weights={f.key: float(st.session_state[f"{_WIDGET_PREFIX}w_{f.key}"]) for f in cfg.FACTORS},
            zero_at={f.key: float(st.session_state[f"{_WIDGET_PREFIX}z_{f.key}"]) for f in cfg.FACTORS},
            max_at={f.key: float(st.session_state[f"{_WIDGET_PREFIX}m_{f.key}"]) for f in cfg.FACTORS},
            umbral_medio=float(medio),
            umbral_alto=float(alto),
        )
        try:
            scoring.validate_config(candidate)
        except scoring.ScoringConfigError as exc:
            st.error(f"No se han aplicado los cambios: {exc}")
        else:
            st.session_state.scoring_cfg = candidate
            st.session_state.k_min = int(st.session_state[f"{_WIDGET_PREFIX}k"])
            st.session_state.max_mb = int(st.session_state[f"{_WIDGET_PREFIX}mb"])
            st.toast("Configuración aplicada.")
            st.rerun()

    if st.button("Restaurar valores por defecto"):
        _reset_config()
        st.rerun()

    current: cfg.ScoringConfig = st.session_state.scoring_cfg
    total = sum(current.weights.values())
    summary = pd.DataFrame(
        [
            {
                "Factor": f.label,
                "Peso": current.weights[f.key],
                "Peso relativo": fmt_pct(current.weights[f.key] / total if total else None),
                "Sin riesgo en": fmt_num(current.zero_at[f.key], 2),
                "Riesgo máximo en": fmt_num(current.max_at[f.key], 2),
            }
            for f in cfg.FACTORS
        ]
    )
    st.markdown("**Configuración vigente**")
    st.dataframe(summary, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main() -> None:
    """Construye la página completa."""
    st.markdown(_CSS, unsafe_allow_html=True)
    init_state()

    st.title(f"{cfg.APP_ICON} {cfg.APP_TITLE}")
    st.markdown(
        f'<p class="contexto">Detección temprana del riesgo de baja voluntaria en un horizonte de '
        f'{cfg.HORIZONTE_PREDICCION}. Combina un score de reglas explicable con un modelo de aprendizaje automático '
        f'entrenado con el histórico de bajas. Todo se procesa localmente, sin servicios externos.</p>'
        f'<div class="aviso-uso">{cfg.AVISO_USO}</div>',
        unsafe_allow_html=True,
    )

    df, report, is_mock = sidebar_data_source()
    if report is not None:
        show_validation_report(report)

    if df is None:
        sidebar_privacy()
        if report is None:
            st.info("Carga un archivo CSV o XLSX desde la barra lateral, o elige «Datos ficticios». Columnas esperadas:")
            show_schema()
        tab_names = ["⚙️ Configuración"]
        (config_tab,) = st.tabs(tab_names)
        with config_tab:
            tab_config()
        return

    config: cfg.ScoringConfig = st.session_state.scoring_cfg
    try:
        with st.spinner("Calculando el score…"):
            scored = cached_score(df, config.as_key())
    except scoring.ScoringConfigError as exc:
        st.error(f"La configuración del scoring no es válida: {exc}")
        return

    with st.spinner("Entrenando el modelo predictivo…"):
        model_result = cached_model(scored, tuple(active_factor_keys(config)))
    if model_result.trained:
        scored = scored.join(model_result.predictions)

    view = sidebar_filters(scored)
    sidebar_privacy()
    if is_mock:
        st.caption("Estás viendo datos ficticios generados en memoria.")

    attributes = fa.available_attributes(scored)
    tab_names = ["📊 Dashboard", "🔍 Diagnóstico individual", "💡 Plan de acción", "🤖 Modelo predictivo"]
    if attributes:
        tab_names.append("⚖️ Auditoría de equidad")
    tab_names.append("⚙️ Configuración")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_dashboard(view, config, is_mock)
    with tabs[1]:
        tab_individual(view, scored, config)
    with tabs[2]:
        tab_action_plan(view)
    with tabs[3]:
        tab_model(view, model_result, is_mock)
    if attributes:
        with tabs[4]:
            tab_fairness(view, attributes, config)
    with tabs[-1]:
        tab_config()

    st.divider()
    st.caption(cfg.AVISO_USO)


def _log_unhandled(exc: Exception) -> None:
    """Registra un error no controlado sin incluir el mensaje (podría contener datos)."""
    frames = traceback.extract_tb(exc.__traceback__)
    location = f"{frames[-1].filename.rsplit('/', 1)[-1]}:{frames[-1].lineno}" if frames else "desconocido"
    logger.error("Error no controlado %s en %s", type(exc).__name__, location)


try:
    main()
except Exception as exc:  # Único except genérico permitido: nivel superior de la interfaz.
    _log_unhandled(exc)
    st.error(
        "Se ha producido un error inesperado. Prueba a recargar la página o a pulsar «Borrar datos de la sesión y caché». "
        "Si persiste, revisa que el archivo siga la plantilla."
    )
