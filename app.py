# ============================================================================
# OUTLIER DETECTION STUDIO - STREAMLIT
# Port of the R Shiny app (Z-Score, IQR, Isolation Forest)
# Run locally:  streamlit run app.py
# ============================================================================
import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Outlier Detection Studio", page_icon="🚀", layout="wide")

# ============================================================================
# CONSTANTES
# ============================================================================
MAX_UPLOAD_MB = 100       # también configurado en .streamlit/config.toml
IF_MAX_ROWS = 20000       # filas máx. para entrenar Isolation Forest
IF_SAMPLE_SIZE = 5000     # max_samples por árbol
PLOT_MAX_ROWS = 5000      # muestreo para el gráfico interactivo
EXCEL_MAX_ROWS = 50000    # filas máx. a leer de Excel


# ============================================================================
# CARGA DE ARCHIVOS (con caché para que no se relea en cada interacción)
# ============================================================================
@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes) -> list:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data(show_spinner=False)
def read_excel(file_bytes: bytes, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, nrows=EXCEL_MAX_ROWS)


# ============================================================================
# FUNCIONES DE DETECCIÓN
# ============================================================================
def detect_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    out = df.dropna(subset=[col]).copy()
    z = (out[col] - out[col].mean()) / out[col].std(ddof=1)
    out["z_score"] = z
    out["outlier"] = z.abs() > threshold
    return out


def detect_iqr(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> pd.DataFrame:
    out = df.dropna(subset=[col]).copy()
    q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    out["outlier"] = (out[col] < lower) | (out[col] > upper)
    out["lower_bound"] = lower
    out["upper_bound"] = upper
    return out


def detect_isolation_forest(df: pd.DataFrame, cols: list, contamination: float = 0.1) -> pd.DataFrame:
    if len(cols) > 10:
        st.warning("Más de 10 columnas seleccionadas, usando las primeras 10.")
        cols = cols[:10]

    out = df.dropna(subset=cols).copy()

    # Se entrena con una muestra (si es grande) y se puntúan TODAS las filas
    train = out
    if len(out) > IF_MAX_ROWS:
        st.warning(
            f"Dataset grande ({len(out):,} filas). Entrenando con una muestra de "
            f"{IF_MAX_ROWS:,} filas; todas las filas se puntúan igualmente."
        )
        train = out.sample(IF_MAX_ROWS, random_state=42)

    model = IsolationForest(
        n_estimators=100,
        max_samples=min(len(train), IF_SAMPLE_SIZE),
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[cols])

    out["anomaly_score"] = -model.score_samples(out[cols])  # más alto = más anómalo
    out["outlier"] = model.predict(out[cols]) == -1
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ============================================================================
# ESTADO DE SESIÓN
# ============================================================================
for key in ("results", "method", "plot_col"):
    st.session_state.setdefault(key, None)

# ============================================================================
# UI
# ============================================================================
st.title("🚀 Outlier Detection Assessment")

tab_upload, tab_detect, tab_results, tab_info = st.tabs(
    ["📂 Cargar Datos", "🔍 Detección", "📋 Resultados", "ℹ️ Info"]
)

# ── Tab 1: Carga ─────────────────────────────────────────────────────────────
df = None
with tab_upload:
    st.write(
        f"Selecciona un archivo CSV o Excel. Límite: {MAX_UPLOAD_MB} MB. "
        f"Para Excel se cargan las primeras {EXCEL_MAX_ROWS:,} filas."
    )
    file = st.file_uploader("Selecciona archivo (CSV o Excel)", type=["csv", "xlsx", "xls"])

    if file is not None:
        file_bytes = file.getvalue()
        size_mb = round(len(file_bytes) / 1024**2, 2)

        if size_mb > MAX_UPLOAD_MB:
            st.error(f"El archivo ({size_mb} MB) supera el límite de {MAX_UPLOAD_MB} MB.")
        else:
            try:
                ext = file.name.rsplit(".", 1)[-1].lower()
                with st.spinner(f"Cargando archivo de {size_mb} MB..."):
                    if ext == "csv":
                        df = read_csv(file_bytes)
                    else:
                        sheets = get_sheet_names(file_bytes)
                        sheet = st.selectbox("Selecciona Hoja", sheets)
                        df = read_excel(file_bytes, sheet)

                numeric_cols = df.select_dtypes("number").columns.tolist()
                if not numeric_cols:
                    st.error("El archivo no tiene columnas numéricas.")
                    df = None
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Filas", f"{len(df):,}")
                    c2.metric("Columnas", df.shape[1])
                    c3.metric("Columnas numéricas", len(numeric_cols))
                    c4.metric("Tamaño", f"{size_mb} MB")

                    st.subheader("👁️ Vista previa (primeras 20 filas)")
                    st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Error al cargar: {e}")
                df = None
    else:
        # Si se quita el archivo, se limpian los resultados anteriores
        st.session_state["results"] = None

# ── Tab 2: Detección ─────────────────────────────────────────────────────────
with tab_detect:
    if df is None:
        st.info("Primero carga un archivo en la pestaña **Cargar Datos**.")
    else:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        left, right = st.columns([1, 2])

        with left:
            st.subheader("⚙️ Configuración")
            method = st.selectbox("🔍 Método de detección", ["Z-Score", "IQR", "Isolation Forest"])

            z_threshold = iqr_multiplier = contamination = None
            ml_columns = []

            if method == "Z-Score":
                z_threshold = st.slider("Umbral Z-Score", 1.0, 5.0, 3.0, 0.1)
            elif method == "IQR":
                iqr_multiplier = st.slider("Multiplicador IQR", 1.0, 5.0, 1.5, 0.1)
            else:
                contamination = st.slider("Contaminación", 0.01, 0.5, 0.1, 0.01)
                st.caption(
                    f"⚠️ Isolation Forest entrena con máximo {IF_MAX_ROWS:,} filas. "
                    "Para datasets completos usa Z-Score o IQR."
                )

            column = st.selectbox("📊 Selecciona columna", numeric_cols)

            if method == "Isolation Forest":
                ml_columns = st.multiselect(
                    "🤖 Columnas para ML", numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))]
                )

            run = st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True)
            st.caption("💡 Z-Score e IQR son más rápidos y consumen menos memoria.")

        if run:
            try:
                with st.spinner("Ejecutando análisis..."):
                    if method == "Z-Score":
                        res = detect_zscore(df, column, z_threshold)
                        plot_col = column
                    elif method == "IQR":
                        res = detect_iqr(df, column, iqr_multiplier)
                        plot_col = column
                    else:
                        if len(ml_columns) < 2:
                            st.warning("Isolation Forest necesita al menos 2 columnas.")
                            st.stop()
                        res = detect_isolation_forest(df, ml_columns, contamination)
                        plot_col = ml_columns[0]

                st.session_state.update(results=res, method=method, plot_col=plot_col)
            except Exception as e:
                st.error(f"Error en análisis: {e}")

        with right:
            st.subheader("📈 Resultados")
            res = st.session_state["results"]
            if res is None:
                st.info("Configura los parámetros y pulsa **Ejecutar análisis**.")
            else:
                n_total = len(res)
                n_out = int(res["outlier"].sum())
                pct = n_out / n_total * 100 if n_total else 0

                m1, m2, m3 = st.columns(3)
                m1.metric("Total filas", f"{n_total:,}")
                m2.metric("Outliers", f"{n_out:,}")
                m3.metric("Porcentaje", f"{pct:.2f}%")

                plot_data = res
                if n_total > PLOT_MAX_ROWS:
                    plot_data = res.sample(PLOT_MAX_ROWS, random_state=42).sort_index()
                plot_data = plot_data.assign(
                    Índice=plot_data.index,
                    Tipo=plot_data["outlier"].map({True: "Outlier", False: "Normal"}),
                )

                fig = px.scatter(
                    plot_data,
                    x="Índice",
                    y=st.session_state["plot_col"],
                    color="Tipo",
                    color_discrete_map={"Normal": "#2E86AB", "Outlier": "#E74C3C"},
                    opacity=0.6,
                    title=f"Detección de Outliers — {st.session_state['method']}",
                )
                fig.update_layout(legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, use_container_width=True)

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                d1, d2 = st.columns(2)
                d1.download_button(
                    "💾 Descargar resultados",
                    to_csv_bytes(res),
                    file_name=f"outliers_completos_{stamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                d2.download_button(
                    "📋 Descargar solo outliers",
                    to_csv_bytes(res[res["outlier"]]),
                    file_name=f"solo_outliers_{stamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ── Tab 3: Tabla de outliers ─────────────────────────────────────────────────
with tab_results:
    res = st.session_state["results"]
    if res is None:
        st.info("Aún no hay resultados. Ejecuta un análisis en la pestaña **Detección**.")
    else:
        outliers_df = res[res["outlier"]]
        st.subheader("📋 Outliers detectados")
        if outliers_df.empty:
            st.success("No se detectaron outliers.")
        else:
            st.dataframe(outliers_df, use_container_width=True)

# ── Tab 4: Info ──────────────────────────────────────────────────────────────
with tab_info:
    st.subheader("Límites de esta versión")
    st.markdown(
        f"""
- Upload máximo: **{MAX_UPLOAD_MB} MB**
- Excel: primeras **{EXCEL_MAX_ROWS:,}** filas
- Isolation Forest: entrena con muestra de **{IF_MAX_ROWS:,}** filas
- Gráfico: muestra de **{PLOT_MAX_ROWS:,}** puntos
"""
    )
