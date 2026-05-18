from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import utils


st.set_page_config(
    page_title="Proyecto Ecommerce Behavior",
    page_icon=None,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_csv(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_feature_scores(path: str, mtime: float, target: str) -> pd.DataFrame:
    df = cached_csv(path, mtime)
    return utils.compute_feature_scores(df, target)


def load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return cached_csv(str(path), path.stat().st_mtime)


def show_missing(message: str, command: str | None = None) -> None:
    st.warning(message)
    if command:
        st.code(command, language="bash")


def metric_label(metric: str) -> str:
    return {"r2": "R²", "mse": "MSE", "cv_r2_mean": "CV R² promedio"}.get(metric, metric)


def format_metric(value) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "No disponible"


def pipeline_section(original: pd.DataFrame | None, processed: pd.DataFrame | None, target: str | None) -> None:
    st.header("1. Pipeline del proyecto")
    st.write(
        "Esta sección muestra el estado verificable del flujo: descarga de datos, preparación, selección de variables "
        "y generación de artefactos para evaluación."
    )

    st.subheader("Estado de archivos")
    status = utils.project_file_status()
    st.dataframe(status, width="stretch", hide_index=True)

    if original is None:
        show_missing("No se encontró data/orders.csv. Ejecuta primero:", "python src/download_data.py")
    if processed is None:
        show_missing(
            "No se encontró data/processed/df_visualizacion.csv. Ejecuta el script de preprocesamiento real encontrado:",
            "python src/eda_selection.py",
        )

    available = {}
    if original is not None:
        available["Dataset original: data/orders.csv"] = original
    if processed is not None:
        available["Dataset procesado: data/processed/df_visualizacion.csv"] = processed

    if available:
        st.subheader("Resumen del dataset")
        selected_name = st.selectbox("Dataset a inspeccionar", list(available.keys()))
        df = available[selected_name]
        profile = utils.dataframe_profile(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{profile['filas']:,}")
        c2.metric("Columnas", f"{profile['columnas']:,}")
        c3.metric("Numéricas", len(profile["numericas"]))
        c4.metric("Categóricas", len(profile["categoricas"]))

        details = st.checkbox("Mostrar detalles técnicos del dataset", value=False)
        st.dataframe(df.head(20), width="stretch")
        if details:
            st.write("Tipos de datos")
            st.dataframe(
                pd.DataFrame({"columna": df.columns, "tipo": [str(dtype) for dtype in df.dtypes]}),
                width="stretch",
                hide_index=True,
            )
            st.write("Columnas numéricas")
            st.write(", ".join(profile["numericas"]) or "No detectadas.")
            st.write("Columnas categóricas")
            st.write(", ".join(profile["categoricas"]) or "No detectadas.")

        st.subheader("Valores nulos")
        only_nulls = st.checkbox("Mostrar solo columnas con nulos", value=True)
        null_sources = {"Original": original, "Procesado": processed}
        null_frames = []
        for name, source in null_sources.items():
            if source is None:
                continue
            table = utils.null_table(source)
            table.insert(0, "dataset", name)
            null_frames.append(table)
        if null_frames:
            nulls = pd.concat(null_frames, ignore_index=True)
            if only_nulls:
                nulls = nulls[nulls["nulos"] > 0]
            st.dataframe(nulls, width="stretch", hide_index=True)
        else:
            st.info("No hay datasets disponibles para revisar nulos.")

    st.subheader("Etapas del pipeline")
    thresholds = utils.detect_thresholds()
    stage = st.selectbox(
        "Selecciona una etapa",
        [
            "Carga de datos",
            "Verificación de nulos",
            "Imputación",
            "Encoding",
            "Selección de características",
            "Dataset final",
        ],
    )
    stage_info = {
        "Carga de datos": (
            "Lee `data/orders.csv` generado por `src/download_data.py`.",
            "Archivo relacionado: `src/download_data.py` y función `cargar_datos()` en `src/eda_selection.py`.",
        ),
        "Verificación de nulos": (
            "Calcula nulos antes y después del tratamiento.",
            "Función relacionada: `verificar_nulos()` en `src/eda_selection.py`.",
        ),
        "Imputación": (
            f"Usa KNNImputer con {thresholds.get('KNN_NEIGHBORS') or 'valor no detectado'} vecinos, según el código.",
            "Función relacionada: `imputar_knn()` en `src/eda_selection.py`.",
        ),
        "Encoding": (
            "Codifica variables categóricas para selección de características y, más adelante, usa OneHotEncoder/TargetEncoder para modelado.",
            "Funciones relacionadas: `encodear()` y `encoded_data()` en `src/eda_selection.py`.",
        ),
        "Selección de características": (
            "Aplica eliminación de constantes, alta correlación y Mutual Information contra el target.",
            "Función relacionada: `seleccionar_features()` en `src/eda_selection.py`.",
        ),
        "Dataset final": (
            "Guarda las columnas seleccionadas más el target en `data/processed/df_visualizacion.csv`.",
            "Bloque `main()` de `src/eda_selection.py`.",
        ),
    }
    st.write(stage_info[stage][0])
    st.caption(stage_info[stage][1])
    if stage == "Dataset final" and processed is not None:
        st.dataframe(processed.head(20), width="stretch")
    elif stage == "Dataset final":
        show_missing("No se puede comprobar el dataset final porque falta data/processed/df_visualizacion.csv.")

    st.subheader("Comparación original vs procesado")
    if original is not None and processed is not None:
        comparison = utils.compare_columns(original, processed, target)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas original", f"{len(original):,}")
        c2.metric("Filas procesado", f"{len(processed):,}")
        c3.metric("Columnas original", original.shape[1])
        c4.metric("Columnas procesado", processed.shape[1])
        st.write(f"Target detectado: `{target}`" if target else "No fue posible detectar el target con certeza.")
        st.dataframe(comparison, width="stretch", hide_index=True)
    else:
        st.info("La comparación requiere que existan tanto el dataset original como el procesado.")


def variables_section(original: pd.DataFrame | None, processed: pd.DataFrame | None, target: str | None) -> None:
    st.header("2. Justificación de variables")
    st.write(
        "Esta sección ayuda a explicar qué variables se conservaron, cuáles quedaron fuera y qué evidencia cuantitativa "
        "puede revisarse mediante Mutual Information."
    )

    if original is None and processed is None:
        show_missing("No hay datasets disponibles para justificar variables.", "python src/download_data.py")
        return

    status = utils.compare_columns(original, processed, target)
    dropped_in_code = utils.detect_columns_dropped_in_code()
    if dropped_in_code:
        st.info(
            "Columnas eliminadas explícitamente en `encoded_data()`: "
            + ", ".join(f"`{col}`" for col in dropped_in_code)
        )

    states = ["Todas"] + sorted(status["estado"].dropna().unique().tolist())
    state_filter = st.selectbox("Filtrar por estado", states)
    show_removed = st.checkbox("Mostrar variables eliminadas", value=True)
    filtered_status = status.copy()
    if state_filter != "Todas":
        filtered_status = filtered_status[filtered_status["estado"] == state_filter]
    if not show_removed:
        filtered_status = filtered_status[filtered_status["estado"] != "Eliminada"]
    st.dataframe(filtered_status, width="stretch", hide_index=True)

    with st.expander("Explicación académica de Mutual Information"):
        st.write(
            "Mutual Information estima cuánta información aporta una variable para explicar el target. "
            "Un score mayor sugiere más dependencia estadística, pero no implica causalidad. En este proyecto se usa "
            "como apoyo para reducir ruido y seleccionar variables; los motivos específicos solo se afirman cuando el "
            "código o los datos permiten comprobarlos."
        )

    score_source = processed if processed is not None else original
    scores = pd.DataFrame()
    if utils.FEATURE_SCORES_PATH.exists():
        loaded_scores = load_if_exists(utils.FEATURE_SCORES_PATH)
        scores = loaded_scores if loaded_scores is not None else pd.DataFrame()
    elif score_source is not None and target is not None and target in score_source.columns:
        try:
            score_path = utils.PROCESSED_PATH if processed is not None else utils.ORDERS_PATH
            scores = cached_feature_scores(str(score_path), score_path.stat().st_mtime, target)
            st.caption("Scores calculados en memoria. Para guardarlos ejecuta: `python src/generate_feature_scores.py`.")
        except Exception as exc:
            st.warning("No fue posible calcular Mutual Information con los datos actuales.")
            with st.expander("Detalles técnicos"):
                st.exception(exc)
    else:
        st.warning("No se puede calcular Mutual Information porque no se detectó el target en el dataset disponible.")

    if not scores.empty:
        score_table = utils.attach_variable_status(scores, status)
        min_score = st.slider(
            "Score mínimo de Mutual Information",
            min_value=0.0,
            max_value=float(max(score_table["mi_score"].max(), 0.001)),
            value=0.0,
            step=0.001,
        )
        score_table = score_table[score_table["mi_score"] >= min_score]
        st.plotly_chart(
            px.bar(score_table.head(25), x="mi_score", y="variable", color="estado", orientation="h", title="Scores de variables"),
            width="stretch",
        )
        st.dataframe(score_table, width="stretch", hide_index=True)
    else:
        show_missing("No se encontraron scores de variables guardados.", "python src/generate_feature_scores.py")

    st.subheader("Explorador de variable")
    all_variables = status["variable"].sort_values().tolist()
    if not all_variables:
        st.info("No hay variables disponibles para explorar.")
        return
    variable = st.selectbox("Variable", all_variables)
    source = processed if processed is not None and variable in processed.columns else original
    source_name = "procesado" if source is processed else "original"
    variable_row = status[status["variable"] == variable].head(1)
    if source is None or variable not in source.columns:
        st.warning("La variable no está disponible en los datasets cargados.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", source_name)
    c2.metric("Tipo", str(source[variable].dtype))
    c3.metric("Valores únicos", int(source[variable].nunique(dropna=True)))
    c4.metric("Nulos", int(source[variable].isna().sum()))
    if not variable_row.empty:
        st.write(f"Estado: `{variable_row.iloc[0]['estado']}`. {variable_row.iloc[0]['comentario']}")
    if not scores.empty and variable in scores["variable"].values:
        st.write(f"Mutual Information: `{float(scores.loc[scores['variable'] == variable, 'mi_score'].iloc[0]):.4f}`")

    if pd.api.types.is_numeric_dtype(source[variable]):
        st.plotly_chart(px.histogram(source, x=variable, nbins=40, title=f"Distribución de {variable}"), width="stretch")
        if target and target in source.columns and variable != target:
            st.plotly_chart(px.scatter(source, x=variable, y=target, opacity=0.45, title=f"{variable} vs {target}"), width="stretch")
    else:
        counts = source[variable].astype(str).value_counts().head(25).reset_index()
        counts.columns = [variable, "conteo"]
        st.plotly_chart(px.bar(counts, x="conteo", y=variable, orientation="h", title=f"Categorías más frecuentes de {variable}"), width="stretch")
        if target and target in source.columns and variable != target:
            st.plotly_chart(px.box(source, x=variable, y=target, title=f"{target} por {variable}"), width="stretch")


def models_section() -> None:
    st.header("3. Evaluación de modelos")
    st.write(
        "Esta sección compara los modelos definidos en el proyecto usando métricas guardadas. "
        "El dashboard no entrena modelos automáticamente al abrirse."
    )

    detected = utils.detect_model_specs_from_source()
    st.subheader("Modelos detectados en src/models.py")
    if detected.empty:
        st.warning("No fue posible detectar modelos desde `src/models.py`.")
    else:
        st.dataframe(detected, width="stretch", hide_index=True)
        if "SVR" in detected["model_name"].tolist():
            st.caption(
                "`SVR` está definido en `src/models.py`, pero el generador auxiliar de resultados lo excluye "
                "porque en pruebas previas puede tardar más de 40 minutos."
            )

    results = load_if_exists(utils.MODEL_RESULTS_PATH)
    predictions = load_if_exists(utils.MODEL_PREDICTIONS_PATH)
    if results is None:
        show_missing("No se encontraron resultados de modelos guardados.", "python src/generate_model_results.py")
        st.info("MSE mide error cuadrático promedio; R² mide proporción de variabilidad explicada; la validación cruzada ayuda a revisar estabilidad del desempeño.")
        return
    if "SVR" not in results.get("model_name", pd.Series(dtype=str)).tolist():
        st.caption("Los resultados cargados no incluyen SVR; se omite por costo de cómputo.")

    show_scaled = st.checkbox("Mostrar modelos con escalado", value=True)
    show_unscaled = st.checkbox("Mostrar modelos sin escalado", value=True)
    filtered = results.copy()
    allowed = []
    if show_scaled:
        allowed.append("con escalado")
    if show_unscaled:
        allowed.append("sin escalado")
    filtered = filtered[filtered["scaling"].isin(allowed)] if allowed else filtered.iloc[0:0]

    metric = st.radio("Métrica principal", ["r2", "mse", "cv_r2_mean"], format_func=metric_label, horizontal=True)
    best = utils.best_model_row(filtered, metric)
    if best is not None:
        st.success(
            f"Según {metric_label(metric)}, el mejor modelo en estos resultados fue "
            f"{best['model_name']} ({best['scaling']}) con valor {format_metric(best[metric])}."
        )

    display_cols = [
        "model_name",
        "scaling",
        "mse",
        "r2",
        "cv_r2_mean",
        "cv_r2_std",
        "cv_folds",
        "train_rows",
        "test_rows",
    ]
    st.dataframe(filtered[[col for col in display_cols if col in filtered.columns]], width="stretch", hide_index=True)

    if not filtered.empty:
        chart = filtered.copy()
        chart["modelo"] = chart["model_name"] + " (" + chart["scaling"] + ")"
        st.plotly_chart(
            px.bar(chart.sort_values(metric, ascending=(metric == "mse")), x="modelo", y=metric, color="scaling", title=f"{metric_label(metric)} por modelo"),
            width="stretch",
        )

        model_options = (filtered["model_name"] + " | " + filtered["scaling"]).tolist()
        selected = st.selectbox("Modelo a revisar", model_options)
        selected_name, selected_scaling = selected.split(" | ", 1)
        row = filtered[(filtered["model_name"] == selected_name) & (filtered["scaling"] == selected_scaling)].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", format_metric(row.get("mse")))
        c2.metric("R²", format_metric(row.get("r2")))
        c3.metric("CV R² promedio", format_metric(row.get("cv_r2_mean")))
        st.caption("Las métricas se calculan sobre la escala usada por el script de modelado. Si existe `target_scale`, se muestra en la tabla de resultados.")

        if predictions is not None:
            pred = predictions[(predictions["model_name"] == selected_name) & (predictions["scaling"] == selected_scaling)]
            if not pred.empty:
                st.plotly_chart(px.scatter(pred, x="y_true", y="y_pred", opacity=0.5, title="Predicción vs valor real"), width="stretch")
                st.plotly_chart(px.histogram(pred, x="residual", nbins=40, title="Distribución de residuales"), width="stretch")
        else:
            st.info("No se encontraron predicciones guardadas para graficar predicción vs valor real o residuales.")

    with st.expander("Interpretación de métricas"):
        st.write(
            "MSE penaliza errores grandes porque eleva los residuos al cuadrado. R² resume qué proporción de la variabilidad del target explica el modelo. "
            "La validación cruzada repite la evaluación en varios folds y permite observar si el desempeño es estable. Comparar varios modelos reduce el riesgo "
            "de elegir una técnica de forma arbitraria."
        )


def main() -> None:
    st.title("Proyecto Ecommerce Behavior")
    st.write(
        "Esta aplicación resume de forma interactiva el flujo académico del proyecto: preparación de datos, "
        "selección de variables y evaluación de modelos predictivos para estimar el monto total de compra."
    )

    original = load_if_exists(utils.ORDERS_PATH)
    processed = load_if_exists(utils.PROCESSED_PATH)
    available_columns = []
    if processed is not None:
        available_columns = processed.columns.tolist()
    elif original is not None:
        available_columns = original.columns.tolist()
    target = utils.detect_target(available_columns)

    st.sidebar.title("Navegación")
    st.sidebar.write(f"Raíz del proyecto: `{utils.ROOT_DIR.name}`")
    st.sidebar.write(f"Target detectado: `{target or 'No determinado'}`")
    section = st.sidebar.radio(
        "Sección",
        ["1. Pipeline del proyecto", "2. Justificación de variables", "3. Evaluación de modelos"],
    )

    if section == "1. Pipeline del proyecto":
        pipeline_section(original, processed, target)
    elif section == "2. Justificación de variables":
        variables_section(original, processed, target)
    else:
        models_section()


if __name__ == "__main__":
    main()
