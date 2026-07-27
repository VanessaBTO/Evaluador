"""Interfaz Streamlit del evaluador de propuestas de Xemelgos Dixitais."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from evaluator import (
    CRITERIA,
    EvaluationError,
    build_pdf_report,
    evaluate_proposal,
    extract_text_from_path,
    extract_text_from_pdf,
)
from monday_integration import (
    MondayIntegrationError,
    monday_is_configured,
    sync_evaluation_to_monday,
)


APP_DIR = Path(__file__).resolve().parent
BASES_PATH = APP_DIR / "01_Bases.pdf"

st.set_page_config(
    page_title="Avaliador Xemelgos Dixitais",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(40, 110, 105, 0.08);
        border: 1px solid rgba(40, 110, 105, 0.20);
        border-radius: 12px;
        padding: 16px;
    }
    .status-ok {color: #138a72; font-weight: 650;}
    .status-error {color: #c23b3b; font-weight: 650;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_bases_text(path: str, modified_time: float) -> str:
    """Cachea las bases y se invalida cuando cambia el archivo."""
    del modified_time
    return extract_text_from_path(path)


def criterion_rows(evaluation: dict) -> list[dict]:
    """Transforma el resultado validado en filas para Streamlit."""
    rows = []
    for key, (label, maximum) in CRITERIA.items():
        item = evaluation["desglose_puntuacion"][key]
        rows.append(
            {
                "Criterio": label,
                "Puntos": item["puntos"],
                "Máximo": maximum,
                "Cumplimiento": round(float(item["puntos"]) / maximum * 100, 1),
                "Justificación técnica": item["justificacion"],
            }
        )
    return rows


with st.sidebar:
    st.header("Configuración")
    if BASES_PATH.is_file():
        st.markdown(
            '<p class="status-ok">● Bases oficiales disponibles</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Archivo: {BASES_PATH.name}")
    else:
        st.markdown(
            '<p class="status-error">● No se encontró 01_Bases.pdf</p>',
            unsafe_allow_html=True,
        )
    if monday_is_configured():
        st.markdown(
            '<p class="status-ok">● Monday conectado</p>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Monday no configurado")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help=(
            "Introduce tu propia clave de OpenAI. Se utiliza únicamente durante "
            "la evaluación y no se guarda en disco."
        ),
    )
    st.divider()
    st.caption("Umbral de entrevista: 70/100")
    st.caption("El resultado es orientativo y debe someterse a revisión humana.")


st.title("🧭 Avaliador de Xemelgos Dixitais")
st.write(
    "Sube una candidatura en PDF para obtener una evaluación técnica conforme "
    "a la Base Sexta del concurso."
)

uploaded_file = st.file_uploader(
    "Formulario de candidatura",
    type=["pdf"],
    accept_multiple_files=False,
    help="PDF basado en 02_Formulario.pdf. Tamaño recomendado: menos de 20 MB.",
)

if uploaded_file:
    st.success(f"Archivo preparado: {uploaded_file.name}")

evaluate_clicked = st.button(
    "🚀 Evaluar candidatura",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)

if evaluate_clicked and uploaded_file is not None:
    if not api_key.strip():
        st.error("Introduce tu OpenAI API Key para realizar la evaluación.")
    elif not BASES_PATH.is_file():
        st.error("No se puede evaluar: falta el archivo 01_Bases.pdf.")
    else:
        try:
            with st.spinner("Extraendo o formulario e aplicando a rúbrica oficial..."):
                uploaded_pdf_bytes = uploaded_file.getvalue()
                proposal_text = extract_text_from_pdf(uploaded_file)
                bases_text = load_bases_text(
                    str(BASES_PATH), BASES_PATH.stat().st_mtime
                )
                result = evaluate_proposal(
                    proposal_text,
                    api_key,
                    provider="OpenAI",
                    model="gpt-4.1-mini",
                    bases_text=bases_text,
                )
                st.session_state["evaluation"] = result
                st.session_state["source_filename"] = uploaded_file.name
                st.session_state.pop("monday_item", None)
                st.session_state.pop("monday_error", None)

                if monday_is_configured():
                    report_es_pdf = build_pdf_report(result, language="es")
                    report_gl_pdf = build_pdf_report(result, language="gl")
                    try:
                        monday_item = sync_evaluation_to_monday(
                            result,
                            uploaded_file.name,
                            uploaded_pdf_bytes,
                            report_es_pdf,
                            report_gl_pdf,
                        )
                        st.session_state["monday_item"] = monday_item
                    except MondayIntegrationError as monday_exc:
                        st.session_state["monday_error"] = str(monday_exc)
        except EvaluationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Se produjo un error inesperado: {exc}")


evaluation = st.session_state.get("evaluation")
if evaluation:
    st.divider()
    st.subheader("Resultado de la evaluación")

    total_col, verdict_col, entity_col = st.columns([1, 1.4, 1.4])
    total_col.metric("Puntuación total", f"{evaluation['puntuacion_total']}/100")
    verdict_col.metric(
        "Dictamen",
        (
            "Apto para entrevista"
            if evaluation["puntuacion_total"] >= 70
            else "No seleccionado"
        ),
    )
    entity_col.metric("Entidad", evaluation["entidad"])
    st.caption(f"Vertical estratégica: {evaluation['vertical']}")

    monday_item = st.session_state.get("monday_item")
    monday_error = st.session_state.get("monday_error")
    if monday_item:
        if monday_item.get("url"):
            st.success(
                f"Evaluación registrada en Monday: "
                f"[{monday_item['name']}]({monday_item['url']})"
            )
        else:
            st.success("Evaluación registrada correctamente en Monday.")
    elif monday_error:
        st.warning(
            "La evaluación se completó, pero no pudo registrarse en Monday: "
            f"{monday_error}"
        )

    st.subheader("Desglose por criterios")
    score_frame = pd.DataFrame(criterion_rows(evaluation))
    st.dataframe(
        score_frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Puntos": st.column_config.NumberColumn(format="%.1f"),
            "Máximo": st.column_config.NumberColumn(format="%d"),
            "Cumplimiento": st.column_config.ProgressColumn(
                "Cumplimiento",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Justificación técnica": st.column_config.TextColumn(width="large"),
        },
    )

    st.subheader("Resumen ejecutivo")
    st.write(evaluation["resumen_ejecutivo"] or "Sin resumen disponible.")

    st.subheader("Recomendaciones de mejora")
    recommendations = evaluation.get("recomendaciones_mejora") or []
    if recommendations:
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
    else:
        st.write("No se indicaron recomendaciones adicionales.")

    report_es = build_pdf_report(evaluation, language="es")
    report_gl = build_pdf_report(evaluation, language="gl")
    has_galician_content = bool(evaluation.get("resumen_ejecutivo_gl")) and all(
        item.get("justificacion_gl")
        for item in evaluation["desglose_puntuacion"].values()
    )
    if not has_galician_content:
        st.info(
            "Vuelve a pulsar «Evaluar candidatura» para generar también los "
            "contenidos técnicos en gallego."
        )
    source_stem = Path(st.session_state.get("source_filename", "candidatura")).stem
    download_es, download_gl = st.columns(2)
    with download_es:
        st.download_button(
            "⬇️ Descargar informe en español",
            data=report_es,
            file_name=f"informe_evaluacion_ES_{source_stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with download_gl:
        st.download_button(
            "⬇️ Descargar informe en galego",
            data=report_gl,
            file_name=f"informe_avaliacion_GL_{source_stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=not has_galician_content,
        )
