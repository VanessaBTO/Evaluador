"""Extracción de candidaturas PDF y evaluación mediante un LLM compatible con OpenAI."""

from __future__ import annotations

import io
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, BinaryIO

from pypdf import PdfReader


CRITERIA: dict[str, tuple[str, int]] = {
    "adecuacion_gemelo_digital": (
        "Adecuación al Xemelgo Dixital y Modelo de Datos",
        25,
    ),
    "impacto_innovacion": ("Impacto e Innovación", 20),
    "consistencia_viabilidad": (
        "Consistencia y Viabilidad de la Propuesta",
        35,
    ),
    "viabilidad_economica": ("Viabilidad Económica", 10),
    "replicabilidad_escalabilidad": ("Replicabilidad y Escalabilidad", 10),
}

DEFAULT_MODELS = {
    "OpenAI": "gpt-4.1-mini",
    "Groq": "llama-3.3-70b-versatile",
    "Ollama": "llama3.1:8b",
}

DEFAULT_BASE_URLS = {
    "OpenAI": None,
    "Groq": "https://api.groq.com/openai/v1",
    "Ollama": "http://localhost:11434/v1",
}

MAX_PROPOSAL_CHARS = 100_000


class EvaluationError(RuntimeError):
    """Error legible para la interfaz durante la evaluación."""


def _read_pdf_bytes(pdf_file: BinaryIO | bytes | bytearray | str | Path) -> bytes:
    """Normaliza rutas, bytes y objetos subidos por Streamlit."""
    if isinstance(pdf_file, (str, Path)):
        return Path(pdf_file).read_bytes()
    if isinstance(pdf_file, (bytes, bytearray)):
        return bytes(pdf_file)
    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)
    data = pdf_file.read()
    if hasattr(pdf_file, "seek"):
        pdf_file.seek(0)
    return data


def _normalise_field_value(value: Any) -> str:
    """Convierte valores PDF (incluidos checkboxes) en texto útil."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(filter(None, (_normalise_field_value(item) for item in value)))
    text = str(value).strip()
    if text in {"/Off", "Off", "None"}:
        return ""
    return text.removeprefix("/") if text.startswith("/") else text


def extract_text_from_pdf(pdf_file: BinaryIO | bytes | bytearray | str | Path) -> str:
    """Extrae texto de páginas y valores de formulario de una candidatura.

    Se leen ambos canales porque un formulario AcroForm puede mostrar respuestas
    que no forman parte del flujo de texto normal de la página.
    """
    try:
        data = _read_pdf_bytes(pdf_file)
        if not data.startswith(b"%PDF"):
            raise EvaluationError("El archivo seleccionado no parece ser un PDF válido.")

        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise EvaluationError("El PDF está protegido y no puede leerse.") from exc

        sections: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                sections.append(f"[PÁGINA {page_number}]\n{page_text}")

        field_lines: list[str] = []
        for name, field in (reader.get_fields() or {}).items():
            value = _normalise_field_value(field.get("/V"))
            if value:
                field_lines.append(f"{name}: {value}")
        if field_lines:
            sections.append("[CAMPOS CUMPLIMENTADOS DEL FORMULARIO]\n" + "\n".join(field_lines))

        extracted = "\n\n".join(sections).strip()
        if not extracted:
            raise EvaluationError(
                "No se pudo extraer texto ni campos cumplimentados. "
                "El PDF podría ser una imagen escaneada sin OCR."
            )
        return extracted
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"No se pudo leer el PDF: {exc}") from exc


def extract_text_from_path(pdf_path: str | Path) -> str:
    """Extrae texto de un PDF local, devolviendo una cadena vacía si no existe."""
    path = Path(pdf_path)
    return extract_text_from_pdf(path) if path.is_file() else ""


def _system_prompt(bases_text: str = "") -> str:
    bases_note = (
        "\n\nCONTEXTO EXTRAÍDO DE LAS BASES OFICIALES:\n"
        + bases_text[:35_000]
        if bases_text
        else ""
    )
    return f"""
Eres un comité evaluador técnico de AMTEGA para el Concurso de Ideas de
Xemelgos Dixitais. Evalúa exclusivamente las evidencias presentes en la
candidatura. No inventes datos, costes, integraciones, resultados ni capacidades.
La candidatura puede estar redactada en español, gallego o combinar ambos
idiomas. Interpreta correctamente los dos idiomas. Extrae como "entidad" el
nombre exacto de la empresa o entidad proponente, no el nombre del proyecto.

RÚBRICA OBLIGATORIA (100 puntos):
1. Adecuación al Xemelgo Dixital y Modelo de Datos: 25 puntos. Valora que el
   gemelo digital sea central, el modelo y fuentes de datos, datos públicos e
   integración con sistemas de la Administración.
2. Impacto e Innovación: 20 puntos. Valora innovación, mejora de gestión o
   servicios y beneficios verificables para ciudadanía y territorio.
3. Consistencia y Viabilidad de la Propuesta: 35 puntos. Valora claridad del
   problema y solución, arquitectura, madurez, plan, recursos, riesgos,
   accesibilidad de datos y viabilidad técnica.
4. Viabilidad Económica: 10 puntos. Valora costes de desarrollo, implantación y
   mantenimiento, sostenibilidad, eficiencia y reutilización de recursos.
5. Replicabilidad y Escalabilidad: 10 puntos. Valora transferencia a otros
   ámbitos o territorios, adaptaciones, estándares y estrategia de escalado.

NORMAS:
- Usa puntuaciones numéricas, admitiendo incrementos de 0.5.
- Cada justificación debe mencionar evidencias concretas o carencias del formulario.
- Redacta cada justificación, el resumen y las recomendaciones en dos versiones
  equivalentes: español y gallego normativo. No resumas ni reduzcas la versión gallega.
- No premies afirmaciones genéricas sin mecanismo, dato, indicador o plan.
- La puntuación total debe ser la suma exacta de los cinco criterios.
- El dictamen será "RECOMENDADO PARA ENTREVISTA" si el total es >= 70; en caso
  contrario será "NO SELECCIONADO".
- La vertical solo puede ser "Medio Rural", "Turismo" o "No identificada".
- Devuelve exclusivamente un objeto JSON válido, sin Markdown ni texto adicional.

ESQUEMA EXACTO:
{{
  "entidad": "string",
  "vertical": "Medio Rural | Turismo | No identificada",
  "desglose_puntuacion": {{
    "adecuacion_gemelo_digital": {{"puntos": 0, "max": 25, "justificacion": "español", "justificacion_gl": "galego"}},
    "impacto_innovacion": {{"puntos": 0, "max": 20, "justificacion": "español", "justificacion_gl": "galego"}},
    "consistencia_viabilidad": {{"puntos": 0, "max": 35, "justificacion": "español", "justificacion_gl": "galego"}},
    "viabilidad_economica": {{"puntos": 0, "max": 10, "justificacion": "español", "justificacion_gl": "galego"}},
    "replicabilidad_escalabilidad": {{"puntos": 0, "max": 10, "justificacion": "español", "justificacion_gl": "galego"}}
  }},
  "puntuacion_total": 0,
  "dictamen": "RECOMENDADO PARA ENTREVISTA | NO SELECCIONADO",
  "resumen_ejecutivo": "español",
  "resumen_ejecutivo_gl": "galego",
  "recomendaciones_mejora": ["español"],
  "recomendaciones_mejora_gl": ["galego"]
}}
""".strip() + bases_note


def _parse_json_response(content: str) -> dict[str, Any]:
    """Tolera cercas Markdown accidentales y valida el objeto JSON."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EvaluationError("El modelo no devolvió un JSON válido.") from exc
    if not isinstance(result, dict):
        raise EvaluationError("La respuesta del modelo no es un objeto JSON.")
    return result


def _validated_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    """Valida límites y recalcula total y dictamen de forma determinista."""
    breakdown = raw.get("desglose_puntuacion")
    if not isinstance(breakdown, dict):
        raise EvaluationError("Falta el desglose de puntuación en la respuesta.")

    validated_breakdown: dict[str, dict[str, Any]] = {}
    total = 0.0
    for key, (_, maximum) in CRITERIA.items():
        item = breakdown.get(key)
        if not isinstance(item, dict):
            raise EvaluationError(f"Falta el criterio obligatorio: {key}.")
        try:
            points = float(item.get("puntos"))
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Puntuación inválida en {key}.") from exc
        if not math.isfinite(points) or points < 0 or points > maximum:
            raise EvaluationError(f"La puntuación de {key} está fuera de rango.")
        points = round(points * 2) / 2
        justification = str(item.get("justificacion", "")).strip()
        if not justification:
            raise EvaluationError(f"Falta la justificación de {key}.")
        validated_breakdown[key] = {
            "puntos": int(points) if points.is_integer() else points,
            "max": maximum,
            "justificacion": justification,
            "justificacion_gl": (
                str(item.get("justificacion_gl", "")).strip() or justification
            ),
        }
        total += points

    total = round(total, 1)
    recommendations = raw.get("recomendaciones_mejora", [])
    if isinstance(recommendations, str):
        recommendations = [recommendations]
    if not isinstance(recommendations, list):
        recommendations = []
    recommendations_gl = raw.get("recomendaciones_mejora_gl", [])
    if isinstance(recommendations_gl, str):
        recommendations_gl = [recommendations_gl]
    if not isinstance(recommendations_gl, list):
        recommendations_gl = []

    vertical = str(raw.get("vertical", "No identificada")).strip()
    if vertical not in {"Medio Rural", "Turismo", "No identificada"}:
        vertical = "No identificada"

    return {
        "entidad": str(raw.get("entidad", "No identificada")).strip() or "No identificada",
        "vertical": vertical,
        "desglose_puntuacion": validated_breakdown,
        "puntuacion_total": int(total) if total.is_integer() else total,
        "dictamen": (
            "RECOMENDADO PARA ENTREVISTA" if total >= 70 else "NO SELECCIONADO"
        ),
        "resumen_ejecutivo": str(raw.get("resumen_ejecutivo", "")).strip(),
        "resumen_ejecutivo_gl": (
            str(raw.get("resumen_ejecutivo_gl", "")).strip()
            or str(raw.get("resumen_ejecutivo", "")).strip()
        ),
        "recomendaciones_mejora": [
            str(item).strip() for item in recommendations if str(item).strip()
        ],
        "recomendaciones_mejora_gl": (
            [str(item).strip() for item in recommendations_gl if str(item).strip()]
            or [str(item).strip() for item in recommendations if str(item).strip()]
        ),
    }


def evaluate_proposal(
    text: str,
    api_key: str,
    *,
    provider: str = "OpenAI",
    model: str | None = None,
    base_url: str | None = None,
    bases_text: str = "",
) -> dict[str, Any]:
    """Evalúa una candidatura usando OpenAI, Groq u Ollama.

    ``api_key`` se mantiene como segundo argumento para facilitar su uso directo.
    Ollama local acepta una clave ficticia y no requiere credenciales reales.
    """
    provider = provider if provider in DEFAULT_MODELS else "OpenAI"
    selected_model = model or DEFAULT_MODELS[provider]
    selected_base_url = base_url or DEFAULT_BASE_URLS[provider]
    effective_key = api_key.strip() if api_key else ""

    if provider != "Ollama" and not effective_key:
        raise EvaluationError(f"Introduce una API Key válida para {provider}.")
    if provider == "Ollama":
        effective_key = effective_key or "ollama"

    proposal = text.strip()
    if not proposal:
        raise EvaluationError("La candidatura no contiene texto evaluable.")
    if len(proposal) > MAX_PROPOSAL_CHARS:
        proposal = proposal[:MAX_PROPOSAL_CHARS]

    try:
        # Importación diferida: permite usar la extracción PDF sin instalar aún
        # el cliente LLM (por ejemplo, durante pruebas o diagnóstico).
        from openai import OpenAI

        client = OpenAI(api_key=effective_key, base_url=selected_base_url)
        response = client.chat.completions.create(
            model=selected_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _system_prompt(bases_text)},
                {
                    "role": "user",
                    "content": "CANDIDATURA A EVALUAR:\n\n" + proposal,
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise EvaluationError("El modelo devolvió una respuesta vacía.")
        return _validated_evaluation(_parse_json_response(content))
    except EvaluationError:
        raise
    except ImportError as exc:
        raise EvaluationError(
            "Falta la dependencia 'openai'. Ejecuta: pip install -r requirements.txt"
        ) from exc
    except Exception as exc:
        raise EvaluationError(f"Error al conectar con {provider}: {exc}") from exc


def build_text_report(evaluation: dict[str, Any]) -> str:
    """Genera un informe TXT legible y portable."""
    lines = [
        "INFORME DE EVALUACIÓN - CONCURSO DE IDEAS DE XEMELGOS DIXITAIS",
        "=" * 68,
        f"Entidad: {evaluation['entidad']}",
        f"Vertical: {evaluation['vertical']}",
        f"Puntuación total: {evaluation['puntuacion_total']}/100",
        f"Dictamen: {evaluation['dictamen']}",
        "",
        "DESGLOSE DE PUNTUACIÓN",
        "-" * 68,
    ]
    for key, (label, maximum) in CRITERIA.items():
        item = evaluation["desglose_puntuacion"][key]
        lines.extend(
            [
                f"{label}: {item['puntos']}/{maximum}",
                f"Justificación: {item['justificacion']}",
                "",
            ]
        )
    lines.extend(["RESUMEN EJECUTIVO", "-" * 68, evaluation["resumen_ejecutivo"], ""])
    lines.extend(["RECOMENDACIONES DE MEJORA", "-" * 68])
    recommendations = evaluation.get("recomendaciones_mejora") or [
        "No se aportaron recomendaciones adicionales."
    ]
    lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines).strip() + "\n"


def build_pdf_report(
    evaluation: dict[str, Any],
    language: str = "es",
) -> bytes:
    """Genera un informe PDF profesional en español (``es``) o gallego (``gl``)."""
    if language not in {"es", "gl"}:
        raise EvaluationError("El idioma del informe debe ser 'es' o 'gl'.")
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise EvaluationError(
            "Falta la dependencia 'reportlab'. Ejecuta: pip install -r requirements.txt"
        ) from exc

    output = io.BytesIO()
    page_width, _ = A4
    primary = colors.HexColor("#176B66")
    primary_dark = colors.HexColor("#104D4A")
    pale = colors.HexColor("#EAF5F3")
    grey = colors.HexColor("#53636A")
    border = colors.HexColor("#CDDAD8")
    verdict_ok = colors.HexColor("#167D62")
    verdict_no = colors.HexColor("#B64343")

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=19 * mm,
        title="Informe de evaluación - Xemelgos Dixitais",
        author="Sistema Evaluador de Xemelgos Dixitais",
        subject="Evaluación técnica de candidatura",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_dark,
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=grey,
        spaceAfter=8 * mm,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=primary_dark,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )
    criterion_style = ParagraphStyle(
        "Criterion",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=primary_dark,
        spaceAfter=1.5 * mm,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#263238"),
        alignment=TA_LEFT,
        spaceAfter=3 * mm,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=8,
        leading=10,
        textColor=grey,
    )
    score_style = ParagraphStyle(
        "Score",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=primary,
        spaceAfter=1.5 * mm,
    )
    total_style = ParagraphStyle(
        "Total",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=primary_dark,
        alignment=TA_CENTER,
    )
    verdict_style = ParagraphStyle(
        "Verdict",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=(
            verdict_ok
            if evaluation["puntuacion_total"] >= 70
            else verdict_no
        ),
        alignment=TA_CENTER,
    )

    def safe(value: Any) -> str:
        return escape(str(value)).replace("\n", "<br/>")

    def format_score(value: Any) -> str:
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return safe(value)

    translations = {
        "es": {
            "title": "Informe de evaluación técnica",
            "subtitle": "Concurso de Ideas de Xemelgos Dixitais - Rúbrica oficial de la Base Sexta",
            "entity": "Entidad",
            "vertical": "Vertical",
            "summary": "Resumen ejecutivo",
            "breakdown": "Desglose de puntuación",
            "score": "Puntuación",
            "points": "puntos",
            "justification": "Justificación",
            "recommendations": "Recomendaciones de mejora",
            "no_recommendations": "No se aportaron recomendaciones adicionales.",
            "generated": "Generado el",
            "page": "Página",
            "criteria": {key: label for key, (label, _) in CRITERIA.items()},
        },
        "gl": {
            "title": "Informe de avaliación técnica",
            "subtitle": "Concurso de Ideas de Xemelgos Dixitais - Rúbrica oficial da Base Sexta",
            "entity": "Entidade",
            "vertical": "Vertical",
            "summary": "Resumo executivo",
            "breakdown": "Desagregación da puntuación",
            "score": "Puntuación",
            "points": "puntos",
            "justification": "Xustificación",
            "recommendations": "Recomendacións de mellora",
            "no_recommendations": "Non se achegaron recomendacións adicionais.",
            "generated": "Xerado o",
            "page": "Páxina",
            "criteria": {
                "adecuacion_gemelo_digital": "Adecuación ao Xemelgo Dixital e Modelo de Datos",
                "impacto_innovacion": "Impacto e Innovación",
                "consistencia_viabilidad": "Consistencia e Viabilidade da Proposta",
                "viabilidad_economica": "Viabilidade Económica",
                "replicabilidad_escalabilidad": "Replicabilidade e Escalabilidade",
            },
        },
    }
    text = translations[language]
    vertical = evaluation["vertical"]
    if language == "gl" and vertical == "No identificada":
        vertical = "Non identificada"
    verdict = evaluation["dictamen"]
    if language == "gl":
        verdict = (
            "RECOMENDADO PARA ENTREVISTA"
            if evaluation["puntuacion_total"] >= 70
            else "NON SELECCIONADO"
        )

    story: list[Any] = [
        Paragraph(text["title"], title_style),
        Paragraph(text["subtitle"], subtitle_style),
    ]

    summary_data = [
        [
            Paragraph(f"<b>{text['entity']}</b>", small_style),
            Paragraph(safe(evaluation["entidad"]), body_style),
            Paragraph(f"<b>{text['vertical']}</b>", small_style),
            Paragraph(safe(vertical), body_style),
        ]
    ]
    summary_table = Table(
        summary_data,
        colWidths=[24 * mm, 62 * mm, 22 * mm, 51 * mm],
        hAlign="LEFT",
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), pale),
                ("BACKGROUND", (2, 0), (2, 0), pale),
                ("BOX", (0, 0), (-1, -1), 0.6, border),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, border),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 7 * mm)])

    result_table = Table(
        [
            [
                Paragraph(
                    f"{format_score(evaluation['puntuacion_total'])}<font size='11'>/100</font>",
                    total_style,
                ),
                Paragraph(safe(verdict), verdict_style),
            ]
        ],
        colWidths=[50 * mm, 109 * mm],
        rowHeights=[27 * mm],
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), pale),
                ("BOX", (0, 0), (-1, -1), 0.8, primary),
                ("LINEAFTER", (0, 0), (0, 0), 0.6, primary),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend(
        [
            result_table,
            Paragraph(text["summary"], section_style),
            Paragraph(
                safe(
                    evaluation.get(
                        "resumen_ejecutivo_gl" if language == "gl" else "resumen_ejecutivo"
                    )
                    or ("Sen resumo dispoñible." if language == "gl" else "Sin resumen disponible.")
                ),
                body_style,
            ),
            Paragraph(text["breakdown"], section_style),
        ]
    )

    for index, (key, (_, maximum)) in enumerate(CRITERIA.items(), start=1):
        item = evaluation["desglose_puntuacion"][key]
        justification_key = "justificacion_gl" if language == "gl" else "justificacion"
        justification = item.get(justification_key) or item["justificacion"]
        criterion_block = [
            Paragraph(f"{index}. {safe(text['criteria'][key])}", criterion_style),
            Paragraph(
                f"{text['score']}: {format_score(item['puntos'])} "
                f"{'de' if language == 'es' else 'de'} {maximum} {text['points']}",
                score_style,
            ),
            Paragraph(
                f"<b>{text['justification']}:</b> {safe(justification)}",
                body_style,
            ),
        ]
        story.append(KeepTogether(criterion_block))

    story.extend(
        [
            Paragraph(text["recommendations"], section_style),
        ]
    )
    recommendations_key = (
        "recomendaciones_mejora_gl" if language == "gl" else "recomendaciones_mejora"
    )
    recommendations = evaluation.get(recommendations_key) or [
        text["no_recommendations"]
    ]
    for recommendation in recommendations:
        story.append(Paragraph(f"- {safe(recommendation)}", body_style))

    def add_page_chrome(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(primary)
        canvas.setLineWidth(1.2)
        canvas.line(18 * mm, A4[1] - 13 * mm, page_width - 18 * mm, A4[1] - 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(grey)
        canvas.drawString(
            18 * mm,
            10 * mm,
            f"{text['generated']} {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        )
        canvas.drawRightString(
            page_width - 18 * mm,
            10 * mm,
            f"{text['page']} {doc.page}",
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_chrome,
        onLaterPages=add_page_chrome,
    )
    return output.getvalue()
