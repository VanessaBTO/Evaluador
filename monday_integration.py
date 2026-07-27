"""Integración de la evaluación con un tablero de monday.com."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests


MONDAY_API_URL = "https://api.monday.com/v2"
MONDAY_FILE_URL = "https://api.monday.com/v2/file"

# Identificadores facilitados para el tablero Concurso de Ideas.
SCORE_COLUMN_ID = "text_mm5nshpj"
FORM_COLUMN_ID = "file_mm5nh93q"
REPORT_ES_COLUMN_ID = "file_mm5n1a0q"
REPORT_GL_COLUMN_ID = "file_mm5ny5s8"


class MondayIntegrationError(RuntimeError):
    """Error legible al sincronizar una evaluación con Monday."""


def load_private_environment(
    env_path: str | Path = "claves_privadas.env",
) -> None:
    """Carga el archivo local sin reemplazar variables definidas por Render.

    En producción se deben configurar las claves desde Render > Environment.
    """
    path = Path(env_path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


def monday_is_configured() -> bool:
    """Indica si existen las credenciales mínimas."""
    load_private_environment()
    return bool(
        os.getenv("MONDAY_API_TOKEN", "").strip()
        and os.getenv("MONDAY_BOARD_ID", "").strip()
    )


def _configuration() -> tuple[str, int, str | None]:
    load_private_environment()
    token = os.getenv("MONDAY_API_TOKEN", "").strip()
    board_value = os.getenv("MONDAY_BOARD_ID", "").strip()
    group_id = os.getenv("MONDAY_GROUP_ID", "").strip() or None
    if not token or not board_value:
        raise MondayIntegrationError(
            "Faltan MONDAY_API_TOKEN o MONDAY_BOARD_ID en la configuración."
        )
    try:
        board_id = int(board_value)
    except ValueError as exc:
        raise MondayIntegrationError("MONDAY_BOARD_ID debe ser un número.") from exc
    return token, board_id, group_id


def _graphql_request(
    token: str,
    query: str,
    variables: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = requests.post(
            MONDAY_API_URL,
            headers={
                "Authorization": token,
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MondayIntegrationError(f"No se pudo conectar con Monday: {exc}") from exc
    if payload.get("errors"):
        message = payload["errors"][0].get("message", "Error desconocido")
        raise MondayIntegrationError(f"Monday rechazó la operación: {message}")
    return payload.get("data") or {}


def create_evaluation_item(
    entity: str,
    score: int | float,
) -> dict[str, str]:
    """Crea un ítem cuyo nombre es la empresa y asigna su puntuación."""
    token, board_id, group_id = _configuration()
    query = """
    mutation CreateEvaluation(
      $boardId: ID!,
      $groupId: String,
      $itemName: String!,
      $columnValues: JSON!
    ) {
      create_item(
        board_id: $boardId,
        group_id: $groupId,
        item_name: $itemName,
        column_values: $columnValues
      ) {
        id
        name
        url
      }
    }
    """
    score_text = f"{float(score):g}"
    data = _graphql_request(
        token,
        query,
        {
            "boardId": board_id,
            "groupId": group_id,
            "itemName": entity.strip() or "Entidad no identificada",
            "columnValues": json.dumps({SCORE_COLUMN_ID: score_text}),
        },
    )
    item = data.get("create_item")
    if not isinstance(item, dict) or not item.get("id"):
        raise MondayIntegrationError("Monday no devolvió el ítem creado.")
    return {
        "id": str(item["id"]),
        "name": str(item.get("name", entity)),
        "url": str(item.get("url", "")),
    }


def upload_file_to_column(
    item_id: str,
    column_id: str,
    filename: str,
    content: bytes,
    mime_type: str | None = None,
) -> str:
    """Adjunta bytes a una columna Files usando el endpoint multipart."""
    token, _, _ = _configuration()
    if not content:
        raise MondayIntegrationError(f"El archivo {filename} está vacío.")
    safe_filename = Path(filename).name or "archivo.pdf"
    detected_type = mime_type or mimetypes.guess_type(safe_filename)[0]
    detected_type = detected_type or "application/octet-stream"
    query = f"""
    mutation ($file: File!) {{
      add_file_to_column(
        item_id: {int(item_id)},
        column_id: "{column_id}",
        file: $file
      ) {{
        id
      }}
    }}
    """
    try:
        response = requests.post(
            MONDAY_FILE_URL,
            headers={"Authorization": token},
            data={
                "query": query,
                "map": json.dumps({"upload": "variables.file"}),
            },
            files={"upload": (safe_filename, content, detected_type)},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MondayIntegrationError(
            f"No se pudo subir {safe_filename} a Monday: {exc}"
        ) from exc
    if payload.get("errors"):
        message = payload["errors"][0].get("message", "Error desconocido")
        raise MondayIntegrationError(
            f"Monday rechazó el archivo {safe_filename}: {message}"
        )
    asset = (payload.get("data") or {}).get("add_file_to_column") or {}
    if not asset.get("id"):
        raise MondayIntegrationError(
            f"Monday no confirmó la carga de {safe_filename}."
        )
    return str(asset["id"])


def sync_evaluation_to_monday(
    evaluation: dict[str, Any],
    form_filename: str,
    form_pdf: bytes,
    report_es_pdf: bytes,
    report_gl_pdf: bytes,
) -> dict[str, str]:
    """Crea el ítem y sube formulario e informes a sus columnas."""
    item = create_evaluation_item(
        entity=str(evaluation.get("entidad", "Entidad no identificada")),
        score=evaluation.get("puntuacion_total", 0),
    )
    entity_slug = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in item["name"].strip()
    ).strip("_") or "entidad"

    upload_file_to_column(
        item["id"],
        FORM_COLUMN_ID,
        form_filename,
        form_pdf,
        "application/pdf",
    )
    upload_file_to_column(
        item["id"],
        REPORT_ES_COLUMN_ID,
        f"informe_evaluacion_ES_{entity_slug}.pdf",
        report_es_pdf,
        "application/pdf",
    )
    upload_file_to_column(
        item["id"],
        REPORT_GL_COLUMN_ID,
        f"informe_avaliacion_GL_{entity_slug}.pdf",
        report_gl_pdf,
        "application/pdf",
    )
    return item
