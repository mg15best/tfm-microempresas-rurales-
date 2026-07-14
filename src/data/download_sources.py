"""
Descarga reproducible de fuentes oficiales del proyecto.

Fase 4:
- Descarga la tabla 2073 del INE.
- Conserva el CSV original sin transformaciones.
- Calcula el hash SHA-256 del fichero.
- Registra la descarga en data/metadata/download_log.csv.

Este script utiliza únicamente librerías incluidas en Python.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------
# Rutas generales del proyecto
# ---------------------------------------------------------------------

# El script está en project-root/src/data/download_sources.py.
# Por eso parents[2] corresponde a la raíz del repositorio.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "ine_ocupacion_rural"
DOWNLOAD_LOG_PATH = PROJECT_ROOT / "data" / "metadata" / "download_log.csv"


# ---------------------------------------------------------------------
# Configuración de la fuente
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SourceConfig:
    """Configuración mínima de una fuente descargable."""

    source_id: str
    source_name: str
    provider: str
    table_id: int
    url: str
    file_format: str
    filename_label: str
    parameters: dict[str, object]
    notes: str


SOURCE = SourceConfig(
    source_id="ine_eotr_demand_province",
    source_name="Viajeros y pernoctaciones por provincias",
    provider="Instituto Nacional de Estadística",
    table_id=2073,
    url="https://www.ine.es/jaxiT3/files/t/csv_bdsc/2073.csv",
    file_format="csv",
    filename_label="ine_2073_demand_province",
    parameters={
        "table_id": 2073,
        "separator": "semicolon",
        "language": "es",
        "download_scope": "complete_table",
    },
    notes=(
        "Fichero raw descargado sin transformaciones desde la "
        "distribución CSV oficial del INE."
    ),
)


# ---------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """
    Calcula el hash SHA-256 de un fichero.

    El hash funciona como una huella digital:
    si el contenido cambia, el hash también cambia.
    """
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def validate_downloaded_file(file_path: Path) -> None:
    """
    Aplica comprobaciones mínimas al fichero descargado.

    No modifica ni interpreta todavía los datos.
    Solo comprueba que parece un CSV y no una página de error HTML.
    """
    file_size = file_path.stat().st_size

    if file_size < 500:
        raise ValueError(
            f"El fichero descargado es demasiado pequeño: {file_size} bytes."
        )

    with file_path.open("rb") as file:
        sample = file.read(2048)

    normalized_sample = sample.lstrip().lower()

    if (
        normalized_sample.startswith(b"<!doctype html")
        or b"<html" in normalized_sample[:500]
    ):
        raise ValueError(
            "La descarga contiene HTML en lugar del CSV esperado."
        )

    if b";" not in sample:
        raise ValueError(
            "No se ha detectado el separador ';' esperado en el CSV."
        )


def append_download_log(
    source: SourceConfig,
    raw_file_path: Path,
    http_status: int,
    content_type: str,
) -> None:
    """
    Añade una fila al registro histórico de descargas.

    Si download_log.csv no existe, crea primero su cabecera.
    """
    DOWNLOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = (
        DOWNLOAD_LOG_PATH.exists()
        and DOWNLOAD_LOG_PATH.stat().st_size > 0
    )

    relative_path = raw_file_path.relative_to(PROJECT_ROOT).as_posix()
    file_hash = calculate_sha256(raw_file_path)
    file_size = raw_file_path.stat().st_size

    log_row = {
        "download_id": str(uuid.uuid4()),
        "download_date": datetime.now(timezone.utc).isoformat(),
        "source_id": source.source_id,
        "source_name": source.source_name,
        "provider": source.provider,
        "source_url_or_endpoint": source.url,
        "raw_file_path": relative_path,
        "file_format": source.file_format,
        "parameters": json.dumps(
            source.parameters,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "http_status": http_status,
        "content_type": content_type,
        "file_size_bytes": file_size,
        "file_hash": file_hash,
        "notes": source.notes,
    }

    fieldnames = list(log_row.keys())

    with DOWNLOAD_LOG_PATH.open(
        mode="a",
        encoding="utf-8",
        newline="",
    ) as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_row)


def download_source(source: SourceConfig) -> Path:
    """
    Descarga una fuente y devuelve la ruta del fichero generado.

    La descarga se guarda primero como archivo temporal .part.
    Solo se renombra como CSV cuando supera las validaciones mínimas.
    """
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{source.filename_label}.csv"

    final_path = RAW_DIRECTORY / filename
    temporary_path = RAW_DIRECTORY / f"{filename}.part"

    request = Request(
        source.url,
        headers={
            "User-Agent": (
                "tfm-microempresas-rurales/1.0 "
                "(academic data project)"
            ),
            "Accept": "text/csv,*/*",
        },
    )

    try:
        with urlopen(request, timeout=120) as response:
            http_status = getattr(response, "status", 200)
            content_type = response.headers.get(
                "Content-Type",
                "not_provided",
            )

            if http_status != 200:
                raise RuntimeError(
                    f"El servidor devolvió el estado HTTP {http_status}."
                )

            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        validate_downloaded_file(temporary_path)

        # El fichero temporal pasa a ser el fichero raw definitivo.
        temporary_path.replace(final_path)

        append_download_log(
            source=source,
            raw_file_path=final_path,
            http_status=http_status,
            content_type=content_type,
        )

        # Ya no es necesario conservar .gitkeep porque la carpeta
        # contiene un fichero real.
        gitkeep_path = RAW_DIRECTORY / ".gitkeep"

        if gitkeep_path.exists():
            gitkeep_path.unlink()

        return final_path

    except (HTTPError, URLError, TimeoutError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No se pudo conectar con la fuente oficial: {error}"
        ) from error

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------

def main() -> int:
    """Ejecuta la descarga de la tabla 2073 del INE."""
    print("=" * 70)
    print("DESCARGA DE DATOS RAW")
    print("=" * 70)
    print(f"Fuente: {SOURCE.source_name}")
    print(f"Proveedor: {SOURCE.provider}")
    print(f"Tabla INE: {SOURCE.table_id}")
    print()

    try:
        downloaded_path = download_source(SOURCE)
        file_hash = calculate_sha256(downloaded_path)

        print("[OK] Descarga completada.")
        print(
            "[OK] Fichero: "
            f"{downloaded_path.relative_to(PROJECT_ROOT).as_posix()}"
        )
        print(f"[OK] Tamaño: {downloaded_path.stat().st_size:,} bytes")
        print(f"[OK] SHA-256: {file_hash}")
        print(
            "[OK] Registro: "
            f"{DOWNLOAD_LOG_PATH.relative_to(PROJECT_ROOT).as_posix()}"
        )
        return 0

    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())