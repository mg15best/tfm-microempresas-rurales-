"""
Descarga reproducible de fuentes oficiales del proyecto.

Fuentes disponibles:
- Tabla INE 2073: viajeros y pernoctaciones por provincias.
- Tabla INE 2070: establecimientos, plazas, ocupación y personal
  empleado por provincias.

El script:
- conserva los CSV originales sin transformaciones;
- valida mínimamente cada descarga;
- calcula el hash SHA-256;
- registra cada ejecución en data/metadata/download_log.csv.
"""

from __future__ import annotations

import argparse
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
# Rutas del proyecto
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ine_ocupacion_rural"
)

DOWNLOAD_LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "download_log.csv"
)


# ---------------------------------------------------------------------
# Configuración de las fuentes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SourceConfig:
    """Configuración de una fuente oficial descargable."""

    source_id: str
    source_name: str
    provider: str
    table_id: int
    url: str
    file_format: str
    filename_label: str
    parameters: dict[str, object]
    notes: str


SOURCES: dict[str, SourceConfig] = {
    "2073": SourceConfig(
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
            "CSV raw de demanda turística rural provincial "
            "descargado sin transformaciones."
        ),
    ),
    "2070": SourceConfig(
        source_id="ine_eotr_supply_province",
        source_name=(
            "Establecimientos, plazas, grados de ocupación "
            "y personal empleado por provincias"
        ),
        provider="Instituto Nacional de Estadística",
        table_id=2070,
        url="https://www.ine.es/jaxiT3/files/t/csv_bdsc/2070.csv",
        file_format="csv",
        filename_label="ine_2070_supply_province",
        parameters={
            "table_id": 2070,
            "separator": "semicolon",
            "language": "es",
            "download_scope": "complete_table",
        },
        notes=(
            "CSV raw de oferta, ocupación y empleo turístico rural "
            "provincial descargado sin transformaciones."
        ),
    ),
}


# ---------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un fichero."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def validate_downloaded_file(file_path: Path) -> None:
    """
    Comprueba que el fichero descargado parece un CSV válido.

    Esta validación no modifica los datos.
    """

    file_size = file_path.stat().st_size

    if file_size < 500:
        raise ValueError(
            f"El fichero es demasiado pequeño: {file_size} bytes."
        )

    with file_path.open("rb") as file:
        sample = file.read(4096)

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
            "No se ha detectado el separador ';' esperado."
        )


def append_download_log(
    source: SourceConfig,
    raw_file_path: Path,
    http_status: int,
    content_type: str,
) -> None:
    """Añade una fila al registro histórico de descargas."""

    DOWNLOAD_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        DOWNLOAD_LOG_PATH.exists()
        and DOWNLOAD_LOG_PATH.stat().st_size > 0
    )

    relative_path = (
        raw_file_path
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )

    log_row = {
        "download_id": str(uuid.uuid4()),
        "download_date": datetime.now(
            timezone.utc
        ).isoformat(),
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
        "file_size_bytes": raw_file_path.stat().st_size,
        "file_hash": calculate_sha256(raw_file_path),
        "notes": source.notes,
    }

    fieldnames = list(log_row.keys())

    with DOWNLOAD_LOG_PATH.open(
        mode="a",
        encoding="utf-8",
        newline="",
    ) as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_row)


def download_source(source: SourceConfig) -> Path:
    """Descarga una fuente oficial y devuelve su ruta local."""

    RAW_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    filename = (
        f"{timestamp}_"
        f"{source.filename_label}.csv"
    )

    final_path = RAW_DIRECTORY / filename
    temporary_path = RAW_DIRECTORY / f"{filename}.part"

    request = Request(
        source.url,
        headers={
            "User-Agent": (
                "tfm-microempresas-rurales/1.0 "
                "(academic data project)"
            ),
            "Accept": "text/csv,text/plain,*/*",
        },
    )

    try:
        with urlopen(
            request,
            timeout=120,
        ) as response:
            http_status = getattr(
                response,
                "status",
                200,
            )

            content_type = response.headers.get(
                "Content-Type",
                "not_provided",
            )

            if http_status != 200:
                raise RuntimeError(
                    "El servidor devolvió el estado "
                    f"HTTP {http_status}."
                )

            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(
                    response,
                    output_file,
                )

        validate_downloaded_file(temporary_path)

        temporary_path.replace(final_path)

        append_download_log(
            source=source,
            raw_file_path=final_path,
            http_status=http_status,
            content_type=content_type,
        )

        gitkeep_path = RAW_DIRECTORY / ".gitkeep"

        if gitkeep_path.exists():
            gitkeep_path.unlink()

        return final_path

    except (
        HTTPError,
        URLError,
        TimeoutError,
    ) as error:
        temporary_path.unlink(missing_ok=True)

        raise RuntimeError(
            "No se pudo conectar con la fuente oficial: "
            f"{error}"
        ) from error

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def print_download_result(
    source: SourceConfig,
    downloaded_path: Path,
) -> None:
    """Muestra un resumen de una descarga correcta."""

    relative_path = (
        downloaded_path
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )

    print(f"[OK] Tabla INE: {source.table_id}")
    print(f"[OK] Fichero: {relative_path}")
    print(
        "[OK] Tamaño: "
        f"{downloaded_path.stat().st_size:,} bytes"
    )
    print(
        "[OK] SHA-256: "
        f"{calculate_sha256(downloaded_path)}"
    )
    print()


# ---------------------------------------------------------------------
# Argumentos de terminal
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Lee la fuente solicitada desde la terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Descarga fuentes oficiales de turismo rural."
        )
    )

    parser.add_argument(
        "--source",
        choices=["2073", "2070", "all"],
        default="all",
        help=(
            "Tabla que se desea descargar. "
            "Opciones: 2073, 2070 o all."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------

def main() -> int:
    """Ejecuta una o varias descargas."""

    arguments = parse_arguments()

    if arguments.source == "all":
        selected_sources = list(SOURCES.values())
    else:
        selected_sources = [
            SOURCES[arguments.source]
        ]

    print("=" * 70)
    print("DESCARGA DE DATOS RAW")
    print("=" * 70)
    print(
        f"Fuentes seleccionadas: "
        f"{len(selected_sources)}"
    )
    print()

    try:
        for source in selected_sources:
            print(
                f"Descargando tabla {source.table_id}: "
                f"{source.source_name}"
            )

            downloaded_path = download_source(source)

            print_download_result(
                source,
                downloaded_path,
            )

        log_relative_path = (
            DOWNLOAD_LOG_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        )

        print(
            "[OK] Registro actualizado: "
            f"{log_relative_path}"
        )

        return 0

    except Exception as error:
        print(
            f"[ERROR] {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())