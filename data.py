"""Carga, saneado y validación de datos.

Este módulo no depende de Streamlit: recibe bytes y devuelve un DataFrame
limpio junto con un informe de validación. Así se puede probar con pytest.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

import config as cfg
from create_mock_data import generate_mock_data

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning", "info"]

__all__ = [
    "DataLoadError",
    "Issue",
    "ValidationReport",
    "generate_mock_data",
    "load_and_validate",
    "normalize_column_name",
    "numeric_matrix",
    "prepare_mock",
    "read_tabular_file",
    "validate_dataframe",
]

_MAX_ROWS_IN_MESSAGE = 10
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_DNI_NIE_RE = re.compile(r"^[XYZ]?\d{7,8}[A-Z]$", re.IGNORECASE)
_TRUE_VALUES = {"true", "1", "si", "sí", "s", "yes", "y", "verdadero"}
_FALSE_VALUES = {"false", "0", "no", "n", "falso"}


class DataLoadError(Exception):
    """Error que impide leer o usar el archivo. Su mensaje es apto para la interfaz."""


@dataclass
class Issue:
    """Incidencia detectada en la validación.

    Attributes:
        severity: "error" (bloqueante), "warning" (se continúa) o "info".
        column: Columna afectada, o "" si afecta al archivo completo.
        message: Descripción comprensible del problema.
        rows: Filas afectadas, numeradas como en una hoja de cálculo
            (la fila 1 es la cabecera, los datos empiezan en la 2).
    """

    severity: Severity
    column: str
    message: str
    rows: list[int] = field(default_factory=list)

    def rows_text(self) -> str:
        """Devuelve las filas afectadas en texto, truncadas si son muchas."""
        if not self.rows:
            return ""
        shown = ", ".join(str(r) for r in self.rows[:_MAX_ROWS_IN_MESSAGE])
        extra = len(self.rows) - _MAX_ROWS_IN_MESSAGE
        return f"{shown} y {extra} más" if extra > 0 else shown


@dataclass
class ValidationReport:
    """Resultado de la validación de un archivo.

    Attributes:
        issues: Lista de incidencias.
        dropped_pii: Columnas eliminadas por parecer identificadores directos.
        dropped_unknown: Columnas descartadas por no pertenecer al esquema.
        rejected_absence_cause: Columnas rechazadas por describir causas de ausencia.
        n_rows_in: Filas leídas.
        n_rows_out: Filas válidas tras la limpieza.
    """

    issues: list[Issue] = field(default_factory=list)
    dropped_pii: list[str] = field(default_factory=list)
    dropped_unknown: list[str] = field(default_factory=list)
    rejected_absence_cause: list[str] = field(default_factory=list)
    n_rows_in: int = 0
    n_rows_out: int = 0

    @property
    def has_blocking_errors(self) -> bool:
        """Indica si hay al menos un error bloqueante."""
        return any(i.severity == "error" for i in self.issues)

    def add(self, severity: Severity, column: str, message: str, rows: list[int] | None = None) -> None:
        """Añade una incidencia al informe."""
        self.issues.append(Issue(severity, column, message, rows or []))

    def to_frame(self) -> pd.DataFrame:
        """Devuelve las incidencias como DataFrame para mostrarlas en la interfaz."""
        etiquetas = {"error": "Error bloqueante", "warning": "Advertencia", "info": "Información"}
        return pd.DataFrame(
            [
                {
                    "Tipo": etiquetas[i.severity],
                    "Columna": i.column or "(archivo)",
                    "Problema": i.message,
                    "Filas": i.rows_text(),
                }
                for i in self.issues
            ],
            columns=["Tipo", "Columna", "Problema", "Filas"],
        )


# ---------------------------------------------------------------------------
# Utilidades de normalización
# ---------------------------------------------------------------------------
def _strip_accents(text: str) -> str:
    """Quita tildes y diacríticos de un texto."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_column_name(name: object) -> str:
    """Normaliza un nombre de columna: sin tildes, minúsculas y con "_".

    Args:
        name: Nombre original (puede no ser texto).

    Returns:
        Nombre normalizado, p. ej. "Evaluación Desempeño 2025" -> "evaluacion_desempeno_2025".
    """
    text = _strip_accents(str(name)).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


_CANONICAL_BY_NORMALIZED: dict[str, str] = {
    **{normalize_column_name(spec.name): spec.name for spec in cfg.SCHEMA},
    **cfg.COLUMN_SYNONYMS,
}


def _matches_any(normalized: str, patterns: tuple[str, ...]) -> bool:
    """Indica si un nombre normalizado contiene alguno de los fragmentos dados.

    Los fragmentos cortos (3 letras o menos) solo coinciden como palabra
    completa entre guiones bajos para evitar falsos positivos (p. ej. "nie"
    dentro de "ingeniero").
    """
    tokens = normalized.split("_")
    for pattern in patterns:
        if len(pattern.strip("_")) <= 3:
            if pattern.strip("_") in tokens:
                return True
        elif pattern in normalized:
            return True
    return False


def _excel_rows(index: pd.Index) -> list[int]:
    """Convierte posiciones de fila (0, 1, ...) a filas de hoja de cálculo (2, 3, ...)."""
    return [int(i) + 2 for i in index]


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def read_tabular_file(content: bytes, filename: str, max_mb: float) -> pd.DataFrame:
    """Lee un CSV o XLSX desde bytes.

    Args:
        content: Contenido del archivo.
        filename: Nombre del archivo (se usa para decidir el formato).
        max_mb: Tamaño máximo permitido en megabytes.

    Returns:
        DataFrame con todas las columnas leídas como texto u objeto.

    Raises:
        DataLoadError: Si el archivo supera el tamaño, tiene un formato no
            admitido, está vacío o no se puede interpretar.
    """
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise DataLoadError(f"El archivo ocupa {size_mb:.1f} MB y el máximo configurado es {max_mb:.0f} MB.")
    if not content:
        raise DataLoadError("El archivo está vacío.")

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if extension == "csv":
            df = _read_csv(content)
        elif extension == "xlsx":
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl", dtype=object, nrows=cfg.MAX_ROWS + 1)
        else:
            raise DataLoadError("Formato no admitido. Usa un archivo .csv o .xlsx.")
    except DataLoadError:
        raise
    except (zipfile.BadZipFile, InvalidFileException) as exc:
        logger.warning("Archivo XLSX no válido: %s", type(exc).__name__)
        raise DataLoadError("El archivo .xlsx parece estar dañado, protegido con contraseña o no es un Excel real.") from exc
    except (ValueError, KeyError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("Fallo al leer archivo tabular: %s", type(exc).__name__)
        raise DataLoadError(
            "No se ha podido leer el archivo. Comprueba que es un CSV o XLSX válido y que no está protegido."
        ) from exc

    if df.empty or len(df.columns) == 0:
        raise DataLoadError("El archivo no contiene datos.")
    if len(df) > cfg.MAX_ROWS:
        raise DataLoadError(f"El archivo supera el máximo de {cfg.MAX_ROWS:,} filas.".replace(",", "."))
    return df


def _read_csv(content: bytes) -> pd.DataFrame:
    """Lee un CSV detectando codificación (UTF-8 o Latin-1) y separador."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodifica cualquier secuencia de bytes
        raise DataLoadError("No se reconoce la codificación del archivo. Guárdalo como UTF-8.")

    first_line = text.split("\n", 1)[0]
    candidates = {sep: first_line.count(sep) for sep in (";", ",", "\t", "|")}
    separator = max(candidates, key=candidates.get)
    if candidates[separator] == 0:
        raise DataLoadError("No se ha detectado el separador del CSV (se esperaba ';', ',', tabulador o '|').")
    return pd.read_csv(io.StringIO(text), sep=separator, dtype=str, keep_default_na=True, nrows=cfg.MAX_ROWS + 1)


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    """Convierte a número aceptando coma decimal ("3,5")."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    as_text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(as_text, errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    """Convierte valores tipo Sí/No, 1/0, True/False a booleano nullable."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.isin(_TRUE_VALUES).fillna(False)] = True
    result[text.isin(_FALSE_VALUES).fillna(False)] = False
    return result


def _normalize_categories(series: pd.Series, allowed: tuple[str, ...]) -> pd.Series:
    """Ajusta mayúsculas y tildes a los valores admitidos ("hibrido" -> "Híbrido")."""
    lookup = {normalize_column_name(value): value for value in allowed}
    lookup.update({"manager": "Mánager"} if "Mánager" in allowed else {})
    text = series.astype("string").str.strip()
    mapped = text.map(lambda v: lookup.get(normalize_column_name(v)) if isinstance(v, str) else pd.NA)
    return mapped.astype("string")


def _validate_id(df: pd.DataFrame, report: ValidationReport) -> None:
    """Valida la columna de ID: presente, única y sin aspecto de identificador directo."""
    ids = df[cfg.COL_ID].astype("string").str.strip()
    df[cfg.COL_ID] = ids

    missing = ids.isna() | (ids == "")
    if missing.any():
        report.add("error", cfg.COL_ID, "Hay filas sin identificador.", _excel_rows(df.index[missing]))

    duplicated = ids.duplicated(keep=False) & ~missing
    if duplicated.any():
        report.add("error", cfg.COL_ID, "Hay identificadores duplicados.", _excel_rows(df.index[duplicated]))

    looks_direct = ids.fillna("").map(lambda v: bool(_EMAIL_RE.search(v) or _DNI_NIE_RE.match(v)))
    if looks_direct.any():
        report.add(
            "error",
            cfg.COL_ID,
            "Algunos IDs parecen un email o un DNI/NIE. Sustitúyelos por un seudónimo (p. ej. EMP-0001) antes de cargar el archivo.",
            _excel_rows(df.index[looks_direct]),
        )


def _validate_column(df: pd.DataFrame, spec: cfg.ColumnSpec, report: ValidationReport) -> None:
    """Convierte una columna a su tipo y marca como vacíos los valores no válidos."""
    original = df[spec.name]
    original_missing = original.isna() | (original.astype("string").str.strip() == "")

    if spec.kind in ("int", "float"):
        converted = _to_numeric(original)
        bad_type = converted.isna() & ~original_missing
        if bad_type.any():
            report.add("warning", spec.name, "Valores no numéricos; se tratan como vacíos.", _excel_rows(df.index[bad_type]))
        out_of_range = pd.Series(False, index=df.index)
        if spec.min_value is not None:
            out_of_range |= converted < spec.min_value
        if spec.max_value is not None:
            out_of_range |= converted > spec.max_value
        out_of_range = out_of_range.fillna(False)
        if out_of_range.any():
            rango = f"{spec.min_value:g}–{spec.max_value:g}" if spec.max_value is not None else f"≥ {spec.min_value:g}"
            report.add(
                "warning",
                spec.name,
                f"Valores fuera de rango ({rango}); se tratan como vacíos.",
                _excel_rows(df.index[out_of_range]),
            )
            converted = converted.mask(out_of_range)
        if spec.kind == "int":
            non_integer = (converted.notna()) & (converted != converted.round())
            if non_integer.any():
                report.add("info", spec.name, "Valores con decimales; se redondean al entero más próximo.", _excel_rows(df.index[non_integer]))
            converted = converted.round().astype("Int64")
        else:
            converted = converted.astype("Float64")
        df[spec.name] = converted

    elif spec.kind in ("ordinal", "category"):
        if spec.allowed:
            converted = _normalize_categories(original, spec.allowed)
            not_allowed = converted.isna() & ~original_missing
            if not_allowed.any():
                report.add(
                    "warning",
                    spec.name,
                    f"Valores no admitidos (se esperaba: {', '.join(spec.allowed)}); se tratan como vacíos.",
                    _excel_rows(df.index[not_allowed]),
                )
        else:
            converted = original.astype("string").str.strip().replace("", pd.NA)
        df[spec.name] = converted

    elif spec.kind == "bool":
        converted = _to_bool(original)
        not_bool = converted.isna() & ~original_missing
        if not_bool.any():
            report.add("warning", spec.name, "Valores no reconocidos como Sí/No; se tratan como vacíos.", _excel_rows(df.index[not_bool]))
        df[spec.name] = converted

    if spec.required and spec.kind != "id":
        missing_now = df[spec.name].isna()
        share = float(missing_now.mean()) if len(df) else 0.0
        if share > 0.2:
            report.add(
                "warning",
                spec.name,
                f"El {share:.0%} de las filas no tiene un valor válido. El score de esas personas se calcula con menos factores.",
            )


def validate_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Sanea y valida un DataFrame leído de un archivo.

    Pasos: renombrado a nombres canónicos, eliminación de identificadores
    directos, rechazo de causas de ausencia, descarte de columnas ajenas al
    esquema (minimización), comprobación de columnas obligatorias y
    conversión de tipos y rangos.

    Args:
        raw: DataFrame tal como se ha leído.

    Returns:
        Tupla (DataFrame limpio, informe). Si el informe tiene errores
        bloqueantes, el DataFrame no debe usarse.
    """
    report = ValidationReport(n_rows_in=len(raw))
    df = raw.copy()
    df.columns = [str(c) for c in df.columns]

    rename: dict[str, str] = {}
    to_drop: list[str] = []
    seen_canonical: set[str] = set()
    for column in df.columns:
        normalized = normalize_column_name(column)
        canonical = _CANONICAL_BY_NORMALIZED.get(normalized)
        if canonical:
            if canonical in seen_canonical:
                report.add("warning", column, f"Columna duplicada de «{canonical}»; se ignora esta copia.")
                to_drop.append(column)
                continue
            seen_canonical.add(canonical)
            rename[column] = canonical
        elif _matches_any(normalized, cfg.ABSENCE_CAUSE_PATTERNS):
            report.rejected_absence_cause.append(column)
            to_drop.append(column)
        elif _matches_any(normalized, cfg.PII_NAME_PATTERNS):
            report.dropped_pii.append(column)
            to_drop.append(column)
        else:
            report.dropped_unknown.append(column)
            to_drop.append(column)

    df = df.drop(columns=to_drop).rename(columns=rename)

    if report.dropped_pii:
        report.add(
            "warning",
            ", ".join(report.dropped_pii),
            "Columnas con posibles identificadores personales directos. Se han eliminado y no se usan.",
        )
    if report.rejected_absence_cause:
        report.add(
            "warning",
            ", ".join(report.rejected_absence_cause),
            "Columnas con causa o diagnóstico de ausencias (posibles datos de salud). Se han rechazado y no se cargan.",
        )
    if report.dropped_unknown:
        report.add(
            "info",
            ", ".join(report.dropped_unknown),
            "Columnas que no forman parte del esquema. Se descartan por minimización de datos.",
        )

    missing_required = [spec.name for spec in cfg.SCHEMA if spec.required and spec.name not in df.columns]
    if missing_required:
        report.add("error", ", ".join(missing_required), "Faltan columnas obligatorias.")
        return df, report

    _validate_id(df, report)
    for spec in cfg.SCHEMA:
        if spec.name in df.columns and spec.kind != "id":
            _validate_column(df, spec, report)

    if cfg.COL_MESES_PROMO in df.columns and cfg.COL_ANTIGUEDAD in df.columns:
        inconsistent = (df[cfg.COL_MESES_PROMO] > df[cfg.COL_ANTIGUEDAD]).fillna(False)
        if inconsistent.any():
            report.add(
                "warning",
                cfg.COL_MESES_PROMO,
                "Meses desde la última promoción superiores a la antigüedad. Revisa el dato; se mantiene tal cual.",
                _excel_rows(df.index[inconsistent]),
            )

    ordered = [spec.name for spec in cfg.SCHEMA if spec.name in df.columns]
    df = df[ordered].reset_index(drop=True)
    report.n_rows_out = len(df)
    if report.n_rows_out == 0:
        report.add("error", "", "No quedan filas válidas tras la validación.")
    return df, report


def load_and_validate(content: bytes, filename: str, max_mb: float) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Lee y valida un archivo en un solo paso.

    Args:
        content: Contenido del archivo.
        filename: Nombre del archivo.
        max_mb: Tamaño máximo permitido en megabytes.

    Returns:
        Tupla (DataFrame limpio o None si hay errores bloqueantes, informe).
    """
    try:
        raw = read_tabular_file(content, filename, max_mb)
    except DataLoadError as exc:
        report = ValidationReport()
        report.add("error", "", str(exc))
        return None, report

    df, report = validate_dataframe(raw)
    if report.has_blocking_errors:
        return None, report
    return df, report


def prepare_mock(n: int, seed: int, include_sensitive: bool) -> pd.DataFrame:
    """Genera datos ficticios y les aplica la misma validación que a un archivo.

    Así los datos ficticios tienen exactamente los mismos tipos que un
    archivo cargado y el resto de la app no necesita distinguirlos.

    Args:
        n: Número de empleados.
        seed: Semilla.
        include_sensitive: Si se generan Edad y Género ficticios.

    Returns:
        DataFrame validado.

    Raises:
        DataLoadError: Si la validación de los datos generados falla (no
            debería ocurrir; indicaría un error en el generador).
    """
    raw = generate_mock_data(n=n, seed=seed, include_sensitive=include_sensitive)
    df, report = validate_dataframe(raw.astype(object))
    if report.has_blocking_errors:
        raise DataLoadError("Los datos ficticios no han superado la validación.")
    return df


def numeric_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Devuelve columnas numéricas como float64 con NaN (útil para cálculos con NumPy)."""
    return df[columns].apply(lambda s: pd.to_numeric(s, errors="coerce")).astype(float).replace({pd.NA: np.nan})
