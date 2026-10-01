"""Valores por defecto y definiciones estáticas de la aplicación.

Todo lo que una persona usuaria puede querer ajustar (pesos, umbrales,
tramos, k mínimo, tamaño máximo de archivo) parte de aquí. La pestaña de
configuración trabaja sobre una copia de estos valores en la sesión.

AVISO IMPORTANTE SOBRE LOS VALORES POR DEFECTO
Los pesos y umbrales de los factores son supuestos razonables elegidos para
una demostración. No se han derivado de evidencia empírica ni de datos de
ninguna organización real. Antes de usar la herramienta con datos reales
deben revisarse con el equipo de People Analytics y, si existe histórico de
bajas, contrastarse con él (pestaña de dashboard, bloque de validación).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Contexto de negocio (texto informativo, no interviene en cálculos)
# ---------------------------------------------------------------------------
APP_TITLE: Final[str] = "Riesgo de rotación voluntaria"
APP_ICON: Final[str] = "📉"
HORIZONTE_PREDICCION: Final[str] = "6–12 meses"
OBJETIVO_NEGOCIO: Final[str] = (
    "Ejemplo de objetivo: reducir la rotación voluntaria no deseada un X % en "
    "12 meses. Sustituye este texto por el objetivo real de tu organización."
)
AVISO_USO: Final[str] = (
    "Herramienta de apoyo a la decisión. El score orienta conversaciones y "
    "prioriza recursos; no debe ser la base única de ninguna decisión sobre "
    "una persona."
)

# ---------------------------------------------------------------------------
# Límites técnicos
# ---------------------------------------------------------------------------
# Tamaño máximo por defecto que acepta la app (MB). El límite duro del
# servidor está en .streamlit/config.toml (server.maxUploadSize); el valor
# configurable en la app nunca puede superarlo.
MAX_UPLOAD_MB_DEFAULT: Final[int] = 10
MAX_UPLOAD_MB_SERVER: Final[int] = 50
MAX_ROWS: Final[int] = 100_000

# Tamaño mínimo de grupo para mostrar resultados agregados en la auditoría
# de equidad. Valor orientativo; ajústalo a la política de tu organización.
K_MIN_DEFAULT: Final[int] = 10
K_MIN_LOWER_BOUND: Final[int] = 3

# Avisos de tamaño de muestra en la validación frente a bajas reales.
# Son umbrales heurísticos, no criterios estadísticos formales.
MIN_N_VALIDACION: Final[int] = 200
MIN_POSITIVOS_VALIDACION: Final[int] = 30

# ---------------------------------------------------------------------------
# Esquema de datos
# ---------------------------------------------------------------------------
COL_ID: Final[str] = "ID_Empleado"
COL_DEPARTAMENTO: Final[str] = "Departamento"
COL_PUESTO: Final[str] = "Puesto"
COL_NIVEL: Final[str] = "Nivel"
COL_ANTIGUEDAD: Final[str] = "Antigüedad_meses"
COL_EVAL_2024: Final[str] = "Evaluación_Desempeño_2024"
COL_EVAL_2025: Final[str] = "Evaluación_Desempeño_2025"
COL_POTENCIAL: Final[str] = "Potencial"
COL_COMPETENCIAS: Final[str] = "Nivel_Competencias"
COL_FORMACION: Final[str] = "Horas_Formación_12m"
COL_MESES_PROMO: Final[str] = "Meses_desde_última_promoción"
COL_COMPA: Final[str] = "Compa_Ratio"
COL_HORAS_EXTRA: Final[str] = "Horas_Extra_Mensuales_Media"
COL_CAMBIOS_RESP: Final[str] = "Cambios_Responsable_24m"
COL_CLIMA: Final[str] = "Puntuación_Clima"
COL_AUSENCIAS: Final[str] = "Días_Ausencia_12m"
COL_MODALIDAD: Final[str] = "Modalidad_Trabajo"
COL_BAJA: Final[str] = "Baja_Voluntaria"

# Columnas sensibles admitidas solo para la auditoría de equidad.
COL_EDAD: Final[str] = "Edad"
COL_GENERO: Final[str] = "Género"
SENSITIVE_COLUMNS: Final[tuple[str, ...]] = (COL_EDAD, COL_GENERO)

NIVELES: Final[tuple[str, ...]] = ("Junior", "Medio", "Senior", "Mánager")
POTENCIALES: Final[tuple[str, ...]] = ("Bajo", "Medio", "Alto")
MODALIDADES: Final[tuple[str, ...]] = ("Presencial", "Híbrido", "Remoto")


@dataclass(frozen=True)
class ColumnSpec:
    """Especificación de una columna del esquema.

    Attributes:
        name: Nombre canónico de la columna.
        kind: Tipo lógico: "id", "category", "ordinal", "int", "float" o "bool".
        required: Si la columna es obligatoria en un archivo cargado.
        min_value: Valor mínimo admitido (numéricas).
        max_value: Valor máximo admitido (numéricas). None = sin máximo.
        allowed: Valores admitidos (categóricas cerradas).
        description: Descripción breve para la interfaz y el README.
    """

    name: str
    kind: str
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed: tuple[str, ...] | None = None
    description: str = ""


SCHEMA: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec(COL_ID, "id", description="Seudónimo, p. ej. EMP-0001"),
    ColumnSpec(COL_DEPARTAMENTO, "category", description="Departamento"),
    ColumnSpec(COL_PUESTO, "category", description="Puesto"),
    ColumnSpec(COL_NIVEL, "ordinal", allowed=NIVELES, description="Nivel jerárquico"),
    ColumnSpec(COL_ANTIGUEDAD, "int", min_value=0, max_value=480, description="Meses en la empresa"),
    ColumnSpec(COL_EVAL_2024, "float", min_value=1, max_value=5, description="Evaluación 2024 (1–5)"),
    ColumnSpec(COL_EVAL_2025, "float", min_value=1, max_value=5, description="Evaluación 2025 (1–5)"),
    ColumnSpec(COL_POTENCIAL, "ordinal", allowed=POTENCIALES, description="Potencial"),
    ColumnSpec(COL_COMPETENCIAS, "int", min_value=1, max_value=5, description="Nivel de competencias (1–5)"),
    ColumnSpec(COL_FORMACION, "int", min_value=0, max_value=120, description="Horas de formación últimos 12 meses"),
    ColumnSpec(COL_MESES_PROMO, "int", min_value=0, max_value=480, description="Meses desde la última promoción"),
    ColumnSpec(COL_COMPA, "float", min_value=0.7, max_value=1.3, description="Salario / punto medio de banda"),
    ColumnSpec(COL_HORAS_EXTRA, "float", min_value=0, max_value=200, description="Media mensual de horas extra"),
    ColumnSpec(COL_CAMBIOS_RESP, "int", min_value=0, max_value=4, description="Cambios de responsable en 24 meses"),
    ColumnSpec(COL_CLIMA, "float", min_value=1, max_value=5, description="Puntuación de clima (1–5)"),
    ColumnSpec(COL_AUSENCIAS, "int", min_value=0, max_value=365, description="Días de ausencia en 12 meses, sin causa"),
    ColumnSpec(COL_MODALIDAD, "category", allowed=MODALIDADES, description="Modalidad de trabajo"),
    ColumnSpec(COL_BAJA, "bool", required=False, description="Baja voluntaria (solo datos históricos)"),
    ColumnSpec(COL_EDAD, "int", required=False, min_value=16, max_value=80, description="Solo auditoría de equidad"),
    ColumnSpec(COL_GENERO, "category", required=False, description="Solo auditoría de equidad"),
)

SCHEMA_BY_NAME: Final[dict[str, ColumnSpec]] = {spec.name: spec for spec in SCHEMA}

# Sinónimos aceptados en archivos cargados (clave normalizada -> canónico).
# La normalización quita tildes, pasa a minúsculas y cambia espacios por "_".
COLUMN_SYNONYMS: Final[dict[str, str]] = {
    "sexo": COL_GENERO,
    "gender": COL_GENERO,
    "genero": COL_GENERO,
    "age": COL_EDAD,
    "id": COL_ID,
    "employee_id": COL_ID,
    "id_empleado": COL_ID,
    "baja": COL_BAJA,
    "attrition": COL_BAJA,
}

# Fragmentos de nombre de columna que indican identificadores directos o
# cuasi-identificadores fuertes. Se eliminan al cargar el archivo.
PII_NAME_PATTERNS: Final[tuple[str, ...]] = (
    "nombre", "apellido", "name", "surname", "email", "e_mail", "correo",
    "mail", "dni", "nie", "nif", "pasaporte", "passport", "telefono",
    "phone", "movil", "mobile", "direccion", "address", "domicilio",
    "codigo_postal", "postal", "iban", "cuenta_bancaria", "seguridad_social",
    "nss", "nacimiento", "birth", "fecha_nac", "foto", "photo", "linkedin",
)

# Fragmentos que indican causa o diagnóstico de ausencias (posibles datos de
# salud, categoría especial del art. 9 RGPD). Estas columnas se rechazan.
ABSENCE_CAUSE_PATTERNS: Final[tuple[str, ...]] = (
    "causa", "motivo", "diagnostico", "diagnosis", "enfermedad", "illness",
    "baja_medica", "incapacidad", "it_", "medico", "medical", "sick",
)


# ---------------------------------------------------------------------------
# Factores de riesgo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorDefault:
    """Definición de un factor de riesgo con sus valores por defecto.

    La intensidad del factor (0–1) crece linealmente desde ``zero_at`` (sin
    riesgo) hasta ``max_at`` (riesgo máximo). Si ``max_at`` < ``zero_at`` el
    riesgo crece cuando el valor baja (p. ej. clima bajo).

    Attributes:
        key: Identificador interno.
        label: Nombre visible.
        source: Descripción de la variable de origen.
        zero_at: Valor a partir del cual el factor no aporta riesgo.
        max_at: Valor en el que el factor alcanza su intensidad máxima.
        weight: Peso relativo (0 = desactivado).
        rationale: Justificación del supuesto, mostrada en la interfaz.
    """

    key: str
    label: str
    source: str
    zero_at: float
    max_at: float
    weight: float
    rationale: str


FACTORS: Final[tuple[FactorDefault, ...]] = (
    FactorDefault(
        key="compa_ratio",
        label="Salario por debajo de banda",
        source=COL_COMPA,
        zero_at=1.00,
        max_at=0.80,
        weight=18,
        rationale="Supuesto: cobrar claramente por debajo del punto medio de la banda aumenta la probabilidad de aceptar ofertas externas.",
    ),
    FactorDefault(
        key="clima",
        label="Clima bajo",
        source=COL_CLIMA,
        zero_at=3.5,
        max_at=2.0,
        weight=18,
        rationale="Supuesto: una percepción de clima baja suele preceder a la intención de salida.",
    ),
    FactorDefault(
        key="estancamiento",
        label="Estancamiento de carrera",
        source=COL_MESES_PROMO,
        zero_at=24,
        max_at=60,
        weight=15,
        rationale="Supuesto: más de dos años sin promoción empieza a pesar; a partir de cinco años se considera máximo.",
    ),
    FactorDefault(
        key="desempeno_bajo",
        label="Desempeño reciente bajo",
        source=COL_EVAL_2025,
        zero_at=3.5,
        max_at=2.0,
        weight=10,
        rationale="Supuesto: un desempeño bajo se asocia a desenganche. Ojo: también puede reflejar problemas de evaluación.",
    ),
    FactorDefault(
        key="caida_desempeno",
        label="Caída de desempeño 2024→2025",
        source=f"{COL_EVAL_2025} − {COL_EVAL_2024}",
        zero_at=0.0,
        max_at=-1.0,
        weight=8,
        rationale="Supuesto: una caída de un punto o más respecto al año anterior es una señal de alerta.",
    ),
    FactorDefault(
        key="sobrecarga",
        label="Sobrecarga (horas extra)",
        source=COL_HORAS_EXTRA,
        zero_at=10,
        max_at=30,
        weight=10,
        rationale="Supuesto: una carga sostenida de horas extra eleva el riesgo de desgaste.",
    ),
    FactorDefault(
        key="antiguedad_temprana",
        label="Antigüedad en ventana crítica",
        source=COL_ANTIGUEDAD,
        zero_at=60,
        max_at=24,
        weight=8,
        rationale="Supuesto: los primeros años concentran más salidas. Posible proxy de edad: revisar en la auditoría de equidad.",
    ),
    FactorDefault(
        key="cambios_responsable",
        label="Inestabilidad de liderazgo",
        source=COL_CAMBIOS_RESP,
        zero_at=1,
        max_at=3,
        weight=8,
        rationale="Supuesto: varios cambios de responsable en dos años debilitan el vínculo con el equipo.",
    ),
    FactorDefault(
        key="formacion_baja",
        label="Baja inversión en formación",
        source=COL_FORMACION,
        zero_at=20,
        max_at=0,
        weight=5,
        rationale="Supuesto: poca formación puede percibirse como falta de desarrollo.",
    ),
    FactorDefault(
        key="ausencias",
        label="Ausencias elevadas",
        source=COL_AUSENCIAS,
        zero_at=8,
        max_at=20,
        weight=0,
        rationale=(
            "Desactivado por defecto: las ausencias pueden reflejar salud, discapacidad o "
            "cuidados, y usarlas en el score puede penalizar indirectamente a esos colectivos. "
            "Actívalo solo tras valorarlo en la EIPD."
        ),
    ),
)

FACTORS_BY_KEY: Final[dict[str, FactorDefault]] = {f.key: f for f in FACTORS}

# Tramos de riesgo: score < UMBRAL_MEDIO -> bajo; < UMBRAL_ALTO -> medio; resto alto.
# Como el score es una media ponderada de muchos factores, es raro que todos
# coincidan a la vez y los scores altos se concentran por debajo de 50. Estos
# umbrales se han fijado mirando la distribución de los DATOS FICTICIOS
# (aprox. 25 % en tramo medio o alto y 5–10 % en alto). Con datos reales deben
# recalibrarse: no son puntos de corte validados.
UMBRAL_MEDIO_DEFAULT: Final[float] = 25.0
UMBRAL_ALTO_DEFAULT: Final[float] = 35.0
TRAMOS: Final[tuple[str, str, str]] = ("Bajo", "Medio", "Alto")
COLORES_TRAMO: Final[dict[str, str]] = {"Bajo": "#2E8B57", "Medio": "#D99A00", "Alto": "#C0392B"}

# Talento clave: desempeño 2025 >= umbral o potencial alto.
UMBRAL_ALTO_DESEMPENO: Final[float] = 4.0

# Una contribución se considera "factor dominante" si aporta al menos estos
# puntos al score. Se muestran como máximo MAX_FACTORES_DOMINANTES.
MIN_PUNTOS_FACTOR_DOMINANTE: Final[float] = 4.0
MAX_FACTORES_DOMINANTES: Final[int] = 3


@dataclass
class ScoringConfig:
    """Configuración editable del scoring.

    Attributes:
        weights: Peso por clave de factor.
        zero_at: Umbral sin riesgo por clave de factor.
        max_at: Umbral de riesgo máximo por clave de factor.
        umbral_medio: Score a partir del cual el tramo es medio.
        umbral_alto: Score a partir del cual el tramo es alto.
    """

    weights: dict[str, float] = field(default_factory=lambda: {f.key: float(f.weight) for f in FACTORS})
    zero_at: dict[str, float] = field(default_factory=lambda: {f.key: float(f.zero_at) for f in FACTORS})
    max_at: dict[str, float] = field(default_factory=lambda: {f.key: float(f.max_at) for f in FACTORS})
    umbral_medio: float = UMBRAL_MEDIO_DEFAULT
    umbral_alto: float = UMBRAL_ALTO_DEFAULT

    def as_key(self) -> tuple:
        """Devuelve una representación inmutable y hasheable (útil para caché)."""
        return (
            tuple(sorted(self.weights.items())),
            tuple(sorted(self.zero_at.items())),
            tuple(sorted(self.max_at.items())),
            self.umbral_medio,
            self.umbral_alto,
        )
