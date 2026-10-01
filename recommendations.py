"""Plan de acción basado en reglas.

Las reglas viven en ``RULES``: un diccionario editable que asocia cada factor
de riesgo con acciones concretas. Para cambiar las recomendaciones no hace
falta tocar la lógica, solo este diccionario.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config as cfg
from scoring import COL_SCORE, COL_TALENTO_CLAVE, COL_TRAMO, CONTRIB_PREFIX


@dataclass(frozen=True)
class ActionRule:
    """Recomendación asociada a un factor de riesgo.

    Attributes:
        titulo: Resumen de la línea de actuación.
        acciones: Acciones concretas sugeridas.
        responsable: Quién debería liderarla.
        plazo: Plazo orientativo.
    """

    titulo: str
    acciones: tuple[str, ...]
    responsable: str
    plazo: str


RULES: dict[str, ActionRule] = {
    "compa_ratio": ActionRule(
        titulo="Revisión salarial",
        acciones=(
            "Comparar el salario con la banda y con referencias de mercado actualizadas.",
            "Valorar un ajuste en el próximo ciclo de revisión o un complemento ligado a objetivos.",
        ),
        responsable="Compensación y beneficios + responsable directo",
        plazo="Próximo ciclo de revisión salarial",
    ),
    "clima": ActionRule(
        titulo="Conversación de experiencia y clima",
        acciones=(
            "Reunión individual centrada en escuchar: qué funciona, qué no y qué cambiaría.",
            "Revisar los resultados de clima del equipo (agregados) para detectar causas compartidas.",
        ),
        responsable="Responsable directo con apoyo del HRBP",
        plazo="Próximas 2–4 semanas",
    ),
    "estancamiento": ActionRule(
        titulo="Conversación de carrera",
        acciones=(
            "Explorar aspiraciones y opciones internas (promoción, movilidad, proyectos).",
            "Acordar un plan de desarrollo con hitos y fechas de revisión.",
        ),
        responsable="Responsable directo + Talento",
        plazo="Próximo trimestre",
    ),
    "desempeno_bajo": ActionRule(
        titulo="Apoyo al desempeño",
        acciones=(
            "Entender las causas del desempeño bajo antes de actuar (carga, encaje, recursos, situación personal).",
            "Definir objetivos claros y seguimiento cercano con apoyo, no solo control.",
        ),
        responsable="Responsable directo",
        plazo="Próximas 4 semanas",
    ),
    "caida_desempeno": ActionRule(
        titulo="Revisión de la caída de desempeño",
        acciones=(
            "Conversación para entender qué ha cambiado respecto al año anterior.",
            "Comprobar si la caída coincide con cambios de equipo, rol o carga.",
        ),
        responsable="Responsable directo con apoyo del HRBP",
        plazo="Próximas 4 semanas",
    ),
    "sobrecarga": ActionRule(
        titulo="Reequilibrio de carga",
        acciones=(
            "Revisar la distribución de trabajo del equipo y priorizar tareas.",
            "Valorar refuerzos temporales o redistribución si la sobrecarga es estructural.",
        ),
        responsable="Responsable directo",
        plazo="Inmediato",
    ),
    "antiguedad_temprana": ActionRule(
        titulo="Refuerzo de integración",
        acciones=(
            "Comprobar que el onboarding se completó y que las expectativas del puesto se cumplen.",
            "Asignar una persona mentora o de referencia si no la tiene.",
        ),
        responsable="Responsable directo + Talento",
        plazo="Próximas 4 semanas",
    ),
    "cambios_responsable": ActionRule(
        titulo="Estabilidad de liderazgo",
        acciones=(
            "Asegurar una reunión de alineamiento con el responsable actual.",
            "Revisar si los cambios de responsable han afectado a objetivos o reconocimiento.",
        ),
        responsable="Responsable actual + HRBP",
        plazo="Próximas 2–4 semanas",
    ),
    "formacion_baja": ActionRule(
        titulo="Plan de formación",
        acciones=(
            "Identificar necesidades de desarrollo y ofrecer formación relevante para su rol o carrera.",
        ),
        responsable="Responsable directo + Formación",
        plazo="Próximo trimestre",
    ),
    "ausencias": ActionRule(
        titulo="Conversación de bienestar",
        acciones=(
            "Conversación de bienestar sin indagar en causas médicas; ofrecer recursos de apoyo disponibles.",
            "No usar este dato para decisiones disciplinarias.",
        ),
        responsable="HRBP",
        plazo="Próximas 4 semanas",
    ),
}

PRIORITY_LABELS: dict[int, str] = {
    1: "P1 · Riesgo alto, talento clave",
    2: "P2 · Riesgo alto",
    3: "P3 · Riesgo medio, talento clave",
    4: "P4 · Riesgo medio",
    5: "P5 · Seguimiento ordinario",
}

_CSV_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def dominant_factors(
    scored_row: pd.Series,
    min_points: float = cfg.MIN_PUNTOS_FACTOR_DOMINANTE,
    max_factors: int = cfg.MAX_FACTORES_DOMINANTES,
) -> list[str]:
    """Devuelve las claves de los factores que más aportan al score de una persona.

    Args:
        scored_row: Fila con columnas ``Contrib_<factor>``.
        min_points: Contribución mínima para considerar un factor dominante.
        max_factors: Número máximo de factores devueltos.

    Returns:
        Lista de claves de factor, de mayor a menor contribución.
    """
    contributions = {
        f.key: float(scored_row.get(f"{CONTRIB_PREFIX}{f.key}", 0.0) or 0.0) for f in cfg.FACTORS
    }
    ranked = sorted(
        ((k, v) for k, v in contributions.items() if v == v and v >= min_points),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [k for k, _ in ranked[:max_factors]]


def recommendations_for(scored_row: pd.Series) -> list[tuple[str, ActionRule]]:
    """Devuelve las recomendaciones para una persona según sus factores dominantes.

    Args:
        scored_row: Fila de un DataFrame puntuado.

    Returns:
        Lista de tuplas (etiqueta del factor, regla de acción).
    """
    return [
        (cfg.FACTORS_BY_KEY[key].label, RULES[key])
        for key in dominant_factors(scored_row)
        if key in RULES
    ]


def priority_level(tramo: object, talento_clave: bool) -> int:
    """Calcula el nivel de prioridad (1 = máxima) a partir del tramo y del talento clave.

    Args:
        tramo: "Bajo", "Medio", "Alto" o nulo.
        talento_clave: Si la persona es talento clave.

    Returns:
        Entero de 1 a 5.
    """
    bajo, medio, alto = cfg.TRAMOS
    if not isinstance(tramo, str):
        return 5
    if tramo == alto:
        return 1 if talento_clave else 2
    if tramo == medio:
        return 3 if talento_clave else 4
    return 5


def build_action_plan(scored: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla del plan de acción priorizada para un colectivo.

    Args:
        scored: DataFrame devuelto por ``scoring.score_dataframe``.

    Returns:
        DataFrame ordenado por prioridad y score, con una fila por persona.
    """
    rows = []
    for _, row in scored.iterrows():
        recs = recommendations_for(row)
        prioridad = priority_level(row.get(COL_TRAMO), bool(row.get(COL_TALENTO_CLAVE, False)))
        rows.append(
            {
                cfg.COL_ID: row[cfg.COL_ID],
                cfg.COL_DEPARTAMENTO: row.get(cfg.COL_DEPARTAMENTO),
                "Prioridad": prioridad,
                "Prioridad_Texto": PRIORITY_LABELS[prioridad],
                COL_SCORE: row.get(COL_SCORE),
                COL_TRAMO: row.get(COL_TRAMO),
                "Talento_Clave": "Sí" if row.get(COL_TALENTO_CLAVE, False) else "No",
                "Factores_Dominantes": "; ".join(label for label, _ in recs) or "Ninguno destacado",
                "Líneas_de_Acción": "; ".join(rule.titulo for _, rule in recs) or "Seguimiento ordinario",
                "Responsable_Sugerido": "; ".join(dict.fromkeys(rule.responsable for _, rule in recs)) or "Responsable directo",
            }
        )
    plan = pd.DataFrame(rows)
    if plan.empty:
        return plan
    return plan.sort_values(["Prioridad", COL_SCORE], ascending=[True, False], na_position="last").reset_index(drop=True)


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Neutraliza la inyección de fórmulas al abrir el CSV en una hoja de cálculo.

    Las celdas de texto que empiezan por ``=``, ``+``, ``-``, ``@``, tabulador o
    retorno de carro se prefijan con un apóstrofo.

    Args:
        df: DataFrame a exportar.

    Returns:
        Copia saneada.
    """
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(
                lambda v: f"'{v}" if isinstance(v, str) and v.startswith(_CSV_INJECTION_PREFIXES) else v
            )
    return safe
