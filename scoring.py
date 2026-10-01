"""Motor de scoring de riesgo de rotación por reglas ponderadas.

Cálculo
-------
1. Cada factor transforma una variable en una intensidad de riesgo entre 0 y 1
   mediante una rampa lineal entre ``zero_at`` (0) y ``max_at`` (1).
2. El score de una persona es la media ponderada de las intensidades de los
   factores con dato disponible, multiplicada por 100:

       score = 100 · Σ (w_i · r_i) / Σ w_i      (sumando solo factores con dato)

3. La contribución de cada factor es ``100 · w_i · r_i / Σ w_i``. Las
   contribuciones suman exactamente el score (descomposición aditiva).

Si una persona no tiene dato en un factor, ese factor se excluye de su
denominador. Así un valor vacío no se interpreta como "sin riesgo", pero el
score se apoya en menos información: la columna ``Factores_Disponibles``
lo hace visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg

COL_SCORE = "Score_Riesgo"
COL_TRAMO = "Tramo_Riesgo"
COL_FACTORES_DISP = "Factores_Disponibles"
COL_TALENTO_CLAVE = "Talento_Clave"
CONTRIB_PREFIX = "Contrib_"
INTENS_PREFIX = "Intens_"


class ScoringConfigError(ValueError):
    """Configuración del scoring no válida. Su mensaje es apto para la interfaz."""


def validate_config(config: cfg.ScoringConfig) -> None:
    """Comprueba que la configuración del scoring es coherente.

    Args:
        config: Configuración a validar.

    Raises:
        ScoringConfigError: Si algún peso es negativo, todos son cero, algún
            factor tiene umbrales iguales o los tramos no son crecientes.
    """
    if any(w < 0 for w in config.weights.values()):
        raise ScoringConfigError("Los pesos no pueden ser negativos.")
    if sum(config.weights.values()) <= 0:
        raise ScoringConfigError("Al menos un factor debe tener un peso mayor que cero.")
    for factor in cfg.FACTORS:
        if config.weights.get(factor.key, 0) > 0 and config.zero_at[factor.key] == config.max_at[factor.key]:
            raise ScoringConfigError(
                f"En «{factor.label}», el umbral sin riesgo y el de riesgo máximo no pueden ser iguales."
            )
    if not 0 <= config.umbral_medio < config.umbral_alto <= 100:
        raise ScoringConfigError("Los umbrales de tramo deben cumplir 0 ≤ medio < alto ≤ 100.")


def ramp_intensity(values: pd.Series, zero_at: float, max_at: float) -> pd.Series:
    """Convierte valores en intensidad de riesgo 0–1 con una rampa lineal.

    Args:
        values: Valores numéricos (NaN se conserva como NaN).
        zero_at: Valor con intensidad 0.
        max_at: Valor con intensidad 1. Puede ser menor que ``zero_at``
            (el riesgo crece al bajar el valor).

    Returns:
        Serie float con valores en [0, 1] o NaN.

    Raises:
        ScoringConfigError: Si ``zero_at`` y ``max_at`` son iguales.
    """
    if zero_at == max_at:
        raise ScoringConfigError("Los umbrales de un factor no pueden ser iguales.")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return ((numeric - zero_at) / (max_at - zero_at)).clip(lower=0.0, upper=1.0)


def factor_values(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Obtiene el valor bruto de cada factor a partir de las columnas del esquema.

    Args:
        df: DataFrame validado.

    Returns:
        Diccionario clave de factor -> Serie float (NaN donde falta el dato).
    """

    def col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").astype(float)

    return {
        "compa_ratio": col(cfg.COL_COMPA),
        "clima": col(cfg.COL_CLIMA),
        "estancamiento": col(cfg.COL_MESES_PROMO),
        "desempeno_bajo": col(cfg.COL_EVAL_2025),
        "caida_desempeno": col(cfg.COL_EVAL_2025) - col(cfg.COL_EVAL_2024),
        "sobrecarga": col(cfg.COL_HORAS_EXTRA),
        "antiguedad_temprana": col(cfg.COL_ANTIGUEDAD),
        "cambios_responsable": col(cfg.COL_CAMBIOS_RESP),
        "formacion_baja": col(cfg.COL_FORMACION),
        "ausencias": col(cfg.COL_AUSENCIAS),
    }


def assign_tier(score: pd.Series, umbral_medio: float, umbral_alto: float) -> pd.Series:
    """Asigna el tramo de riesgo a cada score.

    Args:
        score: Scores 0–100 (NaN permitido).
        umbral_medio: Score mínimo del tramo medio.
        umbral_alto: Score mínimo del tramo alto.

    Returns:
        Serie de texto con "Bajo", "Medio", "Alto" o <NA> si no hay score.
    """
    bajo, medio, alto = cfg.TRAMOS
    tier = pd.Series(pd.NA, index=score.index, dtype="string")
    tier[score < umbral_medio] = bajo
    tier[(score >= umbral_medio) & (score < umbral_alto)] = medio
    tier[score >= umbral_alto] = alto
    return tier


def is_key_talent(df: pd.DataFrame) -> pd.Series:
    """Marca talento clave: evaluación 2025 alta o potencial alto.

    Args:
        df: DataFrame validado.

    Returns:
        Serie booleana.
    """
    eval_2025 = pd.to_numeric(df.get(cfg.COL_EVAL_2025), errors="coerce")
    alto_desempeno = (eval_2025 >= cfg.UMBRAL_ALTO_DESEMPENO).fillna(False)
    alto_potencial = (df.get(cfg.COL_POTENCIAL).astype("string") == "Alto").fillna(False)
    return (alto_desempeno | alto_potencial).astype(bool)


def score_dataframe(df: pd.DataFrame, config: cfg.ScoringConfig) -> pd.DataFrame:
    """Calcula score, tramo, intensidades y contribuciones para cada persona.

    Args:
        df: DataFrame validado.
        config: Configuración del scoring.

    Returns:
        Copia de ``df`` con las columnas ``Score_Riesgo``, ``Tramo_Riesgo``,
        ``Factores_Disponibles``, ``Talento_Clave``, ``Intens_<factor>`` y
        ``Contrib_<factor>``.

    Raises:
        ScoringConfigError: Si la configuración no es válida.
    """
    validate_config(config)
    out = df.copy()
    values = factor_values(df)

    active = [f for f in cfg.FACTORS if config.weights.get(f.key, 0) > 0]
    weighted = pd.DataFrame(index=df.index, dtype=float)
    weights_available = pd.DataFrame(index=df.index, dtype=float)

    for factor in cfg.FACTORS:
        intensity = ramp_intensity(values[factor.key], config.zero_at[factor.key], config.max_at[factor.key])
        out[f"{INTENS_PREFIX}{factor.key}"] = intensity
        if factor in active:
            weight = float(config.weights[factor.key])
            weighted[factor.key] = intensity * weight
            weights_available[factor.key] = intensity.notna().astype(float) * weight

    denominator = weights_available.sum(axis=1)
    safe_denominator = denominator.where(denominator > 0)

    score = 100.0 * weighted.sum(axis=1, min_count=1) / safe_denominator
    for factor in cfg.FACTORS:
        column = f"{CONTRIB_PREFIX}{factor.key}"
        if factor in active:
            out[column] = (100.0 * weighted[factor.key] / safe_denominator).fillna(0.0).where(score.notna())
        else:
            out[column] = np.where(score.notna(), 0.0, np.nan)

    out[COL_SCORE] = score.round(1)
    out[COL_TRAMO] = assign_tier(out[COL_SCORE], config.umbral_medio, config.umbral_alto)
    out[COL_FACTORES_DISP] = (
        pd.DataFrame({f.key: out[f"{INTENS_PREFIX}{f.key}"].notna() for f in active}).sum(axis=1).astype(int).astype(str)
        + f"/{len(active)}"
    )
    out[COL_TALENTO_CLAVE] = is_key_talent(df)
    return out


def contributions_long(scored_row: pd.Series) -> pd.DataFrame:
    """Devuelve las contribuciones de una persona en formato largo, ordenadas.

    Args:
        scored_row: Fila de un DataFrame devuelto por :func:`score_dataframe`.

    Returns:
        DataFrame con columnas Factor, Puntos e Intensidad, de mayor a menor.
    """
    rows = []
    for factor in cfg.FACTORS:
        rows.append(
            {
                "Factor": factor.label,
                "Clave": factor.key,
                "Puntos": float(scored_row.get(f"{CONTRIB_PREFIX}{factor.key}", np.nan)),
                "Intensidad": float(scored_row.get(f"{INTENS_PREFIX}{factor.key}", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("Puntos", ascending=False, na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validación frente a bajas reales
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Métricas del score frente al resultado observado.

    Attributes:
        n: Personas con score y resultado conocidos.
        positives: Número de bajas voluntarias.
        auc: Área bajo la curva ROC (None si no se puede calcular).
        precision_alto: Proporción de bajas dentro del tramo alto (None si el tramo está vacío).
        recall_alto: Proporción de bajas que estaban en el tramo alto.
        confusion: Matriz 2×2 (tramo alto sí/no × baja sí/no).
        warnings: Avisos sobre la fiabilidad del resultado.
    """

    n: int
    positives: int
    auc: float | None
    precision_alto: float | None
    recall_alto: float | None
    confusion: pd.DataFrame
    warnings: list[str]


def roc_auc(score: pd.Series, outcome: pd.Series) -> float | None:
    """Calcula el AUC ROC con el estadístico de Mann-Whitney (empates = 0,5).

    Args:
        score: Puntuaciones (mayor = más riesgo).
        outcome: Resultado observado (True = baja).

    Returns:
        AUC entre 0 y 1, o None si solo hay una clase.
    """
    mask = score.notna() & outcome.notna()
    s = score[mask].astype(float)
    y = outcome[mask].astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = s.rank(method="average")
    sum_ranks_pos = float(ranks[y].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate_against_outcome(scored: pd.DataFrame) -> ValidationResult | None:
    """Compara el score con la columna ``Baja_Voluntaria``, si existe.

    Args:
        scored: DataFrame devuelto por :func:`score_dataframe`.

    Returns:
        Resultado de la validación, o None si no hay columna de resultado
        o no hay filas con resultado conocido.
    """
    if cfg.COL_BAJA not in scored.columns:
        return None
    mask = scored[COL_SCORE].notna() & scored[cfg.COL_BAJA].notna()
    if not mask.any():
        return None

    y = scored.loc[mask, cfg.COL_BAJA].astype(bool)
    high = scored.loc[mask, COL_TRAMO] == cfg.TRAMOS[2]
    n = int(mask.sum())
    positives = int(y.sum())

    tp = int((high & y).sum())
    fp = int((high & ~y).sum())
    fn = int((~high & y).sum())
    tn = int((~high & ~y).sum())
    confusion = pd.DataFrame(
        [[tp, fp], [fn, tn]],
        index=["Tramo alto", "Resto de tramos"],
        columns=["Baja voluntaria", "Sin baja"],
    )

    warnings: list[str] = []
    if n < cfg.MIN_N_VALIDACION or positives < cfg.MIN_POSITIVOS_VALIDACION:
        warnings.append(
            f"Muestra pequeña ({n} personas, {positives} bajas). Las métricas pueden variar mucho con otra muestra."
        )
    warnings.append(
        "Las variables deben medirse antes de la salida. Si el archivo mezcla datos actuales con bajas pasadas, "
        "las métricas pueden estar sesgadas (fuga de información temporal)."
    )

    return ValidationResult(
        n=n,
        positives=positives,
        auc=roc_auc(scored.loc[mask, COL_SCORE], y),
        precision_alto=(tp / (tp + fp)) if (tp + fp) else None,
        recall_alto=(tp / positives) if positives else None,
        confusion=confusion,
        warnings=warnings,
    )
