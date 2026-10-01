"""Auditoría de equidad agregada.

Compara la distribución del score entre grupos de edad o género. Edad y
Género nunca intervienen en el score: solo se usan aquí, de forma agregada y
con supresión de grupos pequeños para reducir el riesgo de reidentificación.

Por qué hace falta aunque el score no use edad ni género: otras variables
pueden actuar como proxy (la antigüedad suele correlacionar con la edad; la
modalidad o las horas extra pueden diferir por género por razones de
cuidados). Excluir la variable no garantiza resultados equitativos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from scoring import COL_SCORE, COL_TRAMO, INTENS_PREFIX

SIN_DATO = "Sin dato"
SUPPRESSED_TEXT = "Suprimido"

_AGE_BINS = (0, 30, 45, 55, 200)
_AGE_LABELS = ("Menos de 30", "30–44", "45–54", "55 o más")


def available_attributes(df: pd.DataFrame) -> list[str]:
    """Devuelve los atributos sensibles presentes y con algún dato.

    Args:
        df: DataFrame validado.

    Returns:
        Lista con "Edad" y/o "Género".
    """
    return [c for c in cfg.SENSITIVE_COLUMNS if c in df.columns and df[c].notna().any()]


def group_labels(df: pd.DataFrame, attribute: str) -> pd.Series:
    """Obtiene la etiqueta de grupo de cada persona para un atributo.

    La edad se agrupa en tramos para no mostrar edades individuales.

    Args:
        df: DataFrame validado.
        attribute: "Edad" o "Género".

    Returns:
        Serie de texto con la etiqueta de grupo ("Sin dato" si falta).

    Raises:
        ValueError: Si el atributo no es uno de los admitidos.
    """
    if attribute == cfg.COL_EDAD:
        ages = pd.to_numeric(df[attribute], errors="coerce")
        bands = pd.cut(ages, bins=_AGE_BINS, labels=_AGE_LABELS, right=False)
        return bands.astype("string").fillna(SIN_DATO)
    if attribute == cfg.COL_GENERO:
        return df[attribute].astype("string").fillna(SIN_DATO)
    raise ValueError(f"Atributo no admitido para la auditoría: {attribute}")


def suppressed_groups(counts: pd.Series, k: int) -> set[str]:
    """Decide qué grupos se suprimen.

    Se suprime todo grupo con menos de ``k`` personas. Si solo se suprime un
    grupo, se suprime también el siguiente más pequeño (supresión
    complementaria), porque de lo contrario su tamaño podría deducirse
    restando del total.

    Args:
        counts: Número de personas por grupo.
        k: Tamaño mínimo de grupo.

    Returns:
        Conjunto de grupos suprimidos.
    """
    suppressed = set(counts[counts < k].index)
    remaining = counts[~counts.index.isin(suppressed)].sort_values()
    if len(suppressed) == 1 and len(remaining) >= 1:
        suppressed.add(remaining.index[0])
    return suppressed


def fairness_table(scored: pd.DataFrame, attribute: str, k: int) -> pd.DataFrame:
    """Tabla agregada del score por grupo, con supresión de grupos pequeños.

    Args:
        scored: DataFrame puntuado.
        attribute: "Edad" o "Género".
        k: Tamaño mínimo de grupo.

    Returns:
        DataFrame con Grupo, Personas, Score medio, Score mediano, % en riesgo
        alto y ratio frente al grupo con menor proporción en riesgo alto.
        Los grupos suprimidos muestran "Suprimido" y no aportan cifras.
    """
    labels = group_labels(scored, attribute)
    frame = pd.DataFrame(
        {
            "Grupo": labels,
            "score": pd.to_numeric(scored[COL_SCORE], errors="coerce"),
            "alto": (scored[COL_TRAMO] == cfg.TRAMOS[2]).fillna(False),
        }
    )
    frame = frame[frame["score"].notna()]
    grouped = frame.groupby("Grupo", observed=True)
    counts = grouped.size()
    suppressed = suppressed_groups(counts, k)

    stats = pd.DataFrame(
        {
            "Personas": counts,
            "Score medio": grouped["score"].mean().round(1),
            "Score mediano": grouped["score"].median().round(1),
            "% riesgo alto": (100 * grouped["alto"].mean()).round(1),
        }
    )
    visible = stats[~stats.index.isin(suppressed)]
    min_rate = visible["% riesgo alto"].min() if not visible.empty else np.nan
    stats["Ratio vs. menor tasa"] = (
        (stats["% riesgo alto"] / min_rate).round(2) if min_rate and min_rate > 0 else np.nan
    )

    display = stats.astype(object)
    for group in suppressed:
        display.loc[group] = [f"< {k}" if counts[group] < k else SUPPRESSED_TEXT] + [SUPPRESSED_TEXT] * 4
    display = display.reset_index().rename(columns={"index": "Grupo"})
    display["Suprimido"] = display["Grupo"].isin(suppressed)
    order = list(_AGE_LABELS) + [SIN_DATO] if attribute == cfg.COL_EDAD else None
    if order:
        display["_orden"] = display["Grupo"].map({g: i for i, g in enumerate(order)})
        display = display.sort_values("_orden").drop(columns="_orden")
    return display.reset_index(drop=True)


def proxy_analysis(scored: pd.DataFrame, attribute: str, k: int, active_keys: list[str]) -> pd.DataFrame:
    """Mide cuánto se relaciona cada factor del score con el atributo sensible.

    - Edad: correlación de Spearman entre la edad y la intensidad del factor.
    - Género: diferencia máxima de intensidad media entre grupos con al menos
      ``k`` personas (excluye "Sin dato").

    Un valor alto no prueba discriminación, pero señala un posible proxy que
    conviene revisar.

    Args:
        scored: DataFrame puntuado.
        attribute: "Edad" o "Género".
        k: Tamaño mínimo de grupo.
        active_keys: Claves de factores con peso mayor que cero.

    Returns:
        DataFrame con Factor y la medida correspondiente, ordenado de mayor a
        menor valor absoluto. Vacío si no hay datos suficientes.
    """
    rows = []
    if attribute == cfg.COL_EDAD:
        ages = pd.to_numeric(scored[attribute], errors="coerce").astype(float)
        if ages.notna().sum() < k:
            return pd.DataFrame(columns=["Factor", "Correlación con la edad (Spearman)"])
        for key in active_keys:
            intensity = scored[f"{INTENS_PREFIX}{key}"].astype(float)
            mask = ages.notna() & intensity.notna()
            if mask.sum() < k or intensity[mask].nunique() < 2:
                continue
            rho = ages[mask].rank().corr(intensity[mask].rank())
            rows.append({"Factor": cfg.FACTORS_BY_KEY[key].label, "Correlación con la edad (Spearman)": round(float(rho), 2)})
        measure = "Correlación con la edad (Spearman)"
    else:
        labels = group_labels(scored, attribute)
        counts = labels[labels != SIN_DATO].value_counts()
        valid_groups = counts[counts >= k].index
        if len(valid_groups) < 2:
            return pd.DataFrame(columns=["Factor", "Diferencia máx. de intensidad media"])
        for key in active_keys:
            intensity = scored[f"{INTENS_PREFIX}{key}"].astype(float)
            means = intensity[labels.isin(valid_groups)].groupby(labels[labels.isin(valid_groups)]).mean()
            rows.append(
                {
                    "Factor": cfg.FACTORS_BY_KEY[key].label,
                    "Diferencia máx. de intensidad media": round(float(means.max() - means.min()), 3),
                }
            )
        measure = "Diferencia máx. de intensidad media"

    result = pd.DataFrame(rows, columns=["Factor", measure])
    if result.empty:
        return result
    return result.reindex(result[measure].abs().sort_values(ascending=False).index).reset_index(drop=True)


def probability_by_group(scored: pd.DataFrame, attribute: str, k: int, proba_col: str) -> pd.DataFrame:
    """Probabilidad media del modelo predictivo por grupo, con la misma supresión.

    Un modelo entrenado con datos históricos puede aprender sesgos del pasado a
    través de variables proxy, aunque no use Edad ni Género.

    Args:
        scored: DataFrame con la columna de probabilidad.
        attribute: "Edad" o "Género".
        k: Tamaño mínimo de grupo.
        proba_col: Nombre de la columna de probabilidad.

    Returns:
        DataFrame con Grupo, Personas y Probabilidad media (%), con los grupos
        pequeños suprimidos. Vacío si no hay probabilidades.
    """
    if proba_col not in scored.columns:
        return pd.DataFrame(columns=["Grupo", "Personas", "Probabilidad media (%)"])
    frame = pd.DataFrame({"Grupo": group_labels(scored, attribute), "p": pd.to_numeric(scored[proba_col], errors="coerce")})
    frame = frame[frame["p"].notna()]
    counts = frame.groupby("Grupo").size()
    suppressed = suppressed_groups(counts, k)
    rows = []
    for group, n in counts.items():
        if group in suppressed:
            rows.append({"Grupo": group, "Personas": f"< {k}" if n < k else SUPPRESSED_TEXT, "Probabilidad media (%)": SUPPRESSED_TEXT})
        else:
            rows.append({"Grupo": group, "Personas": int(n), "Probabilidad media (%)": round(100 * float(frame.loc[frame["Grupo"] == group, "p"].mean()), 1)})
    return pd.DataFrame(rows)
