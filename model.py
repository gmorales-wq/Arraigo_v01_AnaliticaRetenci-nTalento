"""Modelo predictivo de aprendizaje automático (regresión logística).

Complementa al score de reglas: en lugar de pesos fijados por personas,
aprende del histórico de bajas cuánto pesa cada variable y estima una
probabilidad de baja voluntaria.

Diseño
------
- Algoritmo: regresión logística con regularización L2, precedida de
  imputación por la mediana y estandarización. Se elige por ser explicable:
  cada coeficiente indica cómo cambia el riesgo al subir una variable.
- Variables: las mismas que los factores activos del score de reglas
  (valores brutos, no intensidades). Edad y Género nunca se usan.
- Entrenamiento: filas con ``Baja_Voluntaria`` conocida. Si hay filas sin
  resultado (plantilla actual), el modelo las puntúa.
- Evaluación: AUC con validación cruzada estratificada (predicciones fuera
  de muestra), comparado con el AUC del score de reglas en las mismas filas.

Limitaciones que la interfaz debe mostrar: requiere histórico suficiente y
medido antes de las salidas; con datos ficticios la evaluación es circular;
los coeficientes de variables correlacionadas pueden ser inestables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
import scoring

COL_PROBA = "Probabilidad_Baja_ML"
LOGIT_PREFIX = "Logit_"

MIN_POSITIVOS_MODELO = 30
MIN_NEGATIVOS_MODELO = 30
MAX_FOLDS = 5
RANDOM_STATE = 0


@dataclass
class ModelResult:
    """Resultado del entrenamiento del modelo.

    Attributes:
        trained: Si el modelo se ha podido entrenar.
        message: Explicación para la interfaz (motivo si no se entrena).
        feature_keys: Claves de factor usadas como variables.
        coefficients: Coeficiente estandarizado por variable (efecto de +1 desviación típica en el logit).
        auc_model_cv: AUC del modelo con validación cruzada.
        auc_rules: AUC del score de reglas en las mismas filas.
        n_train: Filas usadas para entrenar.
        n_positives: Bajas en el conjunto de entrenamiento.
        n_scored_unlabeled: Filas sin resultado puntuadas por el modelo.
        predictions: DataFrame indexado como la entrada con la probabilidad y
            las contribuciones al logit (``Logit_<factor>``).
        warnings: Avisos sobre la fiabilidad.
    """

    trained: bool
    message: str
    feature_keys: list[str] = field(default_factory=list)
    coefficients: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    auc_model_cv: float | None = None
    auc_rules: float | None = None
    n_train: int = 0
    n_positives: int = 0
    n_scored_unlabeled: int = 0
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def feature_matrix(df: pd.DataFrame, feature_keys: list[str]) -> pd.DataFrame:
    """Construye la matriz de variables del modelo a partir de los factores.

    Args:
        df: DataFrame validado.
        feature_keys: Claves de factor a usar.

    Returns:
        DataFrame float con una columna por factor (NaN donde falta el dato).
    """
    values = scoring.factor_values(df)
    return pd.DataFrame({key: values[key] for key in feature_keys}, index=df.index)


def _build_pipeline() -> Pipeline:
    """Crea el pipeline imputación + estandarización + regresión logística."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )


def train_model(scored: pd.DataFrame, feature_keys: list[str]) -> ModelResult:
    """Entrena la regresión logística con el histórico de bajas y puntúa a todas las personas.

    Args:
        scored: DataFrame puntuado por ``scoring.score_dataframe`` (se usa su
            score de reglas para la comparación).
        feature_keys: Claves de los factores que actúan como variables.

    Returns:
        Resultado con métricas, coeficientes y predicciones. Si no hay datos
        suficientes, ``trained`` es False y ``message`` explica el motivo.
    """
    if cfg.COL_BAJA not in scored.columns:
        return ModelResult(False, "El archivo no incluye la columna Baja_Voluntaria: no hay histórico del que aprender.")
    if len(feature_keys) < 2:
        return ModelResult(False, "Hacen falta al menos dos factores activos para entrenar el modelo.")

    X = feature_matrix(scored, feature_keys)
    usable = X.notna().any(axis=1)
    labeled = scored[cfg.COL_BAJA].notna() & usable
    y = scored.loc[labeled, cfg.COL_BAJA].astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos < MIN_POSITIVOS_MODELO or n_neg < MIN_NEGATIVOS_MODELO:
        return ModelResult(
            False,
            f"Histórico insuficiente: {n_pos} bajas y {n_neg} permanencias con dato. "
            f"Se necesitan al menos {MIN_POSITIVOS_MODELO} de cada.",
        )

    X_train = X.loc[labeled]
    folds = min(MAX_FOLDS, n_pos, n_neg)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(_build_pipeline(), X_train, y, cv=cv, method="predict_proba")[:, 1]
    auc_model = float(roc_auc_score(y, oof))
    auc_rules = scoring.roc_auc(scored.loc[labeled, scoring.COL_SCORE], y)

    pipeline = _build_pipeline().fit(X_train, y)
    coefs = pd.Series(pipeline.named_steps["model"].coef_[0], index=feature_keys)

    X_usable = X.loc[usable]
    transformed = pipeline.named_steps["scaler"].transform(pipeline.named_steps["imputer"].transform(X_usable))
    contributions = pd.DataFrame(transformed * coefs.to_numpy(), index=X_usable.index,
                                 columns=[f"{LOGIT_PREFIX}{k}" for k in feature_keys])
    predictions = contributions.reindex(scored.index)
    predictions[COL_PROBA] = pd.Series(pipeline.predict_proba(X_usable)[:, 1], index=X_usable.index).reindex(scored.index)

    warnings = [
        "La AUC se calcula con validación cruzada (predicciones fuera de muestra). Las probabilidades que se muestran "
        "por persona proceden del modelo final, entrenado con todo el histórico: para quien ya tiene resultado son retrospectivas.",
        "Las variables deben medirse antes de la salida; si no, el modelo aprende de información posterior (fuga temporal).",
        "Si dos variables están muy correlacionadas, sus coeficientes pueden repartirse el efecto de forma inestable.",
    ]
    if n_pos < cfg.MIN_POSITIVOS_VALIDACION * 3:
        warnings.insert(0, f"Pocas bajas para entrenar ({n_pos}). Los coeficientes pueden cambiar mucho con otra muestra.")

    return ModelResult(
        trained=True,
        message=f"Modelo entrenado con {int(labeled.sum())} personas ({n_pos} bajas) y validación cruzada de {folds} particiones.",
        feature_keys=feature_keys,
        coefficients=coefs,
        auc_model_cv=auc_model,
        auc_rules=auc_rules,
        n_train=int(labeled.sum()),
        n_positives=n_pos,
        n_scored_unlabeled=int((usable & scored[cfg.COL_BAJA].isna()).sum()),
        predictions=predictions,
        warnings=warnings,
    )


def coefficient_table(result: ModelResult) -> pd.DataFrame:
    """Tabla de coeficientes legible para la interfaz.

    Args:
        result: Resultado de un modelo entrenado.

    Returns:
        DataFrame con Variable, Coeficiente, Odds ratio (+1 DT) y Efecto,
        ordenado por magnitud del efecto.
    """
    rows = []
    for key, coef in result.coefficients.items():
        rows.append(
            {
                "Variable": cfg.FACTORS_BY_KEY[key].source,
                "Coeficiente": round(float(coef), 3),
                "Odds ratio (+1 DT)": round(float(np.exp(coef)), 2),
                "Efecto": "Sube el riesgo al aumentar" if coef > 0 else "Baja el riesgo al aumentar",
            }
        )
    table = pd.DataFrame(rows)
    return table.reindex(table["Coeficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)
