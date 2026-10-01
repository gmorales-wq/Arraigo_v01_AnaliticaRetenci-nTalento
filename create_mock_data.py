"""Generador de datos ficticios para la app de riesgo de rotación.

Se puede usar como módulo (``generate_mock_data``) o como script:

    python create_mock_data.py --n 500 --seed 42 --out datos_ficticios.csv

RELACIONES INTRODUCIDAS (todas inventadas, con fines de demostración)
- Evaluación 2025 depende de la de 2024 (persistencia) más ruido.
- Potencial y nivel de competencias aumentan con la evaluación.
- Los meses desde la última promoción nunca superan la antigüedad.
- La probabilidad de baja voluntaria (modelo logístico) aumenta con:
  compa-ratio bajo, clima bajo, muchos meses sin promoción, evaluación 2025
  baja, caída de evaluación, horas extra altas, antigüedad baja y cambios de
  responsable. Las ausencias NO intervienen en la probabilidad de baja.
- Si se piden columnas sensibles, la edad se correlaciona con la antigüedad
  (para que la auditoría de equidad pueda mostrar un efecto proxy) y el género
  se asigna al azar, sin relación con ninguna otra variable.

ADVERTENCIA: validar el score de la app con estos datos es circular. La baja
se ha generado a partir de los mismos factores que usa el score, de modo que
un buen AUC aquí no dice nada sobre su rendimiento con datos reales.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as cfg

_DEPARTAMENTOS: dict[str, tuple[str, ...]] = {
    "Tecnología": ("Desarrollador/a", "Analista de datos", "Ingeniero/a de sistemas", "Jefe/a de proyecto"),
    "Ventas": ("Comercial", "Key account", "Responsable de zona"),
    "Operaciones": ("Técnico/a de operaciones", "Coordinador/a logístico", "Responsable de planta"),
    "Finanzas": ("Analista financiero", "Contable", "Controller"),
    "Personas": ("Técnico/a de RR. HH.", "HR Business Partner", "Responsable de selección"),
    "Marketing": ("Especialista digital", "Product marketing", "Responsable de marca"),
    "Atención al Cliente": ("Agente", "Supervisor/a", "Responsable de calidad"),
}
_PESOS_DEPARTAMENTO: tuple[float, ...] = (0.22, 0.18, 0.20, 0.10, 0.07, 0.08, 0.15)
_PESOS_NIVEL: tuple[float, ...] = (0.30, 0.38, 0.22, 0.10)
_PESOS_MODALIDAD: tuple[float, ...] = (0.45, 0.40, 0.15)


def generate_mock_data(n: int = 500, seed: int = 42, include_sensitive: bool = False) -> pd.DataFrame:
    """Genera un DataFrame ficticio y reproducible con el esquema de la app.

    Args:
        n: Número de empleados ficticios (mínimo 10).
        seed: Semilla del generador aleatorio.
        include_sensitive: Si es True, añade Edad y Género ficticios para
            probar la auditoría de equidad. Por defecto no se generan.

    Returns:
        DataFrame con las columnas del esquema, incluida ``Baja_Voluntaria``.

    Raises:
        ValueError: Si ``n`` es menor que 10.
    """
    if n < 10:
        raise ValueError("El número de empleados ficticios debe ser al menos 10.")

    rng = np.random.default_rng(seed)

    departamentos = rng.choice(list(_DEPARTAMENTOS), size=n, p=_PESOS_DEPARTAMENTO)
    puestos = np.array([rng.choice(_DEPARTAMENTOS[d]) for d in departamentos])
    niveles = rng.choice(cfg.NIVELES, size=n, p=_PESOS_NIVEL)

    # Antigüedad: distribución sesgada a la derecha, más alta en niveles superiores.
    plus_nivel = pd.Series(niveles).map({"Junior": 0, "Medio": 24, "Senior": 60, "Mánager": 84}).to_numpy()
    antiguedad = np.clip(rng.gamma(shape=1.8, scale=30, size=n) + plus_nivel, 1, 480).round().astype(int)

    eval_2024 = np.clip(rng.normal(3.4, 0.6, n), 1, 5)
    eval_2025 = np.clip(0.6 * eval_2024 + 0.4 * 3.4 + rng.normal(0, 0.45, n), 1, 5)
    eval_2024 = eval_2024.round(1)
    eval_2025 = eval_2025.round(1)

    pot_latente = eval_2025 + rng.normal(0, 0.5, n)
    potencial = np.where(pot_latente >= 3.9, "Alto", np.where(pot_latente >= 3.0, "Medio", "Bajo"))
    competencias = np.clip(np.round(eval_2025 + rng.normal(0, 0.6, n)), 1, 5).astype(int)

    formacion = np.clip(rng.gamma(shape=2.0, scale=12, size=n), 0, 120).round().astype(int)
    meses_promo = np.minimum(rng.gamma(shape=2.0, scale=16, size=n).round().astype(int), antiguedad)
    compa = np.clip(rng.normal(0.98, 0.09, n), 0.7, 1.3).round(2)
    horas_extra = np.clip(rng.gamma(shape=1.6, scale=6, size=n), 0, 80).round(1)
    cambios_resp = np.clip(rng.poisson(0.7, n), 0, 4).astype(int)
    clima = np.clip(rng.normal(3.5, 0.7, n), 1, 5).round(1)
    ausencias = np.clip(rng.poisson(6, n), 0, 60).astype(int)
    modalidad = rng.choice(cfg.MODALIDADES, size=n, p=_PESOS_MODALIDAD)

    # Probabilidad de baja: logística sobre factores (coeficientes inventados).
    logit = (
        -3.3
        + 2.2 * np.clip(1.0 - compa, 0, None) / 0.2
        + 0.9 * np.clip(3.5 - clima, 0, None)
        + 0.020 * np.clip(meses_promo - 24, 0, None)
        + 0.5 * np.clip(3.5 - eval_2025, 0, None)
        + 0.6 * np.clip(eval_2024 - eval_2025, 0, None)
        + 0.04 * np.clip(horas_extra - 10, 0, None)
        + 0.5 * (antiguedad <= 24)
        + 0.3 * np.clip(cambios_resp - 1, 0, None)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    baja = rng.random(n) < prob

    df = pd.DataFrame(
        {
            cfg.COL_ID: [f"EMP-{i:04d}" for i in range(1, n + 1)],
            cfg.COL_DEPARTAMENTO: departamentos,
            cfg.COL_PUESTO: puestos,
            cfg.COL_NIVEL: niveles,
            cfg.COL_ANTIGUEDAD: antiguedad,
            cfg.COL_EVAL_2024: eval_2024,
            cfg.COL_EVAL_2025: eval_2025,
            cfg.COL_POTENCIAL: potencial,
            cfg.COL_COMPETENCIAS: competencias,
            cfg.COL_FORMACION: formacion,
            cfg.COL_MESES_PROMO: meses_promo,
            cfg.COL_COMPA: compa,
            cfg.COL_HORAS_EXTRA: horas_extra,
            cfg.COL_CAMBIOS_RESP: cambios_resp,
            cfg.COL_CLIMA: clima,
            cfg.COL_AUSENCIAS: ausencias,
            cfg.COL_MODALIDAD: modalidad,
            cfg.COL_BAJA: baja,
        }
    )

    if include_sensitive:
        # Edad correlacionada con la antigüedad (efecto proxy deliberado).
        edad = np.clip(24 + 0.6 * antiguedad / 12 + rng.normal(12, 9, n), 18, 67).round().astype(int)
        genero = rng.choice(("Mujer", "Hombre", "No binario"), size=n, p=(0.48, 0.49, 0.03))
        df[cfg.COL_EDAD] = edad
        df[cfg.COL_GENERO] = genero

    return df


def _main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description="Genera datos ficticios de empleados.")
    parser.add_argument("--n", type=int, default=500, help="Número de empleados (por defecto 500)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla (por defecto 42)")
    parser.add_argument("--sensibles", action="store_true", help="Incluye Edad y Género ficticios")
    parser.add_argument("--out", default="datos_ficticios.csv", help="Ruta del CSV de salida")
    args = parser.parse_args()

    df = generate_mock_data(n=args.n, seed=args.seed, include_sensitive=args.sensibles)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Generados {len(df)} registros en {args.out}")


if __name__ == "__main__":
    _main()
