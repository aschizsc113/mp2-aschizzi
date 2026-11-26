"""Estrategias que se comparan con validación cruzada (SPECS.md §4).

Cada estrategia es una fila de SPECS.md §4, con su Id, su tipo (`baseline`, `minimo`, `estrategia` o
`caracteristica`), su descripción, su referencia y la función que crea el modelo sin entrenar.
`uv run python -m src.experimentos` las evalúa todas y compara cada una con su referencia (R8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

from src.baseline import BaselineMetroCuadrado
from src.decision import ULTIMA_ACEPTADA
from src.preparacion import (ASCENSOR, PARKING, PLANTA, SUPERFICIE_BRUTA, SUPERFICIE_CORREGIDA, crear_preparacion,
                             potencia_de_10)

# Los tipos del contrato del profesorado (CM6, SPECS.md §4)
TIPOS = ("baseline", "minimo", "estrategia", "caracteristica")

# Orden en el que se resuelve «última aceptada» (R8)
CADENA = ["E2", "T1", "T2", "T3", "N1"]


@dataclass(frozen=True)
class Estrategia:
    nombre: str          # el Id de SPECS.md §4: E0, E1, E2, T1…, m_ridge_10…
    tipo: str            # uno de TIPOS
    descripcion: str     # qué hace, en una frase
    crear: Callable[[], BaseEstimator]  # devuelve el modelo SIN entrenar
    referencia: str = ""  # la de SPECS.md §4: un Id, ULTIMA_ACEPTADA o "" (sin referencia)
    base: str = ""        # el Id cuyas variables amplía (la última aceptada según el equipo al escribirla)


VARIABLES_MINIMO = ["superficie_construida", "habitaciones", "banos", "distancia_centro_km", "poblacion_zona"]


def modelo_minimo() -> Pipeline:
    """Regresión lineal con las variables numéricas que no tienen nulos, sin transformar (R5)."""
    seleccion = ColumnTransformer([("numericas", "passthrough", VARIABLES_MINIMO)])
    return Pipeline([("preparacion", seleccion), ("regresion", LinearRegression())])


def crear_pipeline(variables: list[str], estimador: BaseEstimator) -> Pipeline:
    """Preparación, `StandardScaler` y estimador sobre log10(precio) (R14, R16; ARCHITECTURE §1)."""
    return Pipeline([
        ("preparacion", crear_preparacion(variables)),
        ("escalado", StandardScaler()),
        ("regresion", TransformedTargetRegressor(regressor=estimador, func=np.log10,
                                                 inverse_func=potencia_de_10, check_inverse=False)),
    ])


def lineal(variables: list[str]) -> Callable[[], Pipeline]:
    """Fábrica de una regresión lineal de log10(precio) con esas variables (R14)."""
    return lambda: crear_pipeline(variables, LinearRegression())


# Variables de cada estrategia de la cadena, explícitas (ARCHITECTURE §3: cómo se añade una estrategia)
VARIABLES = {
    "E2": [SUPERFICIE_BRUTA],
    "T1": [SUPERFICIE_CORREGIDA],
    "T2": [SUPERFICIE_CORREGIDA, ASCENSOR],  # T1 (aceptada) + ascensor
    "T3": [SUPERFICIE_CORREGIDA, ASCENSOR, PARKING],  # T2 (aceptada) + parking
    "N1": [SUPERFICIE_CORREGIDA, ASCENSOR, PLANTA],  # T2 (última aceptada; T3 descartada) + planta
}

ESTRATEGIAS: list[Estrategia] = [
    Estrategia("E0", "baseline", "€/m² mediano de la zona × superficie construida", BaselineMetroCuadrado),
    Estrategia("E1", "minimo",
               "Regresión lineal con las 5 variables numéricas sin nulos, sin transformar", modelo_minimo),
    Estrategia("E2", "minimo", "Regresión lineal de log10(precio) solo con superficie_construida, sin transformar",
               lineal(VARIABLES["E2"])),
    Estrategia("T1", "estrategia", "Superficie corregida por tipo: ceros → mediana, por encima del p99 → p99",
               lineal(VARIABLES["T1"]), referencia="E2", base="E2"),
    Estrategia("T2", "estrategia", "Añade ascensor normalizado a 0/1, vacíos → 0",
               lineal(VARIABLES["T2"]), referencia=ULTIMA_ACEPTADA, base="T1"),
    Estrategia("T3", "estrategia", "Añade parking: 0 (No) / 1 (Opcional) / 2 (Sí, Incluido), vacíos → 0",
               lineal(VARIABLES["T3"]), referencia=ULTIMA_ACEPTADA, base="T2"),
    Estrategia("N1", "caracteristica", "Añade es_atico, es_bajo y es_casa a partir de planta normalizada",
               lineal(VARIABLES["N1"]), referencia=ULTIMA_ACEPTADA, base="T2"),
]

# Fase 4 (R16): mismas variables que la última aceptada (N1), solo cambia el estimador
VARIABLES_FASE_4 = VARIABLES["N1"]
BASE_FASE_4 = "N1"
ALFAS_RIDGE = [0.1, 1, 10, 100, 1_000, 10_000, 100_000]
ALFAS_LASSO = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
L1_RATIOS = [0.1, 0.5, 0.9]


def penalizado(estimador: Callable[[], BaseEstimator]) -> Callable[[], Pipeline]:
    """Fábrica de un modelo de la fase 4 con las variables de la última aceptada (R16)."""
    return lambda: crear_pipeline(VARIABLES_FASE_4, estimador())


def estrategias_fase_4() -> list[Estrategia]:
    """Las 35 combinaciones de M1-M3: m_ridge_<α>, m_lasso_<α> y m_enet_<α>_<l1_ratio> (R16)."""
    def nueva(nombre: str, descripcion: str, estimador: Callable[[], BaseEstimator]) -> Estrategia:
        return Estrategia(nombre, "estrategia", descripcion, penalizado(estimador),
                          referencia=ULTIMA_ACEPTADA, base=BASE_FASE_4)

    lista = [nueva(f"m_ridge_{a:g}", f"Ridge con α = {a:g}", lambda a=a: Ridge(alpha=a)) for a in ALFAS_RIDGE]
    lista += [nueva(f"m_lasso_{a:g}", f"Lasso con α = {a:g}", lambda a=a: Lasso(alpha=a)) for a in ALFAS_LASSO]
    lista += [nueva(f"m_enet_{a:g}_{r:g}", f"ElasticNet con α = {a:g} y l1_ratio = {r:g}",
                    lambda a=a, r=r: ElasticNet(alpha=a, l1_ratio=r))
              for a in ALFAS_LASSO for r in L1_RATIOS]
    return lista


ESTRATEGIAS += estrategias_fase_4()

# La estrategia que se entrena como modelo final (uv run python -m src.entrenar). R17.
MODELO_FINAL = "m_enet_0.1_0.1"


def buscar(nombre: str) -> Estrategia:
    """Devuelve la estrategia con ese nombre."""
    for estrategia in ESTRATEGIAS:
        if estrategia.nombre == nombre:
            return estrategia
    raise KeyError(f"No hay ninguna estrategia llamada {nombre!r} en ESTRATEGIAS")


def crear_modelo_final() -> BaseEstimator:
    """El modelo de MODELO_FINAL, sin entrenar."""
    return buscar(MODELO_FINAL).crear()
