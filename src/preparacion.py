"""Transformadores de las fases 2 y 3 (SPECS.md R10-R15).

Todo lo que aprende algo de los datos (medianas, topes) lo hace en `fit`, así que dentro del
`Pipeline` solo ve las filas de entrenamiento de cada vuelta. El resto son funciones fila a fila.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

SUPERFICIE = "superficie_construida"
SUPERFICIE_MINIMA, SUPERFICIE_MAXIMA = 1, 400
PERCENTIL_TOPE = 0.99

VALORES_SI = {"si", "s", "true", "1"}
VALORES_PARKING = {"no": 0, "opcional": 1, "si": 2, "incluido": 2}
PLANTAS = {
    "es_atico": {"atico", "at"},
    "es_bajo": {"bajo", "bj", "planta baja", "0"},
    "es_casa": {"casa"},
}

# Variables que puede pedir una estrategia a crear_preparacion()
SUPERFICIE_BRUTA = "superficie_bruta"
SUPERFICIE_CORREGIDA = "superficie_corregida"
ASCENSOR = "ascensor"
PARKING = "parking"
PLANTA = "planta"


def normalizar_texto(valor: object) -> str | None:
    """Minúsculas, sin tildes y sin espacios a los lados; los vacíos quedan en None (R10, R12, R13, R15)."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    if isinstance(valor, (int, float, np.integer, np.floating)) and float(valor).is_integer():
        valor = int(valor)
    sin_tildes = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode("ascii")
    return sin_tildes.lower().strip()


def normalizar_tipo(tipos: pd.Series) -> pd.Series:
    """`tipo` normalizado; `chalet` pasa a `casa` (R10)."""
    return tipos.map(normalizar_texto).replace({"chalet": "casa"})


class CorrectorSuperficie(TransformerMixin, BaseEstimator):
    """Ceros o vacíos → mediana de su tipo; por encima del percentil 99 de su tipo → ese percentil (R11).

    Mediana y percentil se calculan en `fit` con las filas de cada tipo entre 1 y 400 m².
    Un tipo que no estaba en el entrenamiento usa las cifras de todas esas filas.
    Recibe `tipo` y `superficie_construida`; devuelve solo `superficie_construida`.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "CorrectorSuperficie":
        tipos = normalizar_tipo(X["tipo"])
        superficie = X[SUPERFICIE].astype(float)
        validas = superficie.between(SUPERFICIE_MINIMA, SUPERFICIE_MAXIMA)
        por_tipo = superficie[validas].groupby(tipos[validas])
        self.medianas_ = {t: float(v) for t, v in por_tipo.median().items()}
        self.topes_ = {t: float(v) for t, v in por_tipo.quantile(PERCENTIL_TOPE).items()}
        self.mediana_global_ = float(superficie[validas].median())
        self.tope_global_ = float(superficie[validas].quantile(PERCENTIL_TOPE))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        tipos = normalizar_tipo(X["tipo"])
        medianas = tipos.map(self.medianas_).astype(float).fillna(self.mediana_global_)
        topes = tipos.map(self.topes_).astype(float).fillna(self.tope_global_)
        superficie = X[SUPERFICIE].astype(float)
        superficie = superficie.where(superficie.notna() & (superficie != 0), medianas)
        superficie = superficie.where(superficie <= topes, topes)
        return pd.DataFrame({SUPERFICIE: superficie.to_numpy()}, index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.array([SUPERFICIE], dtype=object)


def ascensor_a_binario(X: pd.DataFrame) -> pd.DataFrame:
    """`ascensor`: sí/si/s/true/1 → 1; no/n/false/0, vacíos y desconocidos → 0 (R12)."""
    valores = X["ascensor"].map(normalizar_texto)
    return pd.DataFrame({"ascensor": valores.isin(VALORES_SI).astype(int).to_numpy()}, index=X.index)


def parking_a_ordinal(X: pd.DataFrame) -> pd.DataFrame:
    """`parking`: no → 0, opcional → 1, si/incluido → 2; vacíos y desconocidos → 0 (R13)."""
    valores = X["parking"].map(normalizar_texto).map(VALORES_PARKING).fillna(0)
    return pd.DataFrame({"parking": valores.astype(int).to_numpy()}, index=X.index)


def planta_a_indicadores(X: pd.DataFrame) -> pd.DataFrame:
    """`es_atico`, `es_bajo` y `es_casa` a partir de `planta` normalizada; el resto → 0 (R15)."""
    valores = X["planta"].map(normalizar_texto)
    return pd.DataFrame({nombre: valores.isin(grupo).astype(int).to_numpy() for nombre, grupo in PLANTAS.items()},
                        index=X.index)


def nombres_planta(transformador: object, input_features: object) -> list[str]:
    """Nombres de las columnas que salen de `planta_a_indicadores` (R15)."""
    return list(PLANTAS)


def potencia_de_10(z: np.ndarray) -> np.ndarray:
    """Inversa de log10: la tasación vuelve a euros (R14). Con nombre para que joblib la guarde."""
    return np.power(10.0, z)


def crear_preparacion(variables: list[str]) -> ColumnTransformer:
    """`ColumnTransformer` con las variables pedidas, en este orden de salida (R10-R15).

    - `superficie_bruta`: `superficie_construida` sin tocar (E2).
    - `superficie_corregida`: `superficie_construida` corregida por tipo (R10, R11).
    - `ascensor` (R12), `parking` (R13) y `planta` → `es_atico`, `es_bajo`, `es_casa` (R15).
    """
    piezas = {
        SUPERFICIE_BRUTA: ("superficie", "passthrough", [SUPERFICIE]),
        SUPERFICIE_CORREGIDA: ("superficie", CorrectorSuperficie(), ["tipo", SUPERFICIE]),
        ASCENSOR: ("ascensor", FunctionTransformer(ascensor_a_binario, feature_names_out="one-to-one"),
                   ["ascensor"]),
        PARKING: ("parking", FunctionTransformer(parking_a_ordinal, feature_names_out="one-to-one"), ["parking"]),
        PLANTA: ("planta", FunctionTransformer(planta_a_indicadores, feature_names_out=nombres_planta), ["planta"]),
    }
    desconocidas = set(variables) - set(piezas)
    if desconocidas:
        raise ValueError(f"Variables desconocidas: {sorted(desconocidas)}")
    return ColumnTransformer([piezas[v] for v in variables], remainder="drop", verbose_feature_names_out=False)
