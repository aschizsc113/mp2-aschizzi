"""Carga de los datos, limpieza de la variable objetivo y rutas del proyecto (ejemplo resuelto: R1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
MODELOS = RAIZ / "models"
INFORMES = RAIZ / "reports"
ENTREGABLES = RAIZ / "entregables"

OBJETIVO = "precio"
PARTICIONES_CV = ["1", "2", "3", "4", "5"]
VALIDACION = "validacion"
# Columnas que describen la operación, no la vivienda: no son variables del modelo
NO_VARIABLES = [OBJETIVO, "particion", "estado_operacion"]


def cargar_entrenamiento() -> pd.DataFrame:
    """Lee train_set.csv y le añade la columna `particion` de particiones.csv (R1).

    El índice es la columna `index` de los CSV. No se limpia nada: eso lo decide el equipo.
    """
    train = pd.read_csv(RAW / "train_set.csv")
    particiones = pd.read_csv(RAW / "particiones.csv", dtype={"particion": str})
    datos = train.merge(particiones, on="index", how="left", validate="one_to_one")
    return datos.set_index("index")


def cargar_test() -> pd.DataFrame:
    """Lee test_set.csv tal cual, con `index` como índice. No tiene `precio` (R1)."""
    return pd.read_csv(RAW / "test_set.csv", index_col="index")


PRECIO_MINIMO = 30_000
PRECIO_MAXIMO = 3_000_000


def limpiar_objetivo(datos: pd.DataFrame) -> pd.DataFrame:
    """Quita anuncios retirados (P1) y precios fuera de [30.000, 3.000.000] € (P2) (R9, CM7).

    Decide FILA A FILA con reglas fijas: no usa medianas, percentiles ni nada de otras filas.
    Los precios de exactamente 30.000 € o 3.000.000 € se quedan. No cambia ninguna columna.
    """
    vendida = datos["estado_operacion"] != "retirada"
    precio_valido = datos[OBJETIVO].between(PRECIO_MINIMO, PRECIO_MAXIMO)
    return datos[vendida & precio_valido]


def separar(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables (X) de la variable objetivo (y) (R1).

    Quita `precio`, `particion` y `estado_operacion`: describen la operación, no la vivienda,
    y no están en el test.
    """
    X = datos.drop(columns=NO_VARIABLES, errors="ignore")
    return X, datos[OBJETIVO]
