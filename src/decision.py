"""Regla de decisión y elección del modelo final (SPECS.md §4, R8 y R17).

Funciones puras sobre la tabla de resultados de la validación cruzada: no entrenan, no leen
ficheros y nunca reciben filas de `validacion`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.datos import PARTICIONES_CV

COLUMNAS_MAE = [f"mae_p{p}" for p in PARTICIONES_CV]
ULTIMA_ACEPTADA = "ultima_aceptada"
MEJORA_MINIMA = 1_000.0
PARTICIONES_MINIMAS = 4
FAMILIAS = ("ridge", "lasso", "enet")


def es_verdadero(valor: object) -> bool:
    """`aceptada` leída de un CSV puede ser True, "True", NaN…: solo True o "True" cuentan (R8)."""
    return valor is True or str(valor) == "True"


def es_aceptada(mejora_media: float, particiones_mejor: int) -> bool:
    """Se acepta si mejora en al menos 4 de 5 particiones y al menos 1.000 € de media (R8)."""
    return bool(particiones_mejor >= PARTICIONES_MINIMAS and mejora_media >= MEJORA_MINIMA)


def comparar(errores: list[float], errores_referencia: list[float]) -> dict:
    """`mejora_media`, `particiones_mejor` y `aceptada` de una estrategia frente a su referencia (R8)."""
    errores, referencia = np.asarray(errores, dtype=float), np.asarray(errores_referencia, dtype=float)
    mejora_media = float(referencia.mean() - errores.mean())
    particiones_mejor = int((errores < referencia).sum())
    return {"mejora_media": mejora_media, "particiones_mejor": particiones_mejor,
            "aceptada": es_aceptada(mejora_media, particiones_mejor)}


def ultima_aceptada(aceptadas: dict[str, bool], cadena: list[str]) -> str:
    """La última estrategia de la cadena con `aceptada = True`, o la primera si no hay ninguna (R8, CA6)."""
    ultima = cadena[0]
    for estrategia in cadena[1:]:
        if aceptadas.get(estrategia, False):
            ultima = estrategia
    return ultima


def completar_decisiones(tabla: pd.DataFrame, referencias: dict[str, str], cadena: list[str]) -> pd.DataFrame:
    """Añade `referencia`, `mejora_media`, `particiones_mejor` y `aceptada` a cada fila (R8).

    `referencias` da, para cada estrategia, el Id de su referencia, `ULTIMA_ACEPTADA` o "" (sin referencia).
    `cadena` es el orden en el que «última aceptada» se resuelve (E2 → T1 → … → N1). Las filas se
    procesan en el orden de la tabla, así que una estrategia solo ve las decisiones de las anteriores.
    """
    resultado = tabla.copy()
    errores = {fila.estrategia: [getattr(fila, c) for c in COLUMNAS_MAE] for fila in tabla.itertuples()}
    aceptadas: dict[str, bool] = {}
    columnas: dict[str, list] = {"referencia": [], "mejora_media": [], "particiones_mejor": [], "aceptada": []}
    for estrategia in tabla["estrategia"]:
        declarada = referencias.get(estrategia, "")
        referencia = ultima_aceptada(aceptadas, cadena) if declarada == ULTIMA_ACEPTADA else declarada
        if referencia:
            decision = comparar(errores[estrategia], errores[referencia])
            if estrategia in cadena:
                aceptadas[estrategia] = decision["aceptada"]
        else:
            decision = {"mejora_media": np.nan, "particiones_mejor": np.nan, "aceptada": np.nan}
        columnas["referencia"].append(referencia)
        for clave in ("mejora_media", "particiones_mejor", "aceptada"):
            columnas[clave].append(decision[clave])
    for clave, valores in columnas.items():
        resultado[clave] = pd.Series(valores, index=resultado.index, dtype=object if clave == "aceptada" else None)
    return resultado


def familia_y_alfa(estrategia: str) -> tuple[str, float] | None:
    """`m_ridge_10` → ("ridge", 10.0); `m_enet_0.01_0.5` → ("enet", 0.01); otra cosa → None (R16)."""
    partes = estrategia.split("_")
    if len(partes) < 3 or partes[0] != "m" or partes[1] not in FAMILIAS:
        return None
    return partes[1], float(partes[2])


def mejor_por_familia(tabla: pd.DataFrame) -> dict[str, str]:
    """En cada familia, la combinación con el MAE medio más bajo; si empatan, la de α mayor (R17)."""
    filas = []
    for fila in tabla.itertuples():
        clave = familia_y_alfa(fila.estrategia)
        if clave is not None:
            filas.append({"estrategia": fila.estrategia, "familia": clave[0], "alfa": clave[1],
                          "mae_medio": fila.mae_medio})
    if not filas:
        return {}
    candidatos = pd.DataFrame(filas).sort_values(["mae_medio", "alfa"], ascending=[True, False], kind="stable")
    return candidatos.groupby("familia", sort=False)["estrategia"].first().to_dict()


def elegir_modelo_final(tabla: pd.DataFrame, cadena: list[str]) -> str:
    """La mejor de las tres familias si pasa R8 frente a la última aceptada; si no, la última aceptada (R17).

    `tabla` debe tener ya la columna `aceptada` (de `completar_decisiones`).
    """
    por_id = tabla.set_index("estrategia")
    aceptadas = {e: es_verdadero(por_id.loc[e, "aceptada"]) for e in cadena if e in por_id.index}
    referencia = ultima_aceptada(aceptadas, cadena)
    ganadoras = mejor_por_familia(tabla)
    if not ganadoras:
        return referencia
    mejor = min(ganadoras.values(), key=lambda e: por_id.loc[e, "mae_medio"])
    decision = comparar(por_id.loc[mejor, COLUMNAS_MAE].tolist(), por_id.loc[referencia, COLUMNAS_MAE].tolist())
    return mejor if decision["aceptada"] else referencia
