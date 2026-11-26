"""Evalúa todas las estrategias con validación cruzada y escribe reports/experimentos.csv (R3, R8).

    uv run python -m src.experimentos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.datos import INFORMES, PARTICIONES_CV, cargar_entrenamiento, limpiar_objetivo
from src.decision import completar_decisiones, elegir_modelo_final
from src.estrategias import CADENA, ESTRATEGIAS, MODELO_FINAL, TIPOS
from src.validacion import validacion_cruzada

COLUMNAS = (["estrategia", "tipo", "descripcion", "referencia"] + [f"mae_p{p}" for p in PARTICIONES_CV]
            + ["mae_medio", "mae_sd", "mejora_media", "particiones_mejor", "aceptada"])


def evaluar_estrategias(datos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por estrategia: MAE de cada partición, media y desviación típica (R3)."""
    filas = []
    for estrategia in ESTRATEGIAS:
        if estrategia.tipo not in TIPOS:
            raise ValueError(f"{estrategia.nombre}: tipo {estrategia.tipo!r} no es uno de {TIPOS}")
        errores = validacion_cruzada(estrategia.crear(), datos)
        fila = {"estrategia": estrategia.nombre, "tipo": estrategia.tipo, "descripcion": estrategia.descripcion}
        fila |= {f"mae_p{p}": e for p, e in zip(PARTICIONES_CV, errores)}
        fila |= {"mae_medio": float(np.mean(errores)), "mae_sd": float(np.std(errores, ddof=1))}
        filas.append(fila)
        print(f"{estrategia.nombre:24s} MAE medio {fila['mae_medio']:>12,.0f} €  ± {fila['mae_sd']:>9,.0f}")
    return pd.DataFrame(filas)


def avisar_bases(tabla: pd.DataFrame) -> None:
    """Avisa si una estrategia amplía otra que no es su referencia calculada (ARCHITECTURE §3)."""
    referencias = tabla.set_index("estrategia")["referencia"]
    for estrategia in ESTRATEGIAS:
        if estrategia.base and referencias[estrategia.nombre] != estrategia.base:
            print(f"AVISO: {estrategia.nombre} amplía las variables de {estrategia.base}, pero su referencia "
                  f"(última aceptada) es {referencias[estrategia.nombre]}. Hay que reescribirla.")


def main() -> None:
    resultados = evaluar_estrategias(limpiar_objetivo(cargar_entrenamiento()))
    referencias = {e.nombre: e.referencia for e in ESTRATEGIAS}
    resultados = completar_decisiones(resultados, referencias, CADENA)[COLUMNAS]
    avisar_bases(resultados)
    elegido = elegir_modelo_final(resultados, [e for e in CADENA if e in set(resultados["estrategia"])])
    if elegido != MODELO_FINAL:
        print(f"AVISO: según R17 el modelo final sería {elegido}, y MODELO_FINAL es {MODELO_FINAL}.")
    INFORMES.mkdir(exist_ok=True)
    resultados.to_csv(INFORMES / "experimentos.csv", index=False)
    print(f"Escrito {INFORMES / 'experimentos.csv'}")


if __name__ == "__main__":
    main()
