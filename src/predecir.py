"""Predice el conjunto de test con el modelo guardado y escribe el envío para Kaggle (R7, R19).

    uv run python -m src.predecir
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.datos import ENTREGABLES, INFORMES, MODELOS, cargar_test

TASACION_MAXIMA = 6_000_000


def resumir_envio(precios: np.ndarray) -> dict:
    """Cuántas tasaciones hay, cuántas están fuera de (0, 6.000.000] €, la mínima y la máxima (R19)."""
    precios = np.asarray(precios, dtype=float)
    fuera = int(((precios <= 0) | (precios > TASACION_MAXIMA) | np.isnan(precios)).sum())
    return {"n_tasaciones": int(len(precios)), "n_fuera_de_rango": fuera,
            "minimo": float(np.nanmin(precios)), "maximo": float(np.nanmax(precios))}


def main() -> None:
    modelo = joblib.load(MODELOS / "modelo.joblib")
    test = cargar_test()
    envio = pd.DataFrame({"index": test.index, "precio": modelo.predict(test)})
    ENTREGABLES.mkdir(exist_ok=True)
    envio.to_csv(ENTREGABLES / "envio_kaggle.csv", index=False)
    print(f"Escrito {ENTREGABLES / 'envio_kaggle.csv'} ({len(envio)} filas)")

    resumen = resumir_envio(envio["precio"].to_numpy())
    INFORMES.mkdir(exist_ok=True)
    (INFORMES / "envio_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Tasaciones entre {resumen['minimo']:,.0f} € y {resumen['maximo']:,.0f} €")
    if resumen["n_fuera_de_rango"]:
        print(f"AVISO: {resumen['n_fuera_de_rango']} tasaciones son ≤ 0 € o pasan de {TASACION_MAXIMA:,} € (R19)")


if __name__ == "__main__":
    main()
