"""Entrena el modelo final, lo mide en la partición de validación y lo guarda (R6).

    uv run python -m src.entrenar

Se aplica `limpiar_objetivo`, el modelo se entrena con las particiones 1-5 y se mide una sola vez
en `validacion`.
Ese mismo modelo es el que predice el test para Kaggle: así el error estimado
corresponde exactamente al modelo enviado.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.baseline import BaselineMetroCuadrado
from src.datos import (INFORMES, MODELOS, PARTICIONES_CV, VALIDACION, cargar_entrenamiento, limpiar_objetivo,
                       separar)
from src.estrategias import MODELO_FINAL, crear_modelo_final
from src.validacion import intervalo_bootstrap, mae, rmse


def resumir(y_real: pd.Series, y_pred) -> dict:
    """MAE y RMSE con su intervalo de confianza del 95 % por bootstrap (R6)."""
    return {"mae": intervalo_bootstrap(y_real, y_pred, mae),
            "rmse": intervalo_bootstrap(y_real, y_pred, rmse)}


def guardar_predicciones(X_val: pd.DataFrame, y_val: pd.Series, pred_modelo, pred_baseline) -> None:
    """Precio, tasaciones y columnas originales de cada vivienda de `validacion` (SPECS §5, R6).

    Las columnas originales las usa el cuadro de mando (Dónde falla y Abrir la caja).
    """
    predicciones = pd.DataFrame({"precio": y_val, "prediccion": pred_modelo, "prediccion_baseline": pred_baseline},
                                index=X_val.index)
    pd.concat([predicciones, X_val], axis=1).to_csv(INFORMES / "predicciones_validacion.csv")


def main() -> None:
    datos = limpiar_objetivo(cargar_entrenamiento())
    X_ent, y_ent = separar(datos[datos["particion"].isin(PARTICIONES_CV)])
    X_val, y_val = separar(datos[datos["particion"] == VALIDACION])

    modelo = crear_modelo_final().fit(X_ent, y_ent)
    baseline = BaselineMetroCuadrado().fit(X_ent, y_ent)
    pred_modelo, pred_baseline = modelo.predict(X_val), baseline.predict(X_val)

    metricas = {
        "modelo_final": MODELO_FINAL,
        "n_entrenamiento": len(X_ent),
        "n_validacion": len(X_val),
        "baseline": resumir(y_val, pred_baseline),
        "modelo": resumir(y_val, pred_modelo),
    }
    MODELOS.mkdir(exist_ok=True)
    INFORMES.mkdir(exist_ok=True)
    joblib.dump(modelo, MODELOS / "modelo.joblib")
    (INFORMES / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    guardar_predicciones(X_val, y_val, pred_modelo, pred_baseline)

    for nombre in ("baseline", "modelo"):
        m = metricas[nombre]["mae"]
        print(f"{nombre:9s} MAE {m['valor']:>10,.0f} €  IC 95 % [{m['ic95'][0]:,.0f}, {m['ic95'][1]:,.0f}]")
    print(f"Guardado {MODELOS / 'modelo.joblib'} ({MODELO_FINAL})")


if __name__ == "__main__":
    main()
