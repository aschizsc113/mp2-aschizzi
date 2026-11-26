"""Por qué el modelo tasa lo que tasa: contribuciones y coeficientes (R18, CM5, DESIGN §6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def _partes(modelo: Pipeline) -> tuple[Pipeline, object]:
    """La preparación (todo menos el último paso) y el estimador lineal ya entrenado."""
    preparacion, final = modelo[:-1], modelo[-1]
    if isinstance(final, TransformedTargetRegressor):
        final = final.regressor_
    return preparacion, final


def explicar(modelo: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """`intercepto` + `coeficiente × valor transformado` por columna; cada fila suma la predicción en log10 (R18)."""
    preparacion, estimador = _partes(modelo)
    transformado = np.asarray(preparacion.transform(X), dtype=float)
    nombres = list(preparacion.get_feature_names_out())
    contribuciones = pd.DataFrame(transformado * np.asarray(estimador.coef_, dtype=float), columns=nombres,
                                  index=X.index)
    contribuciones.insert(0, "intercepto", float(estimador.intercept_))
    return contribuciones


def coeficientes(modelo: Pipeline) -> pd.DataFrame:
    """Coeficiente de cada variable (por desviación típica y por unidad) y su factor 10^coef (DESIGN §6).

    - `coeficiente`, `factor`: cuando la variable sube una desviación típica (variables estandarizadas).
    - `coeficiente_unidad`, `factor_unidad`: cuando sube una unidad (1 m², o pasar de 0 a 1).
    """
    preparacion, estimador = _partes(modelo)
    nombres = list(preparacion.get_feature_names_out())
    coef = np.asarray(estimador.coef_, dtype=float)
    escala = np.ones_like(coef)
    if "escalado" in modelo.named_steps:
        escala = np.asarray(modelo.named_steps["escalado"].scale_, dtype=float)
    tabla = pd.DataFrame({"variable": nombres, "coeficiente": coef, "coeficiente_unidad": coef / escala})
    tabla["factor"] = 10 ** tabla["coeficiente"]
    tabla["factor_unidad"] = 10 ** tabla["coeficiente_unidad"]
    return tabla.reindex(tabla["coeficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)
