"""Tests de la fase 3 · Variables nuevas (criterios CA18 y CA19 de SPECS.md)."""
import numpy as np
import pandas as pd
import pytest

from src.datos import limpiar_objetivo
from src.preparacion import planta_a_indicadores


@pytest.mark.criterio("CA18")
@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (" Ático ", (1, 0, 0)), ("AT", (1, 0, 0)),
        ("Planta baja", (0, 1, 0)), ("Bj", (0, 1, 0)), ("0", (0, 1, 0)),
        ("Casa", (0, 0, 1)),
        ("Pral.", (0, 0, 0)), ("Entlo", (0, 0, 0)), ("3º", (0, 0, 0)), ("-", (0, 0, 0)), (np.nan, (0, 0, 0)),
    ],
)
def test_planta_a_indicadores(valor, esperado):
    resultado = planta_a_indicadores(pd.DataFrame({"planta": [valor]}))
    assert list(resultado.columns) == ["es_atico", "es_bajo", "es_casa"]
    assert tuple(resultado.iloc[0]) == esperado


@pytest.mark.criterio("CA19")
def test_planta_en_datos_limpios(entrenamiento):
    indicadores = planta_a_indicadores(limpiar_objetivo(entrenamiento))
    assert indicadores.sum().to_dict() == {"es_atico": 334, "es_bajo": 689, "es_casa": 59}
