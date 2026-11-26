"""Tests de la fase 1 · Limpieza (criterios CA8 y CA9 de SPECS.md)."""
import pandas as pd
import pytest

from src.datos import limpiar_objetivo


@pytest.mark.criterio("CA8")
def test_limpiar_objetivo_seis_filas():
    datos = pd.DataFrame(
        {
            "estado_operacion": ["vendida", "retirada", "vendida", "vendida", "vendida", "vendida"],
            "precio": [250_000.0, 250_000.0, 29_999.0, 30_000.0, 3_000_000.0, 3_000_001.0],
            "barri": ["A", "B", "C", "D", "E", "F"],
            "particion": ["1", "2", "3", "4", "5", "validacion"],
        },
        index=pd.Index([1, 2, 3, 4, 5, 6], name="index"),
    )
    limpios = limpiar_objetivo(datos)
    # Se quedan la 1 (vendida a 250.000), la 4 (30.000 exactos) y la 5 (3.000.000 exactos)
    assert list(limpios.index) == [1, 4, 5]
    pd.testing.assert_frame_equal(limpios, datos.loc[[1, 4, 5]])


@pytest.mark.criterio("CA9")
def test_limpiar_objetivo_train_set(entrenamiento):
    limpios = limpiar_objetivo(entrenamiento)
    assert len(limpios) == 5592
    assert (limpios["estado_operacion"] != "retirada").all()
    assert limpios["precio"].between(30_000, 3_000_000).all()
