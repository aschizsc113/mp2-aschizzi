"""Tests de la fase 2 · Transformaciones (criterios CA10-CA17 de SPECS.md)."""
import numpy as np
import pandas as pd
import pytest

from src.datos import limpiar_objetivo
from src.preparacion import CorrectorSuperficie, ascensor_a_binario, normalizar_tipo, parking_a_ordinal


def _superficies(tipos: list[str], superficies: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"tipo": tipos, "superficie_construida": superficies})


@pytest.mark.criterio("CA10")
def test_normalizar_tipo():
    entrada = pd.Series(["PISO", " piso ", "Ático", "Atico", "Dúplex", "Duplex", "Chalet", "Casa"])
    esperado = ["piso", "piso", "atico", "atico", "duplex", "duplex", "casa", "casa"]
    assert normalizar_tipo(entrada).tolist() == esperado


@pytest.mark.criterio("CA11")
def test_corrector_superficie_pisos():
    corrector = CorrectorSuperficie().fit(_superficies(["Piso"] * 5, [60, 80, 100, 0, 900]))
    # Con 60, 80 y 100 m²: mediana 80; percentil 99 = 80 + 0,98 × (100 − 80) = 99,6
    assert corrector.medianas_["piso"] == pytest.approx(80)
    assert corrector.topes_["piso"] == pytest.approx(99.6)
    resultado = corrector.transform(_superficies(["Piso"] * 4, [0, 60, 100, 900]))
    np.testing.assert_allclose(resultado["superficie_construida"], [80, 60, 99.6, 99.6])


@pytest.mark.criterio("CA12")
def test_corrector_superficie_casa_y_chalet():
    corrector = CorrectorSuperficie().fit(_superficies(["Casa", "Chalet"], [200, 300]))
    # Casa = {200, 300}: mediana 250; percentil 99 = 200 + 0,99 × 100 = 299
    assert corrector.medianas_["casa"] == pytest.approx(250)
    assert corrector.topes_["casa"] == pytest.approx(299)
    resultado = corrector.transform(_superficies(["chalet", "CASA"], [0, 400]))
    np.testing.assert_allclose(resultado["superficie_construida"], [250, 299])


@pytest.mark.criterio("CA13")
def test_corrector_superficie_no_aprende_al_transformar():
    filas_a = _superficies(["Piso", "Piso", "Piso", "Ático", "Ático"], [60, 80, 100, 90, 120])
    filas_b = _superficies(["Piso", "Ático", "Piso"], [0, 500, 70])
    corrector = CorrectorSuperficie().fit(filas_a)
    medianas, topes = dict(corrector.medianas_), dict(corrector.topes_)
    primera = corrector.transform(filas_b)
    otra_b = filas_b.assign(superficie_construida=[5_000, 1, 0])
    corrector.transform(otra_b)
    assert corrector.medianas_ == medianas and corrector.topes_ == topes
    # Las filas de B se corrigen con las cifras de A, no con las suyas
    np.testing.assert_allclose(primera["superficie_construida"], [80, topes["atico"], 70])


@pytest.mark.criterio("CA14")
def test_corrector_superficie_datos_limpios(entrenamiento):
    limpios = limpiar_objetivo(entrenamiento)
    corrector = CorrectorSuperficie().fit(limpios)
    esperado = {"piso": 196.0, "atico": 182.6, "duplex": 253.3, "casa": 357.3, "estudio": 60.9}
    for tipo, tope in esperado.items():
        assert corrector.topes_[tipo] == pytest.approx(tope, abs=0.1), tipo
    tipos = normalizar_tipo(limpios["tipo"])
    topes = tipos.map(corrector.topes_)
    assert int((limpios["superficie_construida"] > topes).sum()) == 73
    transformado = corrector.transform(limpios)["superficie_construida"]
    assert (transformado > 0).all()


@pytest.mark.criterio("CA15")
def test_ascensor_a_binario():
    entrada = pd.DataFrame({"ascensor": ["Sí", "Si", "SI", "si", "S", "True", "1",
                                         "No", "no", "N", "False", "0", np.nan]})
    esperado = [1] * 7 + [0] * 6
    assert ascensor_a_binario(entrada)["ascensor"].tolist() == esperado


@pytest.mark.criterio("CA16")
def test_parking_a_ordinal():
    entrada = pd.DataFrame({"parking": ["No", "Opcional", "Sí", "Incluido", np.nan]})
    assert parking_a_ordinal(entrada)["parking"].tolist() == [0, 1, 2, 2, 0]


@pytest.mark.criterio("CA17")
def test_e2_predice_en_euros_con_log10():
    from src.estrategias import buscar
    X = pd.DataFrame({"superficie_construida": [1.0, 2.0, 3.0]})
    y = pd.Series([10.0, 100.0, 1_000.0])
    modelo = buscar("E2").crear().fit(X, y)
    # log10(precio) = superficie exactamente → superficie 4 da 10^4 = 10.000 €
    assert modelo.predict(pd.DataFrame({"superficie_construida": [4.0]}))[0] == pytest.approx(10_000, abs=1)
