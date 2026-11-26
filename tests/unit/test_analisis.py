"""Tests unitarios de src/analisis.py: cifras y frases del cuadro de mando (DESIGN.md)."""
import numpy as np
import pandas as pd
import pytest

from src import analisis


def _metricas(modelo: float, ic_modelo: list[float], ingenua: float, ic_ingenua: list[float]) -> dict:
    def bloque(valor, ic):
        return {"mae": {"valor": valor, "ic95": ic}, "rmse": {"valor": 2 * valor, "ic95": [2 * ic[0], 2 * ic[1]]}}
    return {"modelo_final": "N1", "modelo": bloque(modelo, ic_modelo), "baseline": bloque(ingenua, ic_ingenua)}


def test_margenes_se_solapan():
    assert analisis.margenes_se_solapan([1, 3], [2, 4])
    assert not analisis.margenes_se_solapan([1, 2], [3, 4])


def test_frases_de_rendimiento_en_los_dos_sentidos():
    mejor = _metricas(100_000, [90_000, 110_000], 150_000, [140_000, 160_000])
    peor = _metricas(150_000, [140_000, 160_000], 100_000, [90_000, 110_000])
    empate = _metricas(100_000, [90_000, 110_000], 105_000, [95_000, 115_000])
    assert "la mejora no es ruido" in analisis.frase_rendimiento(mejor)
    assert "claramente peor" in analisis.frase_rendimiento(peor)
    assert "no podemos asegurar" in analisis.frase_rendimiento(empate)
    assert analisis.titulo_metricas(mejor) == "El modelo se equivoca 50.000 € menos que la tasación ingenua"
    assert analisis.titulo_metricas(peor) == "El modelo todavía se equivoca 50.000 € más que la tasación ingenua"


def _predicciones() -> pd.DataFrame:
    return pd.DataFrame({
        "precio": [100_000.0, 200_000.0, 400_000.0, 800_000.0],
        "prediccion": [110_000.0, 180_000.0, 400_000.0, 600_000.0],
        "prediccion_baseline": [100_000.0, 200_000.0, 300_000.0, 900_000.0],
        "districte": ["Gràcia", "Gràcia", None, None],
        "municipi": ["Barcelona", "Barcelona", "Badalona", "Badalona"],
        "tipo": ["Piso", "Piso", "Casa", "Ático"],
    }, index=[10, 11, 12, 13])


def test_errores_zona_y_rango():
    datos = analisis.con_errores(_predicciones())
    np.testing.assert_allclose(datos["error_pct"], [0.1, -0.1, 0.0, -0.25])
    assert datos["zona"].tolist() == ["Gràcia", "Gràcia", "Badalona", "Badalona"]
    assert datos["rango"].tolist() == ["< 150.000 €", "150.000-250.000 €", "400.000-700.000 €", "400.000-700.000 €"]


def test_error_por_grupo_marca_pocas_viviendas():
    tabla = analisis.error_por_grupo(analisis.con_errores(_predicciones()), "zona")
    assert tabla["pocas"].all()
    assert tabla.iloc[0]["zona"] == "Badalona"  # 100.000 € de error medio frente a 15.000 €


def test_frase_error_pct():
    datos = analisis.con_errores(_predicciones())
    assert analisis.frase_error_pct(datos).startswith("La mitad de las tasaciones se desvían menos de 10,0 %")


def test_elegir_vivienda():
    datos = analisis.con_errores(_predicciones())
    assert analisis.elegir_vivienda(datos, "la que más se equivoca") == 13
    assert analisis.elegir_vivienda(datos, "la que mejor acierta") == 12
    assert analisis.elegir_vivienda(datos, "por número (index)", numero=11) == 11
    assert analisis.elegir_vivienda(datos, "por número (index)", numero=999) is None


def test_desglose_termina_en_la_tasacion():
    contribuciones = pd.Series({"intercepto": 5.0, "superficie_construida": 0.1, "ascensor": -0.05})
    pasos = analisis.desglose(contribuciones)
    assert pasos["paso"].tolist() == ["Vivienda media", "Superficie", "Ascensor", "Tasación"]
    assert pasos.iloc[0]["acumulado"] == pytest.approx(100_000)
    assert pasos.iloc[-1]["acumulado"] == pytest.approx(10 ** 5.05)
    assert pasos.iloc[1:-1]["euros"].sum() == pytest.approx(10 ** 5.05 - 100_000)


def _experimentos() -> pd.DataFrame:
    filas = [("E0", "", None, 70_000), ("E1", "", None, 120_000), ("E2", "", None, 150_000),
             ("T1", "E2", True, 130_000), ("T2", "T1", False, 129_500), ("T3", "T1", False, 131_000),
             ("N1", "T1", True, 128_000), ("m_ridge_1", "N1", False, 127_900), ("m_lasso_0.01", "N1", False, 128_500)]
    tabla = pd.DataFrame(filas, columns=["estrategia", "referencia", "aceptada", "mae_medio"])
    for p in range(1, 6):
        tabla[f"mae_p{p}"] = tabla["mae_medio"]
    return tabla


def test_frase_experimentos_y_ultima_aceptada():
    experimentos = _experimentos()
    assert analisis.frase_experimentos(experimentos, "N1") == (
        "Probamos 6 estrategias y aceptamos 2. El error medio bajó de 150.000 € (E2) a 128.000 € (modelo final).")
    assert analisis.ultima_aceptada(experimentos) == "N1"
    assert analisis.tabla_experimentos(experimentos)["decision"].tolist()[:4] == [
        "referencia", "referencia", "referencia", "aceptada"]
