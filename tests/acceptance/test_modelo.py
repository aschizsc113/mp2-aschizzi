"""Tests de la fase 4 · Modelo y explicación (criterios CA7 y CA20-CA24 de SPECS.md)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from src.datos import PARTICIONES_CV, VALIDACION, limpiar_objetivo, separar
from src.decision import COLUMNAS_MAE, completar_decisiones, elegir_modelo_final, mejor_por_familia

COLUMNAS_R8 = {"estrategia", "tipo", "referencia", "mejora_media", "particiones_mejor", "aceptada",
               "mae_medio", "mae_sd"} | set(COLUMNAS_MAE)


@pytest.mark.criterio("CA7")
def test_experimentos_tiene_todas_las_filas(experimentos):
    assert not COLUMNAS_R8 - set(experimentos.columns), "Faltan columnas de R8"
    for estrategia in ["E0", "E1", "E2", "T1", "T2", "T3", "N1"]:
        assert (experimentos["estrategia"] == estrategia).sum() == 1, estrategia
    assert experimentos["estrategia"].str.startswith("m_").sum() == 35


@pytest.mark.criterio("CA20")
def test_rejilla_fase_4(experimentos):
    ids = experimentos["estrategia"]
    assert ids.str.startswith("m_ridge_").sum() == 7
    assert ids.str.startswith("m_lasso_").sum() == 7
    assert ids.str.startswith("m_enet_").sum() == 21
    assert (experimentos.loc[ids.str.startswith("m_"), "tipo"] == "estrategia").all()


def _tabla(errores: dict[str, float]) -> pd.DataFrame:
    """Tabla de resultados inventada: el mismo MAE en las 5 particiones."""
    filas = [{"estrategia": e, **{c: v for c in COLUMNAS_MAE}, "mae_medio": v} for e, v in errores.items()]
    return pd.DataFrame(filas)


def _decidir(errores: dict[str, float]) -> pd.DataFrame:
    referencias = {e: ("" if e == "N1" else "ultima_aceptada") for e in errores}
    return completar_decisiones(_tabla(errores), referencias, cadena=["N1"])


@pytest.mark.criterio("CA21")
def test_elige_el_mejor_de_cada_familia():
    tabla = _decidir({"N1": 100_000, "m_ridge_1": 99_000, "m_ridge_10": 98_500,
                      "m_lasso_0.001": 97_000, "m_lasso_0.01": 97_000,  # empate: gana α mayor
                      "m_enet_0.001_0.5": 99_500, "m_enet_0.01_0.1": 99_200})
    assert mejor_por_familia(tabla) == {"ridge": "m_ridge_10", "lasso": "m_lasso_0.01", "enet": "m_enet_0.01_0.1"}
    # La mejor (lasso, 97.000) baja 3.000 € en 5 de 5 → se acepta
    assert elegir_modelo_final(tabla, cadena=["N1"]) == "m_lasso_0.01"


@pytest.mark.criterio("CA21")
def test_si_el_mejor_no_pasa_r8_queda_la_regresion_lineal():
    tabla = _decidir({"N1": 100_000, "m_ridge_1": 99_800, "m_lasso_0.001": 99_900, "m_enet_0.001_0.5": 99_700})
    assert elegir_modelo_final(tabla, cadena=["N1"]) == "N1"


def _datos_pequenos(precio_validacion: float) -> pd.DataFrame:
    filas = []
    for p in PARTICIONES_CV:
        for x in (1.0, 2.0, 3.0):
            filas.append({"x": x, "precio": 10 * x + int(p), "particion": p})
    filas.append({"x": 2.0, "precio": precio_validacion, "particion": VALIDACION})
    return pd.DataFrame(filas)


@pytest.mark.criterio("CA21")
def test_la_eleccion_no_mira_validacion():
    from src.validacion import validacion_cruzada

    def elegir(datos: pd.DataFrame) -> str:
        modelos = {"N1": LinearRegression(), "m_ridge_0.1": Ridge(alpha=0.1), "m_ridge_100": Ridge(alpha=100)}
        filas = []
        for nombre, modelo in modelos.items():
            errores = validacion_cruzada(modelo, datos)
            filas.append({"estrategia": nombre, **dict(zip(COLUMNAS_MAE, errores)), "mae_medio": np.mean(errores)})
        referencias = {"N1": "", "m_ridge_0.1": "ultima_aceptada", "m_ridge_100": "ultima_aceptada"}
        return elegir_modelo_final(completar_decisiones(pd.DataFrame(filas), referencias, ["N1"]), ["N1"])

    assert elegir(_datos_pequenos(20.0)) == elegir(_datos_pequenos(1e9))


@pytest.mark.criterio("CA21")
def test_modelo_final_es_el_que_dice_r17(experimentos):
    from src.estrategias import CADENA, MODELO_FINAL
    assert MODELO_FINAL == elegir_modelo_final(experimentos, CADENA)


@pytest.mark.criterio("CA22")
def test_explicar_a_mano():
    from src.estrategias import crear_pipeline
    from src.explicar import explicar
    from src.preparacion import SUPERFICIE_BRUTA
    X = pd.DataFrame({"superficie_construida": [1.0, 2.0, 3.0]})
    modelo = crear_pipeline([SUPERFICIE_BRUTA], LinearRegression()).fit(X, pd.Series([10.0, 100.0, 1_000.0]))
    contribuciones = explicar(modelo, pd.DataFrame({"superficie_construida": [4.0]}))
    # Escalada: (4 − 2) / 0,816 = 2,449; coeficiente 0,816 → 2. Intercepto = media de log10 = 2.
    assert list(contribuciones.columns) == ["intercepto", "superficie_construida"]
    assert contribuciones.loc[0, "intercepto"] == pytest.approx(2)
    assert contribuciones.loc[0, "superficie_construida"] == pytest.approx(2)
    assert contribuciones.sum(axis=1)[0] == pytest.approx(np.log10(10_000))


@pytest.mark.criterio("CA23")
def test_explicar_modelo_final(modelo, entrenamiento):
    from src.explicar import explicar
    limpios = limpiar_objetivo(entrenamiento)
    X_val, _ = separar(limpios[limpios["particion"] == VALIDACION].head(20))
    contribuciones = explicar(modelo, X_val)
    nombres = list(modelo[:-1].get_feature_names_out())
    assert list(contribuciones.columns) == ["intercepto"] + nombres
    np.testing.assert_allclose(contribuciones.sum(axis=1), np.log10(modelo.predict(X_val)), atol=1e-9)


@pytest.mark.criterio("CA24")
def test_tasaciones_del_test_en_rango(modelo, test_crudo):
    prediccion = np.asarray(modelo.predict(test_crudo), dtype=float)
    assert (prediccion > 0).all()
    assert (prediccion <= 6_000_000).all(), f"Máxima: {prediccion.max():,.0f} €"
