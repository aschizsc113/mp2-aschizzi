"""Tests del cuadro de mando (criterios CA25 y CA26 de SPECS.md), con streamlit.testing.v1.AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import datos

APP = str(Path(__file__).resolve().parents[2] / "src" / "app.py")
PANTALLAS = ["Experimentos", "Rendimiento", "Dónde falla", "Abrir la caja"]
FICHEROS = ["experimentos.csv", "metricas.json", "predicciones_validacion.csv", "modelo.joblib"]


def _abrir(pantalla: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    return app.sidebar.radio(key="pantalla").set_value(pantalla).run()


def _textos(app: AppTest) -> str:
    return " ".join(str(e.value) for e in [*app.info, *app.warning, *app.markdown, *app.caption])


@pytest.mark.criterio("CA25")
def test_la_app_arranca_con_las_4_pantallas(modelo, metricas, experimentos):
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert list(app.sidebar.radio(key="pantalla").options) == PANTALLAS
    for pantalla in PANTALLAS:
        app = _abrir(pantalla)
        assert not app.exception, pantalla
        assert not app.error, f"{pantalla}: {[e.value for e in app.error]}"
        assert app.header[0].value == pantalla


@pytest.mark.criterio("CA26")
def test_sin_ficheros_avisa_del_que_falta(tmp_path, monkeypatch):
    monkeypatch.setattr(datos, "INFORMES", tmp_path / "reports")
    monkeypatch.setattr(datos, "MODELOS", tmp_path / "models")
    for pantalla in PANTALLAS:
        app = _abrir(pantalla)
        assert not app.exception, pantalla
        assert not app.error, pantalla
        assert any(f in " ".join(i.value for i in app.info) for f in FICHEROS), \
            f"{pantalla}: no nombra el fichero que falta ({_textos(app)})"
