"""Tests unitarios de src/formato.py (DESIGN.md §2)."""
import math
from datetime import date

from src import formato


def test_euros():
    assert formato.euros(121_100) == "121.100 €"
    assert formato.euros(121_149, centenas=True) == "121.100 €"
    assert formato.euros(math.nan) == "—"


def test_porcentaje():
    assert formato.porcentaje(0.184) == "18,4 %"
    assert formato.porcentaje(0.062, signo=True) == "+6,2 %"
    assert formato.porcentaje(-0.062, signo=True) == "−6,2 %"
    assert formato.porcentaje(None) == "—"


def test_intervalo():
    assert formato.intervalo(121_100, [112_000, 131_200]) == "121.100 € (entre 112.000 y 131.200 €)"


def test_factor():
    assert formato.factor(0.049) == "×1,12 (+12 %)"
    assert formato.factor(-0.049) == "×0,89 (−11 %)"
    assert formato.coeficiente_tecnico(0.0491) == "0,049"


def test_superficie_y_fecha():
    assert formato.superficie(85.4) == "85 m²"
    assert formato.fecha(date(2024, 2, 17)) == "17/02/2024"
    assert formato.fecha("17/02/2024") == "17/02/2024"
    assert formato.fecha(None) == "—"


def test_numero_con_miles_y_decimales():
    assert formato.numero(1_234_567.891, 2) == "1.234.567,89"
