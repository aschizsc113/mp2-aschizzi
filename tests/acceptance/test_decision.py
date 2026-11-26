"""Tests de la regla de decisión (criterios CA5 y CA6 de SPECS.md)."""
import pandas as pd
import pytest

from src.decision import comparar, completar_decisiones

REFERENCIA = [100_000.0] * 5


@pytest.mark.criterio("CA5")
@pytest.mark.parametrize(
    ("nombre", "errores", "mejora", "particiones", "aceptada"),
    [
        ("A", [98_000, 98_000, 98_000, 98_000, 101_000], 1_400, 4, True),
        ("B", [99_500] * 5, 500, 5, False),
        ("C", [97_000, 97_000, 97_000, 101_000, 101_000], 1_400, 3, False),
        ("D", [99_000, 99_000, 99_000, 99_000, 100_000], 800, 4, False),
        ("E", [98_750, 98_750, 98_750, 98_750, 100_000], 1_000, 4, True),
    ],
)
def test_regla_de_decision(nombre, errores, mejora, particiones, aceptada):
    resultado = comparar([float(e) for e in errores], REFERENCIA)
    assert resultado["mejora_media"] == pytest.approx(mejora), nombre
    assert resultado["particiones_mejor"] == particiones, nombre
    assert resultado["aceptada"] is aceptada, nombre


def _fila(estrategia: str, error: float) -> dict:
    return {"estrategia": estrategia, "tipo": "estrategia"} | {f"mae_p{p}": error for p in range(1, 6)}


@pytest.mark.criterio("CA6")
def test_referencia_ultima_aceptada():
    # E2 → T1 (baja 5.000 €: aceptada) → T2 (sube 1.000 €: descartada) → T3
    tabla = pd.DataFrame([_fila("E2", 100_000), _fila("T1", 95_000), _fila("T2", 96_000), _fila("T3", 94_000)])
    tabla["mae_medio"] = tabla[[f"mae_p{p}" for p in range(1, 6)]].mean(axis=1)
    referencias = {"E2": "", "T1": "E2", "T2": "ultima_aceptada", "T3": "ultima_aceptada"}
    resultado = completar_decisiones(tabla, referencias, cadena=["E2", "T1", "T2", "T3"]).set_index("estrategia")
    assert bool(resultado.loc["T1", "aceptada"]) is True
    assert bool(resultado.loc["T2", "aceptada"]) is False
    assert resultado.loc["T2", "referencia"] == "T1"
    assert resultado.loc["T3", "referencia"] == "T1"
