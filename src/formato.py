"""Números en formato español para el cuadro de mando (DESIGN.md §2).

Punto para los miles, coma para los decimales y espacio antes de € y de %.
Un valor vacío (None o NaN) se muestra como `SIN_DATO`.
"""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

SIN_DATO = "—"


def _vacio(valor: object) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def numero(valor: float | None, decimales: int = 0) -> str:
    """`1234567.8` → `1.234.568`; con decimales, `0.5` → `0,5` (DESIGN §2)."""
    if _vacio(valor):
        return SIN_DATO
    texto = f"{float(valor):,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def euros(valor: float | None, centenas: bool = False) -> str:
    """`121100` → `121.100 €`. Con `centenas`, redondeado a centenas (cifras de cabecera) (DESIGN §2)."""
    if _vacio(valor):
        return SIN_DATO
    if centenas:
        valor = round(float(valor) / 100) * 100
    return f"{numero(valor)} €"


def porcentaje(fraccion: float | None, signo: bool = False, decimales: int = 1) -> str:
    """`0.184` → `18,4 %`; con `signo`, `0.062` → `+6,2 %` (DESIGN §2)."""
    if _vacio(fraccion):
        return SIN_DATO
    valor = 100 * float(fraccion)
    texto = numero(abs(valor) if signo else valor, decimales)
    if signo:
        texto = ("+" if valor >= 0 else "−") + texto
    return f"{texto} %"


def intervalo(valor: float | None, ic: tuple[float, float] | list[float], centenas: bool = False) -> str:
    """`121.100 € (entre 112.000 y 131.200 €)` (DESIGN §2)."""
    if _vacio(valor) or ic is None or any(_vacio(x) for x in ic):
        return SIN_DATO
    inferior, superior = (round(float(x) / 100) * 100 if centenas else x for x in ic)
    return f"{euros(valor, centenas)} (entre {numero(inferior)} y {euros(superior)})"


def factor(coeficiente_log10: float | None) -> str:
    """Coeficiente en log10 → factor multiplicativo y su lectura: `0.049` → `×1,12 (+12 %)` (DESIGN §2)."""
    if _vacio(coeficiente_log10):
        return SIN_DATO
    multiplicador = 10 ** float(coeficiente_log10)
    cambio = round(100 * (multiplicador - 1))
    signo = "+" if cambio >= 0 else "−"
    return f"×{numero(multiplicador, 2)} ({signo}{numero(abs(cambio))} %)"


def coeficiente_tecnico(coeficiente_log10: float | None) -> str:
    """El coeficiente en log10 con 3 decimales, solo para el detalle técnico (DESIGN §2)."""
    return SIN_DATO if _vacio(coeficiente_log10) else numero(coeficiente_log10, 3)


def superficie(metros: float | None) -> str:
    """`85.4` → `85 m²` (DESIGN §2)."""
    return SIN_DATO if _vacio(metros) else f"{numero(metros)} m²"


def fecha(valor: object) -> str:
    """Fecha en día/mes/año: `17/02/2024` (DESIGN §2). Acepta fechas o texto día/mes/año."""
    if _vacio(valor) or valor is pd.NaT:
        return SIN_DATO
    if isinstance(valor, (date, datetime, pd.Timestamp)):
        return valor.strftime("%d/%m/%Y")
    convertida = pd.to_datetime(str(valor), dayfirst=True, errors="coerce")
    return SIN_DATO if pd.isna(convertida) else convertida.strftime("%d/%m/%Y")
