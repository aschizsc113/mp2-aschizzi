"""Figuras Plotly del cuadro de mando con el estilo de DESIGN.md §3, §5 y §6.

Cada función recibe los datos y el título ya calculados (de `analisis.py`) y devuelve una figura.
No calcula cifras ni escribe conclusiones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import formato

AZUL = "#2a78d6"      # el modelo; en el mapa y el sesgo, «tasa de menos»
GRIS = "#898781"      # tasación ingenua, referencias y descartadas
NARANJA = "#eb6834"   # alerta: empeora, tasa de más, variables que bajan el precio
GRIS_CLARO = "#d3d1cc"  # grupos con pocas viviendas; «acierta» en el mapa
PLANTILLA = "simple_white"
COLORES_DECISION = {"aceptada": AZUL, "descartada": GRIS, "referencia": GRIS}
COLORES_QUIEN = {"Modelo": AZUL, "Tasación ingenua": GRIS}
CORTE_MAPA = 0.5


def estilo(figura: go.Figure, titulo: str) -> go.Figure:
    """Plantilla, separadores españoles, título y leyenda arriba en horizontal (DESIGN §2, §3)."""
    figura.update_layout(template=PLANTILLA, separators=",.", title=titulo,
                         legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
                         margin={"t": 90})
    return figura


def barras_experimentos(tabla: pd.DataFrame, error_ingenua: float | None, titulo: str) -> go.Figure:
    """Error medio ± desviación por estrategia, en el orden de la cadena; E0 como línea gris (DESIGN §4)."""
    datos = tabla[tabla["estrategia"] != "E0"]
    figura = px.bar(datos, x="mae_medio", y="etiqueta", error_x="mae_sd", orientation="h", color="decision",
                    color_discrete_map=COLORES_DECISION, category_orders={"etiqueta": datos["etiqueta"].tolist()},
                    labels={"mae_medio": "Error medio (€)", "etiqueta": "", "decision": ""},
                    hover_data={"descripcion": True, "mae_sd": ":,.0f", "mae_medio": ":,.0f"})
    if error_ingenua is not None:
        figura.add_vline(x=error_ingenua, line_dash="dash", line_color=GRIS,
                         annotation_text="Tasación ingenua (E0)", annotation_position="top")
    figura.update_xaxes(ticksuffix=" €")
    figura.update_yaxes(autorange="reversed")
    return estilo(figura, titulo)


def particiones(detalle: pd.DataFrame, titulo: str) -> go.Figure:
    """Error de cada partición de una estrategia junto al de su referencia (DESIGN §4)."""
    largo = detalle.melt(id_vars="particion", var_name="estrategia", value_name="error")
    colores = dict(zip(detalle.columns[1:], [AZUL, GRIS]))
    figura = px.bar(largo, x="particion", y="error", color="estrategia", barmode="group",
                    color_discrete_map=colores, labels={"particion": "Partición", "error": "Error medio (€)",
                                                        "estrategia": ""})
    figura.update_yaxes(ticksuffix=" €")
    return estilo(figura, titulo)


def error_frente_alfa(curvas: pd.DataFrame, titulo: str) -> go.Figure:
    """Error medio frente a α en eje logarítmico, una línea por familia y l1_ratio; ganadoras marcadas (DESIGN §4)."""
    figura = px.line(curvas, x="alfa", y="mae_medio", color="serie", markers=True, log_x=True,
                     labels={"alfa": "α (penalización)", "mae_medio": "Error medio (€)", "serie": ""})
    ganadoras = curvas[curvas["ganadora"]]
    figura.add_trace(go.Scatter(x=ganadoras["alfa"], y=ganadoras["mae_medio"], mode="markers", name="Ganadora",
                                marker={"symbol": "star", "size": 14, "color": AZUL}))
    figura.update_yaxes(ticksuffix=" €")
    return estilo(figura, titulo)


def puntos_metricas(tabla: pd.DataFrame, titulo: str) -> go.Figure:
    """Error medio y RMSE con su margen, modelo frente a tasación ingenua (DESIGN §4, Rendimiento 3)."""
    figura = go.Figure()
    for quien, datos in tabla.groupby("quien", sort=False):
        figura.add_trace(go.Scatter(
            x=datos["metrica"], y=datos["valor"], name=quien, mode="markers",
            marker={"size": 12, "color": COLORES_QUIEN[quien]},
            error_y={"type": "data", "symmetric": False, "array": datos["superior"] - datos["valor"],
                     "arrayminus": datos["valor"] - datos["inferior"]},
            customdata=datos["nota"], hovertemplate="%{x}: %{y:,.0f} €<br>%{customdata}<extra>" + quien + "</extra>"))
    figura.update_yaxes(ticksuffix=" €", rangemode="tozero", title="€")
    figura.update_xaxes(title="")
    return estilo(figura, titulo)


def histograma_error_pct(datos: pd.DataFrame, titulo: str) -> go.Figure:
    """Error en % con signo, modelo y tasación ingenua superpuestos, con bandas de ±10 % y ±20 % (DESIGN §4)."""
    figura = go.Figure()
    for columna, nombre, color in (("baseline_error_pct", "Tasación ingenua", GRIS), ("error_pct", "Modelo", AZUL)):
        valores = 100 * datos[columna].clip(-1, 1)
        figura.add_trace(go.Histogram(x=valores, name=nombre, marker_color=color, opacity=0.6,
                                      xbins={"start": -100, "end": 100, "size": 5}))
    for ancho, opacidad in ((20, 0.08), (10, 0.12)):
        figura.add_vrect(x0=-ancho, x1=ancho, fillcolor=GRIS, opacity=opacidad, line_width=0)
    figura.update_layout(barmode="overlay")
    figura.update_xaxes(title="Error de la tasación (%; + = tasa de más)", ticksuffix=" %")
    figura.update_yaxes(title="Viviendas")
    return estilo(figura, titulo)


def real_frente_tasacion(datos: pd.DataFrame, titulo: str) -> go.Figure:
    """Precio real frente a tasación del modelo, ejes logarítmicos y diagonal (DESIGN §4, Rendimiento 6)."""
    figura = px.scatter(datos, x="precio", y="prediccion", log_x=True, log_y=True, opacity=0.6,
                        color_discrete_sequence=[AZUL],
                        labels={"precio": "Precio real (€)", "prediccion": "Tasación del modelo (€)"})
    extremos = [min(datos["precio"].min(), datos["prediccion"].min()),
                max(datos["precio"].max(), datos["prediccion"].max())]
    figura.add_trace(go.Scatter(x=extremos, y=extremos, mode="lines", name="Tasación = precio",
                                line={"dash": "dash", "color": GRIS}))
    return estilo(figura, titulo)


def _colores_grupo(tabla: pd.DataFrame, color: str) -> list[str]:
    return [GRIS_CLARO if pocas else color for pocas in tabla["pocas"]]


def _etiquetas_grupo(tabla: pd.DataFrame, columna: str) -> list[str]:
    return [f"{g} (pocas viviendas)" if p else str(g) for g, p in zip(tabla[columna], tabla["pocas"])]


def error_por_grupo(tabla: pd.DataFrame, columna: str, titulo: str, horizontal: bool = False) -> go.Figure:
    """Barras de error medio en € por grupo, con el número de viviendas; pocas en gris claro (DESIGN §4, §7)."""
    etiquetas = _etiquetas_grupo(tabla, columna)
    texto = [f"{n} viv." for n in tabla["viviendas"]]
    if horizontal:
        barra = go.Bar(y=etiquetas, x=tabla["error_medio"], orientation="h", text=texto,
                       marker_color=_colores_grupo(tabla, AZUL), hovertemplate="%{y}: %{x:,.0f} €<extra></extra>")
    else:
        barra = go.Bar(x=etiquetas, y=tabla["error_medio"], text=texto,
                       marker_color=_colores_grupo(tabla, AZUL), hovertemplate="%{x}: %{y:,.0f} €<extra></extra>")
    figura = go.Figure(barra)
    if horizontal:
        figura.update_xaxes(title="Error medio (€)", ticksuffix=" €")
        figura.update_yaxes(autorange="reversed")
        figura.update_layout(height=max(350, 28 * len(tabla)))
    else:
        figura.update_yaxes(title="Error medio (€)", ticksuffix=" €")
    return estilo(figura, titulo)


def sesgo_por_grupo(tabla: pd.DataFrame, columna: str, titulo: str) -> go.Figure:
    """Error medio en % con signo por grupo: naranja = tasa de más, azul = de menos (DESIGN §3, §4)."""
    colores = [GRIS_CLARO if p else (NARANJA if s > 0 else AZUL) for s, p in zip(tabla["sesgo"], tabla["pocas"])]
    figura = go.Figure(go.Bar(x=_etiquetas_grupo(tabla, columna), y=100 * tabla["sesgo"], marker_color=colores,
                              hovertemplate="%{x}: %{y:+.1f} %<extra></extra>"))
    figura.add_hline(y=0, line_color=GRIS)
    figura.update_yaxes(title="Sesgo (%; + = tasa de más)", ticksuffix=" %")
    return estilo(figura, titulo)


def mapa_residuos(datos: pd.DataFrame, columna_tasacion: str, columna_error: str, titulo: str) -> go.Figure:
    """Cada vivienda en su sitio, coloreada por error en % con signo cortado en ±50 % (DESIGN §5)."""
    con_coordenadas = datos.dropna(subset=["latitud", "longitud"]).copy()
    con_coordenadas["error_color"] = (100 * con_coordenadas[columna_error]).clip(-100 * CORTE_MAPA, 100 * CORTE_MAPA)
    con_coordenadas["lugar"] = con_coordenadas["barri"].fillna(con_coordenadas["municipi"])
    figura = px.scatter_map(
        con_coordenadas, lat="latitud", lon="longitud", color="error_color",
        color_continuous_scale=[(0, AZUL), (0.5, GRIS_CLARO), (1, NARANJA)],
        range_color=[-100 * CORTE_MAPA, 100 * CORTE_MAPA], opacity=0.7, zoom=10,
        center={"lat": 41.39, "lon": 2.16}, map_style="carto-positron",
        custom_data=["lugar", "tipo_vivienda", "superficie_construida", "precio", "prediccion", "prediccion_baseline",
                     columna_error.replace("_pct", ""), columna_error],
    )
    figura.update_traces(marker={"size": 7}, hovertemplate=(
        "<b>%{customdata[0]}</b> · %{customdata[1]} · %{customdata[2]:.0f} m²<br>"
        "Precio real: %{customdata[3]:,.0f} €<br>Tasación del modelo: %{customdata[4]:,.0f} €<br>"
        "Tasación ingenua: %{customdata[5]:,.0f} €<br>"
        "Error: %{customdata[6]:+,.0f} € (%{customdata[7]:+.1%})<extra></extra>"))
    figura.update_layout(coloraxis_colorbar={"title": "Error", "ticksuffix": " %"}, height=600)
    return estilo(figura, titulo)


def coeficientes(tabla: pd.DataFrame, titulo: str) -> go.Figure:
    """Factor 10^coeficiente de cada variable: azul si sube el precio, naranja si lo baja (DESIGN §6)."""
    colores = [AZUL if c >= 0 else NARANJA for c in tabla["coeficiente"]]
    figura = go.Figure(go.Bar(
        y=tabla["nombre"], x=tabla["factor"] - 1, base=1, orientation="h", marker_color=colores,
        text=tabla["texto"], customdata=tabla[["lectura", "tecnico"]],
        hovertemplate="%{customdata[0]}<br>Coeficiente (log10): %{customdata[1]}<extra></extra>"))
    figura.add_vline(x=1, line_color=GRIS)
    figura.update_xaxes(title="Factor sobre la tasación (por desviación típica)", tickprefix="×")
    figura.update_yaxes(autorange="reversed")
    return estilo(figura, titulo)


def cascada(pasos: pd.DataFrame, precio_real: float | None, titulo: str) -> go.Figure:
    """Desglose en euros: vivienda media, un paso por variable y tasación; línea discontinua = precio real (DESIGN §6)."""
    medidas = ["absolute"] + ["relative"] * (len(pasos) - 2) + ["total"]
    texto = [formato.euros(p) if np.isnan(f) else
             f"×{formato.numero(f, 2)} ({'+' if e >= 0 else '−'}{formato.euros(abs(e))})"
             for p, f, e in zip(pasos["acumulado"], pasos["factor"], pasos["euros"])]
    figura = go.Figure(go.Waterfall(
        x=pasos["paso"], y=pasos["euros"], measure=medidas, text=texto, textposition="outside",
        increasing={"marker": {"color": AZUL}}, decreasing={"marker": {"color": NARANJA}},
        totals={"marker": {"color": AZUL}}, connector={"line": {"color": GRIS_CLARO}}))
    if precio_real is not None and not np.isnan(precio_real):
        figura.add_hline(y=precio_real, line_dash="dash", line_color=GRIS,
                         annotation_text=f"Precio real: {formato.euros(precio_real)}")
    figura.update_yaxes(title="€", ticksuffix=" €")
    return estilo(figura, titulo)
