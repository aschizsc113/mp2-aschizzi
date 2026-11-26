"""Cuadro de mando de validación del modelo (R20, R21, DESIGN.md).

    uv run streamlit run src/app.py

Solo lee lo que escriben src.experimentos, src.entrenar y src.predecir: no entrena nada.
Las cifras y frases salen de `analisis.py`, las figuras de `graficos.py` y los formatos de `formato.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar src/ desde Streamlit

import joblib
import pandas as pd
import streamlit as st

from src import analisis, formato, graficos
from src import datos as rutas
from src.decision import elegir_modelo_final
from src.estrategias import CADENA
from src.explicar import coeficientes, explicar

PANTALLAS = ["Experimentos", "Rendimiento", "Dónde falla", "Abrir la caja"]
COMANDOS = {
    "experimentos.csv": "uv run python -m src.experimentos",
    "metricas.json": "uv run python -m src.entrenar",
    "predicciones_validacion.csv": "uv run python -m src.entrenar",
    "modelo.joblib": "uv run python -m src.entrenar",
    "envio_resumen.json": "uv run python -m src.predecir",
}
OPCIONES_VIVIENDA = ["la que más se equivoca", "la que mejor acierta", "una al azar", "por número (index)"]


# ── Lectura de ficheros (R20: solo reports/ y models/) ─────────────────────────────────────────

def ruta(nombre: str) -> Path:
    return (rutas.MODELOS if nombre.endswith(".joblib") else rutas.INFORMES) / nombre


@st.cache_data
def _leer(ruta_fichero: str, modificado: float) -> object:
    camino = Path(ruta_fichero)
    if camino.suffix == ".csv":
        return pd.read_csv(camino, index_col=0 if camino.name == "predicciones_validacion.csv" else None)
    return json.loads(camino.read_text(encoding="utf-8"))


@st.cache_resource
def _leer_modelo(ruta_fichero: str, modificado: float) -> object:
    return joblib.load(ruta_fichero)


def cargar(nombre: str) -> object | None:
    """El contenido del fichero, o None si no existe (R21)."""
    camino = ruta(nombre)
    if not camino.exists():
        return None
    lector = _leer_modelo if camino.suffix == ".joblib" else _leer
    return lector(str(camino), camino.stat().st_mtime)


def falta(nombre: str) -> None:
    """Aviso con el fichero que falta y el comando que lo genera (R21, DESIGN §8)."""
    carpeta = "models" if nombre.endswith(".joblib") else "reports"
    st.info(f"Falta el fichero `{carpeta}/{nombre}`. Genéralo con `{COMANDOS[nombre]}`.")


def modelo_final_de(experimentos: pd.DataFrame) -> str:
    return elegir_modelo_final(experimentos, [e for e in CADENA if e in set(experimentos["estrategia"])])


# ── Experimentos ──────────────────────────────────────────────────────────────────────────────

def pantalla_experimentos() -> None:
    st.header("Experimentos")
    experimentos = cargar("experimentos.csv")
    if experimentos is None:
        return falta("experimentos.csv")
    modelo_final = modelo_final_de(experimentos)
    metricas = cargar("metricas.json")
    if metricas is not None and not analisis.resultados_coherentes(metricas, modelo_final):
        st.warning("Los resultados son de ejecuciones distintas. Vuelve a ejecutar `src.experimentos` y `src.entrenar`.")

    st.markdown(f"**{analisis.frase_experimentos(experimentos, modelo_final)}**")
    st.info(f"**Regla de decisión.** {analisis.REGLA_DECISION}")

    todas = st.checkbox("Mostrar las 35 combinaciones de la fase 4", value=False)
    tabla = analisis.tabla_experimentos(experimentos, todas=todas)
    ingenua = tabla.loc[tabla["estrategia"] == "E0", "mae_medio"]
    figura = graficos.barras_experimentos(tabla, float(ingenua.iloc[0]) if len(ingenua) else None,
                                          analisis.titulo_experimentos(experimentos, modelo_final))
    st.plotly_chart(figura, width="stretch")
    st.caption("Validación cruzada con las particiones 1-5: cada error se mide en las viviendas de la partición "
               "que el modelo no vio al entrenar. Las barras son la desviación entre las 5 particiones.")

    st.dataframe(_tabla_legible(tabla), hide_index=True, width="stretch")

    with st.expander("Detalle por partición"):
        con_referencia = tabla[tabla["decision"] != "referencia"]["estrategia"].tolist()
        if con_referencia:
            defecto = analisis.ultima_aceptada(experimentos)
            elegida = st.selectbox("Estrategia", con_referencia,
                                   index=con_referencia.index(defecto) if defecto in con_referencia else 0)
            detalle = analisis.detalle_particion(experimentos, elegida)
            st.plotly_chart(graficos.particiones(detalle, f"{elegida} frente a su referencia, partición a partición"),
                            width="stretch")

    with st.expander("Detalle técnico · elección de λ"):
        if not analisis.tiene_fase_4(experimentos):
            st.caption("Aún no se han ejecutado M1-M3.")
        else:
            st.plotly_chart(graficos.error_frente_alfa(
                analisis.curvas_alfa(experimentos), "Error medio según la penalización α de cada familia"),
                width="stretch")
            st.caption("Error medio de validación cruzada (particiones 1-5). La estrella marca la ganadora de cada "
                       "familia; nunca se usa la partición de validación.")


def _tabla_legible(tabla: pd.DataFrame) -> pd.DataFrame:
    def particiones(valor: object) -> str:
        return formato.SIN_DATO if pd.isna(valor) else f"{int(valor)} de 5"

    decision = {"aceptada": "✓ aceptada", "descartada": "✗ descartada", "referencia": formato.SIN_DATO}
    return pd.DataFrame({
        "Estrategia": tabla["etiqueta"],
        "Qué cambia": tabla["descripcion"],
        "Referencia": tabla["referencia"].fillna(formato.SIN_DATO).replace("", formato.SIN_DATO),
        "Error medio": tabla["mae_medio"].map(formato.euros),
        "Mejora media": tabla["mejora_media"].map(formato.euros),
        "Particiones en que mejora": tabla["particiones_mejor"].map(particiones),
        "Decisión": tabla["decision"].map(decision),
    })


# ── Rendimiento ───────────────────────────────────────────────────────────────────────────────

def pantalla_rendimiento() -> None:
    st.header("Rendimiento")
    metricas = cargar("metricas.json")
    if metricas is None:
        return falta("metricas.json")
    st.markdown(f"**{analisis.frase_rendimiento(metricas)}**")

    resumen = cargar("envio_resumen.json")
    if resumen is None:
        falta("envio_resumen.json")
    elif (aviso := analisis.aviso_fuera_de_rango(resumen)) is not None:
        st.warning(aviso)

    modelo, ingenua = metricas["modelo"]["mae"]["valor"], metricas["baseline"]["mae"]["valor"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Error medio del modelo", formato.euros(modelo, centenas=True))
    col2.metric("Error medio de la tasación ingenua", formato.euros(ingenua, centenas=True),
                help="La tasación ingenua (*baseline*): €/m² mediano del barrio × superficie.")
    diferencia = ingenua - modelo
    color, texto = ("green", "menos") if diferencia > 0 else ("orange", "más")
    col3.markdown(f"Diferencia\n\n### :{color}[{formato.euros(abs(diferencia), centenas=True)} {texto}]")

    st.plotly_chart(graficos.puntos_metricas(analisis.tabla_metricas(metricas), analisis.titulo_metricas(metricas)),
                    width="stretch")
    n_validacion = formato.numero(metricas.get("n_validacion"))
    st.caption(f"{n_validacion} viviendas de validación que el modelo no vio al entrenar. "
               "Barras: margen (IC 95 %).")

    predicciones = cargar("predicciones_validacion.csv")
    if predicciones is None:
        return falta("predicciones_validacion.csv")
    errores = analisis.con_errores(predicciones)
    st.markdown(analisis.frase_error_pct(errores))
    st.plotly_chart(graficos.histograma_error_pct(errores, "Cuánto se desvía cada tasación del precio real"),
                    width="stretch")
    st.caption(f"{formato.numero(len(errores))} viviendas de validación. Bandas sombreadas: ±10 % y ±20 %.")
    with st.expander("Precio real frente a tasación"):
        st.plotly_chart(graficos.real_frente_tasacion(errores, "Cuanto más cerca de la diagonal, mejor"),
                        width="stretch")


# ── Dónde falla ───────────────────────────────────────────────────────────────────────────────

def pantalla_donde_falla() -> None:
    st.header("Dónde falla")
    predicciones = cargar("predicciones_validacion.csv")
    if predicciones is None:
        return falta("predicciones_validacion.csv")
    todas = analisis.con_errores(predicciones)

    col1, col2, col3 = st.columns(3)
    tipos = col1.multiselect("Tipo de vivienda", sorted(todas["tipo_vivienda"].unique()))
    zonas = col2.multiselect("Zona (distrito o municipio)", sorted(todas["zona"].dropna().unique()))
    minimo, maximo = float(todas["prediccion"].min()), float(todas["prediccion"].max())
    rango = col3.slider("Rango de tasación (€)", min_value=minimo, max_value=maximo, value=(minimo, maximo),
                        step=1_000.0, format="%.0f")
    datos = analisis.filtrar(todas, tipos, zonas, rango)
    st.caption(f"{formato.numero(len(datos))} viviendas de {formato.numero(len(todas))}")
    if datos.empty:
        st.warning("Ninguna vivienda cumple los filtros. Amplía el tipo, la zona o el rango.")
        return
    st.markdown(f"**{analisis.frase_donde_falla(datos)}**")
    origen = f"{formato.numero(len(datos))} viviendas de validación que el modelo no vio al entrenar."

    por_tipo = analisis.error_por_grupo(datos, "tipo_vivienda")
    col1, col2 = st.columns(2)
    col1.plotly_chart(graficos.error_por_grupo(por_tipo, "tipo_vivienda",
                                               analisis.titulo_error(por_tipo, "tipo_vivienda", "tipo")),
                      width="stretch")
    col2.plotly_chart(graficos.sesgo_por_grupo(por_tipo, "tipo_vivienda",
                                               analisis.titulo_sesgo(por_tipo, "tipo_vivienda")), width="stretch")
    st.caption(origen + " Gris claro: menos de 20 viviendas, error poco fiable.")

    por_zona = analisis.error_por_grupo(datos, "zona")
    st.plotly_chart(graficos.error_por_grupo(por_zona, "zona", analisis.titulo_error(por_zona, "zona", "zona"),
                                             horizontal=True), width="stretch")
    st.caption(origen + " Distrito en Barcelona, municipio fuera.")

    por_rango = analisis.error_por_grupo(datos, "rango", orden=analisis.ETIQUETAS_RANGO)
    st.plotly_chart(graficos.sesgo_por_grupo(por_rango, "rango", analisis.titulo_sesgo(por_rango, "rango")),
                    width="stretch")
    st.caption(origen + " Rangos según la tasación del modelo.")

    quien = st.radio("Mapa", ["Modelo", "Tasación ingenua"], horizontal=True)
    columna_error = "error_pct" if quien == "Modelo" else "baseline_error_pct"
    columna_tasacion = "prediccion" if quien == "Modelo" else "prediccion_baseline"
    st.plotly_chart(graficos.mapa_residuos(datos, columna_tasacion, columna_error,
                                           analisis.titulo_mapa(datos, columna_error)), width="stretch")
    sin_coordenadas = int(datos[["latitud", "longitud"]].isna().any(axis=1).sum())
    st.caption(origen + " Naranja = tasa de más; azul = tasa de menos; escala cortada en ±50 %."
               + (f" {sin_coordenadas} viviendas sin coordenadas no aparecen." if sin_coordenadas else ""))

    st.subheader("Las 20 viviendas con más error")
    peores = analisis.peores(datos)
    st.dataframe(pd.DataFrame({
        "index": peores.index,
        "Lugar": peores["barri"].fillna(peores["municipi"]),
        "Tipo": peores["tipo_vivienda"],
        "Superficie": peores["superficie_construida"].map(formato.superficie),
        "Precio real": peores["precio"].map(formato.euros),
        "Tasación": peores["prediccion"].map(formato.euros),
        "Error": peores["error_pct"].map(lambda x: formato.porcentaje(x, signo=True)),
    }), hide_index=True, width="stretch")


# ── Abrir la caja ─────────────────────────────────────────────────────────────────────────────

def pantalla_abrir_la_caja() -> None:
    st.header("Abrir la caja")
    st.info("Los coeficientes dicen cuánto más **se tasan** las viviendas con una característica, no cuánto "
            "subiría el precio si se la añadiéramos. Detalle en «Qué no podemos afirmar».")
    modelo = cargar("modelo.joblib")
    if modelo is None:
        return falta("modelo.joblib")
    try:
        tabla = analisis.tabla_coeficientes(coeficientes(modelo))
    except Exception:
        st.warning("El modelo guardado no se puede explicar. Vuelve a entrenar: `uv run python -m src.entrenar`")
        return
    st.plotly_chart(graficos.coeficientes(tabla, analisis.titulo_coeficientes(tabla)), width="stretch")
    st.caption(f"Las {len(tabla)} variables del modelo final. Factor sobre la tasación cuando la variable sube una "
               "desviación típica; la lectura en su unidad está al pasar el ratón.")

    _desglose(modelo)

    with st.expander("Qué no podemos afirmar"):
        st.markdown(analisis.NO_CAUSAL)


def _desglose(modelo: object) -> None:
    st.subheader("Desglose de una vivienda")
    predicciones = cargar("predicciones_validacion.csv")
    if predicciones is None:
        return falta("predicciones_validacion.csv")
    datos = analisis.con_errores(predicciones)
    criterio = st.radio("Vivienda", OPCIONES_VIVIENDA, horizontal=True)
    numero = None
    if criterio == OPCIONES_VIVIENDA[-1]:
        numero = int(st.number_input("Número (index)", value=int(datos.index[0]), step=1))
    elegida = analisis.elegir_vivienda(datos, criterio, numero)
    if elegida is None:
        st.warning("No hay ninguna vivienda de validación con ese número.")
        return
    fila = datos.loc[[elegida]]
    try:
        contribuciones = explicar(modelo, analisis.variables_de_entrada(fila)).iloc[0]
    except Exception:
        st.warning("El modelo guardado no se puede explicar. Vuelve a entrenar: `uv run python -m src.entrenar`")
        return
    vivienda = fila.iloc[0]
    st.markdown(f"**{elegida}** · {vivienda['barri'] if pd.notna(vivienda['barri']) else vivienda['municipi']} · "
                f"{vivienda['tipo_vivienda']} · {formato.superficie(vivienda['superficie_construida'])} · "
                f"precio real {formato.euros(vivienda['precio'])} · tasación {formato.euros(vivienda['prediccion'])} "
                f"({formato.porcentaje(vivienda['error_pct'], signo=True)})")
    pasos = analisis.desglose(contribuciones)
    titulo = (f"De la vivienda media ({formato.euros(pasos.iloc[0]['acumulado'], True)}) a la tasación "
              f"({formato.euros(pasos.iloc[-1]['acumulado'], True)})")
    st.plotly_chart(graficos.cascada(pasos, float(vivienda["precio"]), titulo), width="stretch")
    st.caption("Los factores no dependen del orden; los euros de cada paso, sí, porque se aplican sobre lo acumulado. "
               "Vivienda de validación que el modelo no vio al entrenar.")


# ── Navegación ────────────────────────────────────────────────────────────────────────────────

FUNCIONES = {
    "Experimentos": pantalla_experimentos,
    "Rendimiento": pantalla_rendimiento,
    "Dónde falla": pantalla_donde_falla,
    "Abrir la caja": pantalla_abrir_la_caja,
}


def main() -> None:
    st.set_page_config(page_title="Tasación · Banc Avellaner", layout="wide")
    pantalla = st.sidebar.radio("Pantalla", PANTALLAS, key="pantalla")
    try:
        FUNCIONES[pantalla]()
    except Exception:  # DESIGN §8: nunca se muestra una traza de error
        st.error("No se ha podido mostrar esta pantalla con los ficheros actuales. Vuelve a ejecutar "
                 "`uv run python -m src.experimentos`, `uv run python -m src.entrenar` y "
                 "`uv run python -m src.predecir`.")


main()
