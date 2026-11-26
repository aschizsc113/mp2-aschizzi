"""Cifras y frases de cada pantalla del cuadro de mando (DESIGN.md §1, §3, §4, §6, §8).

Funciones puras: reciben DataFrames o diccionarios y devuelven tablas o texto.
No importan Streamlit ni Plotly. Las frases se calculan con los datos y tienen los dos sentidos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import formato
from src.decision import COLUMNAS_MAE, es_verdadero, familia_y_alfa, mejor_por_familia
from src.preparacion import normalizar_tipo

MINIMO_FIABLE = 20
CADENA_PANTALLA = ["E0", "E1", "E2", "T1", "T2", "T3", "N1"]
NOMBRES_FAMILIA = {"ridge": "Ridge", "lasso": "Lasso", "enet": "ElasticNet"}
RANGOS_TASACION = [0, 150_000, 250_000, 400_000, 700_000, np.inf]
ETIQUETAS_RANGO = ["< 150.000 €", "150.000-250.000 €", "250.000-400.000 €", "400.000-700.000 €", "> 700.000 €"]
COLUMNAS_RESULTADO = ["precio", "prediccion", "prediccion_baseline"]
NOMBRES_TIPO = {"piso": "Piso", "atico": "Ático", "duplex": "Dúplex", "estudio": "Estudio", "casa": "Casa"}
NOMBRES_VARIABLE = {
    "superficie_construida": "Superficie", "ascensor": "Ascensor", "parking": "Parking",
    "es_atico": "Ático", "es_bajo": "Bajo", "es_casa": "Casa (planta)",
}
REGLA_DECISION = ("Una estrategia se acepta si baja el error medio de su referencia en al menos 4 de las 5 "
                  "particiones y en al menos 1.000 € de media; si no, se descarta.")
NO_CAUSAL = (
    "Un coeficiente dice cuánto más **se tasan**, en estos datos, las viviendas que tienen esa característica, "
    "con el resto de variables del modelo iguales. **No dice cuánto subiría el precio si se la añadiéramos.** "
    "Instalar un ascensor o reformar un piso no lo revaloriza en ese factor: el modelo no sabe en qué barrio está "
    "la vivienda, así que el ascensor o el parking recogen también parte de la zona en la que suelen estar. "
    "Además, las ventas son simuladas (2023-2025)."
)


# ── Experimentos ──────────────────────────────────────────────────────────────────────────────

def tiene_fase_4(experimentos: pd.DataFrame) -> bool:
    """Si `experimentos.csv` tiene las filas de M1-M3 (DESIGN §8)."""
    return bool(experimentos["estrategia"].str.startswith("m_").any())


def tabla_experimentos(experimentos: pd.DataFrame, todas: bool = False) -> pd.DataFrame:
    """Filas de la cadena y la mejor de cada familia de la fase 4 (o las 35 con `todas`) (DESIGN §4, §7).

    Añade `decision` (aceptada / descartada / referencia) y `etiqueta` para los gráficos.
    """
    cadena = experimentos[experimentos["estrategia"].isin(CADENA_PANTALLA)]
    fase_4 = experimentos[experimentos["estrategia"].str.startswith("m_")]
    if not todas:
        ganadoras = set(mejor_por_familia(experimentos).values())
        fase_4 = fase_4[fase_4["estrategia"].isin(ganadoras)]
    tabla = pd.concat([cadena, fase_4]).copy()
    tabla["decision"] = [_decision(f) for f in tabla.itertuples()]
    tabla["etiqueta"] = [_etiqueta(e) for e in tabla["estrategia"]]
    return tabla.reset_index(drop=True)


def _decision(fila: object) -> str:
    if not isinstance(fila.referencia, str) or not fila.referencia:
        return "referencia"
    return "aceptada" if es_verdadero(fila.aceptada) else "descartada"


def _etiqueta(estrategia: str) -> str:
    clave = familia_y_alfa(estrategia)
    if clave is None:
        return estrategia
    partes = estrategia.split("_")
    texto = f"{NOMBRES_FAMILIA[clave[0]]} α={formato.numero(clave[1], 4).rstrip('0').rstrip(',')}"
    return texto + (f" l1={partes[3]}" if len(partes) > 3 else "")


def frase_experimentos(experimentos: pd.DataFrame, modelo_final: str) -> str:
    """«Probamos N estrategias y aceptamos X. El error medio pasó de A € (E2) a B € (modelo final)» (DESIGN §4)."""
    probadas = experimentos[experimentos["referencia"].fillna("").astype(str) != ""]
    familias = len(mejor_por_familia(experimentos))
    cadena = probadas[~probadas["estrategia"].str.startswith("m_")]
    n_probadas = len(cadena) + familias
    n_aceptadas = int(cadena["aceptada"].map(es_verdadero).sum()) + int(modelo_final.startswith("m_"))
    por_id = experimentos.set_index("estrategia")
    texto = f"Probamos {n_probadas} estrategias y aceptamos {n_aceptadas}."
    if "E2" in por_id.index and modelo_final in por_id.index:
        inicial, final = por_id.loc["E2", "mae_medio"], por_id.loc[modelo_final, "mae_medio"]
        verbo = "bajó" if final < inicial else "subió"
        texto += (f" El error medio {verbo} de {formato.euros(inicial, centenas=True)} (E2) a "
                  f"{formato.euros(final, centenas=True)} (modelo final).")
    return texto


def titulo_experimentos(experimentos: pd.DataFrame, modelo_final: str) -> str:
    """Conclusión del gráfico de estrategias: el modelo final frente a la tasación ingenua (DESIGN §3, §4)."""
    por_id = experimentos.set_index("estrategia")
    if "E0" not in por_id.index or modelo_final not in por_id.index:
        return "Error medio de cada estrategia en la validación cruzada"
    diferencia = por_id.loc["E0", "mae_medio"] - por_id.loc[modelo_final, "mae_medio"]
    if diferencia > 0:
        return f"El modelo final se equivoca {formato.euros(diferencia, True)} menos que la tasación ingenua"
    return f"Ninguna estrategia baja de la tasación ingenua: el modelo final se equivoca " \
           f"{formato.euros(-diferencia, True)} más"


def detalle_particion(experimentos: pd.DataFrame, estrategia: str) -> pd.DataFrame:
    """Error de cada partición de una estrategia y de su referencia (DESIGN §4, Experimentos 5)."""
    por_id = experimentos.set_index("estrategia")
    fila = por_id.loc[estrategia]
    datos = {"particion": [c.replace("mae_p", "") for c in COLUMNAS_MAE], estrategia: fila[COLUMNAS_MAE].tolist()}
    referencia = fila["referencia"]
    if isinstance(referencia, str) and referencia in por_id.index:
        datos[referencia] = por_id.loc[referencia, COLUMNAS_MAE].tolist()
    return pd.DataFrame(datos)


def ultima_aceptada(experimentos: pd.DataFrame) -> str:
    """La última estrategia de la cadena aceptada, o E2 (DESIGN §7: estrategia por defecto del detalle)."""
    aceptadas = experimentos[experimentos["estrategia"].isin(CADENA_PANTALLA)
                             & experimentos["aceptada"].map(es_verdadero)]
    return aceptadas["estrategia"].iloc[-1] if len(aceptadas) else "E2"


def curvas_alfa(experimentos: pd.DataFrame) -> pd.DataFrame:
    """Error medio frente a α, una serie por familia (y por l1_ratio), con la ganadora marcada (DESIGN §4)."""
    ganadoras = set(mejor_por_familia(experimentos).values())
    filas = []
    for fila in experimentos.itertuples():
        clave = familia_y_alfa(fila.estrategia)
        if clave is None:
            continue
        partes = fila.estrategia.split("_")
        serie = NOMBRES_FAMILIA[clave[0]] + (f" (l1_ratio {partes[3]})" if len(partes) > 3 else "")
        filas.append({"estrategia": fila.estrategia, "serie": serie, "alfa": clave[1],
                      "mae_medio": fila.mae_medio, "ganadora": fila.estrategia in ganadoras})
    return pd.DataFrame(filas)


# ── Rendimiento ───────────────────────────────────────────────────────────────────────────────

def margenes_se_solapan(ic_a: list[float], ic_b: list[float]) -> bool:
    """Si dos intervalos de confianza se solapan (DESIGN §4, Rendimiento 1)."""
    return bool(ic_a[0] <= ic_b[1] and ic_b[0] <= ic_a[1])


def frase_rendimiento(metricas: dict) -> str:
    """Error medio del modelo y de la tasación ingenua con sus márgenes, y si la mejora es ruido (DESIGN §4)."""
    modelo, ingenua = metricas["modelo"]["mae"], metricas["baseline"]["mae"]
    texto = (f"En viviendas que no vio al entrenar, el modelo se equivoca de media "
             f"{formato.intervalo(modelo['valor'], modelo['ic95'], centenas=True)}. La tasación ingenua, "
             f"{formato.intervalo(ingenua['valor'], ingenua['ic95'], centenas=True)}.")
    if margenes_se_solapan(modelo["ic95"], ingenua["ic95"]):
        return texto + " Los márgenes se solapan: no podemos asegurar que uno sea mejor que el otro."
    if modelo["valor"] < ingenua["valor"]:
        return texto + " Los márgenes no se solapan: la mejora no es ruido."
    return texto + " Los márgenes no se solapan: el modelo es claramente peor que la tasación ingenua."


def titulo_metricas(metricas: dict) -> str:
    """Título del gráfico de error medio y RMSE, en el sentido que digan los datos (DESIGN §3)."""
    diferencia = metricas["baseline"]["mae"]["valor"] - metricas["modelo"]["mae"]["valor"]
    if diferencia > 0:
        return f"El modelo se equivoca {formato.euros(diferencia, centenas=True)} menos que la tasación ingenua"
    return f"El modelo todavía se equivoca {formato.euros(-diferencia, centenas=True)} más que la tasación ingenua"


def tabla_metricas(metricas: dict) -> pd.DataFrame:
    """Error medio y RMSE de modelo y tasación ingenua con sus márgenes, para el gráfico de puntos."""
    filas = []
    for quien, nombre in (("modelo", "Modelo"), ("baseline", "Tasación ingenua")):
        for metrica, etiqueta in (("mae", "Error medio"), ("rmse", "RMSE")):
            m = metricas[quien][metrica]
            nota = ("Raíz del error cuadrático medio: castiga más los errores grandes" if metrica == "rmse"
                    else "Lo que se equivoca de media, en euros")
            filas.append({"quien": nombre, "metrica": etiqueta, "valor": m["valor"],
                          "inferior": m["ic95"][0], "superior": m["ic95"][1], "nota": nota})
    return pd.DataFrame(filas)


def con_errores(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Añade error en € y en % con signo ((tasación − real) / real) del modelo y de la ingenua (DESIGN §5)."""
    datos = predicciones.copy()
    for prefijo, columna in (("", "prediccion"), ("baseline_", "prediccion_baseline")):
        datos[f"{prefijo}error"] = datos[columna] - datos["precio"]
        datos[f"{prefijo}error_abs"] = datos[f"{prefijo}error"].abs()
        datos[f"{prefijo}error_pct"] = datos[f"{prefijo}error"] / datos["precio"]
        datos[f"{prefijo}error_pct_abs"] = datos[f"{prefijo}error_pct"].abs()
    datos["zona"] = zona(datos)
    datos["rango"] = rango_tasacion(datos["prediccion"])
    if "tipo" in datos.columns:
        datos["tipo_vivienda"] = normalizar_tipo(datos["tipo"]).map(NOMBRES_TIPO).fillna(formato.SIN_DATO)
    return datos


def frase_error_pct(datos: pd.DataFrame) -> str:
    """«La mitad de las tasaciones se desvían menos de X %; 1 de cada 5, más de Y %» (DESIGN §4)."""
    mediana = datos["error_pct_abs"].median()
    p80 = datos["error_pct_abs"].quantile(0.8)
    return (f"La mitad de las tasaciones se desvían menos de {formato.porcentaje(mediana)}; "
            f"1 de cada 5, más de {formato.porcentaje(p80)}.")


def aviso_fuera_de_rango(resumen: dict | None) -> str | None:
    """Aviso si alguna tasación del test es ≤ 0 € o pasa de 6.000.000 € (R19, DESIGN §8)."""
    if not resumen or not resumen.get("n_fuera_de_rango"):
        return None
    return (f"{resumen['n_fuera_de_rango']} de {formato.numero(resumen['n_tasaciones'])} tasaciones del test "
            f"son de 0 € o menos, o pasan de 6.000.000 €.")


def resultados_coherentes(metricas: dict, modelo_final_experimentos: str) -> bool:
    """Si `metricas.json` es del mismo modelo final que dice `experimentos.csv` (DESIGN §8)."""
    return metricas.get("modelo_final") == modelo_final_experimentos


# ── Dónde falla ───────────────────────────────────────────────────────────────────────────────

def zona(datos: pd.DataFrame) -> pd.Series:
    """Distrito en Barcelona; municipio fuera (DESIGN §4, Dónde falla 4)."""
    return datos["districte"].fillna(datos["municipi"])


def rango_tasacion(tasacion: pd.Series) -> pd.Series:
    """Rango de tasación de DESIGN §4: < 150.000, 150.000-250.000, … > 700.000 €."""
    return pd.cut(tasacion, RANGOS_TASACION, labels=ETIQUETAS_RANGO, right=False).astype(str)


def filtrar(datos: pd.DataFrame, tipos: list[str] | None = None, zonas: list[str] | None = None,
            rango: tuple[float, float] | None = None) -> pd.DataFrame:
    """Filtros de *Dónde falla*: tipo, zona y rango de tasación (DESIGN §7). Vacío = sin filtrar."""
    seleccion = pd.Series(True, index=datos.index)
    if tipos:
        seleccion &= datos["tipo_vivienda"].isin(tipos)
    if zonas:
        seleccion &= datos["zona"].isin(zonas)
    if rango is not None:
        seleccion &= datos["prediccion"].between(rango[0], rango[1])
    return datos[seleccion]


def error_por_grupo(datos: pd.DataFrame, columna: str, orden: list[str] | None = None) -> pd.DataFrame:
    """Viviendas, error medio en €, error medio en % y sesgo (% con signo) por grupo, y si son pocas (DESIGN §4, §7)."""
    tabla = datos.groupby(columna, observed=True).agg(
        viviendas=("precio", "size"), error_medio=("error_abs", "mean"),
        error_pct=("error_pct_abs", "mean"), sesgo=("error_pct", "mean")).reset_index()
    tabla["pocas"] = tabla["viviendas"] < MINIMO_FIABLE
    if orden is not None:
        tabla[columna] = pd.Categorical(tabla[columna], categories=orden, ordered=True)
        return tabla.sort_values(columna).reset_index(drop=True)
    return tabla.sort_values("error_medio", ascending=False).reset_index(drop=True)


def titulo_error(tabla: pd.DataFrame, columna: str, nombre: str) -> str:
    """«Más error en X: N € de media por vivienda» con los grupos fiables (DESIGN §3)."""
    fiables = tabla[~tabla["pocas"]].sort_values("error_medio", ascending=False)
    if fiables.empty:
        return f"Error medio por {nombre} (todos los grupos tienen pocas viviendas)"
    peor = fiables.iloc[0]
    return f"Más error en {peor[columna]}: {formato.euros(peor['error_medio'], True)} de media por vivienda"


def titulo_sesgo(tabla: pd.DataFrame, columna: str) -> str:
    """Qué grupo tasa más de más y cuál más de menos (grupos fiables) (DESIGN §3)."""
    fiables = tabla[~tabla["pocas"]]
    if fiables.empty:
        return "Sesgo por grupo (todos los grupos tienen pocas viviendas)"
    de_mas, de_menos = fiables.loc[fiables["sesgo"].idxmax()], fiables.loc[fiables["sesgo"].idxmin()]
    partes = []
    if de_mas["sesgo"] > 0:
        partes.append(f"tasa de más en {de_mas[columna]} ({formato.porcentaje(de_mas['sesgo'], signo=True)})")
    if de_menos["sesgo"] < 0:
        partes.append(f"de menos en {de_menos[columna]} ({formato.porcentaje(de_menos['sesgo'], signo=True)})")
    return ("Sobre todo " + " y ".join(partes)) if partes else "No tasa sistemáticamente de más ni de menos"


def _sentido(sesgo: float) -> str:
    return "tasa de más" if sesgo > 0 else "tasa de menos"


def frase_donde_falla(datos: pd.DataFrame) -> str:
    """La zona y el rango de tasación con más error, y si tasa de más o de menos (DESIGN §4)."""
    partes = []
    for columna, nombre in (("zona", "La zona"), ("rango", "el rango de tasación")):
        tabla = error_por_grupo(datos, columna)
        fiables = tabla[~tabla["pocas"]]
        if fiables.empty:
            continue
        peor = fiables.iloc[0]
        partes.append(f"{nombre} con más error es {peor[columna]} ({formato.euros(peor['error_medio'], True)} "
                      f"de media; {_sentido(peor['sesgo'])}, {formato.porcentaje(peor['sesgo'], signo=True)})")
    if not partes:
        return "Ningún grupo tiene al menos 20 viviendas: el error por grupo no es fiable."
    return "; ".join(partes) + "."


def titulo_mapa(datos: pd.DataFrame, columna_error: str = "error_pct") -> str:
    """Dónde tasa más de más y más de menos (zonas con al menos 20 viviendas) (DESIGN §5)."""
    tabla = datos.groupby("zona").agg(viviendas=("precio", "size"), sesgo=(columna_error, "mean"))
    tabla = tabla[tabla["viviendas"] >= MINIMO_FIABLE]
    if tabla.empty:
        return "Error de cada vivienda en el mapa"
    de_mas, de_menos = tabla["sesgo"].idxmax(), tabla["sesgo"].idxmin()
    partes = []
    if tabla.loc[de_mas, "sesgo"] > 0:
        partes.append(f"tasa de más en {de_mas} ({formato.porcentaje(tabla.loc[de_mas, 'sesgo'], signo=True)})")
    if tabla.loc[de_menos, "sesgo"] < 0:
        partes.append(f"de menos en {de_menos} ({formato.porcentaje(tabla.loc[de_menos, 'sesgo'], signo=True)})")
    return ("Sobre todo " + " y ".join(partes)) if partes else "No hay zonas con un sesgo claro"


def peores(datos: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Las n viviendas con más error en % (DESIGN §4, Dónde falla 7)."""
    return datos.sort_values("error_pct_abs", ascending=False).head(n)


# ── Abrir la caja ─────────────────────────────────────────────────────────────────────────────

def nombre_variable(variable: str) -> str:
    return NOMBRES_VARIABLE.get(variable, variable)


def lectura_coeficiente(variable: str, coeficiente_unidad: float) -> str:
    """Lectura en la unidad de la variable, para el *tooltip* (DESIGN §6)."""
    if variable == "superficie_construida":
        return f"Por cada 10 m² más, {formato.factor(10 * coeficiente_unidad)}, con el resto igual"
    if variable == "parking":
        return f"Cada nivel de parking más (no → opcional → incluido), {formato.factor(coeficiente_unidad)}"
    descripcion = {"ascensor": "una vivienda con ascensor", "es_atico": "un ático", "es_bajo": "un bajo",
                   "es_casa": "una casa"}.get(variable, variable)
    return f"{descripcion.capitalize()} se tasa {formato.factor(coeficiente_unidad)} respecto a una que no lo es, " \
           f"con el resto igual"


def tabla_coeficientes(coeficientes: pd.DataFrame) -> pd.DataFrame:
    """Nombre legible, factor en texto, lectura en la unidad de la variable y coeficiente técnico (DESIGN §6)."""
    tabla = coeficientes.copy()
    tabla["nombre"] = tabla["variable"].map(nombre_variable)
    tabla["texto"] = tabla["coeficiente"].map(formato.factor)
    tabla["lectura"] = [lectura_coeficiente(v, c) for v, c in zip(tabla["variable"], tabla["coeficiente_unidad"])]
    tabla["tecnico"] = tabla["coeficiente"].map(formato.coeficiente_tecnico)
    return tabla


def titulo_coeficientes(tabla: pd.DataFrame) -> str:
    """La variable que más pesa y en qué sentido (DESIGN §3)."""
    primera = tabla.iloc[0]
    sentido = "sube" if primera["coeficiente"] > 0 else "baja"
    return f"{nombre_variable(primera['variable'])} es lo que más pesa: {sentido} la tasación " \
           f"{formato.factor(primera['coeficiente'])} por desviación típica"


def elegir_vivienda(datos: pd.DataFrame, criterio: str, numero: int | None = None,
                    semilla: int | None = None) -> object | None:
    """El `index` de la vivienda del desglose (DESIGN §6). Devuelve None si no existe."""
    if datos.empty:
        return None
    if criterio == "la que más se equivoca":
        return datos["error_pct_abs"].idxmax()
    if criterio == "la que mejor acierta":
        return datos["error_pct_abs"].idxmin()
    if criterio == "una al azar":
        return datos.sample(1, random_state=semilla).index[0]
    return numero if numero in datos.index else None


def desglose(contribuciones: pd.Series) -> pd.DataFrame:
    """Cascada en euros: empieza en la vivienda media (10^intercepto) y cada variable multiplica (DESIGN §6).

    Devuelve una fila por paso con `paso`, `factor`, `euros` (lo que suma o resta ese paso) y `acumulado`.
    La última fila es la tasación (10^suma).
    """
    acumulado = 10 ** float(contribuciones["intercepto"])
    filas = [{"paso": "Vivienda media", "factor": np.nan, "euros": acumulado, "acumulado": acumulado}]
    for variable, valor in contribuciones.drop("intercepto").items():
        factor = 10 ** float(valor)
        nuevo = acumulado * factor
        filas.append({"paso": nombre_variable(str(variable)), "factor": factor, "euros": nuevo - acumulado,
                      "acumulado": nuevo})
        acumulado = nuevo
    filas.append({"paso": "Tasación", "factor": np.nan, "euros": acumulado, "acumulado": acumulado})
    return pd.DataFrame(filas)


def variables_de_entrada(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Las columnas originales de la vivienda, sin precio ni tasaciones ni columnas calculadas aquí."""
    calculadas = [c for c in predicciones.columns if c.startswith(("error", "baseline_error"))]
    return predicciones.drop(columns=COLUMNAS_RESULTADO + calculadas + ["zona", "rango", "tipo_vivienda"], errors="ignore")
