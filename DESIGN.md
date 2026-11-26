# DESIGN · Cuadro de mando de validación del modelo

> **Cómo se ve y cómo se comporta.** Lo escribe el equipo; el agente lo lee pero no lo edita.

## 1. Para quién

**Núria Casals, directora de riesgos (usuaria principal).** Conoce el riesgo de crédito y el mercado inmobiliario. Entiende una mediana, un porcentaje y un «margen de error», pero no sabe qué es un MAE, un RMSE, un logaritmo ni un λ. Abrirá el cuadro **unos 10 minutos antes de un comité** y en revisiones periódicas. En la primera pantalla que abra tiene que ver, sin bajar, cuánto se equivoca el modelo y si mejora lo que ya hace el banco.

**Tasadores.** Entran en *Dónde falla* y *Abrir la caja* para ver una vivienda concreta y saber en qué zonas o tipos no fiarse.

**Auditoría.** Entra en *Experimentos* para seguir qué se probó, con qué regla se decidió y qué se descartó.

Consecuencias para todo el cuadro:

- **Vocabulario de negocio.** «Error medio» y no MAE; «tasación ingenua» y no *baseline* (se menciona *baseline* una sola vez, entre paréntesis); «margen (IC 95 %)» y no «intervalo bootstrap». Los términos técnicos solo aparecen en *tooltips* o en desplegables «Detalle técnico».
- **Nada en logaritmos en pantalla.** Todo se expresa en euros, en % o como factor multiplicativo (×1,12).
- **Cada pantalla empieza con una frase que responde a su pregunta**, calculada con los datos (no fija).

## 2. Formatos

Locale español: punto para los miles, coma para los decimales, espacio antes de € y de %. En Plotly: `separators=",."`.

| Qué | Formato | Ejemplo |
| --- | --- | --- |
| Precios y errores en euros | Sin decimales, con punto de miles. En las cifras grandes de cabecera, redondeado a centenas | `121.100 €` |
| Errores en porcentaje | Un decimal. Con signo cuando es error con dirección (de más / de menos) | `18,4 %` · `+6,2 %` |
| Intervalos de confianza | Valor y, debajo o entre paréntesis, «entre A y B». Mismas reglas que el valor | `121.100 € (entre 112.000 y 131.200 €)` |
| Coeficientes | Factor multiplicativo con dos decimales y su lectura en %. El coeficiente en log10 (3 decimales) solo en el detalle técnico | `×1,12 (+12 %)` · técnico: `0,049` |
| Superficies | Sin decimales | `85 m²` |
| Fechas | día/mes/año | `17/02/2024` |
| Sin dato | Raya larga, con *tooltip* «sin dato» | `—` |

## 3. Estilo de los gráficos

- Plotly Express con plantilla `simple_white`. Excepción: el desglose de una vivienda (§6) usa `go.Waterfall`, que Plotly Express no tiene.
- **Colores fijos en todo el cuadro:**
  - **Azul `#2a78d6` = el modelo.** Siempre.
  - **Gris `#898781` = la tasación ingenua** y todo lo que es referencia o está descartado.
  - **Naranja `#eb6834` = alerta**: estrategia que empeora, tasación **por encima** del precio real (riesgo para el banco: la garantía vale menos de lo que creemos) y variables que **bajan** el precio.
  - En el mapa y en los gráficos de sesgo, «tasa de menos» va en azul. Como ahí solo aparece el modelo, no se confunde.
- **El título dice la conclusión, y se escribe a partir de los datos**, con plantilla para los dos sentidos. Ejemplo: «El modelo se equivoca 21.000 € menos que la tasación ingenua» o «El modelo todavía se equivoca más que la tasación ingenua». Nunca un título que quede falso si cambian las cifras.
- Ejes con unidades (`€`, `%`, `m²`) y formato de §2. Leyenda arriba y horizontal. Sin gráficos de tarta ni 3D.
- Bajo cada gráfico, una línea de `st.caption` con cuántas viviendas entran y de dónde salen («1.250 viviendas de validación que el modelo no vio al entrenar»).

## 4. Pantallas

| Pantalla | Pregunta | Contenido, de arriba abajo |
| --- | --- | --- |
| Experimentos | 1 · ¿Qué habéis hecho con los datos y por qué? | 1) Frase resumen: «Probamos N estrategias y aceptamos X. El error medio pasó de A € (E2) a B € (modelo final)». 2) Recuadro con la regla de decisión de SPECS §4, en una frase. 3) Gráfico de barras horizontales en el orden de la cadena (E0, E1, E2, T1, T2, T3, N1 y la mejor de M1, M2 y M3): error medio ± desviación de las 5 particiones. Aceptadas en azul, descartadas en gris, E0 como línea vertical gris discontinua. 4) Tabla: estrategia, qué cambia, referencia, error medio, mejora media (€), particiones en que mejora («4 de 5»), decisión (✓ aceptada / ✗ descartada). 5) Desplegable «Detalle por partición»: eliges una estrategia y ves sus 5 errores junto a los de su referencia. 6) Desplegable «Detalle técnico · elección de λ»: error medio frente a α (eje logarítmico), una línea por familia (y por `l1_ratio` en ElasticNet), con la ganadora marcada |
| Rendimiento | 2 · ¿Cuánto se equivoca? | 1) Frase: «En viviendas que no vio al entrenar, el modelo se equivoca de media X € (entre A y B). La tasación ingenua, Y € (entre C y D)». Si los márgenes no se solapan: «la mejora no es ruido»; si se solapan: «no podemos asegurar que mejore». 2) Tres tarjetas: error medio del modelo, error medio de la tasación ingenua, diferencia (en verde si el modelo mejora, en naranja si empeora). 3) Gráfico de puntos con barra de error: error medio y RMSE, modelo frente a tasación ingenua, con sus márgenes. El RMSE se explica en el *tooltip* («castiga más los errores grandes»). 4) Error en %: «La mitad de las tasaciones se desvían menos de X %; 1 de cada 5, más de Y %». 5) Histograma del error en % con signo, modelo y tasación ingenua superpuestos, con bandas sombreadas en ±10 % y ±20 %. 6) Desplegable: precio real frente a tasación (ejes logarítmicos), con la diagonal |
| Dónde falla | 3 · ¿Dónde falla? | 1) Filtros (§7) y «N viviendas de 1.250». 2) Frase: la zona y el rango de precio con más error y si tasa de más o de menos. 3) Error por tipo de vivienda: barras de error medio en € y, al lado, error medio en % con signo (sesgo). 4) Error por zona (distrito en Barcelona, municipio fuera): barras horizontales ordenadas de mayor a menor, con el número de viviendas. 5) Error por rango de tasación (< 150.000, 150.000-250.000, 250.000-400.000, 400.000-700.000, > 700.000 €), con signo: se espera que tase de más las baratas y de menos las caras. 6) Mapa de residuos (§5). 7) Tabla de las 20 viviendas con más error en %, para los tasadores |
| Abrir la caja | 4 · ¿Por qué tasa lo que tasa? · 5 · ¿Qué no podemos afirmar? | 1) Aviso fijo (`st.info`) de que los coeficientes no son efectos causales (§6). 2) Coeficientes: barras horizontales con el factor de cada variable. 3) Selector de vivienda. 4) Desglose en cascada de esa vivienda, con su precio real marcado. 5) Desplegable «Qué no podemos afirmar», con el texto de §6 |

## 5. El mapa de residuos

- **Cada punto es una vivienda de `validacion`**, en su `latitud` y `longitud`. Fondo `carto-positron`, centrado en Barcelona, zoom 10.
- **Color = error en % con signo**: (tasación − precio real) / precio real. Escala divergente centrada en 0: **naranja = tasa de más**, gris claro = acierta, **azul = tasa de menos**. La escala se corta en ±50 % para que unos pocos casos extremos no apaguen el resto.
- **Tamaño fijo** (pequeño, opacidad 0,7). El error ya va en el color; usar también el tamaño duplica la información y tapa puntos.
- **Al pasar el ratón:** barrio (o municipio), tipo, superficie, precio real, tasación del modelo, tasación ingenua, error en € y en %.
- Un selector permite ver el mismo mapa con la tasación ingenua, para comparar.
- Título: la conclusión, por ejemplo «El modelo tasa de más en el norte de Barcelona y de menos en la zona alta».

## 6. Abrir la caja

**Coeficientes.**

- Se muestran **todos** (son 6 variables). Como el `Pipeline` estandariza las variables, los coeficientes ya son comparables entre sí: miden cuánto cambia la tasación cuando la variable sube una desviación típica. Se ordenan por tamaño.
- Cada barra muestra el **factor** 10^coeficiente: `×1,25` = la tasación sube un 25 %. Azul si sube el precio, naranja si lo baja.
- El *tooltip* da la lectura en la unidad de la variable: para las 0/1, «un ático se tasa ×1,15 respecto a uno que no lo es, con el resto igual»; para la superficie, «por cada 10 m² más, ×1,04».

**Desglose de una vivienda.**

- **Cómo se elige:** un selector con cuatro opciones: «la que más se equivoca», «la que mejor acierta», «una al azar» y «por número (`index`)».
- **Cómo se lee:** cascada en euros. Empieza en la **vivienda media** (10^intercepto: como las variables están estandarizadas, el intercepto es la tasación de una vivienda con todas las variables en su media) y cada variable multiplica: «Ascensor: ×1,03 (+6.200 €)». Termina en la tasación. Una línea discontinua marca el precio real.
- Nota bajo el gráfico: «Los factores no dependen del orden; los euros de cada paso, sí, porque se aplican sobre lo acumulado».

**Qué no podemos afirmar** (aviso fijo arriba y texto completo en el desplegable):

> Un coeficiente dice cuánto más **se tasan**, en estos datos, las viviendas que tienen esa característica, con el resto de variables del modelo iguales. **No dice cuánto subiría el precio si se la añadiéramos.** Instalar un ascensor o reformar un piso no lo revaloriza en ese factor: el modelo no sabe en qué barrio está la vivienda, así que el ascensor o el parking recogen también parte de la zona en la que suelen estar. Además, las ventas son simuladas (2023-2025).

## 7. Filtros e interacción

No hay filtros globales: cada pantalla tiene los suyos y solo afectan a esa pantalla.

| Pantalla | Filtro | Por defecto | Afecta a |
| --- | --- | --- | --- |
| Experimentos | Casilla «Mostrar las 35 combinaciones de la fase 4» | Desactivada: solo la mejor de cada familia | Tabla y gráfico |
| Experimentos | Estrategia del detalle por partición | La última aceptada | Solo ese desplegable |
| Dónde falla | Tipo de vivienda (selección múltiple) | Todos | Todos los gráficos, el mapa y la tabla |
| Dónde falla | Zona: distritos de Barcelona y municipios (selección múltiple) | Todas | Todos los gráficos, el mapa y la tabla |
| Dónde falla | Rango de tasación (deslizador, €) | Todo el rango | Todos los gráficos, el mapa y la tabla |
| Dónde falla | Mapa: modelo o tasación ingenua | Modelo | Solo el mapa |
| Abrir la caja | Vivienda (§6) | La que más se equivoca | Solo el desglose |

Los grupos con **menos de 20 viviendas** se muestran en gris claro con la etiqueta «pocas viviendas»: su error no es fiable.

## 8. Casos límite

Nunca se muestra una traza de error.

| Situación | Qué ve el usuario |
| --- | --- |
| Falta un fichero de `reports/` o `models/` | En esa pantalla, `st.info` con el fichero que falta y el comando que lo genera (R21). Las demás pantallas funcionan |
| `experimentos.csv` no tiene las filas de la fase 4 | Se oculta el desplegable de λ, con «Aún no se han ejecutado M1-M3» |
| Los filtros no dejan ninguna vivienda | `st.warning`: «Ninguna vivienda cumple los filtros. Amplía el tipo, la zona o el rango». No se dibujan gráficos vacíos |
| Un grupo tiene menos de 20 viviendas | Gris claro y «pocas viviendas» (§7) |
| El modelo es peor que la tasación ingenua | Se dice tal cual: títulos en el sentido contrario y diferencia en naranja. Nunca se oculta |
| Alguna tasación ≤ 0 € o > 6.000.000 € | Aviso en *Rendimiento* con cuántas son (R19) |
| Viviendas sin coordenadas | No salen en el mapa; debajo, «N viviendas sin coordenadas no aparecen» |
| El modelo guardado no se puede explicar | En *Abrir la caja*: «Vuelve a entrenar: `uv run python -m src.entrenar`» |
| `metricas.json` y `experimentos.csv` no coinciden en el modelo final | Aviso: «Los resultados son de ejecuciones distintas. Vuelve a ejecutar `src.experimentos` y `src.entrenar`» |
