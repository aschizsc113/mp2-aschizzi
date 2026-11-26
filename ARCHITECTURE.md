# ARCHITECTURE · Tasación de viviendas para Banc Avellaner

> **Cómo está construido.** Lo escribe el equipo. El agente puede proponer cambios, pero no aplicarlos sin aprobación.

## 1. Flujo de datos y del modelo

```text
data/raw/*.csv
   │
   ▼
src/datos.py ── cargar_entrenamiento() ─► limpiar_objetivo() (R9, fila a fila) ─► separar()
   │
   ▼
src/preparacion.py   transformadores de R10-R15 (se ajustan dentro del Pipeline)
   │
   ▼
src/estrategias.py   ESTRATEGIAS (E0, E1, E2, T1-T3, N1, m_*), MODELO_FINAL, crear_modelo_final()
   │
   ├─► src/experimentos.py ── validación cruzada 1-5 (src/validacion.py)
   │        └─ src/decision.py: referencia, mejora_media, particiones_mejor, aceptada (R8)
   │                            y elección de la fase 4 (R17)
   │        ──► reports/experimentos.csv
   │
   └─► src/entrenar.py ── entrena con 1-5, mide una vez en `validacion` (R6)
            ──► models/modelo.joblib
            ──► reports/metricas.json
            ──► reports/predicciones_validacion.csv

models/modelo.joblib + test_set.csv ──► src/predecir.py (aviso R19) ──► entregables/envio_kaggle.csv
                                                              ──► reports/envio_resumen.json

reports/ + models/ ──► src/app.py (solo lee; no entrena)
                         ├─ src/analisis.py   cifras y frases de cada pantalla
                         ├─ src/explicar.py   coeficientes y desglose (Abrir la caja)
                         ├─ src/graficos.py   figuras Plotly con los colores de DESIGN §3
                         └─ src/formato.py    números, € y % en formato español (DESIGN §2)
```

**El modelo** (a partir de E2) es siempre un `Pipeline` con esta forma:

```text
Pipeline([
  ("preparacion", ColumnTransformer(...)),       # R10-R15; verbose_feature_names_out=False
  ("escalado",    StandardScaler()),             # desde E2 (R14, R16)
  ("regresion",   TransformedTargetRegressor(    # R14
                     regressor=LinearRegression() | Ridge | Lasso | ElasticNet,
                     func=np.log10, inverse_func=potencia_de_10)),
])
```

- `modelo.predict()` recibe las filas tal como vienen en `test_set.csv` y devuelve euros.
- Las columnas que salen de `preparacion` se llaman como en SPECS: `superficie_construida`, `ascensor`, `parking`, `es_atico`, `es_bajo`, `es_casa`. Son las que devuelve `get_feature_names_out()` y las que usa `explicar()`.
- `potencia_de_10` es una función con nombre en `preparacion.py` (no una `lambda`), para que `joblib` pueda guardar el modelo.
- `tipo` entra en `preparacion` solo para corregir la superficie (R10, R11); no sale como columna.

## 2. Carpetas

| Carpeta | Qué contiene |
| --- | --- |
| `src/` | Todo el código del proyecto |
| `tests/acceptance/` | Tests del contrato del profesorado, del ejemplo resuelto y de los criterios CA5-CA26, un fichero por fase: `test_decision.py`, `test_limpieza.py`, `test_transformaciones.py`, `test_variables_nuevas.py`, `test_modelo.py`, `test_cuadro.py` |
| `tests/unit/` | Tests unitarios (los puede escribir el agente) |
| `data/raw/` | CSV descargados. Solo lectura |
| `models/` | Modelo entrenado. No va a Git: se regenera con `src.entrenar` |
| `reports/` | Resultados de los experimentos y del modelo final |
| `entregables/` | Envío a Kaggle |
| `notebooks/` | Exploración. Importan de `src/`; nunca al revés |
| `scripts/` | Descarga de datos y comprobación de trazabilidad |

## 3. Módulos y responsabilidades

### Modelo

| Módulo | Hace | No hace |
| --- | --- | --- |
| `datos.py` | Carga los CSV, `separar()` y `limpiar_objetivo()`: quita retiradas y precios fuera de [30.000, 3.000.000] mirando solo cada fila (R9) | No transforma variables: eso va en el `Pipeline` |
| `preparacion.py` | Los transformadores de las fases 2 y 3: `normalizar_texto()` (minúsculas, sin tildes, sin espacios a los lados), `normalizar_tipo()` (R10), `CorrectorSuperficie` (R11: aprende mediana y percentil 99 por tipo en `fit`), `ascensor_a_binario()` (R12), `parking_a_ordinal()` (R13), `planta_a_indicadores()` (R15), `potencia_de_10()` (R14) y `crear_preparacion(variables)`, que monta el `ColumnTransformer` con las variables pedidas | No elige qué variables usa cada estrategia ni entrena estimadores. No calcula nada fuera de `fit` |
| `baseline.py` | La tasación ingenua (R2) | |
| `validacion.py` | Validación cruzada e intervalos por bootstrap (R3, R4) | No sabe qué estrategias existen |
| `estrategias.py` | Una `Estrategia` por fila de SPECS §4, con su Id (`E0`, `E1`, `E2`, `T1`, `T2`, `T3`, `N1`, `m_ridge_<α>`, `m_lasso_<α>`, `m_enet_<α>_<l1_ratio>`), su tipo (`baseline`, `minimo`, `estrategia` o `caracteristica`: SPECS §4), su descripción y su referencia. `crear_pipeline(variables, estimador, escalar)` monta el `Pipeline` de §1. `MODELO_FINAL` y `crear_modelo_final()` | No entrena ni evalúa. No decide si una estrategia se acepta |
| `decision.py` | Funciones puras sobre la tabla de resultados: resolver «última aceptada» (R8, CA6), calcular `mejora_media`, `particiones_mejor` y `aceptada` (R8, CA5), y elegir la mejor de cada familia de la fase 4 y el modelo final (R17, CA21) | No entrena, no lee ficheros y nunca recibe filas de `validacion` |
| `experimentos.py` | Evalúa las estrategias en el orden de `ESTRATEGIAS`, añade las columnas de R8 con `decision.py` y escribe `experimentos.csv`. Avisa si la referencia escrita en una estrategia no coincide con la «última aceptada» calculada | No toca `validacion`. No cambia `MODELO_FINAL` |
| `entrenar.py` | Entrena `MODELO_FINAL` con las particiones 1-5, lo mide una vez en `validacion` junto al baseline y lo guarda (R6) | No compara estrategias |
| `predecir.py` | Predice el test con el modelo guardado, avisa si alguna tasación es ≤ 0 o > 6.000.000 € y escribe `reports/envio_resumen.json` (R7, R19) | No entrena. Es el único módulo que lee el test |
| `explicar.py` | `explicar(modelo, X)`: intercepto + `coeficiente × valor transformado` por columna, en log10 (R18, CM5). `coeficientes(modelo)`: coeficiente y factor 10^coef de cada variable (DESIGN §6) | No entrena ni lee ficheros |

**Cómo se elige `MODELO_FINAL`.** `decision.py` calcula cuál debería ser según R17 a partir de `experimentos.csv`. El agente lo copia en la constante `MODELO_FINAL` de `estrategias.py` en la tarea correspondiente, una vez que el equipo ha revisado los resultados. Un test comprueba que la constante coincide con lo que dice `decision.py`.

**Cómo se añade una estrategia.** Una tarea = una fila de SPECS §4. El agente escribe la estrategia con la lista explícita de variables de su referencia más lo que cambia. Si la referencia es «Última aceptada», usa la que el equipo ha dado por aceptada en `TASKS.md`.

### Cuadro de mando

| Módulo | Hace | No hace |
| --- | --- | --- |
| `app.py` | Navegación y las 4 pantallas (R20): lee `reports/` y `models/`, llama a `analisis.py`, `explicar.py` y `graficos.py`, y pinta con Streamlit. Gestiona los ficheros que faltan y los casos límite de DESIGN §8 (R21) | No entrena ni recalcula experimentos. No hace cálculos: los pide a `analisis.py` |
| `analisis.py` | Funciones puras que reciben `DataFrame`s y devuelven lo que muestra cada pantalla: error en € y en % con signo, error por tipo, zona y rango de tasación, grupos con menos de 20 viviendas, si los márgenes se solapan, la vivienda que más y menos se equivoca, la comprobación `metricas.json` ↔ `experimentos.csv` y las frases de cabecera con sus dos sentidos (DESIGN §1, §3, §4) | No importa Streamlit ni Plotly |
| `graficos.py` | Una función por gráfico de DESIGN §4-§6 que devuelve una figura Plotly: plantilla `simple_white`, colores fijos (azul modelo, gris ingenua, naranja alerta), `separators=",."`, mapa `carto-positron` y cascada con `go.Waterfall` | No calcula cifras ni escribe títulos: los recibe de `analisis.py` |
| `formato.py` | `euros()`, `porcentaje()`, `intervalo()`, `factor()`, `superficie()`, `fecha()` y el `—` de sin dato, según DESIGN §2 | |

Se separa así para que todo lo que se puede comprobar sin navegador (cifras, frases, formatos) se pruebe con tests normales, y `app.py` se pruebe con `streamlit.testing.v1.AppTest` (CA25, CA26).

## 4. Salidas

Las de SPECS §5, con estos detalles:

| Fichero | Columnas o claves |
| --- | --- |
| `reports/experimentos.csv` | `estrategia` (el Id de SPECS §4), `tipo`, `descripcion`, `referencia`, `mae_p1`…`mae_p5`, `mae_medio`, `mae_sd`, `mejora_media`, `particiones_mejor`, `aceptada`. Una fila por estrategia, en el orden de la cadena |
| `reports/metricas.json` | Lo que ya escribe la plantilla: `modelo_final`, `n_entrenamiento`, `n_validacion` y MAE y RMSE con `ic95` del baseline y del modelo |
| `reports/predicciones_validacion.csv` | `index`, `precio`, `prediccion`, `prediccion_baseline` y, **además**, las columnas originales de la vivienda (las de `X_val`; SPECS §5) |
| `entregables/envio_kaggle.csv` | `index`, `precio` |
| `reports/envio_resumen.json` | `n_tasaciones`, `n_fuera_de_rango`, `minimo`, `maximo` (R19) |

## 5. Dependencias

No hace falta ninguna librería nueva. Lo instalado en `pyproject.toml` cubre todo:

| Librería | Para qué |
| --- | --- |
| pandas, numpy | Datos y cálculos |
| scikit-learn | `Pipeline`, `ColumnTransformer`, `FunctionTransformer`, `StandardScaler`, `TransformedTargetRegressor` y los estimadores lineales |
| joblib | Guardar y cargar el modelo |
| streamlit | Cuadro de mando y `AppTest` para sus tests |
| plotly | Gráficos y mapa (`carto-positron` no necesita clave) |
| pytest | Tests |

Las tildes se quitan con `unicodedata`, de la librería estándar.

## 6. Qué está protegido

El agente no modifica `SPECS.md`, `DESIGN.md`, `data/raw/` ni este fichero sin aprobación del equipo. Los tests de `tests/acceptance/` los escribe el agente en sus tareas, una persona del equipo los valida y después nadie los cambia para que pasen. El equipo no escribe en `src/` ni en `tests/`.

## 7. Pendiente de decidir

Puntos en los que SPECS.md, DESIGN.md y el contrato del profesorado no encajan. La arquitectura de arriba no los resuelve: los decide el equipo.

1. ~~**Nombre de la columna del Id.**~~ Resuelto el 26/11/2026: se llama `estrategia` (SPECS R8, CM6).
2. ~~**Tipo `estrategia` en CM6.**~~ Resuelto el 26/11/2026: los tipos son `baseline`, `minimo`, `estrategia` (T1-T3, M1-M3) y `caracteristica` (N1) (SPECS §4, R16). Falta confirmarlo con el profesorado (SPECS §9).
3. ~~**Datos de la vivienda en el cuadro.**~~ Resuelto el 26/11/2026: `src.entrenar` guarda en `predicciones_validacion.csv` también las columnas de `X_val` (SPECS §5).
4. ~~**Aviso de R19 en *Rendimiento*.**~~ Resuelto el 26/11/2026 con la opción (a): `src.predecir` escribe `reports/envio_resumen.json` (SPECS R19, §5).
5. ~~**Intercepto como «vivienda media».**~~ Resuelto el 26/11/2026 con la opción (a): `StandardScaler` desde E2 (SPECS R14, R16).
