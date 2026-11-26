# SPECS · Tasación de viviendas para Banc Avellaner

> **Qué construimos.** Lo escribe el equipo; el agente lo lee pero no lo edita.
> Las indicaciones entre `<!-- -->` son de la plantilla: borradlas cuando escribáis cada sección.
> Lo marcado como **ejemplo resuelto** ya está implementado y testeado: no lo borréis, ampliadlo.

## 1. Objetivo

Dar a Banc Avellaner una **primera tasación automática** de cada vivienda que entra como garantía de una hipoteca en Barcelona y su área metropolitana, junto con una medida de cuánto puede equivocarse. Con ella, el departamento de riesgos decide **qué tasaciones acepta tal cual y cuáles manda revisar a un tasador**. Cada cifra tiene que poder explicarse variable a variable a un cliente o a un auditor.

Lo que buscamos no es el mejor R². Buscamos un error en euros **menor que el de la tasación ingenua (€/m² mediano del barrio × superficie)**, con un intervalo de confianza que muestre que la mejora no es ruido, y saber en qué viviendas no conviene fiarse del modelo.

## 2. Usuarios y preguntas

| Usuario | Cuándo lo usa | Qué sabe | Qué necesita |
| --- | --- | --- | --- |
| **Núria Casals**, directora de riesgos (usuaria principal) | Antes de aprobar el modelo para producción y en revisiones periódicas | Riesgo de crédito y mercado inmobiliario. No sabe de *machine learning* | Cifras de error en euros y en %, con su incertidumbre, y una recomendación clara de uso |
| **Tasadores** del banco | Cuando revisan un caso que el modelo marca como dudoso | Saben tasar viviendas, pero no conocen el modelo | Ver por qué el modelo da ese precio a una vivienda concreta y en qué zonas o tipos suele fallar |
| **Auditoría interna o supervisor** | En auditorías del proceso de concesión | Regulación y control. Piden trazabilidad | Qué se hizo con los datos y por qué, qué variables usa el modelo y qué no se puede concluir de él |

**Preguntas de negocio**

- **1 · ¿Qué habéis hecho con los datos y por qué?** Problemas encontrados, estrategias que habéis probado, cuánto ha cambiado el error con cada una y qué habéis decidido. Incluye al menos una característica nueva creada por vosotros, y si ha servido o no.
- **2 · ¿Cuánto se equivoca?** Error medio en euros y en porcentaje, comparado con una tasación ingenua (el *baseline*), y con su intervalo de confianza: ¿la mejora es real o puede ser ruido?
- **3 · ¿Dónde falla?** Error por tipo de vivienda, por zona (distrito o municipio) y por rango de precio, y en el mapa. ¿Hay zonas o viviendas que tasa sistemáticamente de más o de menos?
- **4 · ¿Por qué tasa lo que tasa?** Qué variables pesan más y cómo se construye la tasación de una vivienda concreta, variable a variable.
- **5 · ¿Qué no podemos afirmar?** Qué dicen los coeficientes y qué no (un coeficiente no es el efecto de reformar el piso o de instalar un ascensor).
- **6 · ¿Lo pondríamos en producción?** Para qué viviendas sí, para cuáles no (por ejemplo, zonas o rangos de precio con mucho error) y qué haría falta para fiarse más. *Fuente: preguntas 2 y 3. Se responde en el resumen para la dirección de `README.md` y en la defensa, no en una pantalla propia.*


## 3. Datos de entrada y problemas

Lo encontramos con `notebooks/exploracion.ipynb`. Entrenamiento: 6.237 filas. Test: 4.002.

| Id | Columna | Qué pasa (con cifras) | Por qué importa | Fase |
| --- | --- | --- | --- | --- |
| P1 | `estado_operacion` | 620 filas (10 %) son anuncios retirados, que no se vendieron. El test solo tiene ventas | Su precio es lo que se pedía, no lo que se pagó. Se quitan desde el train dataset! | 1 · Limpieza |
| P2 | `precio` | 22 registros con precio < 30000€ y 4 con precio mayor que 3000000€ | Se consideran problematicos y se quitan desde el train dataset! | 1 · Limpieza |
| P3 | `planta` | Valores distinto para las mismas plantas | Hay que normalizar los valores (minúsculas, sin tildes, sin espacios a los lados) y crear tres variables 0/1. Grupo "Ático": `atico`, `at` (334 filas tras la limpieza). Grupo "Bajo": `bajo`, `bj`, `planta baja`, `0` (689). Grupo "Casa": `casa` (59). El resto (plantas numeradas, `principal`/`pral`/`pr`, `entresuelo`/`entlo`/`entl.`/`en`, `-` y vacíos) vale 0 en las tres. | 3 · Variables nuevas |
| P4 | `ascensor` | Valores distintos para SI/NO. Además hay 935 missing values. | Hay que normalizar transformar en 0 (NO) - 1 (SI). Imputa el valor 0 a los missing (me asumo el riesgo de introducir algun pequeño error, pero la mayoría de las veces será correcto!)| 2 · Transformación |
| P5 | `parking` | 626 missing values y valores distintos para Si/Opcional/No | hay que normalizar y transformar en 0 (NO) - 1 (Opcional) y 2 (SI, Incluido). Imputa el valor 0 a los missing (me asumo el riesgo de introducir algun pequeño error, pero la mayoría de las veces será correcto!) | 2 · Transformación |
| P6 | `superficie_construida` | Hay 20 ceros y 17 registros con más de 400 m² (hasta 1.590 m² en pisos y áticos) | Mediana y tope por tipo, calculados con las filas de entrenamiento de ese tipo que tienen entre 1 y 400 m². Los ceros pasan a la mediana de su tipo. Toda superficie mayor que el percentil 99 de su tipo pasa a ese percentil (no solo las de más de 400 m²), así que no hay saltos: un piso de 390 m² y uno de 410 m² quedan igual. Con los datos limpios los topes son: piso 196, ático 183, dúplex 253, casa 357 y estudio 61 m²; afectan a 73 filas (1,3 %). | 2 · Transformación |
| P7 | `tipo` | Valores distinto para los mismos. | Hay que normalizar. Chalet va con casa. Solo sirve para mediana y tope. | 2 · Transformación |

## 4. Estrategias y experimentos

**Regla de decisión** (escrita antes de ejecutar nada): una estrategia se acepta si baja el MAE de su referencia en al menos 4 de las 5 particiones y en al menos 1.000 € de media. Si no lo cumple, se descarta, y la estrategia siguiente se compara con la última aceptada. Si la mejora media es menor de 1.000 €, aunque sea en 5 de 5, se descarta: no compensa complicar el modelo.

**Tipos** (los del contrato del profesorado, CM6): `baseline` (E0), `minimo` (E1, E2), `caracteristica` (variables nuevas: N1) y `estrategia` (el resto: transformaciones T1-T3 y modelos M1-M3). La fase se lee en el Id y en la sección de esta tabla.

Cómo leemos los resultados: `src.experimentos` da el MAE de cada partición (`mae_p1` … `mae_p5`). Comparamos cada estrategia con su referencia partición a partición, porque hay particiones más difíciles que otras para todas las estrategias a la vez.

**Variables:** a partir de E2, los modelos solo usan variables de §3: `superficie_construida`, `ascensor`, `parking` y las tres de `planta`. `tipo` no entra en el modelo: solo sirve para corregir la superficie (P6). La mediana y el percentil 99 por tipo se calculan dentro del `Pipeline`, solo con las particiones de entrenamiento de cada vuelta.

**Objetivo:** a partir de E2, todos los modelos predicen `log10(precio)` con `TransformedTargetRegressor` (`func=np.log10`, `inverse_func=potencia_de_10`: 10 elevado a la predicción). Las variables se estandarizan (`StandardScaler`) antes del estimador. La tasación se devuelve en euros y el MAE se mide siempre en euros. E1 (ejemplo resuelto) sigue prediciendo el precio en euros.

### Punto de partida

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | baseline | €/m² mediano de la zona × superficie construida | — | — | | — | La tasación que hay que batir |
| E1 | minimo | Regresión lineal con `superficie_construida`, `habitaciones`, `banos`, `distancia_centro_km` y `poblacion_zona`, sin transformar | — | Peor que E0: no sabe en qué barrio está la vivienda, solo a qué distancia del centro | | — | Solo como referencia (ejemplo resuelto) |
| E2 | minimo | Regresión lineal de `log10(precio)` solo con `superficie_construida`, sin transformar (la única variable de §3 que ya es numérica) | — | Peor que E0 y que E1: sin ubicación, el m² vale lo mismo en Pedralbes que en Nou Barris. Con el log, el precio crece exponencialmente con la superficie, y las de 800-1.600 m² sin corregir pueden dar tasaciones absurdas | | — | Referencia de la cadena |

### Fase 1 · Limpieza

No hay estrategias en esta fase. P1 y P2 van en `limpiar_objetivo()`, que decide fila a fila: quita los anuncios retirados (620) y los precios < 30.000 € o > 3.000.000 €. Quedan 5.592 filas. Cada vez que cambie `limpiar_objetivo()` se vuelven a ejecutar E0, E1 y E2 y se anotan aquí sus cifras.

### Fase 2 · Transformaciones

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | estrategia | Superficie corregida (P6, P7): `tipo` normalizado (Chalet → Casa); ceros → mediana de su tipo; por encima del percentil 99 de su tipo → ese percentil | E2 | Mejora mucho: con el log, las superficies de 400 a 1.600 m² y las de 0 m² tuercen la pendiente y disparan las tasaciones | | | |
| T2 | estrategia | Añade `ascensor` normalizado a 0/1, vacíos → 0 (P4) | Última aceptada | Mejora poco: el ascensor sube el precio, pero los 935 vacíos puestos a 0 añaden ruido | | | |
| T3 | estrategia | Añade `parking` normalizado a 0 (No) / 1 (Opcional) / 2 (Sí, Incluido), vacíos → 0 (P5) | Última aceptada | Mejora poco: con plaza incluida, la vivienda suele ser más cara | | | |

### Fase 3 · Variables nuevas

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | caracteristica | Añade `es_atico`, `es_bajo` y `es_casa` a partir de `planta` normalizada, con los grupos de P3; `-`, vacíos y el resto de plantas → 0 en las tres | Última aceptada | Mejora: un ático vale más y un bajo menos que el resto de plantas | | | |

### Fase 4 · Modelo

Las mismas variables que la última estrategia aceptada; solo cambia el estimador. Las variables ya se estandarizan (`StandardScaler`) dentro del `Pipeline` desde E2, así que la penalización trata a todas por igual. En `experimentos.csv` hay una fila por combinación (35 en total: 7 Ridge, 7 Lasso y 21 ElasticNet): `m_ridge_<α>`, `m_lasso_<α>` y `m_enet_<α>_<l1_ratio>`. En esta tabla va una fila por familia, con el mejor valor.

**Cómo elegimos:** en cada familia, el valor con el MAE medio más bajo en la validación cruzada (nunca en `validacion`). El mejor de las tres familias se compara con la última aceptada usando la regla de decisión. Si ninguno la cumple, el modelo final es la regresión lineal de la última aceptada.

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | estrategia | Ridge con α ∈ {0,1; 1; 10; 100; 1.000; 10.000; 100.000} | Última aceptada | Con 6 variables y unas 3.600 filas por vuelta no hay sobreajuste: hasta α = 100 da lo mismo que la regresión lineal y con α ≥ 10.000 empeora | | | |
| M2 | estrategia | Lasso con α ∈ {0,0001; 0,0003; 0,001; 0,003; 0,01; 0,03; 0,1} | Última aceptada | Con α pequeños, igual que la regresión lineal. Al crecer, anula primero las variables débiles (ascensor, parking) y empeora poco a poco; hacia α ≈ 0,19 todos los coeficientes valen 0 | | | |
| M3 | estrategia | ElasticNet con α ∈ {0,0001; 0,0003; 0,001; 0,003; 0,01; 0,03; 0,1} × l1_ratio ∈ {0,1; 0,5; 0,9} | Última aceptada | Entre Ridge y Lasso, sin mejorar a ninguno de los dos: con tan pocas variables, combinar las penalizaciones no aporta | | | |

## 5. Salidas

| Fichero | Qué contiene | Lo escribe |
| --- | --- | --- |
| `reports/experimentos.csv` | Una fila por estrategia con su MAE por partición | `src.experimentos` |
| `models/modelo.joblib` | El modelo final entrenado con las particiones 1-5 | `src.entrenar` |
| `reports/metricas.json` | MAE y RMSE del baseline y del modelo en `validacion`, con IC 95 % | `src.entrenar` |
| `reports/predicciones_validacion.csv` | Precio real, predicción del modelo y del baseline de cada vivienda de `validacion`, más sus columnas originales (las de `X_val`), que usa el cuadro de mando | `src.entrenar` |
| `entregables/envio_kaggle.csv` | Predicción de cada vivienda del test | `src.predecir` |
| `reports/envio_resumen.json` | Cuántas tasaciones del test hay, cuántas están fuera de rango (R19), la mínima y la máxima | `src.predecir` |

## 6. Requisitos

**Ejemplo resuelto: datos y validación**

- R1. `cargar_entrenamiento()` devuelve las 6.237 filas de `train_set.csv` con la columna `particion` de `particiones.csv`; `cargar_test()` devuelve las 4.002 del test. No se limpia nada al cargar. Las filas que quita `limpiar_objetivo()` se definen en los requisitos de la fase 1.
- R2. Baseline: la tasación de una vivienda es el €/m² mediano de las operaciones de entrenamiento de su zona (el barrio en Barcelona; el municipio fuera de Barcelona o si falta el barrio) × su `superficie_construida`. Si la zona no tiene operaciones, el €/m² mediano global; si la vivienda no tiene superficie (vacía o 0), el precio mediano de su zona.
- R3. Validación cruzada: cada estrategia se entrena cinco veces, cada vez con cuatro de las particiones 1-5, y se mide el MAE en la que queda fuera. Las filas de `validacion` no se usan.
- R4. El error en `validacion` se da con un intervalo de confianza del 95 % por bootstrap percentil de las filas (2.000 remuestras, semilla 2026).
- R5. Modelo mínimo: regresión lineal con `superficie_construida`, `habitaciones`, `banos`, `distancia_centro_km` y `poblacion_zona`, sin transformar (E1). Se mantiene como referencia del ejemplo resuelto; la cadena de estrategias de §4 parte de E2.
- R6. El modelo final se entrena con las particiones 1-5 y se mide una vez en `validacion`, junto al baseline entrenado con las mismas filas.
- R7. El envío a Kaggle son las predicciones del modelo guardado para las 4.002 viviendas del test.

**Cómo decidimos**

- R8. `experimentos.csv` tiene, además de `mae_p1`…`mae_p5`, la media y la desviación, estas columnas: `estrategia` (el Id de §4: E0, E1, E2, T1…; uno distinto en cada fila), `tipo`, `descripcion`, `referencia`, `mejora_media` (MAE medio de la referencia − MAE medio de la estrategia, en €), `particiones_mejor` (en cuántas de las 5 particiones el MAE baja) y `aceptada`. Una estrategia se acepta si `particiones_mejor ≥ 4` **y** `mejora_media ≥ 1.000 €`. Su referencia es la estrategia de la columna «Referencia» de §4. Cuando pone «Última aceptada», es la última estrategia de la cadena E2 → T1 → T2 → T3 → N1 con `aceptada = True`, o E2 si no se ha aceptado ninguna. E0 y E1 no tienen referencia.

**Fase 1 · Limpieza**

- R9. `limpiar_objetivo(datos)` quita las filas con `estado_operacion = "retirada"` (P1) y las que tienen `precio < 30.000` o `precio > 3.000.000` (P2). Se quedan las que valen exactamente 30.000 € o 3.000.000 €. Cada fila se decide solo con sus propios valores. Con `train_set.csv` quedan 5.592 filas. No cambia ninguna otra columna.

**Fase 2 · Transformaciones** (todo dentro del `Pipeline`, ajustado solo con las filas de entrenamiento de cada vuelta)

- R10. `tipo` se normaliza: minúsculas, sin espacios a los lados y sin tildes. `chalet` pasa a `casa`. Quedan 5 tipos: `piso`, `atico`, `duplex`, `estudio` y `casa` (P7). `tipo` no entra en el modelo.
- R11. Superficie corregida (P6, T1). Para cada tipo se calculan la mediana y el percentil 99 de `superficie_construida`. Se usan solo las filas de entrenamiento de ese tipo que tienen entre 1 y 400 m². El percentil se calcula por interpolación lineal (la de `pandas.quantile`). Al transformar:
  - una superficie de 0 o vacía pasa a la mediana de su tipo;
  - una superficie mayor que el percentil 99 de su tipo pasa a ese percentil;
  - el resto no cambia.

  Si el tipo no aparece en el entrenamiento, se usan la mediana y el percentil 99 de todas las filas entre 1 y 400 m².
- R12. `ascensor` (P4, T2): `sí`, `si`, `s`, `true` y `1` pasan a 1, y `no`, `n`, `false` y `0` a 0, comparando después de normalizar (minúsculas, sin tildes, sin espacios a los lados). Los vacíos pasan a 0. Un valor que no esté en la lista también pasa a 0.
- R13. `parking` (P5, T3), normalizado igual: `no` pasa a 0, `opcional` a 1, y `si` e `incluido` a 2. Los vacíos y los valores desconocidos pasan a 0.
- R14. Desde E2, el estimador va dentro de `TransformedTargetRegressor(func=np.log10, inverse_func=potencia_de_10)`, donde `potencia_de_10(z)` devuelve `10**z` y es una función con nombre (no una `lambda`), para que `joblib` pueda guardar el modelo. Desde E2, las variables pasan por un `StandardScaler` antes del estimador. La tasación sale en euros y el MAE se mide en euros. E2 usa solo `superficie_construida` en bruto. T1-N1 añaden variables siguiendo la cadena de §4.

**Fase 3 · Variables nuevas**

- R15. `planta` se normaliza (minúsculas, sin tildes, sin espacios a los lados) y da 3 variables 0/1 (P3, N1):
  - `es_atico` = 1 si vale `atico` o `at`;
  - `es_bajo` = 1 si vale `bajo`, `bj`, `planta baja` o `0`;
  - `es_casa` = 1 si vale `casa`.

  Cualquier otro valor, `-` o un vacío dan 0 en las tres.

**Fase 4 · Modelo y explicación**

- R16. M1-M3 usan las variables de la última aceptada, con el mismo `StandardScaler` antes del estimador que E2-N1 (R14) y la misma rejilla de α y `l1_ratio` que §4. Son 35 filas en `experimentos.csv`, llamadas `m_ridge_<α>`, `m_lasso_<α>` y `m_enet_<α>_<l1_ratio>`, todas con tipo `estrategia`.
- R17. En cada familia gana la combinación con el MAE medio de validación cruzada más bajo. Si hay empate, la de α mayor. La mejor de las tres familias se compara con la última aceptada mediante R8. Si se acepta, es `MODELO_FINAL`; si no, lo es la regresión lineal de la última aceptada. `validacion` no se usa en ningún momento.
- R18. `explicar(modelo, X)` devuelve un `DataFrame` con una fila por fila de `X`. Sus columnas son `intercepto` más una por cada nombre de `get_feature_names_out()` del paso anterior al estimador. Cada valor es `coeficiente × valor transformado` (ya escalado). Sumando una fila sale la predicción en `log10` (CM5).
- R19. Ninguna tasación del test es ≤ 0 ni pasa de 6.000.000 €. Si alguna lo supera, `src.predecir` lo avisa. Además, `src.predecir` escribe `reports/envio_resumen.json` con `n_tasaciones`, `n_fuera_de_rango` (≤ 0 € o > 6.000.000 €), `minimo` y `maximo`, que el cuadro de mando usa para el aviso de *Rendimiento* (DESIGN §8).

**Cuadro de mando**

- R20. `src/app.py` tiene las pantallas Experimentos, Rendimiento, Dónde falla y Abrir la caja. Solo lee `reports/` y `models/`, no entrena.
- R21. Si falta un fichero que necesita una pantalla, esa pantalla dice cuál falta y con qué comando se genera. Nunca muestra una traza de error.

## 7. Criterios de aceptación

**Ejemplo resuelto** (tests en `tests/acceptance/test_ejemplo_resuelto.py`)

- CA1 (R1). Hay 6.237 filas de entrenamiento; cada una está en `1`-`5` o en `validacion`, que tiene entre el 18 % y el 22 % de las filas, y ningún inmueble (`ref_catastral`) aparece en dos particiones.
- CA2 (R2). Con tres ventas de ejemplo (barrio A: 100 m² por 300.000 € y 50 m² por 200.000 €; Badalona: 80 m² por 160.000 €), el baseline tasa un piso de 60 m² en A en 210.000 €, uno de Badalona sin superficie en 160.000 €, uno de 100 m² en un barrio sin ventas en 300.000 € (€/m² global) y uno de A sin superficie en 250.000 €.
- CA3 (R3). La validación cruzada devuelve cinco errores, y cambiar el precio de las filas de `validacion` no los cambia.
- CA4 (R4). Con precios reales 100, 200, 300 y 400 y predicciones 110, 190, 330 y 400, el MAE es 12,5, su intervalo lo contiene y repetir el cálculo da el mismo intervalo.

**Cómo decidimos**

- CA5 (R8). La referencia tiene un MAE de 100.000 € en las 5 particiones.
  - A: 98.000, 98.000, 98.000, 98.000 y 101.000 → mejora 1.400 € en 4 de 5 → **se acepta**.
  - B: 99.500 en las 5 → mejora 500 € en 5 de 5 → **se descarta**.
  - C: 97.000, 97.000, 97.000, 101.000 y 101.000 → mejora 1.400 € en 3 de 5 → **se descarta**.
  - D: 99.000, 99.000, 99.000, 99.000 y 100.000 → mejora 800 € → **se descarta**.
  - E: 98.750 en 4 particiones y 100.000 en 1 → mejora 1.000 € exactos en 4 de 5 → **se acepta**.
- CA6 (R8). En la cadena E2 → T1 (aceptada) → T2 (descartada) → T3, la referencia de T3 es T1.
- CA7 (R8). `experimentos.csv` tiene una fila por E0, E1, E2, T1, T2, T3 y N1, más las 35 de la fase 4, con las columnas de R8.

**Fase 1 · Limpieza**

- CA8 (R9). Con 6 filas inventadas:
  - vendida a 250.000 €;
  - retirada a 250.000 €;
  - vendida a 29.999 €;
  - vendida a 30.000 €;
  - vendida a 3.000.000 €;
  - vendida a 3.000.001 €.

  `limpiar_objetivo` deja la 1, la 4 y la 5, con el resto de columnas igual.
- CA9 (R9). Con `train_set.csv`, `limpiar_objetivo` deja 5.592 filas. Ninguna está retirada ni tiene un precio fuera de [30.000, 3.000.000].

**Fase 2 · Transformaciones**

- CA10 (R10). `PISO`, ` piso `, `Ático`, `Atico`, `Dúplex`, `Duplex`, `Chalet` y `Casa` pasan a `piso`, `piso`, `atico`, `atico`, `duplex`, `duplex`, `casa` y `casa`.
- CA11 (R11). Se entrena con pisos de 60, 80, 100, 0 y 900 m². Con los que tienen entre 1 y 400 m², la mediana es 80 y el percentil 99 es 99,6. Al transformar, los pisos de 0, 60, 100 y 900 m² quedan en 80, 60, 99,6 y 99,6.
- CA12 (R10, R11). Se entrena con una `Casa` de 200 m² y un `Chalet` de 300 m², así que la mediana de casa es 250 y el percentil 99 es 299. Al transformar, un `chalet` de 0 m² pasa a 250 y una `CASA` de 400 m² pasa a 299.
- CA13 (R11). Se entrena con las filas A y se transforman las filas B. Si se cambian las superficies de B, las medianas y los topes no cambian.
- CA14 (R11). Se ajusta con las 5.592 filas limpias. Los topes son: piso 196,0, ático 182,6, dúplex 253,3, casa 357,3 y estudio 60,9 m² (±0,1). Hay 73 filas que pasan al tope, y después de transformar ninguna fila tiene 0 m².
- CA15 (R12). `Sí`, `Si`, `SI`, `si`, `S`, `True` y `1` dan 1. `No`, `no`, `N`, `False`, `0` y el vacío dan 0.
- CA16 (R13). `No` da 0, `Opcional` da 1, `Sí` da 2, `Incluido` da 2 y el vacío da 0.
- CA17 (R14). Se entrena con superficies de 1, 2 y 3 y precios de 10, 100 y 1.000 €. Para una superficie de 4, la tasación es 10.000 € (±1 €).

**Fase 3 · Variables nuevas**

- CA18 (R15).

  | Valor | (`es_atico`, `es_bajo`, `es_casa`) |
  | --- | --- |
  | ` Ático `, `AT` | (1, 0, 0) |
  | `Planta baja`, `Bj`, `0` | (0, 1, 0) |
  | `Casa` | (0, 0, 1) |
  | `Pral.`, `Entlo`, `3º`, `-`, vacío | (0, 0, 0) |
- CA19 (R15). En las 5.592 filas limpias, hay 334 filas con `es_atico = 1`, 689 con `es_bajo = 1` y 59 con `es_casa = 1`.

**Fase 4 · Modelo y explicación**

- CA20 (R16). `experimentos.csv` tiene 7 filas `m_ridge_*`, 7 `m_lasso_*` y 21 `m_enet_*`.
- CA21 (R17). Con MAE de validación cruzada inventados, se elige el más bajo de cada familia. Si el mejor no pasa R8, `MODELO_FINAL` es la regresión lineal. Cambiar el precio de las filas de `validacion` no cambia la elección.
- CA22 (R18). Es un `Pipeline` con `StandardScaler` y `LinearRegression` sobre `log10(precio)`, entrenado con superficies 1, 2 y 3 y precios 10, 100 y 1.000. Para una superficie de 4, `explicar` da `intercepto` = 2 y `superficie_construida` = 2: suma 4, que es log10(10.000).
- CA23 (R18). Con el modelo final y 20 filas de `validacion`, las columnas son `intercepto` más los nombres de `get_feature_names_out()`. Cada fila suma `log10(predicción)` (±1e-9).
- CA24 (R19). Todas las tasaciones del test son mayores que 0 € y no pasan de 6.000.000 €.

**Cuadro de mando**

- CA25 (R20). Con `streamlit.testing.v1.AppTest`, la app arranca sin excepciones y tiene las 4 pantallas.
- CA26 (R21). Con `reports/` y `models/` vacíos, la app arranca sin excepciones y muestra un aviso que nombra el fichero que falta.
- Que los gráficos se entiendan no se puede comprobar con un test. Lo revisa una persona del equipo en cada pantalla, contra DESIGN.md, y lo anota en `Verificado por:`.

## 8. Fuera de alcance

- Modelos no lineales (árboles, *ensembles*, redes, k vecinos): llegan en las sesiones 14 y 15.
- Usar el test o sus precios públicos para entrenar o para elegir el modelo.

## 9. Preguntas abiertas

- Confirmar con el profesorado que el tipo `estrategia` de CM6 incluye transformaciones y modelos (decisión del 26/11/2026 en §10). Si no, solo cambian las etiquetas de `tipo`.

## 10. Decisiones

- 2026-11-25 · A partir de E2 se predice `log10(precio)` · E2, M1-M3 · Los precios están muy sesgados (de 30.000 € a 3.000.000 €). Con el log, el error es más proporcional al precio de cada vivienda y las rejillas de α de Lasso y ElasticNet tienen sentido (α_max ≈ 0,19 en log10 frente a ≈ 185.000 en euros).
- 2026-11-25 · El tope de superficie es el percentil 99 de cada tipo y se aplica a todas las superficies que lo superan · P6, T1 · Con "más de 400 m² → percentil 95", un piso de 390 m² se quedaba en 390 y uno de 410 m² pasaba a 150. El percentil se calcula sin los ceros ni las superficies de más de 400 m²: si no, los áticos de hasta 1.590 m² suben el tope de ático a 591 m². Afecta a 73 filas (1,3 %).
- 2026-11-25 · Grupos de `planta`: Ático = `atico`, `at`; Bajo = `bajo`, `bj`, `planta baja`, `0`; Casa = `casa`; el resto vale 0 · P3, N1 · `0` es la planta baja. Principal y entresuelo no son bajos: quedan como cualquier otra planta.
- 2026-11-26 · `inverse_func` es la función con nombre `potencia_de_10`, no una `lambda` · R14 · `joblib` no puede guardar una `lambda` y `src.entrenar` fallaría.
- 2026-11-26 · El tope de las tasaciones del test es 6.000.000 € también en CA24 · R19, CA24 · CA24 decía 10.000.000 € y no coincidía con R19 ni con DESIGN §8.
- 2026-11-26 · `StandardScaler` desde E2, no solo en M1-M3 · R14, R16 · Así el intercepto es la tasación de la «vivienda media» de DESIGN §6 sea cual sea el modelo final. A `LinearRegression` no le cambia las predicciones.
- 2026-11-26 · `predicciones_validacion.csv` incluye las columnas originales de cada vivienda · §5 · *Dónde falla* y *Abrir la caja* las necesitan, y el cuadro solo puede leer `reports/` y `models/`.
- 2026-11-26 · Tipos de `experimentos.csv`: `baseline`, `minimo`, `estrategia` y `caracteristica`. T1-T3 y M1-M3 pasan de `transformacion` y `modelo` a `estrategia` · §4, R16 · CM6 exige al menos 3 filas `estrategia` y 1 `caracteristica`; con los tipos anteriores no podía pasar. La fase sigue en el Id (T, N, `m_`).
- 2026-11-26 · La columna del Id en `experimentos.csv` se llama `estrategia`, no `id` · R8 · Es el nombre que exige el contrato del profesorado (CM6), que además pide `descripcion`.
- 2026-11-26 · `src.predecir` escribe `reports/envio_resumen.json` · R19, §5 · El aviso de tasaciones fuera de rango de DESIGN §8 va en *Rendimiento*, y el cuadro no lee `entregables/`.
