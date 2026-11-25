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
| P6 | `superficie_construida` | Hay 21 ceros y 17 registros con más de 400 m² (hasta 1.590 m² en pisos y áticos) | Mediana y tope por tipo, calculados con las filas de entrenamiento de ese tipo que tienen entre 1 y 400 m². Los ceros pasan a la mediana de su tipo. Toda superficie mayor que el percentil 99 de su tipo pasa a ese percentil (no solo las de más de 400 m²), así que no hay saltos: un piso de 390 m² y uno de 410 m² quedan igual. Con los datos limpios los topes son: piso 196, ático 183, dúplex 253, casa 357 y estudio 61 m²; afectan a 73 filas (1,3 %). | 2 · Transformación |
| P7 | `tipo` |  | Valores distinto para los mismos. Hay que normalizar. Chalet va con casa. Solo sirve para mediana y tope. | 2 · Transformación |

## 4. Estrategias y experimentos

**Regla de decisión** (escrita antes de ejecutar nada): una estrategia se acepta si baja el MAE de su referencia en al menos 4 de las 5 particiones y en al menos 1.000 € de media. Si no lo cumple, se descarta, y la estrategia siguiente se compara con la última aceptada. Si la mejora media es menor de 1.000 €, aunque sea en 5 de 5, se descarta: no compensa complicar el modelo.

Cómo leemos los resultados: `src.experimentos` da el MAE de cada partición (`mae_p1` … `mae_p5`). Comparamos cada estrategia con su referencia partición a partición, porque hay particiones más difíciles que otras para todas las estrategias a la vez.

**Variables:** a partir de E2, los modelos solo usan variables de §3: `superficie_construida`, `ascensor`, `parking` y las tres de `planta`. `tipo` no entra en el modelo: solo sirve para corregir la superficie (P6). La mediana y el percentil 99 por tipo se calculan dentro del `Pipeline`, solo con las particiones de entrenamiento de cada vuelta.

**Objetivo:** a partir de E2, todos los modelos predicen `log10(precio)` con `TransformedTargetRegressor` (`func=np.log10`, `inverse_func`: 10 elevado a la predicción). La tasación se devuelve en euros y el MAE se mide siempre en euros. E1 (ejemplo resuelto) sigue prediciendo el precio en euros.

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
| T1 | transformacion | Superficie corregida (P6, P7): `tipo` normalizado (Chalet → Casa); ceros → mediana de su tipo; por encima del percentil 99 de su tipo → ese percentil | E2 | Mejora mucho: con el log, las superficies de 400 a 1.600 m² y las de 0 m² tuercen la pendiente y disparan las tasaciones | | | |
| T2 | transformacion | Añade `ascensor` normalizado a 0/1, vacíos → 0 (P4) | Última aceptada | Mejora poco: el ascensor sube el precio, pero los 935 vacíos puestos a 0 añaden ruido | | | |
| T3 | transformacion | Añade `parking` normalizado a 0 (No) / 1 (Opcional) / 2 (Sí, Incluido), vacíos → 0 (P5) | Última aceptada | Mejora poco: con plaza incluida, la vivienda suele ser más cara | | | |

### Fase 3 · Variables nuevas

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1 | caracteristica | Añade `es_atico`, `es_bajo` y `es_casa` a partir de `planta` normalizada, con los grupos de P3; `-`, vacíos y el resto de plantas → 0 en las tres | Última aceptada | Mejora: un ático vale más y un bajo menos que el resto de plantas | | | |

### Fase 4 · Modelo

Las mismas variables que la última estrategia aceptada; solo cambia el estimador. Las variables se estandarizan (`StandardScaler`) dentro del `Pipeline` para que la penalización trate a todas por igual. En `experimentos.csv` hay una fila por combinación (35 en total: 7 Ridge, 7 Lasso y 21 ElasticNet): `m_ridge_<α>`, `m_lasso_<α>` y `m_enet_<α>_<l1_ratio>`. En esta tabla va una fila por familia, con el mejor valor.

**Cómo elegimos:** en cada familia, el valor con el MAE medio más bajo en la validación cruzada (nunca en `validacion`). El mejor de las tres familias se compara con la última aceptada usando la regla de decisión. Si ninguno la cumple, el modelo final es la regresión lineal de la última aceptada.

| Id | Tipo | Qué cambia | Referencia | Hipótesis | MAE medio ± sd | Mejora media · particiones | Conclusión |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | modelo | Ridge con α ∈ {0,1; 1; 10; 100; 1.000; 10.000; 100.000} | Última aceptada | Con 6 variables y unas 3.600 filas por vuelta no hay sobreajuste: hasta α = 100 da lo mismo que la regresión lineal y con α ≥ 10.000 empeora | | | |
| M2 | modelo | Lasso con α ∈ {0,0001; 0,0003; 0,001; 0,003; 0,01; 0,03; 0,1} | Última aceptada | Con α pequeños, igual que la regresión lineal. Al crecer, anula primero las variables débiles (ascensor, parking) y empeora poco a poco; hacia α ≈ 0,19 todos los coeficientes valen 0 | | | |
| M3 | modelo | ElasticNet con α ∈ {0,0001; 0,0003; 0,001; 0,003; 0,01; 0,03; 0,1} × l1_ratio ∈ {0,1; 0,5; 0,9} | Última aceptada | Entre Ridge y Lasso, sin mejorar a ninguno de los dos: con tan pocas variables, combinar las penalizaciones no aporta | | | |

## 5. Salidas

| Fichero | Qué contiene | Lo escribe |
| --- | --- | --- |
| `reports/experimentos.csv` | Una fila por estrategia con su MAE por partición | `src.experimentos` |
| `models/modelo.joblib` | El modelo final entrenado con las particiones 1-5 | `src.entrenar` |
| `reports/metricas.json` | MAE y RMSE del baseline y del modelo en `validacion`, con IC 95 % | `src.entrenar` |
| `reports/predicciones_validacion.csv` | Precio real, predicción del modelo y del baseline de cada vivienda de `validacion` | `src.entrenar` |
| `entregables/envio_kaggle.csv` | Predicción de cada vivienda del test | `src.predecir` |

## 6. Requisitos

<!--
Reglas numeradas, cada una verificable y sin ambigüedad. Cada problema de §3 y cada estrategia
de §4 debería acabar en un requisito que diga exactamente qué se hace.
Continuad la numeración del ejemplo resuelto (R8, R9…) y escribid los de cada fase antes de empezarla.
-->

**Ejemplo resuelto: datos y validación**

- R1. `cargar_entrenamiento()` devuelve las 6.237 filas de `train_set.csv` con la columna `particion` de `particiones.csv`; `cargar_test()` devuelve las 4.002 del test. No se limpia nada al cargar. Las filas que quita `limpiar_objetivo()` se definen en los requisitos de la fase 1.
- R2. Baseline: la tasación de una vivienda es el €/m² mediano de las operaciones de entrenamiento de su zona (el barrio en Barcelona; el municipio fuera de Barcelona o si falta el barrio) × su `superficie_construida`. Si la zona no tiene operaciones, el €/m² mediano global; si la vivienda no tiene superficie (vacía o 0), el precio mediano de su zona.
- R3. Validación cruzada: cada estrategia se entrena cinco veces, cada vez con cuatro de las particiones 1-5, y se mide el MAE en la que queda fuera. Las filas de `validacion` no se usan.
- R4. El error en `validacion` se da con un intervalo de confianza del 95 % por bootstrap percentil de las filas (2.000 remuestras, semilla 2026).
- R5. Modelo mínimo: regresión lineal con `superficie_construida`, `habitaciones`, `banos`, `distancia_centro_km` y `poblacion_zona`, sin transformar (E1). Se mantiene como referencia del ejemplo resuelto; la cadena de estrategias de §4 parte de E2.
- R6. El modelo final se entrena con las particiones 1-5 y se mide una vez en `validacion`, junto al baseline entrenado con las mismas filas.
- R7. El envío a Kaggle son las predicciones del modelo guardado para las 4.002 viviendas del test.

**Cómo decidimos**

<!-- La regla de decisión de §4, escrita como requisito: con mejora_media y particiones_mejor de experimentos.csv, ¿cuándo se acepta una estrategia? ¿Y si no cambia casi nada? -->

**Fase 1 · Limpieza**

<!-- Qué hace limpiar_objetivo() con cada fila (qué quita, qué corrige) y cómo se corrigen los demás valores erróneos. -->

**Fase 2 · Transformaciones**

<!-- Para cada variable: cómo se convierte, cómo se rellenan sus vacíos y qué pasa con un valor que no aparece en el entrenamiento. -->

**Fase 3 · Variables nuevas**

<!-- La fórmula de cada variable nueva, con qué hacer si le falta un dato. -->

**Fase 4 · Modelo y explicación**

<!-- Qué valores de λ probáis, cómo elegís el modelo final y qué devuelve explicar() (CM5). -->

**Cuadro de mando**

<!-- Las pantallas de DESIGN.md y qué hace la app si falta algún fichero. -->

## 7. Criterios de aceptación

<!--
Cada criterio (CA5, CA6…) dice qué requisito verifica y cómo se comprueba con una cifra o un hecho
concreto. Cada uno tendrá al menos un test en tests/acceptance/ marcado con
@pytest.mark.criterio("CA…"). scripts/trazabilidad.py lee los CA de esta sección.

Escribid los criterios de cada fase antes de empezarla: el agente escribe sus tests en una tarea propia,
vosotros los revisáis y solo después implementa la fase.

Tres tipos útiles:
- Con un ejemplo pequeño calculado a mano: "CA5 (R9). Con cinco filas inventadas (una vendida normal,
  una retirada…), limpiar_objetivo deja estas tres: …"
- Con los datos reales: "CA6 (R10). Tras corregir la superficie, ninguna vivienda tiene 0 m²."
- Sobre los experimentos: "CA9 (R8). experimentos.csv tiene una fila para cada estrategia de §4."

Lo que no se pueda testear (p. ej. si un gráfico se entiende) indicad cómo se verifica.
-->

**Ejemplo resuelto** (tests en `tests/acceptance/test_ejemplo_resuelto.py`)

- CA1 (R1). Hay 6.237 filas de entrenamiento; cada una está en `1`-`5` o en `validacion`, que tiene entre el 18 % y el 22 % de las filas, y ningún inmueble (`ref_catastral`) aparece en dos particiones.
- CA2 (R2). Con tres ventas de ejemplo (barrio A: 100 m² por 300.000 € y 50 m² por 200.000 €; Badalona: 80 m² por 160.000 €), el baseline tasa un piso de 60 m² en A en 210.000 €, uno de Badalona sin superficie en 160.000 €, uno de 100 m² en un barrio sin ventas en 300.000 € (€/m² global) y uno de A sin superficie en 250.000 €.
- CA3 (R3). La validación cruzada devuelve cinco errores, y cambiar el precio de las filas de `validacion` no los cambia.
- CA4 (R4). Con precios reales 100, 200, 300 y 400 y predicciones 110, 190, 330 y 400, el MAE es 12,5, su intervalo lo contiene y repetir el cálculo da el mismo intervalo.

## 8. Fuera de alcance

- Modelos no lineales (árboles, *ensembles*, redes, k vecinos): llegan en las sesiones 14 y 15.
- Usar el test o sus precios públicos para entrenar o para elegir el modelo.

## 9. Preguntas abiertas

- ¿Qué significa "≥3 de tipo estrategia" en CM6? Hay que preguntarlo al profesorado.

## 10. Decisiones

- 2026-11-25 · A partir de E2 se predice `log10(precio)` · E2, M1-M3 · Los precios están muy sesgados (de 30.000 € a 3.000.000 €). Con el log, el error es más proporcional al precio de cada vivienda y las rejillas de α de Lasso y ElasticNet tienen sentido (α_max ≈ 0,19 en log10 frente a ≈ 185.000 en euros).
- 2026-11-25 · El tope de superficie es el percentil 99 de cada tipo y se aplica a todas las superficies que lo superan · P6, T1 · Con "más de 400 m² → percentil 95", un piso de 390 m² se quedaba en 390 y uno de 410 m² pasaba a 150. El percentil se calcula sin los ceros ni las superficies de más de 400 m²: si no, los áticos de hasta 1.590 m² suben el tope de ático a 591 m². Afecta a 73 filas (1,3 %).
- 2026-11-25 · Grupos de `planta`: Ático = `atico`, `at`; Bajo = `bajo`, `bj`, `planta baja`, `0`; Casa = `casa`; el resto vale 0 · P3, N1 · `0` es la planta baja. Principal y entresuelo no son bajos: quedan como cualquier otra planta.
