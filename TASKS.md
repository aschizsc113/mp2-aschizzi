# TASKS · Tasación de viviendas para Banc Avellaner

> **La lista de tareas, en pasos pequeños.** El agente propone el desglose a partir de SPECS.md; el equipo lo aprueba.
> El agente marca `[x]` y añade la línea `Agente:` cuando los tests pasan. Una persona del equipo completa `Verificado por:`.

<!--
Formato de cada tarea (una tarea = algo que se implementa y verifica en una sola conversación):

- [ ] 1. Qué se hace, en una línea (R…, CA…)

Cuando el agente la termina:

- [x] 1. Qué se hace, en una línea (R…, CA…)
  - Agente: qué hizo y cómo lo comprobó (tests, fila de experimentos.csv).
  - Verificado por: nombre · cómo lo comprobó (diff leído, cifra recalculada a mano, pantalla revisada).
  - Fallos del agente: lo que hizo mal o fuera de SPECS y cómo lo detectasteis (si no hubo, "ninguno").

Pistas:
- Pedid el desglose de una fase cada vez, cuando su parte de SPECS.md esté escrita.
- La primera tarea de cada fase son los tests de sus criterios (CA…). Se revisan antes de implementar nada.
- En los experimentos, una tarea = una fila de SPECS.md §4.
-->

## Ejemplo resuelto

- [x] 0. Carga de datos, baseline, validación cruzada, modelo mínimo, entrenamiento y envío (R1-R7, CA1-CA4)
  - Agente: viene en la plantilla. `uv run pytest`: CA1-CA4, CM2, CM3, CM4 y CM7 en verde; CM1, CM5 y CM6 fallan.
  - Verificado por: profesorado · baseline recalculado a mano en CA2.

> **Pendiente del equipo:** confirmar con el profesorado el significado del tipo `estrategia` de CM6 (SPECS §9).

## Fase 1 · Limpieza (sesión 12)

- [x] 1. Tests de los criterios de limpieza en `tests/acceptance/test_limpieza.py` (CA8, CA9)
  - Agente: 2 tests con las 6 filas de CA8 y con `train_set.csv` (CA9). Escritos antes de implementar. Sin validar por el equipo: se implementó a continuación por indicación de seguir sin parar.
  - Verificado por: 
  - Fallos del agente: 
- [x] 2. `limpiar_objetivo()`: quita retiradas y precios fuera de [30.000, 3.000.000]; volver a ejecutar E0 y E1 (R9, CA8, CA9, CM7)
  - Agente: `limpiar_objetivo()` en `src/datos.py` con reglas fijas por fila. CA8, CA9 y CM7 en verde. Quedan 5.592 filas. E0 = 70.381 € y E1 = 123.353 € de MAE medio (antes, sin limpiar: 74.980 € y 127.934 €).
  - Verificado por: 
  - Fallos del agente: 

## Cómo decidimos (antes de la primera estrategia)

- [x] 3. Tests de la regla de decisión en `tests/acceptance/test_decision.py` (CA5, CA6)
  - Agente: tests de los casos A-E de CA5 (parametrizados) y de la cadena E2 → T1 → T2 → T3 de CA6.
  - Verificado por: 
  - Fallos del agente: 
- [x] 4. `src/decision.py` (referencia, `mejora_media`, `particiones_mejor`, `aceptada`) y sus columnas en `experimentos.csv`; Ids E0 y E1 en `ESTRATEGIAS`; `TIPOS` = `baseline`, `minimo`, `estrategia`, `caracteristica` (R8, SPECS §4, CA5, CA6)
  - Agente: `src/decision.py` con `comparar`, `ultima_aceptada` y `completar_decisiones`. `experimentos.csv` tiene `estrategia`, `tipo`, `descripcion`, `referencia`, `mae_p1`…`mae_p5`, `mae_medio`, `mae_sd`, `mejora_media`, `particiones_mejor` y `aceptada`. Ids E0/E1 y `TIPOS` del contrato. Cada `Estrategia` lleva `referencia` (SPECS) y `base` (qué variables amplía): `src.experimentos` avisa si no coinciden. CA5 y CA6 en verde.
  - Verificado por: 
  - Fallos del agente: 

## Fase 2 · Transformaciones (sesión 12)

- [x] 5. Tests de las transformaciones en `tests/acceptance/test_transformaciones.py` (CA10-CA17)
  - Agente: tests de CA10-CA17 con los ejemplos de SPECS (CA13: se ajusta con A y se comprueba que transformar B, o B cambiada, no cambia medianas ni topes).
  - Verificado por: 
  - Fallos del agente: 
- [x] 6. `src/preparacion.py`: `normalizar_texto`, `normalizar_tipo`, `CorrectorSuperficie`, `ascensor_a_binario`, `parking_a_ordinal`, `potencia_de_10` y `crear_preparacion` (R10-R13, CA10-CA16)
  - Agente: `src/preparacion.py` con todas las funciones de ARCHITECTURE §3. `crear_preparacion` recibe `superficie_bruta`, `superficie_corregida`, `ascensor`, `parking` y `planta`. CA10-CA16 en verde (topes: piso 196,0 · ático 182,6 · dúplex 253,3 · casa 357,3 · estudio 60,9; 73 filas al tope).
  - Verificado por: 
  - Fallos del agente: 
- [x] 7. Estrategia E2: `log10(precio)` solo con `superficie_construida`, con `StandardScaler` (R14, CA17)
  - Agente: E2 = `crear_pipeline([superficie_bruta], LinearRegression())`, con `StandardScaler` y `TransformedTargetRegressor(log10, potencia_de_10)`. CA17 en verde. Fila: `E2 · minimo · MAE 2.626.964 ± 4.944.457 €` (p1-p5: 133.891 · 11.427.930 · 1.279.310 · 147.527 · 146.161). Tasaciones absurdas en p2 y p3, como decía la hipótesis.
  - Verificado por: 
  - Fallos del agente: 
- [x] 8. Estrategia T1: superficie corregida por tipo (R10, R11) · referencia E2
  - Agente: T1 = superficie corregida. Fila: `T1 · ref E2 · MAE 128.572 ± 10.315 € · mejora 2.498.392 € · 5 de 5 · aceptada=True` (p1-p5: 124.856 · 139.741 · 113.226 · 135.710 · 129.330). Referencia E2: la de arriba.
  - Verificado por: 
  - Fallos del agente: 
- [x] 9. Estrategia T2: añade `ascensor` (R12) · referencia: última aceptada
  - Agente: T2 = T1 + ascensor. Fila: `T2 · ref T1 · MAE 126.297 ± 9.460 € · mejora 2.276 € · 5 de 5 · aceptada=True` (p1-p5: 122.540 · 136.190 · 112.081 · 132.706 · 127.966). Referencia T1: la de arriba.
  - Verificado por: 
  - Fallos del agente: 
- [x] 10. Estrategia T3: añade `parking` (R13) · referencia: última aceptada
  - Agente: T3 = T2 + parking. Fila: `T3 · ref T2 · MAE 126.295 ± 9.462 € · mejora 1 € · 3 de 5 · aceptada=False` (p1-p5: 122.588 · 136.151 · 112.054 · 132.759 · 127.925). Referencia T2: la de arriba.
  - Verificado por: 
  - Fallos del agente: 

## Fase 3 · Variables nuevas (entre sesiones)

- [x] 11. Tests de las variables nuevas en `tests/acceptance/test_variables_nuevas.py` (CA18, CA19)
  - Agente: tests de CA18 (cada valor de la tabla, parametrizado) y CA19 (334 / 689 / 59).
  - Verificado por: 
  - Fallos del agente: 
- [x] 12. `planta_a_indicadores()` en `src/preparacion.py` (R15, CA18, CA19)
  - Agente: `planta_a_indicadores()` con los grupos de R15; nombres de salida `es_atico`, `es_bajo`, `es_casa`. CA18 y CA19 en verde.
  - Verificado por: 
  - Fallos del agente: 
- [x] 13. Estrategia N1: añade `es_atico`, `es_bajo` y `es_casa` (R15) · referencia: última aceptada
  - Agente: N1 = T2 (última aceptada; T3 descartada) + planta. Fila: `N1 · caracteristica · ref T2 · MAE 124.715 ± 9.978 € · mejora 1.582 € · 4 de 5 · aceptada=True` (p1-p5: 120.390 · 136.467 · 110.351 · 130.538 · 125.827).
  - Verificado por: 
  - Fallos del agente: 

## Fase 4 · Modelo y explicación (sesión 13)

- [x] 14. Tests del modelo en `tests/acceptance/test_modelo.py` (CA7, CA20-CA24)
  - Agente: tests de CA7, CA20, CA21 (tabla inventada con empate de α, caso en que no pasa R8, datos pequeños cambiando el precio de `validacion`, y `MODELO_FINAL` = lo que dice R17 con `experimentos.csv`), CA22 (a mano), CA23 y CA24.
  - Verificado por: 
  - Fallos del agente: 
- [x] 15. Estrategias M1-M3: las 35 combinaciones de Ridge, Lasso y ElasticNet (R16, CA7, CA20)
  - Agente: 35 estrategias `m_ridge_<α>`, `m_lasso_<α>`, `m_enet_<α>_<l1_ratio>`, tipo `estrategia`, con las variables de N1. CA7 y CA20 en verde. Mejor de cada familia (ref N1): `m_ridge_1000 · 123.621 ± 5.218 € · mejora 1.094 € · 3 de 5 · False`; `m_lasso_0.01 · 124.334 ± 8.475 € · mejora 381 € · 2 de 5 · False`; `m_enet_0.1_0.1 · 123.459 ± 6.707 € · mejora 1.256 € · 4 de 5 · True` (p1-p5: 119.309 · 128.432 · 113.905 · 129.900 · 125.749).
  - Verificado por: 
  - Fallos del agente: 
- [x] 16. Elección de la mejor de cada familia y del modelo final en `src/decision.py` (R17, CA21)
  - Agente: `mejor_por_familia` (empate → α mayor) y `elegir_modelo_final` en `src/decision.py`. CA21 en verde. R17 elige `m_enet_0.1_0.1`.
  - Verificado por: 
  - Fallos del agente: 
- [x] 17. Fijar `MODELO_FINAL` cuando el equipo haya revisado los resultados; `src.entrenar` guarda las columnas de `X_val` en `predicciones_validacion.csv`; `src.predecir` avisa y escribe `reports/envio_resumen.json` (R6, R7, R17, R19, CA24, CM1-CM4, CM6)
  - Agente: `MODELO_FINAL = "m_enet_0.1_0.1"` (lo que dice R17; **pendiente de revisión del equipo**). `src.entrenar` guarda también las columnas de `X_val`; `src.predecir` escribe `envio_resumen.json` y avisa. En `validacion`: modelo 116.719 € (IC 109.337-124.812), ingenua 71.518 € (IC 61.263-83.827). Test: de 141.424 a 4.058.751 €, 0 fuera de rango. CM1-CM7 y CA24 en verde.
  - Verificado por: 
  - Fallos del agente: 
- [x] 18. `src/explicar.py`: `explicar()` y `coeficientes()` (R18, CA22, CA23, CM5)
  - Agente: `explicar()` (intercepto + coef × valor escalado) y `coeficientes()` (por desviación típica y por unidad, con su factor 10^coef). CA22, CA23 y CM5 en verde.
  - Verificado por: 
  - Fallos del agente: 

## Cuadro de mando (entre sesiones y sesión 13)

- [x] 19. Tests del cuadro en `tests/acceptance/test_cuadro.py` (CA25, CA26)
  - Agente: CA25 (arranca, radio con las 4 pantallas, cada una sin excepción ni `st.error`) y CA26 (`reports/` y `models/` vacíos con `monkeypatch`: cada pantalla nombra el fichero que falta). Navegación con `st.sidebar.radio`: `AppTest` no puede cambiar de página con `st.navigation` de funciones.
  - Verificado por: 
  - Fallos del agente: 
- [x] 20. `src/formato.py`: euros, %, intervalos, factores, superficies, fechas y `—` (DESIGN §2), con tests unitarios
  - Agente: `src/formato.py` con `euros`, `porcentaje`, `intervalo`, `factor`, `coeficiente_tecnico`, `superficie`, `fecha` y `SIN_DATO`. 6 tests unitarios en `tests/unit/test_formato.py`.
  - Verificado por: 
  - Fallos del agente: 
- [x] 21. `src/analisis.py`: cifras y frases de cabecera de cada pantalla (DESIGN §1, §3, §4), con tests unitarios
  - Agente: `src/analisis.py` con las cifras, títulos y frases de cada pantalla (dos sentidos), grupos con menos de 20 viviendas, desglose en cascada y comprobación `metricas.json` ↔ `experimentos.csv`. 8 tests unitarios en `tests/unit/test_analisis.py`.
  - Verificado por: 
  - Fallos del agente: 
- [x] 22. `src/graficos.py`: una función por gráfico, con los colores y la plantilla de DESIGN §3, §5 y §6
  - Agente: `src/graficos.py` con una función por gráfico: `simple_white`, `separators=",."`, azul/gris/naranja fijos, mapa `carto-positron` cortado en ±50 % y cascada `go.Waterfall`.
  - Verificado por: 
  - Fallos del agente: 
- [x] 23. Pantalla *Experimentos* (R20, DESIGN §4, §7)
  - Agente: frase, regla, barras (E0 como línea), tabla, detalle por partición (por defecto la última aceptada), detalle de λ y casilla de las 35. Revisada con `AppTest`: sin errores.
  - Verificado por: 
  - Fallos del agente: 
- [x] 24. Pantalla *Rendimiento*, con el aviso de R19 leído de `envio_resumen.json` (R20, DESIGN §4, §8)
  - Agente: frase con márgenes, 3 tarjetas (diferencia en naranja si empeora), puntos con RMSE explicado en el *tooltip*, frase de error en %, histograma con bandas y desplegable real frente a tasación. Aviso de R19 desde `envio_resumen.json`.
  - Verificado por: 
  - Fallos del agente: 
- [x] 25. Pantalla *Dónde falla*, con filtros y mapa de residuos (R20, DESIGN §4, §5, §7)
  - Agente: filtros de tipo, zona y rango, «N viviendas de 1.123», frase, error y sesgo por tipo, por zona y por rango, mapa con selector modelo/ingenua y tabla de las 20 peores. Comprobado el filtro «Estudio» (64 viviendas) y el mapa de la ingenua.
  - Verificado por: 
  - Fallos del agente: 
- [x] 26. Pantalla *Abrir la caja*: coeficientes, desglose en cascada y aviso causal (R20, DESIGN §4, §6)
  - Agente: aviso causal fijo, coeficientes (factor y lectura en su unidad en el *tooltip*), selector de vivienda con las 4 opciones, cascada en euros con el precio real y desplegable «Qué no podemos afirmar». Probadas las 4 opciones con `AppTest`.
  - Verificado por: 
  - Fallos del agente: 
- [x] 27. Ficheros que faltan y casos límite de DESIGN §8 (R21, CA25, CA26)
  - Agente: cada fichero que falta da `st.info` con su ruta y comando; filtros vacíos → `st.warning`; sin fase 4 → «Aún no se han ejecutado M1-M3»; modelo no explicable → «Vuelve a entrenar»; resultados de ejecuciones distintas → aviso; cualquier otro fallo → `st.error` sin traza. CA25 y CA26 en verde.
  - Verificado por: 
  - Fallos del agente: 
