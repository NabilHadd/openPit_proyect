# Continuidad — validación por etapas (2026-10-23)

## Trabajo posterior a los cuatro controles

El usuario guardó y publicó los controles anteriores en `8e20087`. Después pidió:
validar el entorno; resolver una mina diminuta por enumeración y aprendizaje;
revisar empates/Stop/truncamientos/cobertura; y consolidar gamma=1 con diez semillas
y presupuestos iguales de transiciones. No convertirlo automáticamente en modelo final.

- Nuevo CLI aislado: `python -B -m src.validation --transitions 10000`.
- Pruebas exhaustivas del contrato físico con capacidad, períodos y descuento.
  Se comprobaron también 20.684 transiciones reales guardadas, sin duplicación,
  infracción de precedencias ni discrepancia entre recompensa y VAN.
- Oráculo independiente de seis bloques: óptimo alcanzable 15, requiere pérdida
  intermedia −5 y evitar pérdidas opcionales. No entra al agente.
- Con horizonte finito 12, semillas 0…9 y 10.000 transiciones por semilla: **200/200
  rollouts finales óptimos**; el test automatizado también lo exige en cuatro bloques.
- Desempate uniforme de evaluación con RNG separado; mejor observado solo de
  aprendizaje, mejor episodio completo y distribución de políticas separados.
- Horizonte `finite` sigue predeterminado. `external` usa reloj estacionario y
  truncamientos con bootstrap; **no son el mismo problema**. Un piloto external con
  límite 12 no reprodujo el óptimo en todos los rollouts: ciclos gratuitos gamma=1,
  no una prueba superada que deba escalarse sin más.
- Protocolo, artefactos y comandos: [`learning_validation.md`](learning_validation.md).
- Comparación real terminada: 100.000 transiciones por semilla 0…9, gamma=1, alpha=1,
  epsilon=0,3, horizonte finito 500 y 20 evaluaciones por semilla. **336.000 observado
  en 4/10 semillas** (2,4,7,9); episodios completos óptimos en 3/10; política final
  30/200 aciertos, solo semillas 2 (16/20) y 9 (14/20). Media 305.250. Ninguna semilla
  reproduce el óptimo en todos sus rollouts. Los 200 terminan por Stop, no horizonte.
- Reproducidos y verificados 10 mejores prefijos y 200 políticas; los 34 óptimos contienen
  los 42 positivos sin negativos. Referencia revalidada post hoc; Excel original intacto.
  Artefactos: `artifacts/reports/20261023T193418-learning-validation-aba1d7d5/`.
- Ablación con las mismas Q: ordered agota 500 pasos en 10/10, media 315.000; uniform
  media 305.250 y 200/200 Stop. Corregir sesgo de orden no asegura mejorar VAN.
  Los Stop uniformes siempre fueron empates; no atribuirlos sin más a parada aprendida.
- La revisión independiente detectó y motivó dos regresiones nuevas: el oráculo
  external no puede usar el corte del recolector como horizonte del problema, y un
  rollout truncado con valor óptimo no puede pasar la puerta de política completada.
- Suite ampliada: 85 pruebas en total, con integración privada opcional. Comando:
  `python -B -m unittest discover -s tests -v`. Las nuevas funciones no cambian los
  defaults de `main.py`, ni recompensas, negativos o acciones de los controles guardados.
  Con el Excel habilitado pasaron 85/85; inspecciones IDE sin errores y Graphify actualizado.
  Puerta diminuta estricta reejecutada: `20261023T194311-learning-validation-e2e0c9b1`,
  200/200 políticas completadas óptimas y las diez Q idénticas a la corrida inicial.

## Registro anterior — controles experimentales

## Rama y último trabajo

- Se revisó `main` limpio, se hizo `git fetch origin main` y se comprobó 0/0 commits
  de diferencia. No había cambios que integrar. Se creó `optimizacion-van` desde
  `eb7cf86`, sin sobrescribir cambios ni hacer push/commit.
- El usuario aprobó cuatro controles **solo experimentales**, no modelo final:
  posición/física exacta × gamma 0,7/1, mismo entorno de conos con seis direcciones
  en ambas fases. Prioridad: hito 1; se mantienen las métricas de diagnóstico.
- Nuevo modo `main.py experiments`, modelos aislados en
  `src/models/experimental_q_learning.py`, generador por control/semilla y métricas
  pasivas. La clave exacta conserva horizonte y normaliza solo fase bajo paridad 6/6.
- Pruebas de encodings, métricas calculables a mano, independencia Q/RNG, orden
  post hoc de referencia/benchmark, suite incompleta y exportación no final.
  Suite ampliada: 65 pruebas sintéticas pasan y una integración privada opcional;
  habilitando el Excel real pasan **66/66**. Revisión independiente estática sin
  hallazgos accionables. Se comprobó liberación de modelos sin GC cíclico.
- Corridas reales: smoke 5 episodios × 2 semillas × 4; 100 episodios × 10 × 4;
  después 1.000 episodios × 10 × 4. En la última, máximos A=294.000, B=286.000,
  C=326.000, D=296.000. **Ningún control alcanzó el hito 1**.
- El máximo C/semilla2 contiene los 42 positivos y el bloque 37 de −10.000. Se
  reprodujeron los 40 caminos seleccionados y se verificaron independientemente.
  El Excel original conserva su hash.
- La instrumentación distingue aliasing de falta de cobertura: claves compactas
  muy reutilizadas pero inconsistentes; físicas sin colisiones pero ~83% de claves
  únicas solo visitadas una vez y ~97,5% de visitas greedy a claves no aprendidas.
  Todos los greedy de estos controles terminaron por límite de pasos.
- Protocolo, comandos y resultados completos:
  [`controlled_experiments.md`](controlled_experiments.md). Artefactos principales:
  `artifacts/reports/20261023T140217-experimental-controls-02c11961/`.
- Las corridas normales se conservan; no se añadió VND, Dyna-Q ni dependencias.
  No usar la ruta diagnóstica de 336.000 para inicializar o guiar Q.

## Registro anterior — actualización con paper (2026-10-17 UTC)

## Últimas confirmaciones y selección del camino

El usuario confirmó superficie del Excel en Z=3, recompensa **B incremental**, revisitas
sin nueva extracción y conservación del camino observado de mayor valor, admitiendo
pérdidas intermedias. No hace falta volver a pedir estas definiciones. Conservar el
mejor camino no significa conocer el óptimo ni define por sí solo cuándo acaba un
episodio; no se cambiaron stop/todos/presupuesto ni las actualizaciones Q.

Se añadió un observador de prefijos factibles. Conserva el mayor VAN realmente
observado durante learning o construction, con episodio/fase/pasos/acciones/plan.
Los episodios completos y greedy completos siguen separados. La selección nueva
se exporta, tras verificarla, a `seed-N/selected/conditional-plan.json`. Un prefijo
puede terminar antes de una pérdida posterior sin fingir que el episodio real
terminó allí. No se agrega un camino vacío contrafactual antes del inicio obligatorio.

Reejecución real: 1.000 episodios × diez semillas 0…9, mismos parámetros 1/0,7/0,3
y horizonte 500. Máximo seleccionado **294.000**, media **284.600**; mejor episodio
completo sigue en 286.000 y mejor greedy completo en 240.000. **0/10 alcanza 336.000**.
Artefactos: `artifacts/reports/20261017T020932-train-68af937e/`. Generación/verificación
de episodios y prefijos: 10,349 s; comparar tiempos teniendo en cuenta ese trabajo extra.

Las diez tablas Q son idénticas byte por byte a las previas. Los campos originales
de los 10.000 episodios y los planes greedy tampoco cambiaron: la mejora corresponde
a conservar un camino observado, no a aprendizaje nuevo. Se reprodujeron exactamente
las diez rutas seleccionadas y se verificaron sus planes independientemente. El Excel
original conserva su hash. Pruebas nuevas cubren costos negativos, pérdidas posteriores,
empates, trayectoria negativa, episodios vacíos, greedy, descuento, independencia de
Q/RNG y correspondencia de exportación. Revisión estática independiente sin hallazgos.

## Estado actual

El usuario aportó el artículo de Quelopana/Navarra 2021 y autorizó intentar la
variante espacial. Se implementaron `spatial-cone` (conos atómicos, seis direcciones
al aprender y cinco al construir), Q compacta posición/período, inicio superficial
contabilizado, revisitas sin doble valor y presupuesto explícito. El baseline
`full-state` se conserva por defecto. No hay Dyna-Q ni VND.

Correspondencia, decisiones y dudas: [`paper_mapping.md`](paper_mapping.md).
No se afirma réplica exacta: estado Q, bordes, terminación operativa, movimientos por
fase y avance de período todavía requieren confirmación. El paper numera Z=1
superficial, pero el usuario confirmó Z=3 en el Excel.

### Validación nueva

- 48 pruebas sintéticas aprobadas sin datos privados; con integración privada,
  **49/49 aprobadas**. Benchmark contrastado con enumeración exhaustiva en 30 DAGs pequeños.
- Máximo cierre por min-cut racional, independiente de referencia y aprendizaje:
  Z máximo superficial → 336.000 / 42 bloques; Z mínimo → 8.000 / 1 bloque.
  Son óptimos bajo las reglas elegidas, no certificación de la geometría del cliente.
- Los 53 candidatos coinciden con conos completamente contenidos en la grilla:
  35 en Z=3, 15 en Z=2, 3 en Z=1. Solo diagnóstico, no máscara del agente.
- Se comprobó independencia al cambiar colores/valor declarado y habilitar benchmark:
  historial y tabla Q idénticos. El Excel original conserva el hash indicado abajo.
- Reevaluación CLI de los planes guardados: 286.000 (mejor episodio) y 240.000
  (greedy), ambos factibles condicionalmente. Replay del baseline original, semilla
  0: tabla Q y valores mejor/greedy idénticos a los artefactos de la primera versión.
- `compileall` y `git diff --check` sin errores; índice Graphify actualizado.

### Experimentos reales ejecutados con parámetros publicados (1 / 0,7 / 0,3)

Semillas 0…9; mismo Excel, sin capacidad/descuento. Horizonte espacial 500 pasos.
Son comparaciones por episodios, no igual presupuesto de transiciones/tiempo;
una acción espacial puede extraer varios bloques.

| Variante | Episodios/semilla | Máx. mejor episodio | Media mejores | Máx. greedy | Media greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-state | 100 | 198.000 | 191.000 | −294.000 | −294.000 |
| Spatial, construcción 5 direcciones | 100 | 280.000 | 246.800 | 230.000 | 130.000 |
| Spatial, construcción 5 direcciones | 1.000 | 286.000 | 281.200 | 240.000 | 157.600 |
| Spatial, extensión construcción 6 direcciones | 1.000 | 286.000 | 281.200 | 170.000 | 111.800 |

**Ninguna corrida RL alcanzó 336.000.** El benchmark sí lo calcula, pero no es un
resultado aprendido y no se utiliza para seleccionar acciones, premiar o detener.
Todos los planes generados pasan el verificador con factibilidad condicional.
La política greedy de 5 direcciones terminó por presupuesto en 7/10 semillas de
1.000 episodios; la extensión de 6, en 10/10. Son ciclos/tránsito sin progreso,
**no convergencia**. Habilitar subir no solucionó el problema de política compacta.

Artefactos locales (configuración, huella de código, planes, recorridos, Q y tiempos):

- Baseline 100: `artifacts/reports/20261017T013559-train-a7301a97/`.
- Spatial 100: `artifacts/reports/20261017T013539-train-1e45edcd/`.
- Spatial 1.000: `artifacts/reports/20261017T013619-train-67ddf2c6/`.
- Extensión up: `artifacts/reports/20261017T013951-train-f212d7b4/`.

La ablation up conserva historiales de aprendizaje y tablas Q idénticos en las diez
semillas; solo cambia la máscara del rollout final. Generación/verificación: 0,849 s,
0,741 s, 7,836 s y 8,030 s respectivamente; excluye benchmark y serialización.
No son benchmarks comparables al tiempo industrial del paper.

Próximo paso: contrastar el estado Q original, los bordes y la regla de
construcción/terminación. No ajustar la política usando los 42 bloques
de `Resultado`. Mantener la diferencia entre mejor episodio y política greedy.
Revisión independiente estática de la nueva lógica: sin hallazgos accionables.
No se añadieron dependencias ni se hizo commit/push.

## Registro histórico — primera versión (2026-10-16)

## Alcance implementado

Se conservó la estructura inicial y se implementaron carga/selecciones, verificador,
entorno determinista, Q-learning tabular **experimental**, entrenamiento, CLI y
artefactos JSON/SVG reproducibles. `losses.py` sigue vacío: no hace falta para una
actualización tabular. No se implementó Dyna-Q ni VND. No se hizo commit ni push.

## Hechos observados

- Git estaba limpio; los módulos, dependencias y CI estaban vacíos.
- Solo existe `data/raw/Bloques105_SinRestricción.xlsx` (acento Unicode descompuesto),
  no la copia revisada `(1)`. No hay paper local. Una búsqueda web no recuperó uno.
- SHA-256 del libro: `60f7b45b25245fda551112f74b174428aef5c2040e4fc5ebe95e913155528aa6`.
- Encabezados exactos ID/X/Y/Z/Valor; 105 IDs y coordenadas únicos; X=1…7,
  Y=1…5, Z=1…3. Los valores están en unidades económicas no declaradas.
- Se recuperan 53 candidatos y 42 seleccionados. La referencia suma 336.000 desde
  la tabla, coincidiendo con `Resultado!L13`. Su color es de tema (índice de tema 8),
  no RGB directo; el lector resuelve el tema, no estilos internos.
- La referencia es cerrada bajo la hipótesis geométrica `grid9-z-up` y admite un
  orden testigo en el mismo período. **Esto es factibilidad condicional**, no
  validación de las reglas del cliente ni certificación de optimalidad.

## Decisiones técnicas (no hechos del paper)

- Sin reglas, solo comprobación económica y `feasible=null`; entrenar exige una
  opción explícita de precedencias.
- Hipótesis opcional: superficie en Z máximo, predecesores 3×3 en Z+1, bordes
  recortados. Alternativa: archivo JSON de precedencias con fuente y todos los IDs.
- Plan ordenado; precedencia permite el mismo período con orden anterior.
- Estado físico completo, extraer/avanzar/parar, descuento desde período 0, rechazo
  atómico de acciones inválidas, epsilon constante. Todo es experimental hasta
  contrastarlo con el paper. Los 53 candidatos no restringen el modelo.
- Referencia aislada del entrenamiento (leída después de terminar las semillas).
- Mejor episodio y greedy final se reportan por separado; ambos se verifican.
- Exportación interna documentada, no formato del cliente; nunca se rotula como
  validación minera definitiva.

## Verificación y experimentos

- `python -m unittest discover -s tests -v`: 25 pruebas sintéticas aprobadas y
  una integración privada omitida.
- `OPENPIT_TEST_XLSX="$PWD/data/raw/Bloques105_SinRestricción.xlsx" python -m unittest discover -s tests -v`:
  26/26 aprobadas, incluyendo integración real y hash del original intacto.
- `python -m compileall -q main.py src tests` y `git diff --check`: sin errores.
- Revisión independiente estática: detectó rechazo no controlado de JSON de tipo
  incorrecto; corregido con pruebas CLI para listas/null en planes y precedencias.
- CLI `reference` sin reglas: VAN 336.000, `feasible=null`, sin exportación como
  plan comprobado. Con `--precedence grid9-z-up`: VAN 336.000, factibilidad condicional.
- Smoke real: 5 episodios, semilla 0, alpha=0,2 / gamma=1 / epsilon=0,2;
  mejor episodio 140.000 y greedy final −294.000. SVG generado.
- Experimento real **ejecutado**, no solo habilitado: 100 episodios × semillas
  0…9, mismos hiperparámetros; sin capacidad ni descuento, precedencias provisionales.

| Semilla | Mejor episodio (VAN) | Greedy final (VAN) |
| --- | ---: | ---: |
| 0 | 200.000 | −294.000 |
| 1 | 182.000 | −294.000 |
| 2 | 204.000 | −294.000 |
| 3 | 186.000 | −294.000 |
| 4 | 200.000 | −294.000 |
| 5 | 190.000 | −294.000 |
| 6 | 200.000 | −294.000 |
| 7 | 188.000 | −294.000 |
| 8 | 196.000 | −294.000 |
| 9 | 214.000 | −294.000 |

**No se alcanzó 336.000 en ninguna ejecución (0/10)**. Mejor observado 214.000;
media de mejores episodios 196.000. Todos los planes pasan el verificador bajo
la hipótesis elegida. El greedy final puede ir por estados no visitados, con
valores Q cero y desempate determinista: esos planes pobres son resultados reales,
no se sustituyen por la referencia ni se ocultan. No demuestra eficacia del método
de Quelopana, que todavía no se replicó.

Se repitió la corrida de diez semillas tras las correcciones finales: todos los
planes, historiales y tablas Q son idénticos byte por byte. Artefactos más recientes:
`artifacts/reports/20261016T190601-train-244ecb94/`. Tiempo total de generación y
verificación: 0,946 s; no incluye carga/serialización/gráficos/reportes. Estos
tiempos cortos son diagnósticos locales, no un benchmark. Python 3.13.12,
openpyxl 3.1.5, commit base `0861507154bc4b5b05ada59a4a10fffdf4b7bbd9` con cambios
locales y huella por archivo en cada reporte. Revisar los reportes para la versión
exacta utilizada: los cambios de documentación posteriores tienen otra huella.

La CLI `evaluate-plan` reevaluó el plan exportado de la semilla 9: VAN 214.000,
factibilidad condicional, sin errores. Se conservó el hash original del Excel.

Se configuró CI sintético, pero no se ejecutó GitHub Actions ni se creó un entorno
Conda nuevo. `graphify update .` generó el índice local (ignorado por Git).

## Insumos que pedir a Aldo

1. Paper completo y/o pseudocódigo exacto: estados de la matriz Q, acciones,
   recompensa, exploración, inicialización, actualización y terminación.
2. Copia revisada `(1)` y confirmación de si reemplaza el libro local (comparar hash).
3. Orientación de Z, vecindad/talud y tratamiento de bordes; reglas de precedencia
   dentro de un período y transición entre períodos.
4. Significado exacto de los 53 candidatos y su relación con estados/acciones.
5. Tonelajes, unidades monetarias, convenciones de capacidad y descuento para
   escenarios extendidos. El caso actual no permite inferir toneladas.
6. Especificación del formato de solución/plan del cliente.

No presentar la línea base genérica como réplica. Primero incorporar esas reglas,
validar el caso reducido y luego evaluar diez o más semillas bajo el método real.
