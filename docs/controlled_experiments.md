# Controles VAN 2×2 — uso exclusivamente experimental

Estos controles no son el modelo final ni una réplica validada del paper. Su objeto
es separar el efecto de la representación del estado y de gamma sobre el mismo
problema. No se promueve automáticamente ninguna variante.

## Diseño fijo

| ID | Clave Q | Gamma |
| --- | --- | ---: |
| A_position_g07 | posición y período | 0,7 |
| B_position_g1 | posición y período | 1 |
| C_physical_g07 | estado físico completo | 0,7 |
| D_physical_g1 | estado físico completo | 1 |

En los cuatro casos se usa `SpatialConeEnv`, seis direcciones tanto al aprender como
al evaluar, recompensa incremental neta del cono, revisitas sin extracción adicional,
inicio superficial obligatorio, stop y horizonte finito. Se permiten costos negativos.
No se cambian precedencias ni se filtran acciones por los colores del Excel.

Defaults comunes: alpha=1, epsilon=0,3 constante, 100 episodios por semilla,
500 pasos como máximo y semillas 0…9. `--alpha`, `--epsilon`, `--episodes`,
`--max-steps` y `--seeds` cambian el presupuesto común, no una celda individual.
Gamma queda fijado por la matriz; `--gamma` y el selector normal `--model` no sirven
para elegir una variante experimental. El horizonte se incluye en la clave física:
no confundir este ensayo finito con una prueba de convergencia.

La clave física conserva extraídos, posición, período, uso de capacidad, terminal y
pasos restantes. Normaliza **únicamente la etiqueta de fase**, bajo validación de
dinámicas idénticas 6/6. Sin esto, las claves aprendidas con `learning` no coincidirían
con las consultadas con `construction`. Esta normalización no se aplica al modelo
normal ni se admite para un control 6/5.

La suite reducida requiere un período sin capacidad ni descuento económico. Otras
configuraciones económicas se rechazan en este protocolo, no se aproximan.

## Comandos

Desde la raíz del proyecto (o usando la ruta absoluta de `main.py`):

```bash
# Toda la suite sintética; no necesita datos privados.
python -B -m unittest discover -s tests -v

# Smoke: cuatro celdas × dos semillas. No acredita el hito de diez semillas.
python -B main.py experiments --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --episodes 5 --max-steps 100 --seeds 0 1

# Corrida de comparación ejecutada: cuatro celdas × diez semillas.
python -B main.py experiments --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --episodes 1000 --max-steps 500 --seeds 0 1 2 3 4 5 6 7 8 9

# Incluye también la integración privada existente.
OPENPIT_TEST_XLSX="$PWD/data/raw/Bloques105_SinRestricción.xlsx" python -B -m unittest discover -s tests -v
```

No se selecciona el primer Excel. Las rutas relativas se resuelven desde la raíz del
repositorio. `grid9-z-up` corresponde a la superficie Z=3 confirmada para este libro;
los bordes recortados continúan siendo una interpretación pendiente.

## Archivos e independencia

- `src/models/experimental_q_learning.py`: proyecciones de control. Reutiliza la
  actualización y RNG existentes; no se registra en el selector de modelos finales.
- `src/training/controlled_experiments.py`: generador de cuatro celdas y semillas;
  crea un modelo independiente en cada ejecución. No importa referencia ni benchmark.
- `src/evaluation/experiment_metrics.py`: observación pasiva de transiciones y
  visitas; no cambia premios, elecciones ni terminación.
- `src/evaluation/control_report.py`: análisis post hoc de historias guardadas,
  gaps, primer hit y diferencias pareadas descriptivas.
- `main.py experiments`: orquestación, validación del protocolo, exportación y
  comparación posterior. `train`, `reference` y `benchmark` siguen disponibles.

Se escribe un control/semilla por vez, sin retener cuarenta tablas Q. Se comprueba
que los modelos anteriores se liberan incluso sin recolección cíclica. Cada carpeta
`artifacts/reports/...-experimental-controls-.../` incluye:

- `report.json`: diseño, procedencia/hashes, estado running/completed/failed,
  ejecuciones esperadas/completadas, resultados y resumen.
- `summary.md`: tabla comparativa, expresamente rotulada como experimental.
- `ID/seed-N/training.json`: episodios completos, mejores prefijos y greedy final.
- `ID/seed-N/diagnostics.json`: métricas por episodio y greedy, definiciones y tiempos.
- `ID/seed-N/q-table.json`: tabla Q de ese control.
- `selected/`, `best-episode/`, `greedy/`: planes comprobados por el verificador,
  etiquetados como experimentales y no finales. La factibilidad geométrica continúa
  siendo condicional. `--plot` añade una curva SVG por ejecución.

Un error o una celda/semilla faltante deja la suite fallida. No se calcula una tasa
de éxito con un denominador parcial. Primero deben terminar las 4×N ejecuciones;
solo después se calcula el máximo cierre y se leen las selecciones visuales. Ningún
valor objetivo, camino testigo ni selección de referencia entra al agente.

## Definición de métricas

- Se separan mejor prefijo observado, mejor episodio completo y greedy completo.
- Gaps y hits se calculan después contra el óptimo independiente. El primer hit
  identifica fase, episodio, paso y transiciones de aprendizaje acumuladas. Los
  episodios no se detienen al alcanzarlo.
- El hito 1 requiere al menos diez semillas distintas, alguna coincidencia y una
  referencia compatible económica y factiblemente con el benchmark. Sin referencia
  válida o con menos semillas queda no evaluado, aunque se muestre el hit del benchmark.
- Una visita Q es el estado previo de una transición de aprendizaje, no cada consulta
  bootstrap. Se cuentan claves únicas y pares estado/acción únicos por separado.
- Reuso = `(visitas − únicos) / visitas`. Singleton = proporción de claves/pares
  visitados una sola vez. Denominadores cero se representan como `null`.
- Se distingue una fila compartida por estados físicos diferentes de una compartida
  por contextos mineros diferentes: el segundo ignora reloj y etiqueta de fase.
- Las visitas greedy desconocidas se cuentan por clave y por par clave/acción. Un
  valor Q aprendido igual a cero no es desconocido.
- Pasos sin extracción incluyen stop/avance; movimientos sin extracción los excluyen.
  Una extracción con valor cero sí es progreso.
- Un retorno de navegación vuelve a extraídos/posición/período/uso ya vistos en el
  tramo sin extracción, ignorando el reloj. No prueba un ciclo infinito del estado
  exacto, no invalida el plan y no poda el recorrido.
- Tiempo de generación: creación del entorno, aprendizaje, greedy, observación y
  verificaciones. Excluye agregación diagnóstica, exportación, gráficos, benchmark y
  reportes. Se separan tiempos de aprendizaje y greedy. Mismo presupuesto de episodios
  y horizonte no implica mismas transiciones ni tiempo.

## Resultados ejecutados — 2026-10-23

Datos originales sin modificar; 1.000 episodios por semilla, máximo 500 pasos,
semillas 0…9, alpha=1 y epsilon=0,3. Benchmark calculado: 336.000.

| Control | Máximo observado | Media observada | Máximo greedy | Media greedy | Hits observados |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 294.000 | 284.600 | 170.000 | 111.800 | 0/10 |
| B | 286.000 | 264.600 | 140.000 | 41.400 | 0/10 |
| C | 326.000 | 289.200 | 306.000 | 233.200 | 0/10 |
| D | 296.000 | 278.000 | 270.000 | 257.000 | 0/10 |

Artefactos: `artifacts/reports/20261023T140217-experimental-controls-02c11961/`.
Generación y verificación total: 49,768 s; 559.246 transiciones de aprendizaje.
Se reprodujeron y verificaron independientemente los 40 caminos seleccionados.

**El hito 1 no se alcanzó.** El mejor camino, C/semilla 2, contiene los 42 positivos
y el bloque 37 de −10.000 (43 bloques en total). No se eliminó ese bloque para reparar
la solución. La brecha restante es un costo observado de 10.000. El bloque 37 está
en un borde, fuera de los 53 candidatos coloreados; no se usó esa selección como
filtro para evitarlo.

### Qué muestran los controles (sin promover un ganador)

- A/B tienen reuso de claves ≈99,5%, pero filas compartidas por historias mineras
  distintas. Reuso alto no implica una representación adecuada.
- C/D no presentan esas colisiones, pero ≈83% de sus claves únicas se visitaron
  solo una vez. El ≈71% de reuso total se concentra en estados frecuentes; no
  significa que todos los estados tengan experiencia suficiente.
- En C/D, ≈97,5% de las visitas greedy consultaron claves sin actualizaciones de
  aprendizaje. La fase está normalizada y probada: esta falta de cobertura no es
  el error trivial de separar learning/construction.
- Los 40 greedy llegaron al horizonte. Es un síntoma de recorridos improductivos,
  no convergencia ni invalidez de las selecciones observadas.
- Gamma=1 no mejoró el máximo observado con este presupuesto. En física exacta sí
  aumentó la media del greedy (257.000 frente a 233.200). No concluir superioridad
  general de gamma=0,7 ni abandonar la alineación económica por este ensayo pequeño.

La lectura para retomar es separar cobertura, representación y horizonte antes de
ampliar presupuesto o introducir otras técnicas. No hubo promoción a modelo final,
cambios de recompensa, eliminación de negativos, VND ni Dyna-Q. El código exacto y
las versiones de cada corrida constan en su reporte; retoques posteriores de tipos,
metadatos y documentación no deben confundirse con reentrenamientos.
