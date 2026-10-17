# Quelopana y Navarra (2021): correspondencia e interpretaciones

Fuente: *Integration of strategic open-pit mine planning into hierarchical artificial
intelligence*, JSAIMM 121(12), pp. 643–652. DOI `10.17159/2411-9717/1367/2021`.
El usuario aportó el PDF y su texto en la conversación. No hay una copia PDF local
que se haya versionado o cuyo hash se pueda registrar.

## Confirmaciones posteriores del usuario

- La superficie del Excel reducido es **Z=3**.
- La recompensa corresponde a **B: valor incremental neto de los bloques nuevos
  extraídos por la acción**, no un premio terminal. Se admiten pérdidas intermedias.
- Una revisita desplaza al agente sin extraer ni contabilizar bloques nuevamente.
- Se conserva el camino observado de mayor valor. Esto define la selección de la
  solución, **no una condición que requiera conocer el óptimo antes de aprender**.

Estas confirmaciones proceden de la conversación, no se atribuyen como fórmulas
literales al artículo. Cierre natural de episodio, estado Q exacto y bordes siguen
pendientes. Se mantienen stop/todos extraídos/presupuesto como reglas operativas
explícitas; no se inventó un criterio de detenerse ante el primer movimiento negativo.

## Qué está documentado y qué se implementó

| Evidencia | Código | Límite |
| --- | --- | --- |
| p. 647, Fig. 4: nueve superiores; t' ≤ t | `transforms.py`, verificador independiente | Superficie Z=3 confirmada por usuario; recorte de bordes provisional |
| p. 648, Fig. 7: seis movimientos de exploración | `SpatialConeEnv`, fase learning | Distinción por fase es interpretación; no confundir recorrido con plan |
| p. 649: cinco movimientos al construir, sin subir | fase construction | Se puede habilitar up como extensión explícita; no se atribuye al paper |
| p. 649: inicio en bloque superficial de mayor valor | acción obligatoria start | Empates por ID es decisión técnica |
| p. 649: destino más todos sus predecesores | cierre transitivo y extracción atómica | Recompensa incremental confirmada por usuario; manejo de capacidad aún interpretado |
| pp. 648–649, Ec. 3–4 | actualización Q estándar y epsilon-greedy | Se conservan actualización terminal y máscara física |
| pp. 649–650: α=1, γ=0,7, ε=0,3; Q=0 | defaults de spatial-cone | No se trasladan capacidades/tasas/tonelajes industriales al Excel reducido |

Se mantiene `full-state` para reproducir la primera línea base. No se implementó
VND ni Dyna-Q, aunque el artículo incluya VND como refinamiento posterior.

## Estado físico frente a tabla Q

`SpatialState` es inmutable e incluye: extraídos, período, capacidad usada, posición,
fase y presupuesto restante. Determina máscaras, transición y factibilidad.

`SpatialQLearning` proyecta a `PositionKey(position, period)`. Dos estados físicos
distintos pueden compartir fila Q y tener recompensas/máscaras distintas. Se trata
de **aliasing y una aproximación**, no de una representación Markov completa ni de
una definición exacta demostrada del estado del artículo. El evaluador no usa esta
proyección. La máscara del entorno se aplica a selección y bootstrap.

## Interpretaciones experimentales explícitas

1. El entrenamiento utiliza seis direcciones y el rollout constructivo cinco.
   Esta transición de fases no está desarrollada completamente en el artículo.
2. Recompensa: suma del VAN de bloques **nuevos** del cono. Se incluyen costos
   negativos. Una revisita permite tránsito, sin extracción ni recompensa adicional.
   Estos puntos ahora están confirmados por el usuario.
3. Cono: destino y cierre transitivo de precedencias, excluyendo extraídos; orden
   topológico. Se consume capacidad de todos los nuevos en el período actual.
   Si no caben, la acción se enmascara/rechaza atómicamente; no se divide el cono.
4. Avance de período voluntario; conserva posición y libera capacidad. No se adivina
   una regla automática de calendarización del paper.
5. Termina por stop, todos extraídos o presupuesto de pasos. El presupuesto incluye
   inicio, revisitas y avances. `step_limit` es horizonte artificial, no convergencia.
6. Inicio: mayor Valor en la superficie global declarada, empate por menor ID;
   se extrae mediante acción contabilizada, no como una ganancia gratuita al reset.

Quedan por confirmar el estado Q, el cierre natural de los episodios, los bordes
y el paso de aprendizaje a generación del plan; no se vuelve a pedir la recompensa,
las revisitas ni la orientación del caso reducido ya confirmadas.

## Coordenadas y candidatos del caso reducido

El paper numera la superficie como Z=1 y el descenso Z+1. El usuario confirmó
Z=3 superficial en el Excel; se mantienen intactas sus coordenadas:

- `grid9-z-up`: superficie Z máximo, predecesores en Z+1, descenso Z−1.
- `grid9-z-down`: superficie Z mínimo, predecesores en Z−1, descenso Z+1.
- N/S corresponden a X−1/X+1; O/E a Y−1/Y+1, según Fig. 7, no al dibujo de la hoja.
- Ambas reglas recortan los vecinos fuera de la grilla: es una decisión provisional
  conservada del baseline, no una instrucción documentada del cliente.

Diagnóstico del Excel: la referencia tiene cero violaciones de cierre con Z máximo
superficial y 40 bloques con predecesores faltantes con Z mínimo superficial. El
máximo cierre independiente calcula 336.000 y 8.000 respectivamente. La orientación
**se elige explícitamente**, no mediante selección automática de la que alcance el objetivo.

Los 53 resaltados coinciden exactamente con bloques cuyo cono superior completo
está dentro de la grilla con superficie en Z máximo: 35 en Z=3, 15 en Z=2, 3 en Z=1.
Esto es una coincidencia geométrica verificada, no una confirmación del significado
de la matriz Q del cliente. Ni estos IDs ni los 42 de referencia se filtran al agente.

## Benchmark y alcance de los resultados

El máximo cierre solo usa valores de la tabla y aristas de precedencia. Se transforma
en min-cut: source→bloques positivos, negativos→sink, bloque→predecesor con capacidad
mayor que la suma positiva. Se usa Edmonds–Karp con fracciones exactas; el lado source
del corte da la selección cerrada. Se comprueba contra enumeración en minas pequeñas
y contra el evaluador físico. El valor de referencia no entra en el cálculo.

Este benchmark no tiene restricciones de recorrido local ni bloque inicial forzado.
Por tanto, es un óptimo del problema de selección y una cota superior para la variante
espacial, **no una garantía de que su espacio de recorridos alcance ese valor**. En
particular, construir con cinco direcciones impide volver a niveles superiores; la
referencia visual no especifica un recorrido que satisfaga esta restricción.

Se conservan los mejores episodios completos de learning (seis direcciones) y el
plan greedy completo de construction (cinco, salvo extensión). Además, `PathObserver`
verifica cada prefijo con cambios de extracción y guarda el mayor VAN observado,
incluyendo los de greedy. `selected_path` identifica el prefijo seleccionado con
plan, evaluación, fase, episodio, pasos y acciones. Un prefijo es una selección
factible observada, no un episodio al que se haya impuesto una terminación nueva.
Los empates retienen el primero. No se elimina una pérdida intermedia ni se interrumpe
el aprendizaje al conservar un prefijo. Se prueba que Q y RNG son idénticos con y
sin observación. Episodios terminales sin pasos conservan un camino vacío verificado.

Un camino de alto valor no demuestra una política greedy igualmente buena ni
optimalidad. Se registran recorridos, causa de terminación, semillas, hiperparámetros,
hashes, tiempos y evaluaciones independientes. La verificación de prefijos añade
costo al tiempo de generación; no comparar tiempos sin considerar esta diferencia.

Referencia, colores y benchmark no participan en inicio, acciones, recompensas,
selección del mejor episodio, tabla Q ni parada; se leen/calculan después de todas
las semillas. Las pruebas alteran los metadatos de referencia y habilitan benchmark
para comprobar que planes, historiales y tabla Q no cambian.
