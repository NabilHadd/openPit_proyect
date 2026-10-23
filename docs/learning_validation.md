# Validación de Q-learning por etapas

Protocolo **experimental**, separado de `main.py train` y de los cuatro controles
históricos. No es un modelo final, una réplica confirmada del paper ni una garantía
de escalabilidad. Solo usa biblioteca estándar y la dependencia existente `openpyxl`.

## 1. Contrato del entorno

`tests/test_environment_contract.py` recorre todos los estados alcanzables de una
mina pequeña con horizonte finito. Comprueba cada transición bajo tres escenarios:
sin restricciones de capacidad; tres períodos con descuento y capacidad por bloques;
y tres períodos con descuento y capacidad en toneladas.

Se verifica extracción única, cierre de precedencias, capacidad, rechazo atómico,
consistencia estado/acción y equivalencia de fases con seis direcciones. Tras cada
transición se compara la recompensa acumulada con una suma económica directa y con
el VAN del verificador independiente. Las revisitas no vuelven a cobrar.

Además se reprodujeron los 40 caminos seleccionados y los 40 greedy de los controles
guardados del Excel: **20.684 transiciones verificadas**, sin discrepancias de VAN,
duplicados ni planes infactibles bajo las reglas declaradas. Esto valida contratos
de implementación, no confirma la hipótesis geométrica de bordes del cliente.

## 2. Mina diminuta y oráculo independiente

`src/evaluation/tiny_oracle.py` define seis bloques:

| ID | Valor | Predecesores directos |
| --- | ---: | --- |
| 1 | +5 | ninguno |
| 2 | −4 | 1, 4 |
| 3 | +15 | 2 |
| 4 | −1 | ninguno |
| 5 | −8 | ninguno |
| 6 | −6 | 1, 4, 5 |

El inicio obligatorio es el bloque 1. La ruta `start, down, down, stop` obtiene
recompensas **+5, −5, +15, 0 = 15**. Exige atravesar una pérdida; los bloques 5 y 6
deben evitarse. No se suministra esa ruta, su valor ni el conjunto óptimo al agente.

El oráculo enumera subconjuntos cerrados y estados/acciones espaciales alcanzables,
con cálculos propios: no llama a `SpatialConeEnv.step`, `_new_blocks`, las tablas Q
ni máximo cierre. En modo finite distingue el óptimo económico del óptimo alcanzable
dentro del horizonte. Por ejemplo, con solo dos pasos son respectivamente 15 y 5.
En external enumera el grafo estacionario completo, sin convertir el límite de
recolección en un horizonte del problema: el objetivo sigue siendo 15 incluso con
un corte de un paso. Una prueba de regresión impide certificar el prefijo inicial 5.
También prueba que un inicio negativo obligatorio puede hacer inalcanzable el
subconjunto vacío de valor cero.

**Prueba ejecutada:** semillas 0…9, 10.000 transiciones por semilla, alpha=1,
epsilon=0,3, gamma=1, horizonte finito de 12 y 20 evaluaciones por semilla.
En la mina de seis bloques: **10/10 mejores observaciones y 200/200 políticas
evaluadas alcanzaron 15**. Las pruebas automatizadas repiten el criterio sobre
una variante de cuatro bloques: otras 200 evaluaciones óptimas.

Artefacto de seis bloques:
`artifacts/reports/20261023T193045-learning-validation-9871deb3/`.
Revalidación con la puerta estricta de terminación:
`artifacts/reports/20261023T194311-learning-validation-e2e0c9b1/`: 200/200 completados
óptimos, y las diez tablas Q iguales a las de la primera corrida.

Esto demuestra que la implementación puede aprender y reproducir el óptimo pequeño;
no basta para afirmar que resolverá los 105 bloques.

## 3. Exploración, evaluación y cortes

La clase opt-in `ValidationQLearning` reutiliza la actualización Q y la exploración
existentes. Usa el estado físico exacto y normaliza únicamente la etiqueta de fase
bajo dinámica 6/6. No usa la proyección posición/período.

En evaluación, los empates se muestrean uniformemente con un RNG separado, derivado
de `f'{training_seed}:evaluation:{rollout_index}'`. Se registran todos los rollouts,
no el mejor. Evaluar no actualiza Q ni consume el RNG de aprendizaje. La opción
`--evaluation-ties ordered` conserva el desempate anterior solo como diagnóstico.
Cambiar el desempate no inventa conocimiento donde todos los Q son cero.

| Evento | Terminal | Truncado | Bootstrap |
| --- | --- | --- | --- |
| `stop` | sí | no | no |
| todos extraídos | sí | no | no |
| horizonte del problema, modo `finite` | sí | no | no |
| límite de recolección por episodio, modo `external` | no | sí | sí |
| agotamiento del presupuesto total, estado no terminal | no | sí | sí |

El modo **finite**, predeterminado e idéntico a la semántica histórica, conserva los
pasos restantes en Q. En **external** el reloj físico queda fijo en cero y el nuevo
recolector aplica el límite; la actualización usa el sucesor real antes de reiniciar.
Si el último paso también termina realmente el episodio, prevalece esa terminación.
Los modos representan problemas diferentes y no deben compararse como si solo
cambiara la compresión del estado. Los cuatro controles históricos rechazan external.

Un diagnóstico de external sobre seis bloques y límite 12, con el mismo presupuesto
y semillas, encontró el óptimo observado en 10/10 semillas, pero solo 154/200 rollouts
alcanzaron su valor (media 11,59). Artefacto:
`artifacts/reports/20261023T193330-learning-validation-1454842f/`.
La puerta falló y el CLI devolvió código 1. Con gamma=1, movimientos
gratuitos pueden empatar con acciones de progreso para el retorno sin horizonte;
aleatorizar empates no garantiza completar antes de un límite externo corto. **No
escalar ese protocolo suponiendo que ya pasó la prueba pequeña.** La validación
principal y la consolidación inicial conservan el horizonte finito explícito.

El nuevo recolector cuenta exactamente T transiciones, incluyendo inicio, stop y
revisitas. Un episodio parcial por presupuesto no se convierte en un episodio
completo. Se guardan por separado:

- Mejor prefijo de **aprendizaje** y mejor episodio completo de aprendizaje.
- Distribución de rollouts completos de la política final y sus mejores prefijos.
- Estados/pares actualizados y visitados una sola vez, reuso y tamaño Q.
- Decisiones con todos los Q en cero, claves/acciones desconocidas y empates máximos.
- Primer paso desconocido, primer paso sin preferencia numérica y cola sin extracción.
- Contextos mineros únicos desconocidos y visitas desconocidas solo por el reloj.
- Retornos de navegación, movimientos sin extracción y causas de terminación/corte.

Los porcentajes sobre el espacio alcanzable completo solo se calculan para la mina
diminuta finita. En 105 bloques se informa cobertura observada, no una fracción
inventada de los 2^105 subconjuntos.

Q se contrasta con un límite por estado: la suma de los valores positivos aún no
extraídos. Una violación falla la ejecución, sin recortar Q ni alterar recompensas.
Este control habría detectado las estimaciones imposibles de 9,5–13,3 millones del
control compacto gamma=1, frente a solo 336.000 positivos disponibles en total.

## 4. Comparación reproducible por transiciones

```bash
# Suite completa de pruebas sintéticas; incluye las puertas de aprendizaje pequeñas.
python -B -m unittest discover -s tests -v

# Validación sin Excel: diez semillas y veinte rollouts por semilla.
python -B -m src.validation --transitions 10000 --evaluation-rollouts 20

# Comparación de 105 bloques: gamma=1, estado exacto, seis direcciones, RNG de
# evaluación separado. Ejecutar solo después de aprobar las pruebas anteriores.
python -B -m src.validation --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --transitions 100000 --max-steps 500 --evaluation-rollouts 20 --seeds 0 1 2 3 4 5 6 7 8 9
```

Cada carpeta `...-learning-validation-.../` guarda procedencia, configuración,
presupuestos, reportes por semilla, historias compactas, tablas Q, rollouts y planes
comprobados con etiquetas `is_final_model: false`. Solo se mantiene un modelo por
semilla. El tiempo incluye entrenamiento, evaluación y verificaciones internas,
no exportación ni cálculo del benchmark.

Todas las semillas reciben exactamente el mismo número de actualizaciones; la
evaluación se contabiliza aparte. Menos de diez semillas se etiqueta como smoke.
La puerta de la mina pequeña exige que cada semilla alcance el objetivo en **todos**
sus rollouts predeclarados, terminados y no truncados; si falla, el CLI termina con
código 1. Los aciertos de valor en rollouts truncados se cuentan aparte: no se
presentan como políticas que completaron la tarea. Una suite sin
completar queda fallida sin resumen de éxito parcial.

Oráculo y benchmark se calculan solamente al terminar todas las semillas. En minas
grandes, máximo cierre se etiqueta como **cota económica superior**: no certifica
por sí mismo alcance espacial, inicio obligatorio ni horizonte. Este protocolo no
lee colores del Excel ni declara automáticamente alcanzado el hito 1 histórico.

## Resultado de 105 bloques — 2026-10-23

Se ejecutó el comando anterior: diez semillas 0…9, **exactamente 100.000 transiciones
por semilla**, gamma=1, alpha=1, epsilon=0,3, horizonte finito 500 y 20 rollouts de
evaluación independientes por semilla. No se detuvo al encontrar el objetivo.

| Semilla | Mejor prefijo de aprendizaje | Mejor episodio completo | Media política final | Aciertos 336.000 /20 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 306.000 | 306.000 | 300.500 | 0 |
| 1 | 326.000 | 326.000 | 320.500 | 0 |
| 2 | 336.000 | 336.000 | 314.500 | 16 |
| 3 | 326.000 | 326.000 | 297.500 | 0 |
| 4 | 336.000 | 326.000 | 304.500 | 0 |
| 5 | 306.000 | 306.000 | 291.000 | 0 |
| 6 | 316.000 | 316.000 | 307.500 | 0 |
| 7 | 336.000 | 336.000 | 293.000 | 0 |
| 8 | 326.000 | 326.000 | 301.500 | 0 |
| 9 | 336.000 | 336.000 | 322.000 | 14 |

- **Observado 336.000 en 4/10 semillas**, episodios completos óptimos en 3/10.
- Política final: **30/200 aciertos**, concentrados en solo 2/10 semillas; **ninguna**
  semilla lo reproduce en todos sus 20 rollouts. Media global 305.250, mínimo 116.000.
  No tratar los 200 rollouts como 200 agentes entrenados independientemente.
- Los 200 rollouts terminaron por Stop; ninguno agotó 500 pasos. Evaluación total:
  4.354 transiciones. Aprendizaje: 1.000.000; generación y verificación: 110,371 s.
- Se reprodujeron y verificaron independientemente los 10 mejores prefijos y los 200
  rollouts completos. Los 34 caminos de valor 336.000 contienen los 42 positivos y
  ningún costo negativo. Se revalidó la referencia del libro después de aprender;
  coincide económica y factiblemente bajo grid9-z-up. El hash del Excel no cambió.
- El 12,72% de las decisiones de evaluación tenía todos los Q en cero y el 7,95%
  consultó claves sin actualizaciones, frente a la cola desconocida dominante de
  los controles históricos. Las diferencias no aíslan causalmente el desempate:
  también aumentó el presupuesto de aprendizaje.
- Todos los 200 Stop se eligieron entre máximos empatados; 101 tuvieron todos los Q
  en cero y 97 consultaron un par estado/Stop no actualizado. Terminar por Stop no
  demuestra por sí mismo haber aprendido una regla de parada. Con movimientos
  gratuitos, parar y seguir transitando pueden ser económicamente indiferentes.

Artefactos: `artifacts/reports/20261023T193418-learning-validation-aba1d7d5/`.

### Ablación del desempate con las mismas tablas congeladas

Sin entrenar otra vez, se evaluaron esas diez tablas usando el desempate ordenado
anterior: las diez agotaron 500 pasos; media 315.000, dos alcanzaron 336.000. El desempate
uniforme dio media 305.250 y terminación Stop en 200/200. **Quitar el sesgo de orden no
garantiza aumentar el VAN**: también permite salir de recorridos gratuitos hacia
acciones poco conocidas y costosas. No elegir el desempate mirando cuál gana este
Excel ni agregar una preferencia Stop oculta.

Las semillas 4 y 7 observaron 336.000, pero su Q inicial final es 326.000. Al seguir los
caminos guardados, aparecen acciones de ese camino valoradas por debajo de otras
alternativas: la experiencia óptima aún no está consolidada en la tabla. Haber
visitado un buen camino una vez no equivale a propagar todo su retorno con TD de
un paso. La semilla 4 incluso continúa desde 336.000 y no completa un episodio óptimo.

El objetivo de encontrar un camino 336.000 quedó demostrado bajo las reglas actuales;
el de una **política final robusta** sigue pendiente. El siguiente análisis debe
centrarse en repetición de los pares relevantes y propagación del valor, manteniendo
los presupuestos por transiciones y sin confundir observación, reproducción y parada.
No se ha promovido ningún modelo a final ni introducido otra técnica de aprendizaje.
