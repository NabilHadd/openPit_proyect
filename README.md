# openPit_proyect

Primera versión funcional de planificación de extracción minera con Aldo Quelopana.
Incluye carga del libro, verificador independiente, entorno determinista, Q-learning
tabular full-state y una variante **espacial con extracción conjunta de conos**.
El paper de Quelopana & Navarra (2021) ya fue aportado por el usuario. La variante
se inspira en él, pero **no es una réplica confirmada**: ciertos detalles del método
y la correspondencia con el Excel aún requieren aclaración. No incluye Dyna-Q ni VND.

## Estructura

```
.
├── main.py                        # Orquestador principal
├── environment.yml                # Librerías para crear el entorno conda
├── README.md                      # Descripción del proyecto
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml              # Pipeline de GitHub Actions: corre los tests en cada push
├── data/                          # Datos del proyecto (no se versionan)
│   ├── raw/                       # Datos originales (modelo de bloques)
│   └── processed/                 # Datos transformados, listos para usar
├── src/
│   ├── config.py                  # Hiperparámetros y variables de configuración
│   ├── data/                      # Scripts para manejar los datos (no contiene datos)
│   │   ├── dataset.py             # Lectura de los datos según su formato
│   │   └── transforms.py          # Transformaciones de los datos, si son necesarias
│   ├── envs/                      # Entornos de aprendizaje por refuerzo (no confundir con archivos .env)
│   │   └── mine_env.py            # La mina modelada como entorno de RL: estados, acciones, recompensa y restricciones
│   ├── models/
│   │   ├── base.py                # Interfaz común para todos los modelos
│   │   └── q_learning.py          # Baseline Q-learning (Quelopana & Navarra, 2021)
│   ├── training/
│   │   ├── losses.py              # Funciones de pérdida (según el método)
│   │   ├── rewards.py             # Funciones de recompensa para los modelos de RL
│   │   ├── trainer.py             # Rutinas de entrenamiento por modelo
│   │   └── experiment_runner.py   # Catálogo completo de experimentos
│   └── evaluation/
│       ├── metrics.py             # Definición de las métricas de evaluación
│       ├── plots.py               # Generación de gráficos
│       └── reporter.py            # Generación de reportes
├── tests/                         # Tests automatizados
└── artifacts/                     # Resultados generados (no se versionan)
    ├── plots/<nombre>/            # Gráficos por modelo
    ├── reports/<nombre>/          # Reportes por modelo
    └── checkpoints/<nombre>/      # Modelos entrenados por modelo (p. ej. tablas Q)
```

## Instalación y ejecución

```bash
conda env create -f environment.yml
conda activate openpit
python -m unittest discover -s tests -v
```

Solo se necesita `openpyxl` además de la biblioteca estándar. No se ha creado un
entorno Conda nuevo durante el desarrollo; las verificaciones locales usaron Python
3.13.12. CI está configurado para Python 3.12 y pruebas sintéticas, sin datos privados.

El archivo local encontrado es **`Bloques105_SinRestricción.xlsx`**, no la copia
`Bloques105_SinRestricción(1).xlsx` mencionada por Aldo. El acento del nombre local
está descompuesto en Unicode. Estos comandos copian su nombre exacto; si se recibe
otra copia, pasar su ruta explícitamente. Nunca se elige el primer Excel disponible.
Las rutas relativas se resuelven desde la raíz del proyecto, no desde el directorio
de ejecución. También se admiten rutas absolutas.

```bash
# Solo suma económica; factibilidad desconocida sin reglas.
python main.py reference --data 'data/raw/Bloques105_SinRestricción.xlsx'

# Factibilidad CONDICIONAL bajo una hipótesis geométrica explícita.
python main.py reference --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up

# Prueba corta de integración, no réplica científica.
python main.py train --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --episodes 5 --seeds 0 --plot

# Experimento acotado con diez semillas distintas.
python main.py train --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --episodes 100 --seeds 0 1 2 3 4 5 6 7 8 9

# Variante espacial, parámetros publicados por defecto: alpha=1, gamma=0.7, epsilon=0.3.
# Benchmark exacto calculado DESPUÉS de terminar todas las semillas.
python main.py train --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --model spatial-cone --episodes 1000 --max-steps 500 --seeds 0 1 2 3 4 5 6 7 8 9 --exact-benchmark

# Diagnóstico exacto independiente: no lee las selecciones del Excel ni entrena.
python main.py benchmark --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up

# Integración privada opcional; las pruebas normales no necesitan el Excel.
OPENPIT_TEST_XLSX="$PWD/data/raw/Bloques105_SinRestricción.xlsx" python -m unittest discover -s tests -v
```

Cada comando crea una carpeta única bajo `artifacts/reports/`, ignorada por Git.
`report.json` registra configuración efectiva, semillas, commit/estado local,
huella del código y del archivo, versiones, trazabilidad fila→ID→índice interno,
reglas, evaluaciones y tiempos. Cada semilla guarda `training.json` (planes y
resultados por episodio, mejores prefijos, mejor episodio y rollout greedy final)
y `q-table.json`. `selected/` guarda el camino de mayor VAN observado y verificado,
conservando también `best/` (episodio completo) y `greedy/` (rollout completo).
`--plot` agrega una curva SVG local del VAN por episodio. Los tiempos de generación
incluyen entrenamiento y verificación, pero excluyen carga del Excel, metadatos,
serialización, gráficos y reportes. No comparar esos tiempos como benchmark del paper.

## Datos y validación de referencia

- `load_mine` lee únicamente `105 Bloques`: exige `ID`, `X`, `Y`, `Z`, `Valor`,
  números finitos, IDs/coordenadas únicos y coordenadas enteras; conserva fila original.
- `load_selections` es una lectura aparte. Reconoce los paneles rotulados `Z=`, sus
  coordenadas y sus IDs; compara rellenos sólidos con `K3`, resolviendo RGB, paleta,
  tema y tintado. No usa índices de estilo. El adaptador visual corresponde al
  diseño 7×5×3 de este libro; otros diseños requieren otro adaptador, no se adivinan.
- `X-Y View`: 53 candidatos, conservados como metadatos. **No** filtran estados ni
  acciones mientras no se confirme su significado en el paper.
- `Resultado`: 42 seleccionados; la suma desde `Valor` es 336.000 y coincide con
  `L13`. No proporciona orden ni períodos. La moneda y dimensiones físicas de los
  bloques no están declaradas: no se inventan unidades.
- La referencia nunca se pasa al entorno/modelo/entrenador. En entrenamiento, sus
  metadatos se leen **después** de terminar todas las semillas. Una prueba altera
  selecciones y valor declarado y comprueba que planes y tabla Q no cambian.

La comprobación económica es independiente de la factibilidad. Sin precedencias,
el verificador devuelve `feasible: null`, no `true`. La referencia puede recibir
un **orden testigo construido** topológicamente, rotulado como tal, en un período
sin capacidad ni descuento. Ese orden no se atribuye al Excel. Para evaluar otros
escenarios hay que proporcionar un plan con sus períodos: no se calendariza la
referencia visual silenciosamente.

## Reglas explícitas y límites de dominio

El paper documenta nueve bloques superiores (Fig. 4) y precedencias en períodos
anteriores o iguales (p. 647). El usuario confirmó que la superficie del Excel
reducido es **Z=3**, además de la recompensa incremental y las revisitas sin nueva
extracción. Los bordes siguen pendientes. Entrenar sin elegir reglas se rechaza:

1. `--precedence grid9-z-up`: orientación confirmada para este Excel, no elegida
   automáticamente por la solución. Mayor Z es más superficial; cada bloque requiere los
   bloques existentes del cuadrado 3×3 en Z+1 (incluye diagonales y el directamente
   superior). Los bordes se recortan al dominio. Esto admite la referencia pero
   **no confirma** el tratamiento de bordes del cliente. La factibilidad sigue
   siendo condicional a esas reglas. `grid9-z-down` permite probar otra orientación (Z mínimo
   superficial, como numera el paper), sin reescribir coordenadas ni valores.
2. `--precedences-file ruta.json`: reglas explícitas suministradas por el usuario,
   con fuente obligatoria y todos los IDs. Se comprueban referencias y ciclos.
   `specified` significa verificado contra esas reglas, no aprobación del cliente.

Formato del segundo caso, ejemplo **sintético** de dos bloques (no corresponde al Excel):

```json
{"source": "Ejemplo sintético", "predecessors": {"10": [], "20": [10]}}
```

La convención experimental es un **plan ordenado**: los predecesores deben aparecer
antes, pero pueden estar en el mismo período. No se exige período estrictamente
anterior. El alcance geométrico y el tratamiento de bordes aún requieren confirmación.

### Baseline full-state y diferencias con el paper

| Aspecto | Implementación actual | Estado respecto al paper |
| --- | --- | --- |
| Estado | Conjunto completo extraído, período, capacidad utilizada, terminal | No se identifica con el estado tabular de Quelopana |
| Acciones | Extraer un ID factible, `advance_period`, `stop` | El paper describe movimientos espaciales |
| Transición | Determinista; extraer no avanza el período; avance explícito libera capacidad | Convención experimental, pendiente de confirmar |
| Recompensa | `Valor / (1 + discount_rate)^period`, período inicial 0; parar/avanzar dan 0 | Fórmula económica provisional, no recompensa atribuida al paper |
| Actualización | `Q ← Q + alpha * (r + gamma * max Q(s',a') − Q)`; terminal sin bootstrap | Q-learning estándar, no prueba de réplica |
| Exploración | Epsilon constante; epsilon-greedy con desempate aleatorio local | El paper también usa epsilon estático |
| Terminación | Parada voluntaria, todos extraídos o sin extracciones posibles en último período | Definición del paper pendiente |

Una acción inválida lanza `ValueError` antes de modificar el entorno; no recibe una
penalización inventada. Reiniciar restaura el estado y plan. Una semilla controla
el único generador aleatorio (`random.Random` del modelo); evaluación greedy sin
aleatoriedad ni actualizaciones. La tabla Q es dispersa, inicialmente cero. El
estado físico puede tener hasta 2^105 subconjuntos: esta implementación **no promete
convergencia**. Las acciones incluyen los 105 bloques, no solo los 53 marcados.

### Variante espacial con conos

`--model spatial-cone` conserva el verificador y cambia las decisiones del agente:

- Posición en la grilla; movimiento N/S/O/E/abajo/arriba durante aprendizaje.
- Construcción greedy con N/S/O/E/abajo, como la lista de p. 649. La separación por
  fase es una interpretación explícita. `--construction-up` habilita seis direcciones
  también al construir: es una extensión diagnóstica, no la construcción publicada.
- Inicio obligatorio en el bloque de mayor Valor de la superficie declarada; empates
  por ID. La acción inicial cuenta en recompensas, plan y presupuesto de pasos.
- Un destino nuevo extrae su cierre transitivo de precedencias pendiente, ordenado
  topológicamente. Se comprueba la capacidad conjunta antes de modificar nada.
- Se permite transitar por bloques ya extraídos con valor/capacidad adicionales cero.
- El estado físico conserva extraídos, período, uso, posición, fase y pasos restantes.
  La tabla Q usa **posición y período**, una proyección con aliasing, no un MDP completo.
- Recompensa = VAN incremental del conjunto nuevo, **confirmado por el usuario
  como opción B**. No se sustituyó por recompensa terminal ni se excluyen costos negativos.
- Horizonte explícito `--max-steps` (default 500); cada acción consume un paso.
  Se registra `step_limit` y se trata como terminal de este problema de horizonte
  finito, sin bootstrap. No se presenta como convergencia ni terminación del paper.
- Avance de período sigue siendo explícito. Los conos no se dividen ni se difieren
  automáticamente cuando no caben. Con JSON de precedencias hay que declarar
  `--surface min-z|max-z`; con una regla geométrica se usa su orientación y se
  rechazan contradicciones.

El baseline anterior sigue siendo el default y conserva alpha=0,2 / gamma=1 /
epsilon=0,2. La variante espacial usa 1 / 0,7 / 0,3 salvo configuración explícita.
Ver [`docs/paper_mapping.md`](docs/paper_mapping.md) para correspondencias y dudas.

### Benchmark exacto del caso reducido

`benchmark` resuelve **máximo cierre por corte mínimo**, usando capacidades racionales
para evitar redondeo en el algoritmo. No es RL ni utiliza `Resultado`. Solo admite un
período, sin capacidad ni descuento. Verifica el plan obtenido y guarda valor/corte
exactos junto al VAN recalculado. `--exact-benchmark` lo ejecuta después de entrenar
y registra las brechas; nunca alimenta la política, recompensas ni parada.

Con la orientación confirmada Z máximo superficial calcula 336.000 y 42 bloques;
con la orientación alternativa Z mínimo calcula 8.000. Son óptimos **bajo las reglas
elegidas**, no una certificación de todos los supuestos geométricos. El diagnóstico de conos completos
coincide con los 53 candidatos del libro (35/15/3 por profundidad), pero no se utiliza
como máscara. El significado de esa selección y el tratamiento de bordes siguen pendientes.

`--capacity` se aplica por período en `--capacity-unit blocks` o `tonnes`.
No hay capacidad por defecto. En toneladas se exige tonelaje por bloque; el Excel
no lo tiene y ese modo se rechaza. La API soporta tonelajes en ejemplos sintéticos.
`--periods` limita el horizonte y `--discount-rate` es una tasa económica no negativa,
**distinta** de `--gamma` de Q-learning. Las pruebas manuales incluyen -4 ahora y
15 en período 1 con tasa 0,5: VAN = -4 + 15/1,5 = 6; capacidad 2+3 toneladas = 5,
no dos toneladas por ser dos bloques.

La selección final conserva el camino de mayor VAN **entre los prefijos factibles
realmente observados** durante los episodios y el rollout greedy. Un camino puede
continuar tras una pérdida intermedia y mejorar después: el observador no modifica
acciones, recompensas, Q ni terminación. Tampoco necesita saber el óptimo.
Se guardan fase, episodio, pasos y acciones exactas del prefijo. No se registra ese
prefijo como si el episodio hubiese terminado ahí: se mantienen por separado el
episodio completo, su mejor prefijo y el greedy completo. Los empates conservan el
primer camino observado. También se guarda cada suma de recompensas por separado;
el VAN siempre se recalcula desde el plan. El objetivo
de referencia se usa solo para comparar después, nunca para cortar episodios,
inicializar Q, filtrar acciones, formar recompensas o seleccionar el mejor plan.

## Plan interno y evaluación independiente

No se conoce el formato de exportación del cliente. Se guarda este formato interno
documentado (orden de la lista = orden de extracción, períodos desde cero):

```json
{
  "schema_version": 1,
  "format": "openpit-internal-ordered-plan",
  "period_origin": 0,
  "extractions": [{"block_id": 71, "period": 0}, {"block_id": 72, "period": 0}]
}
```

```bash
python main.py evaluate-plan --data 'data/raw/Bloques105_SinRestricción.xlsx' --precedence grid9-z-up --plan '/ruta/al/plan.json'
```

El verificador detecta desconocidos, duplicados, precedencias, períodos inválidos o
decrecientes y capacidad excedida. No repara el plan. Un valor numérico de un plan
inválido es solo diagnóstico: su factibilidad es falsa. Sin verificación completa
no se exporta un plan como comprobado. Con reglas provisionales se llama
`conditional-plan.json`; con reglas explícitas, `verified-plan.json`. Ambos indican
que la aprobación minera y compatibilidad del formato del cliente están pendientes.

## Continuidad

Ver [`docs/continuity.md`](docs/continuity.md) para resultados ejecutados, limitaciones
y próximos insumos necesarios. Mantener intacto el Excel original. Datos y artefactos
locales no se versionan.
