# openPit_proyect

Planificación de la extracción de una mina a cielo abierto con **Q-learning tabular**
como línea base, comparado con Dyna-Q, Prioritized Sweeping y Beam Search.
El agente recorre el modelo de bloques y decide qué bloques extraer para maximizar el
VAN del plan, respetando las precedencias del talud. La formulación se basa en
Quelopana & Navarra (2021).

Estado actual: línea base validada en el caso reducido de 105 bloques entregado por el
cliente (VAN óptimo de referencia: 336.000).

## Modelos disponibles

| Clave | Archivo | Qué hace | Presupuesto (`cost_unit`) |
|---|---|---|---|
| `q_learning` | `src/models/q_learning.py` | Q-learning tabular (línea base): una actualización por paso real | `transitions` (pasos reales) |
| `dyna_q` | `src/models/dyna_q.py` | Q-learning + modelo aprendido de las transiciones vividas: `planning_steps` actualizaciones simuladas por paso real, con pares sorteados | `transitions`; además `planning_updates` |
| `prioritized_sweeping` | `src/models/prioritized_sweeping.py` | Como Dyna-Q, pero actualiza primero los pares de mayor \|error TD\| (> `theta`) y propaga hacia sus predecesores | `transitions`; además `planning_updates`, `queue_pushes` |
| `beam_search` | `src/models/beam_search.py` | Búsqueda directa sin aprendizaje: cada nivel extrae el cono completo de un bloque y conserva los `width` mejores planes según `heuristic` (`npv` u `optimistic`); determinista | `simulator_calls`; además `expansions` |

Los tres primeros caminan con el cursor de 6 direcciones y parten del bloque inicial forzado;
`beam_search` usa la vista por conos del entorno (sin cursor ni inicio forzado), así que su
conjunto de planes posibles es más amplio. Con `planning_steps: 0`, Dyna-Q y Prioritized
Sweeping son exactamente Q-learning. Los hiperparámetros de cada uno, con su justificación,
están comentados en `configs/<clave>.yml`.

## Estructura

```
.
├── main.py                    # Punto de entrada: modos train, experiments y evaluate-plan
├── configs/                   # Un YAML + un módulo de lectura por tema (ver Configuración)
│   ├── __init__.py            # PROJECT_ROOT, rutas y lectura estricta común
│   ├── config.yml             # Común a todos los modelos: datos, problema, entorno, semillas, modelos a correr
│   ├── config_settings.py
│   ├── model_settings.py      # Lector genérico del YAML de cualquier modelo (params + distribuciones)
│   ├── q_learning.yml         # Hiperparámetros de Q-learning y sus distribuciones (convención: <clave>.yml)
│   ├── dyna_q.yml, prioritized_sweeping.yml, beam_search.yml
├── environment.yml            # Entorno conda (Python 3.12)
├── data/                      # Datos (no se versionan, salvo los ejemplos)
│   ├── raw/                   # Archivos de origen (p. ej. el Excel del cliente)
│   └── processed/             # CSV estándar que lee el pipeline (+ *.example.csv versionados)
├── tools/                     # Conversión de archivos de origen a CSV estándar (ver tools/README.md)
├── src/
│   ├── runner.py              # Bucle común de los modos que entrenan: una corrida por semilla y
│   │                          # comparación con la referencia, sin conocer el modelo
│   ├── data/
│   │   ├── dataset.py         # Lectura de bloques y planes; bloques alcanzables
│   │   ├── instance.py        # MineInstance: mina, precedencias, problema y entorno que recibe cada modelo
│   │   └── precedences.py     # Regla de precedencia del talud
│   ├── envs/mine_env.py       # Entorno: simulador puro (transition), episodio vivo (reset/step) y vista
│   │                          # por conos para búsqueda (cone_actions, extract, search_key)
│   ├── models/
│   │   ├── base.py            # MineEstimator: contrato común (estimador de scikit-learn, costo y traza)
│   │   ├── register.py        # @register y REGISTRY: clave -> estimador y dataclass de hiperparámetros;
│   │                          # importa solo src/models/<clave>.py de los modelos pedidos
│   │   ├── tabular.py         # TabularQ: base de los modelos tabulares (bucle de entrenamiento, gancho observe)
│   │   ├── q_learning.py      # Q-learning tabular (línea base): TabularQ sin planificación
│   │   ├── dyna_q.py          # Dyna-Q: TabularQ + modelo aprendido y planificación uniforme
│   │   ├── prioritized_sweeping.py  # Prioritized sweeping: TabularQ + cola de prioridades
│   │   ├── beam_search.py     # Beam search sobre la vista por conos (sin aprendizaje)
│   │   └── exploration.py     # Curvas de epsilon (exploración)
│   ├── training/
│   │   └── rewards.py         # Recompensa: VAN incremental de lo extraído
│   └── evaluation/
│       ├── metrics.py         # Verificador independiente de planes (factibilidad y VAN)
│       ├── reporter.py        # report.json, planes CSV y trazabilidad
│       ├── plots.py           # Curva de aprendizaje (SVG)
│       ├── benchmark.py       # Óptimo exacto por máximo cierre (caso de 1 período, usado en tests)
│       └── tiny_oracle.py     # Mina de 6 bloques con óptimo por enumeración (usado en tests)
├── tests/                     # Tests automatizados (unittest): test_estimator_contract.py revisa a todo
│                              # modelo registrado y test_<modelo>.py su comportamiento propio;
│                              # golden/ guarda las huellas de Q-learning (test_golden_q_learning.py)
└── artifacts/                 # Resultados: train/, experiments/, evaluate-plan/ (no se versionan)
```

## Instalación

```bash
conda env create -f environment.yml
conda activate openpit
python -m unittest discover -s tests -v
```

Dependencias: Python 3.12, `pyyaml`, `scikit-learn`, `scipy` (intervalos de confianza),
`pandas` y `openpyxl` (estas dos últimas solo para `tools/`). GitHub Actions corre los tests en cada push y pull request.

## Datos

El pipeline solo lee dos CSV estándar:

- **Bloques** (`data/processed/data.csv`): `id,x,y,z,value[,tonnes]`, con **Z mayor = superficie**.
- **Plan o solución** (`data/processed/solution.csv`): `order,id,period`. `order` puede ir
  vacío si la solución es solo un conjunto de bloques.

Los archivos de origen se convierten con los scripts de `tools/`. Por ejemplo, para el
Excel del cliente:

```bash
python -m tools.prepare_blocks data/raw/data.xlsx --surface z-max --sheet "105 Bloques" \
    --map ID=id X=x Y=y Z=z Valor=value
python -m tools.extract_aldo_solution data/raw/data.xlsx
```

Detalle de formatos y validaciones: [`tools/README.md`](tools/README.md). Hay ejemplos
versionados en `data/processed/*.example.csv`.

## Configuración

Toda ejecución se define en el YAML común que recibe `--config` (por defecto
`configs/config.yml`). Ese archivo indica qué modelos correr; los hiperparámetros del modelo
`<clave>` están, por convención, en `configs/<clave>.yml` (junto al YAML común). Todo lo que
define la tarea va en el común y lo que define cómo aprende un modelo va en el YAML del
modelo. Una clave desconocida, en cualquiera de los dos, es un error.

`configs/config.yml` (común):

| Sección | Parámetros |
|---|---|
| raíz | `data`, `solution` (`null` = sin referencia), `output` (carpeta raíz de resultados), `seeds`, `name` (opcional: se agrega al nombre de cada carpeta de resultados) |
| `problem` | `periods`, `discount_rate`, `capacity` (`null` = sin límite), `capacity_unit` (`blocks` \| `tonnes`) |
| `environment` | `max_steps` (corte de seguridad por episodio), `horizon` (`external` \| `finite`) |
| `train` | modelo del modo `train` (una clave; su YAML es `configs/<clave>.yml`) |
| `experiments` | lista de modelos del modo `experiments`, todos con las mismas `seeds` |
| `tuning` | búsqueda aleatoria del modo tuning (aún no implementado), igual para todos los modelos: `n_iter` (configuraciones sorteadas), `search_seed` (semilla del sorteo), `seeds` (semillas de entrenamiento de cada configuración; no pueden repetir `seeds`) |

`configs/q_learning.yml`:

| Sección | Parámetros |
|---|---|
| `params` | `state` (`physical` \| `position`), `alpha`, `gamma`, `epsilon`, `epsilon_schedule` (`linear` \| `constant`), `transitions` (presupuesto por semilla), `policy_rollouts` (planes construidos con la política aprendida) |
| `tuning.distributions` | distribución de cada parámetro de `params` a ajustar: `uniform` y `loguniform` (`low`, `high`), `randint` (`low`, `high` enteros, ambos incluidos) o `choice` (`values`); un parámetro entero solo acepta `randint` o `choice`. Los que no aparecen quedan fijos. Se valida al leer el archivo. `transitions` y `policy_rollouts` no se ajustan |

`configs/dyna_q.yml` y `configs/prioritized_sweeping.yml`: los mismos `params` que Q-learning
(con `state: physical` obligatorio) más `planning_steps` (actualizaciones de planificación por
paso real; 0 = Q-learning) y, en Prioritized Sweeping, `theta` (prioridad mínima para entrar a
la cola, en unidades de VAN). `configs/beam_search.yml`: `width` (planes por nivel; 1 = voraz),
`heuristic` (`npv` \| `optimistic`) y `max_simulator_calls` (tope de seguridad, no se ajusta).

Configuración de referencia: `state: physical`, `horizon: external`, `alpha: 1`, `gamma: 1`,
`epsilon: 0.5` con `epsilon_schedule: linear`, 100.000 transiciones.

## Ejecución

```bash
# Entrena el modelo de train, una vez por semilla, y exporta sus resultados
python main.py train --config configs/config.yml

# Entrena cada modelo de experiments con las mismas semillas y los resume
python main.py experiments --config configs/config.yml

# Verifica un plan y recalcula su VAN (sin --plan evalúa la solución del YAML)
python main.py evaluate-plan --config configs/config.yml [--plan RUTA.csv]
```

Carpetas de resultados (hora local; si el nombre ya existe se agrega `_2`, `_3`…; `name`
del YAML, si está, va al final):

```
artifacts/train/2026-11-26_16-02_train_q_learning/      config.yml, q_learning.yml, report.json, seed-<n>/
artifacts/experiments/2026-11-26_15-30_experiments/     config.yml, summary.csv, summary.json,
                                                        <modelo>/{<modelo>.yml, report.json, seed-<n>/}
artifacts/evaluate-plan/2026-11-26_16-10_evaluate-plan/ config.yml, report.json, plan.csv
```

`config.yml` es una copia exacta del YAML usado, y `<modelo>.yml` la del YAML del modelo.

**Modo experiments.** Antes de entrenar, lee y valida el YAML de todos los modelos de la
lista: un error ahí aborta sin entrenar nada. Después entrena los modelos en orden; cada uno
escribe su `report.json` apenas termina. Si un modelo falla al entrenar, su error (tipo,
mensaje y traceback) queda en `summary.json` y en su `report.json`, y se sigue con el
siguiente. Un modelo **determinista** (atributo de clase `deterministic = True`: su resultado
no depende de la semilla) se ejecuta solo con la primera semilla, así que su resumen tiene
n = 1 y no tiene desviación estándar ni IC. `summary.csv` tiene una fila por modelo: estado,
semillas, VAN medio y desviación estándar, IC 95 %, brecha media, semillas que alcanzan la
referencia, costo hasta la referencia y hasta el mejor plan (media y desviación, en la
unidad del modelo, `cost_unit`), presupuesto gastado e hiperparámetros.

Contenido de `report.json` (uno por modelo):

- Configuración común, modelo y sus hiperparámetros (`model`), versión del código y hash de
  los datos. Por semilla, los campos comunes a todo modelo (`score`: VAN del plan entregado,
  `blocks`, `seconds`, `artifacts`, y `cost`: todos los contadores gastados y sus valores en
  el momento en que encontró el plan entregado, con `unit` = la unidad del presupuesto del
  modelo) y los propios del modelo (`diagnostics`).
- El resumen agrega media, desviación estándar e IC 95 % del VAN (t de Student; nulos con 1
  semilla) y el costo hasta el mejor plan.
- La comparación con la solución de referencia se lee **después** de entrenar y guardar
  todas las semillas del modelo: si el plan entregado la iguala (`matches_reference`), su
  brecha (`gap_percent` = 100·(ref − VAN)/|ref|), los contadores en el momento en que la
  alcanzó (`cost_to_reference`, nulo si nunca) y los conteos propios del modelo
  (`reference`). En el resumen, `cost_to_reference_mean` y `_std` promedian **solo** las
  semillas que la alcanzaron (`reached_reference`); los valores por semilla están en
  `cost_to_reference_by_seed`.
- `seed-<n>/`: los archivos de cada modelo (por defecto, `plan.csv`). En Q-learning:
  - `best_observed.csv`: mejor plan observado durante el entrenamiento (el plan entregado).
  - `policy_best.csv`: mejor plan construido con la política aprendida.
  - `training.json` y `q-table.json`: historial de episodios, rollouts y tabla Q.
  - `training.svg`: curva de aprendizaje.

En Q-learning, el presupuesto (`cost_unit`) son las transiciones reales; `cost` también
reporta `simulator_calls` (llamadas a la transición del entorno; no cuentan las consultas
de acciones disponibles) y `updates` (actualizaciones de Q).

En Q-learning, los conteos contra la referencia son `policy_matches` (rollouts de la política
que la igualan), `policy_rollouts` y `policy_always_matches` (semillas en que todos la igualan).

## Cómo añadir un modelo

Basta con dos archivos; no se edita nada más (en `config.yml` solo se nombra el modelo en
`train` o en `experiments`):

1. `src/models/<clave>.py`: la dataclass de hiperparámetros (con sus reglas en
   `__post_init__`) y el estimador, registrado con `@register`:

   ```python
   @dataclass(frozen=True)
   class MiModeloParams:
       width: int = 10
       FIXED: ClassVar[tuple] = ("max_calls",)    # opcional: campos que el tuning no ajusta
       SMOKE: ClassVar[dict] = {"width": 2}       # valores baratos para el test de contrato

   @register("mi_modelo", params=MiModeloParams)
   class MiModelo(MineEstimator):
       cost_unit = "simulator_calls"   # contador en que se mide el presupuesto (default: transitions)
       deterministic = True            # opcional: no depende de la semilla
       def __init__(self, width=10, random_state=0): ...   # mismos nombres y defaults que Params
       def fit(self, instance): ...    # guarda instance_, counters_ y trace_ (ver src/models/base.py)
       def predict(self, instance): ...
       def diagnostics(self): ...
   ```

   La clave, el nombre del módulo y el del YAML coinciden. Un modelo tabular puede heredar de
   `TabularQ` (`src/models/tabular.py`) y redefinir solo `observe`.
2. `configs/<clave>.yml`: `params` (y, si se quiere, `tuning.distributions`).

`tests/test_estimator_contract.py` lo revisa solo (se descubre por su archivo): parámetros
iguales a la dataclass, clonado, `fit`/`predict`/`score` con planes factibles en dos
instancias (una con 2 períodos, capacidad y horizonte external), `counters_` y `trace_`
válidos, reproducibilidad por semilla y que su YAML de `configs/` cargue.

## Tests

```bash
python -m unittest discover -s tests -v
```

`tests/test_golden_q_learning.py` compara Q-learning con huellas guardadas en
`tests/golden/q_learning.json`: mismos resultados por semilla tras cualquier cambio. Las
huellas dependen de la versión de Python, así que ese test solo corre con Python 3.12 (la
del proyecto) y se omite en otra versión.

Incluyen una mina de 6 bloques cuyo óptimo se obtiene por enumeración exhaustiva: en
10 semillas, el mejor plan observado debe alcanzarlo en todas las combinaciones de estado
y horizonte, y la política también, con `state: physical`.
