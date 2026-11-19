# openPit_proyect

Planificación de la extracción de una mina a cielo abierto con **Q-learning tabular**.
El agente recorre el modelo de bloques y decide qué bloques extraer para maximizar el
VAN del plan, respetando las precedencias del talud. La formulación se basa en
Quelopana & Navarra (2021).

Estado actual: línea base validada en el caso reducido de 105 bloques entregado por el
cliente (VAN óptimo de referencia: 336.000).

## Estructura

```
.
├── main.py                    # Punto de entrada: modos train y evaluate-plan
├── configs/                   # Un YAML + un módulo de lectura por tema (ver Configuración)
│   ├── __init__.py            # PROJECT_ROOT, rutas y lectura estricta común
│   ├── config.yml             # Común a todos los modelos: datos, problema, entorno, semillas
│   ├── config_settings.py
│   ├── q_learning.yml         # Hiperparámetros de Q-learning y sus distribuciones de tuning
│   └── q_learning_settings.py
├── environment.yml            # Entorno conda
├── data/                      # Datos (no se versionan, salvo los ejemplos)
│   ├── raw/                   # Archivos de origen (p. ej. el Excel del cliente)
│   └── processed/             # CSV estándar que lee el pipeline (+ *.example.csv versionados)
├── tools/                     # Conversión de archivos de origen a CSV estándar (ver tools/README.md)
├── src/
│   ├── data/
│   │   ├── dataset.py         # Lectura de bloques y planes; bloques alcanzables
│   │   └── precedences.py     # Regla de precedencia del talud
│   ├── envs/mine_env.py       # Entorno: estados, acciones, extracción de conos y recompensa
│   ├── models/
│   │   ├── base.py            # Interfaz común de los modelos
│   │   ├── q_learning.py      # Q-learning tabular
│   │   └── exploration.py     # Curvas de epsilon (exploración)
│   ├── training/
│   │   ├── rewards.py         # Recompensa: VAN incremental de lo extraído
│   │   └── trainer.py         # Entrenamiento por presupuesto de transiciones y evaluación de la política
│   └── evaluation/
│       ├── metrics.py         # Verificador independiente de planes (factibilidad y VAN)
│       ├── reporter.py        # report.json, planes CSV y trazabilidad
│       ├── plots.py           # Curva de aprendizaje (SVG)
│       ├── benchmark.py       # Óptimo exacto por máximo cierre (caso de 1 período, usado en tests)
│       └── tiny_oracle.py     # Mina de 6 bloques con óptimo por enumeración (usado en tests)
├── tests/                     # Tests automatizados (unittest)
└── artifacts/reports/         # Resultados de cada ejecución (no se versionan)
```

## Instalación

```bash
conda env create -f environment.yml
conda activate openpit
python -m unittest discover -s tests -v
```

Dependencias: Python 3.12, `pyyaml`, `scikit-learn`, `pandas` y `openpyxl` (estas dos
últimas solo para `tools/`). GitHub Actions corre los tests en cada push y pull request.

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
`configs/config.yml`). Ese archivo indica qué modelo correr y dónde están sus hiperparámetros.
Todo lo que define la tarea va en el común y lo que define cómo aprende un modelo va en el
YAML del modelo. Una clave desconocida, en cualquiera de los dos, es un error.

`configs/config.yml` (común):

| Sección | Parámetros |
|---|---|
| raíz | `data`, `solution` (`null` = sin referencia), `output`, `seeds` |
| `problem` | `periods`, `discount_rate`, `capacity` (`null` = sin límite), `capacity_unit` (`blocks` \| `tonnes`) |
| `environment` | `max_steps` (corte de seguridad por episodio), `horizon` (`external` \| `finite`) |
| `model`, `models` | modelo a entrenar y ruta del YAML de cada modelo (por ahora solo `q_learning`) |
| `tuning` | búsqueda aleatoria del modo tuning (aún no implementado), igual para todos los modelos: `n_iter` (configuraciones sorteadas), `search_seed` (semilla del sorteo), `seeds` (semillas de entrenamiento de cada configuración; no pueden repetir `seeds`) |

`configs/q_learning.yml`:

| Sección | Parámetros |
|---|---|
| `params` | `state` (`physical` \| `position`), `alpha`, `gamma`, `epsilon`, `epsilon_schedule` (`linear` \| `constant`), `transitions` (presupuesto por semilla), `policy_rollouts` (planes construidos con la política aprendida) |
| `tuning.distributions` | distribución de cada parámetro de `params` a ajustar: `uniform` y `loguniform` (`low`, `high`) o `choice` (`values`); los que no aparecen quedan fijos. Se valida al leer el archivo. `transitions` y `policy_rollouts` no se ajustan |

Configuración de referencia: `state: physical`, `horizon: external`, `alpha: 1`, `gamma: 1`,
`epsilon: 0.5` con `epsilon_schedule: linear`, 100.000 transiciones.

## Ejecución

```bash
# Entrena un agente por semilla y exporta sus planes
python main.py train --config configs/config.yml [--plot]

# Verifica un plan y recalcula su VAN (sin --plan evalúa la solución del YAML)
python main.py evaluate-plan --config configs/config.yml [--plan RUTA.csv]
```

Cada ejecución crea `artifacts/reports/<fecha>-<modo>-<id>/` con:

- `report.json`: configuración común, modelo y sus hiperparámetros (`model`), versión del
  código, hash de los datos, resultado por semilla y resumen (incluida la comparación con la
  solución de referencia, que se lee **después** de entrenar todas las semillas).
- `seed-<n>/best_observed.csv`: mejor plan observado durante el entrenamiento.
- `seed-<n>/policy_best.csv`: mejor plan construido con la política aprendida.
- `seed-<n>/training.json` y `q-table.json`: historial de episodios, rollouts y tabla Q.
- `seed-<n>/training.svg`: curva de aprendizaje (solo con `--plot`).


## Tests

```bash
python -m unittest discover -s tests -v
```

Incluyen una mina de 6 bloques cuyo óptimo se obtiene por enumeración exhaustiva: en
10 semillas, el mejor plan observado debe alcanzarlo en todas las combinaciones de estado
y horizonte, y la política también, con `state: physical`.
