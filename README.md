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
├── configs/config.yml         # Configuración de una ejecución (datos, problema, agente, entrenamiento)
├── environment.yml            # Entorno conda
├── data/                      # Datos (no se versionan, salvo los ejemplos)
│   ├── raw/                   # Archivos de origen (p. ej. el Excel del cliente)
│   └── processed/             # CSV estándar que lee el pipeline (+ *.example.csv versionados)
├── tools/                     # Conversión de archivos de origen a CSV estándar (ver tools/README.md)
├── src/
│   ├── config.py              # Lectura y validación del YAML
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

Dependencias: Python 3.12, `pyyaml`, `pandas` y `openpyxl` (estas dos últimas solo para
`tools/`). GitHub Actions corre los tests en cada push y pull request.

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

Toda ejecución se define en un YAML (por defecto `configs/config.yml`):

| Sección | Parámetros |
|---|---|
| raíz | `data`, `solution` (`null` = sin referencia), `output`, `seeds` |
| `problem` | `periods`, `discount_rate`, `capacity` (`null` = sin límite), `capacity_unit` (`blocks` \| `tonnes`) |
| `agent` | `state` (`physical` \| `position`), `alpha`, `gamma`, `epsilon`, `epsilon_schedule` (`linear` \| `constant`) |
| `training` | `transitions` (presupuesto por semilla), `max_steps` (corte de seguridad por episodio), `horizon` (`external` \| `finite`) |
| `evaluation` | `policy_rollouts` (planes construidos con la política aprendida) |

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

- `report.json`: configuración, versión del código, hash de los datos, resultado por
  semilla y resumen (incluida la comparación con la solución de referencia, que se lee
  **después** de entrenar todas las semillas).
- `seed-<n>/best_observed.csv`: mejor plan observado durante el entrenamiento.
- `seed-<n>/policy_best.csv`: mejor plan construido con la política aprendida.
- `seed-<n>/training.json` y `q-table.json`: historial de episodios, rollouts y tabla Q.
- `seed-<n>/training.svg`: curva de aprendizaje (solo con `--plot`).

## Modelo

**Precedencias.** Talud de 45° con bloques cúbicos: un bloque requiere los 9 bloques
inmediatamente superiores. Un bloque cuyo cono superior no está completo dentro del
modelo (bordes) es **inextraíble**. Regla confirmada por el cliente.

**Entorno.** Un cursor parte en el bloque de superficie de mayor valor y se mueve en
6 direcciones (N, S, E, O, arriba, abajo). Al llegar a un bloque nuevo extrae ese bloque
y todo su cono pendiente, por lo que todo plan intermedio es factible. Pasar por bloques
ya extraídos solo mueve el cursor. Acciones adicionales: `stop` y, con varios períodos,
`advance_period`.

**Recompensa.** VAN incremental de los bloques extraídos en el paso:
`valor / (1 + discount_rate)^período`.

**Q-learning.** Actualización estándar `Q ← Q + α·(r + γ·max Q(s',a') − Q)`, sin valor
futuro cuando la tarea termina de verdad.

- `state: physical`: la tabla Q distingue posición y bloques extraídos (estado markoviano).
  `position` solo recuerda la posición; con `horizon: external` produce ciclos y no se
  recomienda.
- `horizon: external`: `max_steps` solo interrumpe episodios y el agente no ve el reloj.
  `finite`: los pasos restantes son parte del estado.

**Exploración.** Epsilon-greedy con epsilon recalculado en cada transición
(`src/models/exploration.py`). `linear` baja de `epsilon` a 0 a lo largo del presupuesto;
`constant` lo mantiene fijo, para comparar. La exploración al azar nunca elige `stop`:
`stop` solo se elige cuando es la acción de mayor Q.

**Resultado de cada semilla.**
- *Mejor observado*: plan de mayor VAN visto durante el entrenamiento.
- *Política*: planes construidos sin exploración siguiendo la mayor Q (los empates se
  rompen al azar, por eso se construyen varios para medir consistencia).

Todos los planes pasan por el verificador independiente (`src/evaluation/metrics.py`),
que rechaza bloques desconocidos o repetidos, precedencias violadas, períodos inválidos
y capacidad excedida.

## Tests

```bash
python -m unittest discover -s tests -v
```

Incluyen una mina de 6 bloques cuyo óptimo se obtiene por enumeración exhaustiva: en
10 semillas, el mejor plan observado debe alcanzarlo en todas las combinaciones de estado
y horizonte, y la política también, con `state: physical`.
