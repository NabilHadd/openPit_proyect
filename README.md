# openPit_proyect

Optimización de la planificación de minas a rajo abierto. Como baseline se replica el enfoque de Q-learning de Quelopana & Navarra (2021). El repositorio está pensado para incorporar más modelos.

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
