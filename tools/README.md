# tools/ — conversión de datos de origen (fuera del pipeline)

Scripts auxiliares que convierten archivos de origen a los **dos formatos estándar**
que lee el pipeline. `main.py` no los usa ni los importa: el pipeline solo lee CSV.

```
data/raw/<origen>  ──tools/──►  data/processed/data.csv      + data.meta.json
                                data/processed/solution.csv  + solution.meta.json
```

## Formatos estándar

**Bloques** (`data.csv`): una fila por bloque.

| columna | tipo | obligatoria |
|---|---|---|
| `id` | entero único | sí |
| `x`, `y`, `z` | enteros; combinación única; **Z mayor = superficie** | sí |
| `value` | número finito | sí |
| `tonnes` | número positivo (vacío = sin dato) | no |

**Plan / solución** (`solution.csv` y los `plan.csv` que exporta el pipeline): una fila por
bloque extraído. Una solución de referencia es un plan, con o sin orden.

| columna | contenido |
|---|---|
| `order` | posición en la secuencia de extracción (1..n), o **vacío en todas las filas** si el plan es solo un conjunto (el pipeline construye un orden válido para verificarlo) |
| `id` | bloque extraído |
| `period` | período de extracción (entero ≥ 0) |

Ejemplos versionados: `data/processed/data.example.csv` y `data/processed/solution.example.csv`.

## `prepare_blocks.py` (general)

Convierte cualquier tabla CSV o Excel. La hoja y los nombres de columnas son argumentos:

```bash
python -m tools.prepare_blocks data/raw/data.xlsx --surface z-max --sheet "105 Bloques" \
    --map ID=id X=x Y=y Z=z Valor=value
```

`--surface` es obligatorio: indica dónde está la superficie en el archivo de origen
(`z-max` = Z mayor, `z-min` = Z menor). El pipeline asume siempre **Z mayor = superficie**;
con `z-min` el script invierte Z.

Valida cada fila (nulos, no numéricos, coordenadas no enteras, duplicados) y no repara
datos: si algo falla, informa las filas. Las columnas sin mapear se descartan y quedan
registradas en `data.meta.json`, junto al hash del archivo de origen y del CSV generado.

## `extract_aldo_solution.py` (específico)

Lee la solución pintada en la hoja `Resultado` del Excel de Aldo (celdas con el color
de la leyenda `K3`). Está acoplado a ese archivo a propósito.

```bash
python -m tools.extract_aldo_solution data/raw/data.xlsx
```

La solución de Aldo no tiene orden ni períodos: se exporta con `order` vacío y `period=0`.
