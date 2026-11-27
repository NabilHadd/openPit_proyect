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

## `extract_solution.py` (formato del cliente)

Lee una solución **pintada** en un Excel con el formato que usa el cliente y la exporta como
`solution.csv`. No depende del tamaño de la grilla ni del número de períodos.

```bash
python -m tools.extract_solution data/raw/data.xlsx              # 105 bloques, 1 período
python -m tools.extract_solution data/raw/Bloques495_2p.xlsx     # 495 bloques, 2 períodos
```

| Argumento | Default | Qué indica |
|---|---|---|
| `source` | — | Excel de origen; relativo al repositorio o absoluto |
| `--sheet` | `Resultado` | hoja con la solución pintada |
| `--blocks-sheet` | la única hoja con encabezados `ID, X, Y, Z` | hoja con la tabla de bloques |
| `--out` | `data/processed/solution.csv` | CSV de salida (el `.meta.json` va al lado) |

Qué espera encontrar en la hoja de la solución:

- **Paneles por nivel**: una etiqueta `Z=<nivel>` en la columna A; los valores de X en una fila
  (desde la columna C) y los de Y en la columna B. El tamaño de cada panel se lee de esas
  filas y columnas. Cada celda tiene el ID de un bloque y debe coincidir con sus coordenadas
  en la tabla de bloques; la vista debe contener todos los bloques una sola vez.
- **Leyenda**: cada celda pintada y vacía con un texto a su derecha. Si los textos nombran un
  período (`Bloques extraídos Período 2`), cada color es ese período; si no, debe haber una
  sola leyenda y todo lo pintado con su color va al período 0. Las celdas pintadas con un
  color que no está en la leyenda (por ejemplo, el amarillo de los bloques considerados) no
  forman parte de la solución.

Salida:

- `solution.csv` con `order` vacío (la solución es un conjunto por período) y
  `period` = período del Excel − 1 (`Período 1` → `0`).
- `solution.meta.json` con el hash del Excel, la hoja de bloques usada, los paneles leídos
  (nivel, columnas, filas), las leyendas con su período, los **bloques por período** y los
  **valores informados** por el Excel junto a un rótulo `Valor Máximo` o `Capacidad Mina`
  (número a la derecha o debajo). Esos valores son solo informativos: sirven para comprobar
  con `evaluate-plan` que la extracción reproduce el VAN del cliente.

La tool **no deduce** las variables del problema. `periods`, `capacity`, `capacity_unit` y
`discount_rate` se fijan en `configs/config.yml`; por ejemplo, el caso de 495 bloques informa
2 períodos y capacidad 108 bloques por período, pero la tasa de descuento no viene en el
Excel y se debe definir aparte.

Errores que detiene en vez de adivinar: leyendas que mezclan textos con y sin período, varias
leyendas sin período, dos leyendas con el mismo color o el mismo período, `Período 0`,
paneles sin valores de X o Y, IDs que no corresponden a sus coordenadas, bloques repetidos o
faltantes, y ninguna celda pintada con un color de la leyenda.
