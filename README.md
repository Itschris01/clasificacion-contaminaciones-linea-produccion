# Formato de Imagenes C16760

Repositorio para la etapa de preprocesamiento del proyecto: convertir imagenes de arroz y clips en una representacion matricial binaria lista para analisis o entrenamiento.

## Objetivo

El flujo principal de este proyecto hace lo siguiente:

1. Lee imagenes originales desde las carpetas de cada clase.
2. Redimensiona cada imagen a `128x128`.
3. Genera versiones en gris y binarias.
4. Convierte la imagen binaria en una matriz de `0` y `1`.
5. Aplana esa matriz y la guarda en `dataset_imagenes.csv`.

## Estructura

```text
.
|-- Arroz_C16760/
|-- Clips_C16760/
|-- arroz_procesado/
|-- clips_procesado/
|-- formato.py
|-- visualizacion.py
|-- dataset_imagenes.csv
|-- requirements.txt
|-- .gitignore
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecutar el procesamiento

```bash
python formato.py
```

Esto genera:

- Imagen redimensionada: `*_128.png`
- Imagen en escala de grises: `*_gris.png`
- Imagen binaria: `*_binaria.png`
- Dataset tabular: `dataset_imagenes.csv`

## Visualizar una fila del dataset

```bash
python visualizacion.py --fila 0
```

## Etiquetas usadas

- `1`: arroz
- `0`: clip

## Nota tecnica

Cada fila del CSV contiene `128 x 128 = 16384` pixeles binarios y una columna final llamada `etiqueta`.
