# DATASET: Clasificación de Contaminaciones en Línea de Producción

## 1. Descripción General
Este conjunto de datos tabular contiene representaciones vectorizadas de imágenes en blanco y negro, diseñado para entrenar modelos de aprendizaje automático clásico en la detección de anomalías. El objetivo principal es identificar la presencia de granos de arroz (contaminación) sobre un fondo blanco que simula una línea de producción industrial.

## 2. Proceso de Recolección
La recolección de los datos fue un esfuerzo colaborativo de todo el grupo de trabajo. El proceso se llevó a cabo bajo las siguientes condiciones:
* **Entorno simulado:** Se utilizó una hoja blanca como fondo para simular la línea de producción.
* **Objetos:** Las contaminaciones positivas corresponden a granos de arroz. Los ejemplos negativos incluyen la ausencia de objetos o la presencia de otros elementos como aros o clips.
* **Aporte individual:** Cada estudiante aportó un total de 30 fotografías, divididas equitativamente en 15 imágenes de ejemplos positivos y 15 de ejemplos negativos.

## 3. Preprocesamiento de las Imágenes
Antes de consolidar el conjunto de datos, cada imagen original en formato fotográfico pasó por el siguiente flujo de preprocesamiento:
1. **Escalado y Binarización:** Cada imagen fue redimensionada a 128x128 píxeles y convertida a blanco y negro. 
2. **Transformación a Matriz:** Se generó una matriz donde el valor `1` representa un píxel completamente blanco (fondo) y el valor `0` representa un píxel con presencia de un objeto.
3. **Aplanamiento (Flattening):** La matriz bidimensional se convirtió en un vector fila de 16,384 columnas (128x128).
4. **Etiquetado:** Se agregó una última columna (columna 16,385) con la etiqueta objetivo: `1` para indicar la presencia de arroz y `0` para indicar su ausencia.

## 4. Limpieza Avanzada y Consolidación (Pipeline)
Debido a la naturaleza colaborativa de la recolección, los datos crudos presentaban múltiples inconsistencias de formato y ruido visual. Se desarrolló un pipeline de limpieza automático en dos etapas:

* **Filtro Estructural (`concatenar_matrices.py`):** Se detectaron y eliminaron aportes defectuosos que carecían de la columna de etiquetas (archivos con 16,384 columnas) y se estandarizaron los encabezados para asegurar la uniformidad matricial de todos los estudiantes.
* **Filtro Morfológico y Outliers (`limpieza_matriz_concatenada.py`):** Se implementaron técnicas de visión por computadora (OpenCV). Se descartaron imágenes con umbralización extrema (más del 99.5% blancas o menos del 50% blancas). A las imágenes restantes se les aplicaron transformaciones morfológicas de Apertura (Opening) y Cierre (Closing) para eliminar el ruido de fondo y rellenar vacíos dentro de los píxeles de los objetos.

El dataset final utilizado para entrenar los modelos (`dataset_grupal_super_limpio.csv`) es el resultado directo de este exhaustivo proceso de curación de datos.

## 5. Limitaciones Conocidas
* **Falta de uniformidad en la captura:** Al utilizar diferentes dispositivos móviles para fotografiar la hoja blanca, las condiciones de iluminación, enfoque y perspectiva varían drásticamente entre los lotes de cada estudiante.
* **Sensibilidad a la binarización:** La calidad del dataset depende en gran medida del umbral seleccionado por cada estudiante durante el paso a blanco y negro. Si el umbral fue muy agresivo, las sombras se convirtieron en objetos falsos; si fue muy débil, los objetos pequeños desaparecieron.
* **Dimensiones planas:** Al convertir la imagen en un vector de una dimensión, los algoritmos clásicos de clasificación pierden el contexto de adyacencia espacial de los pixeles, lo que dificulta la detección de formas complejas frente a modelos modernos de Deep Learning.