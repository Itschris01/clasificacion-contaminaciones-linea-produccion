# Model Card: Detección de Contaminaciones (Arroz) con SVM

## Model Name + Version
**Nombre:** Clasificador SVM (Support Vector Machine) con Kernel RBF.
**Versión:** 1.0
**Autor:** Cristhian Rojas Alvarez
**Fecha:** Mayo 2026

## Intended Use / Out-of-scope
**Uso previsto:** Este modelo clásico de Machine Learning fue entrenado como prueba de concepto para detectar visualmente anomalías (específicamente granos de arroz) simulando una línea de producción industrial sobre un fondo blanco.
**Fuera de alcance:** No está diseñado para entornos industriales reales, imágenes a color, ni para clasificar múltiples tipos de objetos simultáneamente. Tampoco es apto para imágenes con resoluciones distintas a 128x128 píxeles.

## Data Summary
El modelo fue entrenado con un conjunto de datos tabular (CSV) creado a partir de imágenes de 128x128 píxeles convertidas a blanco y negro. Los datos fueron recolectados de manera colaborativa por el grupo de trabajo. Debido a la recolección descentralizada, el conjunto presenta variaciones significativas en iluminación, enfoque y calidad de la cámara del teléfono de cada estudiante. Tras un proceso de limpieza morfológica y descarte de imágenes ruidosas (mal umbralizadas), el conjunto final para entrenamiento y pruebas consistió en 182 imágenes.

## Labeling Process
El etiquetado se realizó de forma manual a nivel de archivo antes de aplanar las imágenes. Se utilizó una codificación binaria:
* **Clase 1 (Positivo):** Presencia de granos de arroz en la imagen.
* **Clase 0 (Negativo):** Ausencia de arroz (puede contener la hoja en blanco, aros o clips).
La consistencia del etiquetado dependió del criterio individual de cada estudiante durante la recolección.

## Metrics
El modelo fue evaluado utilizando una partición del 20% de los datos para prueba (37 imágenes en total), asegurando que evaluara datos nunca antes vistos. 

**Métricas Finales obtenidas:**
* **Accuracy (Precisión Global):** 70.27%
* **Precision (Clase 1):** 70%
* **Recall (Clase 1):** 74%
* **F1-Score (Macro):** 70%

*Nota del desarrollador:* Aunque un 70% de precisión refleja que el modelo logra generalizar patrones, el rendimiento se vio afectado por la altísima dimensionalidad de los datos (16,384 características) frente al bajo volumen de muestras limpias disponibles, un reto clásico en el aprendizaje automático tradicional.

## Ethical / Safety Notes
El modelo presenta sesgos relacionados con el entorno de captura. Si la iluminación cambia drásticamente o si se utiliza un color de fondo diferente al blanco estandarizado, el modelo fallará. Además, los umbrales de binarización variaron según la cámara de cada participante, lo que introduce un sesgo hacia las condiciones de luz de los teléfonos con mejores lentes.

## Limitations
* **Pérdida espacial:** Al aplanar la matriz de la imagen a un vector de una sola dimensión, se pierde información importante sobre la forma geométrica de los granos.
* **Oclusión y tamaño:** El modelo puede tener dificultades si los granos de arroz están muy agrupados (oclusión) o si la cámara estaba muy lejos, haciendo que el objeto se vuelva ruido visual de pocos píxeles.
* **Sensibilidad al fondo:** Es altamente dependiente de que el fondo sea perfectamente blanco y limpio tras la aplicación de filtros morfológicos.

## Reproducibility
Para reproducir estos resultados exactos, se debe contar con el archivo `dataset_grupal_super_limpio.csv` (generado por `limpieza_matriz_concatenada.py`) y ejecutar el script principal.
**Comando exacto de ejecución:** `python modelos.py`
**Librerías principales:** scikit-learn, pandas, numpy. (Ver `requirements.txt`).
**Hardware utilizado:** Entrenamiento realizado localmente en CPU estándar de computadora portátil.