# Proyecto: Clasificación de Contaminaciones en Línea de Producción

## Descripción General
Este repositorio contiene el sistema completo de Machine Learning desarrollado para detectar granos de arroz sobre superficies estandarizadas. El flujo abarca desde la binarización de imágenes originales hasta la exportación de un clasificador **SVM**.

## Estructura del Repositorio

```text
.
├── 📁 __pycache__/                      # Archivos temporales de Python
│
├── 📁 Formato_imagenes_I_etapa/         # Procesamiento inicial de imágenes
│   ├── 📁 Arroz_C16760/                # Imágenes originales de arroz
│   ├── 📁 arroz_procesado/             # Imágenes de arroz binarizadas (128x128)
│   ├── 📁 Clips_C16760/                # Imágenes originales de clips
│   ├── 📁 clips_procesado/             # Imágenes de clips binarizadas (128x128)
│   ├── dataset_imagenes.csv            # Dataset generado a partir de imágenes
│   ├── formato.py                      # Script de procesamiento y binarización inicial
│   └── visualizacion.py                # Visualización de datos/imágenes
│
├── 📁 Modelos/                          # Desarrollo y entrenamiento de modelos
│   ├── 📁 Datos grupo/                 # Datasets individuales del grupo
│   │   ├── conjunto_de_datos.csv
│   │   ├── dataset(in).csv
│   │   ├── dataset.csv
│   │   ├── dataset_3_3.csv
│   │   ├── dataset_C26797.csv
│   │   ├── dataset_carlos.csv
│   │   ├── dataset_daniel_valverde.csv
│   │   ├── dataset_danna.csv
│   │   ├── dataset_imagenes.csv
│   │   └── matriz_final_alex.csv
│   │
│   ├── C16760_Cristhian_Rojas_Alvarez.joblib  # Modelo ganador exportado (SVM)
│   ├── concatenar_matrices.py          # Script 1: Unificación de CSVs del grupo
│   ├── dataset_grupal_super_limpio.csv # Dataset final tras limpieza
│   ├── dataset_matriz_concatenada.csv  # Resultado de la concatenación grupal
│   ├── limpieza_matriz_concatenada.py  # Script 2: Filtros OpenCV y limpieza
│   └── models.py                       # Script 3: Entrenamiento y evaluación de modelos
│
├── 📁 reports/                          # Reportes e informe final 
│
├── .gitignore                           # Archivos omitidos por Git (__pycache__, etc.)
├── DATASET                              # Documentación de los datos
├── MODEL_CARD                           # Métricas y detalles del modelo
├── README                               # Guía de uso (este archivo)
└── requirements                         # Dependencias del proyecto (librerías)
```

## Flujo de Trabajo

1. **Procesamiento de Imágenes** (`Formato_imagenes_I_etapa/`)
   - Binarización de imágenes originales → `formato.py`
   - Generación de datasets individuales → `dataset_imagenes.csv`
   - Visualización de resultados → `visualizacion.py`

2. **Consolidación de Datos** (`Modelos/`)
   - Unificación de datasets del grupo → `concatenar_matrices.py`
   - Limpieza y filtrado → `limpieza_matriz_concatenada.py`
   - Dataset final → `dataset_grupal_super_limpio.csv`

3. **Entrenamiento y Exportación** (`Modelos/`)
   - Entrenamiento de modelos → `models.py`
   - Modelo final → `C16760_Cristhian_Rojas_Alvarez.joblib`

## Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `formato.py` | Script de binarización inicial de imágenes |
| `concatenar_matrices.py` | Unificación de CSVs individuales del grupo |
| `limpieza_matriz_concatenada.py` | Aplicación de filtros OpenCV y limpieza de datos |
| `models.py` | Entrenamiento, evaluación y exportación del modelo SVM |
| `C16760_Cristhian_Rojas_Alvarez.joblib` | Modelo clasificador final entrenado |
| `DATASET` | Documentación completa de los datos utilizados |
| `MODEL_CARD` | Métricas de rendimiento y especificaciones del modelo |

---

**Nota:** Consulta `DATASET` y `MODEL_CARD` para información detallada sobre los datos y el modelo respectivamente.
