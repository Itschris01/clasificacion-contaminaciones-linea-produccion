import pandas as pd
import numpy as np
import cv2

def limpiar_datos_csv():
    # LEE EL ARCHIVO CONCATENADO
    archivo_entrada = "dataset_matriz_concatenada.csv"
    
    print(f"Cargando {archivo_entrada} para limpieza morfológica...")
    try:
        df = pd.read_csv(archivo_entrada, sep=';')
    except FileNotFoundError:
        print(f" Error: No se encontró {archivo_entrada}. Corre primero el script de concatenación.")
        return

    X = df.drop('etiqueta', axis=1).values
    y = df['etiqueta'].values
    
    X_limpio, y_limpio = [], []
    filas_eliminadas = 0
    
    for i in range(len(X)):
        fila, etiqueta = X[i], y[i]
        porcentaje_blanco = np.mean(fila)
        
        # Filtro de Outliers
        if porcentaje_blanco > 0.995 or porcentaje_blanco < 0.50:
            filas_eliminadas += 1
            continue
            
        # Limpieza Morfología (Opening/Closing)
        imagen_2d = fila.reshape(128, 128).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        imagen_limpia = cv2.morphologyEx(imagen_2d, cv2.MORPH_OPEN, kernel)
        imagen_limpia = cv2.morphologyEx(imagen_limpia, cv2.MORPH_CLOSE, kernel)
        
        X_limpio.append(imagen_limpia.flatten())
        y_limpio.append(etiqueta)

    df_nuevo = pd.DataFrame(X_limpio, columns=[f"pixel_{i}" for i in range(16384)])
    df_nuevo['etiqueta'] = y_limpio
    
    # Mantiene este nombre para que el script de modelos.py no falle
    df_nuevo.to_csv("dataset_grupal_super_limpio.csv", index=False, sep=';')
    
    print(f"Limpieza terminada. Eliminadas {filas_eliminadas} imágenes ruidosas.")
    print(f"Archivo final listo: dataset_grupal_super_limpio.csv")

if __name__ == "__main__":
    limpiar_datos_csv()