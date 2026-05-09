import pandas as pd
import glob
import os

def unificar_datasets():
    # busca todos los archivos CSV en la carpeta de datos
    ruta_datos = "datos_grupo" 
    archivos_csv = glob.glob(os.path.join(ruta_datos, "*.csv"))
    
    lista_df = []
    columnas_esperadas = 128 * 128 + 1  # 16385 columnas en total
    
    print("Iniciando revisión automática de archivos paso a paso...\n")
    
    for archivo in archivos_csv:
        nombre_archivo = os.path.basename(archivo)
        
        # Evitar leer archivos maestros ya generados
        if nombre_archivo in ["dataset_grupal_limpio.csv", "dataset_grupal_super_limpio.csv"]:
            continue
            
        try:
            # PASO 1: Detectar el separador dinámicamente
            with open(archivo, 'r', encoding='utf-8') as f:
                primera_linea = f.readline()
                separador = ';' if ';' in primera_linea else ','
            
            # PASO 2: Leer el CSV asumiendo inicialmente que no hay encabezado
            df = pd.read_csv(archivo, sep=separador, header=None, low_memory=False)
            
            # PASO 3: Limpieza de encabezado
            # Si el archivo tiene 31 filas (1 encabezado + 30 fotos) o si la primera celda es texto,
            # eliminamos la fila 0 para quedarnos solo con las 30 imágenes.
            if df.shape[0] == 31 or isinstance(df.iloc[0, 0], str):
                df = df.iloc[1:].reset_index(drop=True)
            
            # PASO 4: Convertir todo a numérico (fuerza errores a NaN si queda basura)
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # PASO 5: Verificación de dimensiones
            if df.shape[1] == columnas_esperadas:
                lista_df.append(df)
                print(f" {nombre_archivo}: OK ({df.shape[0]} imágenes)")
            else:
                print(f" {nombre_archivo}: ERROR de formato. Tiene {df.shape[1]} columnas, se esperaban {columnas_esperadas}.")
                
        except Exception as e:
            print(f" Error procesando {nombre_archivo}: {e}")

    # PASO 6: Concatenar y guardar
    if lista_df:
        print("\nConcatenando archivos válidos...")
        df_final = pd.concat(lista_df, ignore_index=True)
        
        # Estandarizar el encabezado para el dataset maestro
        nombres_columnas = [f"pixel_{i}" for i in range(16384)] + ["etiqueta"]
        df_final.columns = nombres_columnas
        
        ruta_salida = os.path.join(ruta_modelos, "dataset_matriz_concatenada.csv")
        df_final.to_csv(ruta_salida, index=False, sep=';')
        
        print(f"¡Éxito! Dataset maestro creado: {ruta_salida}")
        print(f"Total de imágenes listas para entrenar: {df_final.shape[0]}")
    else:
        print("\nNo se encontraron archivos válidos para unir.")

if __name__ == "__main__":
    unificar_datasets()