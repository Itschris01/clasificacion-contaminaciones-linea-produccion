import os
import csv
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

RUTA_ARROZ = "Arroz_C16760"
RUTA_CLIPS = "Clips_C16760"

SALIDA_ARROZ = "arroz_procesado"
SALIDA_CLIPS = "clips_procesado"

ARCHIVO_CSV = "dataset_imagenes.csv"
TAMANO = (128, 128)


def crear_carpeta(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)


def procesar_imagen(ruta_imagen, carpeta_salida, etiqueta):
    try:
        with Image.open(ruta_imagen) as img:
            img = img.convert("RGB")
            img = img.resize(TAMANO, Image.Resampling.LANCZOS)

            nombre_base = os.path.splitext(os.path.basename(ruta_imagen))[0]

            # Guardar version reducida
            ruta_rgb = os.path.join(carpeta_salida, f"{nombre_base}_128.png")
            img.save(ruta_rgb)

            # Convertir a numpy y luego a gris con OpenCV
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Suavizar un poco para reducir ruido
            gris_suave = cv2.GaussianBlur(gris, (5, 5), 0)

            # Umbral adaptativo:
            # fondo blanco = 1
            # objeto = 0
            binaria = cv2.adaptiveThreshold(
                gris_suave,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                8
            )

            # Limpiar pequenos puntos
            kernel = np.ones((3, 3), np.uint8)
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

            # Guardar gris y binaria
            cv2.imwrite(os.path.join(carpeta_salida, f"{nombre_base}_gris.png"), gris)
            cv2.imwrite(os.path.join(carpeta_salida, f"{nombre_base}_binaria.png"), binaria)

            # Convertir a matriz de 1 y 0
            # blanco -> 1, objeto -> 0
            matriz_binaria = (binaria == 255).astype(np.uint8)

            vector = matriz_binaria.flatten().tolist()
            vector.append(etiqueta)

            return vector

    except Exception as e:
        print(f"Error con {ruta_imagen}: {e}")
        return None


def procesar_carpeta(carpeta_entrada, carpeta_salida, etiqueta):
    datos = []
    extensiones_validas = (".heic", ".HEIC", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(extensiones_validas):
            ruta_imagen = os.path.join(carpeta_entrada, archivo)
            vector = procesar_imagen(ruta_imagen, carpeta_salida, etiqueta)
            if vector is not None:
                datos.append(vector)

    return datos


def guardar_csv(datos, ruta_csv):
    encabezado = [f"pixel_{i}" for i in range(128 * 128)] + ["etiqueta"]

    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo, delimiter=';')
        writer.writerow(encabezado)
        writer.writerows(datos)


def main():
    crear_carpeta(SALIDA_ARROZ)
    crear_carpeta(SALIDA_CLIPS)

    datos_arroz = procesar_carpeta(RUTA_ARROZ, SALIDA_ARROZ, 1)
    datos_clips = procesar_carpeta(RUTA_CLIPS, SALIDA_CLIPS, 0)

    dataset_total = datos_arroz + datos_clips
    guardar_csv(dataset_total, ARCHIVO_CSV)

    print("Proceso terminado.")
    print(f"Arroz procesadas: {len(datos_arroz)}")
    print(f"Clips procesadas: {len(datos_clips)}")
    print(f"Archivo generado: {ARCHIVO_CSV}")

    print("Verificacion del dataset:")
    print(f"Total de imagenes: {len(dataset_total)}")
    print(f"Columnas por fila: {len(dataset_total[0])}")
if __name__ == "__main__":
    main()
