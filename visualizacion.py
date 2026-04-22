import csv
import numpy as np
from PIL import Image

# usar valor grande pero seguro
csv.field_size_limit(10**7)

with open("dataset_imagenes.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    fila = next(reader)

vector = np.array(fila[:-1], dtype=np.uint8)
imagen = vector.reshape((128, 128)) * 255

img = Image.fromarray(imagen)
img.show()

print("Etiqueta:", fila[-1])
