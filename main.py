import os
import sys

# Asegura que la raiz del proyecto y la carpeta src esten disponibles en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
for path in (BASE_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.autenticador import ejecutar_aplicacion

if __name__ == "__main__":
    ejecutar_aplicacion()
