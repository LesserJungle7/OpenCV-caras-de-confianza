import cv2

# Rutas del sistema
CARPETA_CONFIANZA = "imagenes/confianza"
CARPETA_INTRUSOS = "imagenes/intrusos"
CARPETA_DATOS = "datos"
ARCHIVO_CONFIG = "datos/usuarios.json"
RUTA_CASCADE = "haarcascades/haarcascade_frontalface_default.xml"

# Parametros de comparacion facial y presencia
UMBRAL_COINCIDENCIA = 0.22
TIEMPO_AUSENCIA_GUARDIA = 10  # Tiempo en segundos sin detectar usuario antes de reactivar la guardia
AREA_MINIMA_CARA = 8000       # Area minima (ancho * alto) para descartar detecciones falsas (Semana 6)
BRILLO_OBJETIVO = 120         # Nivel medio de iluminacion deseado calculado con histograma (Semana 3 y 5)

# Estilos visuales para OpenCV
COLOR_TEXTO = (0, 255, 255)
COLOR_RECUADRO = (0, 255, 0)
GROSOR = 2
FUENTE = cv2.FONT_HERSHEY_COMPLEX
ESCALA = 0.7
