import os
import cv2

# Rutas del sistema
CARPETA_CONFIANZA = "imagenes/confianza"
CARPETA_INTRUSOS = "imagenes/intrusos"
CARPETA_DATOS = "datos"
CARPETA_AUDIOS = "audios"
ARCHIVO_CONFIG = "datos/usuarios.json"
RUTA_CASCADE_FACE = "haarcascades/haarcascade_frontalface_default.xml"
RUTA_CASCADE_EYE = "haarcascades/haarcascade_eye.xml"
RUTA_CASCADE = RUTA_CASCADE_FACE

# Rutas de audios de feedback
AUDIO_SUCCESS = os.path.join(CARPETA_AUDIOS, "success.mp3")
AUDIO_ERROR = os.path.join(CARPETA_AUDIOS, "error.mp3")

# Parametros de comparacion facial y presencia
TAM_CARA = (120, 120)              # Dimension fija para normalizacion de rostros
MARGEN_RECORTE_CARA = 0.08         # Fraccion de margen a descartar en bordes (8 %)
ALINEAR_OJOS = True                # Interruptor para activar/desactivar alineacion por ojos
ANGULO_MAXIMO_ALINEACION = 25      # Grados maximos de inclinacion admitidos para rotar
UMBRAL_COINCIDENCIA = 0.07
TIEMPO_AUSENCIA_GUARDIA = 10  # Tiempo en segundos sin detectar usuario antes de reactivar la guardia
AREA_MINIMA_CARA = 8000       # Area minima (ancho * alto) para descartar detecciones falsas
BRILLO_OBJETIVO = 120         # Nivel medio de iluminacion deseado calculado con histograma

# Parametros de estabilidad y fijacion de rostro en guardia
FRAMES_ESTABILIDAD_GUARDIA = 12     # Fotogramas consecutivos con rostro quieto requeridos
UMBRAL_MOVIMIENTO_ESTABLE = 14      # Desplazamiento maximo en pixeles del centroide entre frames
VARIANZA_MINIMA_TEXTURA = 65.0      # Varianza minima del Laplaciano para descartar oclusiones lisas

# Gestos disponibles para el reto facial (prueba de vida)
GESTOS_DISPONIBLES = [
    "levantar_cejas",
    "cerrar_ambos_ojos",
    "cerrar_ojo_izq",
    "cerrar_ojo_der",
    "sonreir"
]

GESTOS_DESCRIPCION = {
    "levantar_cejas": "Levanta las cejas",
    "cerrar_ambos_ojos": "Cierra ambos ojos",
    "cerrar_ojo_izq": "Cierra el ojo izquierdo",
    "cerrar_ojo_der": "Cierra el ojo derecho",
    "sonreir": "Sonrie mostrando los dientes"
}

# Parametros de validacion de gestos
TIEMPO_MAXIMO_GESTO = 30.0         # Segundos permitidos por cada gesto
FRAMES_CONFIRMACION_GESTO = 8      # Fotogramas consecutivos requeridos para confirmar
FRAMES_CALIBRACION = 12            # Fotogramas iniciales para calibrar rostro neutro

# Parametros para deteccion de ojos (Regla K y ventana temporal)
FACTOR_K_CIERRE_OJO = 0.78          # Fraccion del aspecto neutro para considerar ojo cerrado
VENTANA_TEMPORAL_GESTO = 8          # Tamano de la ventana deslizante para gestos de ojos
MIN_FRAMES_CONFIRMACION_GESTO = 6   # Fotogramas minimos requeridos dentro de la ventana

# Discriminacion colorimetrica de esmalte dental
DIF_MAXIMA_ROJO_AZUL_DIENTES = 35  # Diferencia maxima |R - B| para aceptar tono marfil neutro
DIF_MAXIMA_ROJO_VERDE_DIENTES = 30 # Diferencia maxima |R - G|
ANCHO_MINIMO_FRANJA_DIENTES = 14   # Anchura minima del contorno de dientes en pixeles
PORCENTAJE_MINIMO_DIENTES = 0.04   # Porcentaje minimo de area de dientes en centro labial

# Estilos visuales estandarizados (formato BGR de OpenCV)
COLOR_TEXTO = (0, 255, 255)
COLOR_TEXTO_HUD = (255, 255, 255)
COLOR_INACTIVO = (0, 0, 255)       # Rojo para estados en reposo
COLOR_ACTIVO = (0, 255, 0)         # Verde para estados activos
COLOR_RECUADRO = COLOR_ACTIVO
COLOR_REGION_ACTIVA = COLOR_ACTIVO
COLOR_RETO_ACTIVO = COLOR_ACTIVO
GROSOR = 2
FUENTE = cv2.FONT_HERSHEY_COMPLEX
ESCALA = 0.7
