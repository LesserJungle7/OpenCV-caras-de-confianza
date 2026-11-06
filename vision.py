import os
import cv2
from config import RUTA_CASCADE, CARPETA_CONFIANZA, UMBRAL_COINCIDENCIA

def obtener_clasificador():
    """Instancia el clasificador Haar Cascade."""
    return cv2.CascadeClassifier(RUTA_CASCADE)

def detectar_caras(frame, clasificador=None, scale_factor=1.2, min_neighbors=5):
    """
    Detecta rostros en un fotograma.
    Retorna la lista de coordenadas (x, y, w, h) y el frame en escala de grises.
    """
    if clasificador is None:
        clasificador = obtener_clasificador()

    if len(frame.shape) == 3:
        frame_byn = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_byn = frame

    caras = clasificador.detectMultiScale(frame_byn, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    return caras, frame_byn

def cargar_caras_confianza():
    """Carga en memoria los nombres y plantillas de rostros autorizados."""
    lista_nombres = []
    lista_caras = []

    if not os.path.exists(CARPETA_CONFIANZA):
        return lista_nombres, lista_caras

    for archivo in os.listdir(CARPETA_CONFIANZA):
        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            nombre_base = archivo.split("_")[0]
            img = cv2.imread(os.path.join(CARPETA_CONFIANZA, archivo), 0)
            if img is not None:
                lista_nombres.append(nombre_base)
                lista_caras.append(img)

    return lista_nombres, lista_caras

def identificar_rostro(roi_cara_byn, lista_nombres, lista_caras, umbral=UMBRAL_COINCIDENCIA):
    """
    Compara el rostro recortado contra las plantillas guardadas.
    Retorna el nombre correspondiente si supera el umbral, o None en caso contrario.
    """
    mejor_score = 1.0
    nombre_detectado = None
    alto_roi, ancho_roi = roi_cara_byn.shape[:2]

    for indice in range(len(lista_caras)):
        patron = cv2.resize(lista_caras[indice], (ancho_roi, alto_roi))
        res = cv2.matchTemplate(roi_cara_byn, patron, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)

        if min_val < mejor_score:
            mejor_score = min_val
            nombre_detectado = lista_nombres[indice]

    if mejor_score < umbral:
        return nombre_detectado, mejor_score
    return None, mejor_score
