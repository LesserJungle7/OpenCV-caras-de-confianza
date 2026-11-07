import os
import cv2
import numpy as np
from config import (
    RUTA_CASCADE,
    CARPETA_CONFIANZA,
    UMBRAL_COINCIDENCIA,
    AREA_MINIMA_CARA,
    BRILLO_OBJETIVO
)

def obtener_clasificador():
    """Instancia el clasificador Haar Cascade."""
    return cv2.CascadeClassifier(RUTA_CASCADE)

def ajustar_brillo_histograma(frame, brillo_objetivo=BRILLO_OBJETIVO):
    """
    Paso 1: Usamos el histograma (Semana 3) para calcular el nivel medio de luz,
    y si la imagen está oscura, aumentamos el brillo en el canal V de HSV (Semana 5).
    """
    if len(frame.shape) == 3:
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gris = frame

    # Calculamos el histograma en escala de grises como en Semana 3 (histograma.py)
    hist = cv2.calcHist([frame_gris], [0], None, [256], [0, 256])
    total_pixeles = np.sum(hist)
    if total_pixeles == 0:
        return frame

    # Sacamos el brillo medio multiplicando cada nivel de luz por su frecuencia en el histograma
    brillo_medio = int(np.sum(np.arange(256) * hist.flatten()) / total_pixeles)

    # Si la escena está poco iluminada, le subimos el brillo con la función vista en Semana 5
    if brillo_medio < brillo_objetivo and len(frame.shape) == 3:
        incremento = brillo_objetivo - brillo_medio
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Evitamos pasar de 255 sumando el incremento con tope, igual que en clase
        limite = 255 - incremento
        v[v > limite] = 255
        v[v <= limite] += incremento

        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame

def detectar_caras(frame, clasificador=None, scale_factor=1.15, min_neighbors=5, area_min=AREA_MINIMA_CARA):
    """
    Detecta rostros tras corregir la iluminacion y filtra falsos positivos por area minima.
    Retorna la lista de coordenadas (x, y, w, h) y el frame en escala de grises.
    """
    if clasificador is None:
        clasificador = obtener_clasificador()

    # Paso 1: Ajustamos el brillo antes de la deteccion para que el clasificador funcione mejor
    frame_ajustado = ajustar_brillo_histograma(frame)

    if len(frame_ajustado.shape) == 3:
        frame_byn = cv2.cvtColor(frame_ajustado, cv2.COLOR_BGR2GRAY)
    else:
        frame_byn = frame_ajustado

    caras = clasificador.detectMultiScale(frame_byn, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Paso 3: Filtramos las detecciones por area minima (Semana 6) para descartar sombras o falsos positivos pequeños
    caras_filtradas = []
    for (x, y, ancho, alto) in caras:
        if (ancho * alto) >= area_min:
            caras_filtradas.append((x, y, ancho, alto))

    return caras_filtradas, frame_byn

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
    Compara el rostro recortado contra las plantillas guardadas usando matchTemplate.
    Aplica suavizado con blur y redimensiona con la interpolacion adecuada de clase.
    """
    mejor_score = 1.0
    nombre_detectado = None

    # Paso 2: Usamos cv2.blur con kernel (3, 3) como en Semana 3 para quitar ruido y grano de la camara
    roi_suavizada = cv2.blur(roi_cara_byn, (3, 3))
    alto_roi, ancho_roi = roi_suavizada.shape[:2]

    for indice in range(len(lista_caras)):
        patron = lista_caras[indice]

        # Paso 5: Como vimos en Semana 3 (Escalado.py), elegimos la interpolacion segun si agrandamos o reducimos:
        # INTER_AREA para reducir (mas nítida) e INTER_LINEAR para ampliar sin pixelar
        if ancho_roi < patron.shape[1] or alto_roi < patron.shape[0]:
            tipo_interpolacion = cv2.INTER_AREA
        else:
            tipo_interpolacion = cv2.INTER_LINEAR

        patron_redimensionado = cv2.resize(patron, (ancho_roi, alto_roi), interpolation=tipo_interpolacion)

        # Suavizamos tambien la plantilla para que la comparacion de pixeles sea justa
        patron_suavizado = cv2.blur(patron_redimensionado, (3, 3))

        # Busqueda por plantilla con TM_SQDIFF_NORMED (Semana 4: busquedaSimilitud.py y almacenar-buscarCaras.py)
        res = cv2.matchTemplate(roi_suavizada, patron_suavizado, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)

        if min_val < mejor_score:
            mejor_score = min_val
            nombre_detectado = lista_nombres[indice]

    if mejor_score < umbral:
        return nombre_detectado, mejor_score
    return None, mejor_score
