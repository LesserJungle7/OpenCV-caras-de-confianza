import os
import cv2
import numpy as np
from config import (
    RUTA_CASCADE,
    RUTA_CASCADE_EYE,
    CARPETA_CONFIANZA,
    UMBRAL_COINCIDENCIA,
    AREA_MINIMA_CARA,
    BRILLO_OBJETIVO,
    TAM_CARA,
    MARGEN_RECORTE_CARA,
    ALINEAR_OJOS,
    ANGULO_MAXIMO_ALINEACION
)

# Instancias globales a nivel de modulo
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_clasificador_ojos = None
if os.path.exists(RUTA_CASCADE_EYE):
    try:
        _clasificador_ojos = cv2.CascadeClassifier(RUTA_CASCADE_EYE)
        if _clasificador_ojos.empty():
            _clasificador_ojos = None
    except Exception:
        _clasificador_ojos = None

if ALINEAR_OJOS and _clasificador_ojos is None:
    print(f"[AVISO] ALINEAR_OJOS está activo pero no se pudo cargar el clasificador de ojos desde '{RUTA_CASCADE_EYE}'. La alineación queda desactivada.")

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

    # Calculamos el histograma en escala de grises
    hist = cv2.calcHist([frame_gris], [0], None, [256], [0, 256])
    total_pixeles = np.sum(hist)
    if total_pixeles == 0:
        return frame

    # Sacamos el brillo medio multiplicando cada nivel de luz por su frecuencia en el histograma
    brillo_medio = int(np.sum(np.arange(256) * hist.flatten()) / total_pixeles)

    # Si la escena está poco iluminada, le subimos el brillo
    if brillo_medio < brillo_objetivo and len(frame.shape) == 3:
        incremento = brillo_objetivo - brillo_medio
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Evitamos pasar de 255 sumando el incremento con tope
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

    # Paso 1: Ajustamos el brillo antes de la detección para que el clasificador funcione mejor
    frame_ajustado = ajustar_brillo_histograma(frame)

    if len(frame_ajustado.shape) == 3:
        frame_byn = cv2.cvtColor(frame_ajustado, cv2.COLOR_BGR2GRAY)
    else:
        frame_byn = frame_ajustado

    caras = clasificador.detectMultiScale(frame_byn, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Paso 3: Filtramos las detecciones por area mínima para descartar sombras o falsos positivos pequeños.
    caras_filtradas = []
    for (x, y, ancho, alto) in caras:
        if (ancho * alto) >= area_min:
            caras_filtradas.append((x, y, ancho, alto))

    return caras_filtradas, frame_byn

def alinear_ojos(cara):
    """
    Nivela horizontalmente los ojos si se detectan ambos en la mitad superior de la cara.
    No rota si no hay exactamente un ojo a la izquierda y otro a la derecha,
    si la separacion entre ellos es anomala o si el angulo supera ANGULO_MAXIMO_ALINEACION.
    """
    if _clasificador_ojos is None:
        return cara

    alto, ancho = cara.shape[:2]
    if alto < 40 or ancho < 40:
        return cara

    # 1. Buscar ojos solo entre el 10 % y 60 % de la altura de la cara
    y1_ojos = int(alto * 0.10)
    y2_ojos = int(alto * 0.60)
    roi_ojos = cara[y1_ojos:y2_ojos, :]

    min_s = (int(ancho * 0.12), int(alto * 0.10))
    max_s = (int(ancho * 0.45), int(alto * 0.35))

    ojos = _clasificador_ojos.detectMultiScale(
        roi_ojos,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=min_s,
        maxSize=max_s
    )

    # 2. Clasificar detecciones en mitad izquierda y derecha
    ojos_izq = []
    ojos_der = []
    mitad_x = ancho / 2.0

    for (ox, oy, ow, oh) in ojos:
        cx = ox + ow / 2.0
        cy = y1_ojos + oy + oh / 2.0
        if cx < mitad_x:
            ojos_izq.append((cx, cy))
        else:
            ojos_der.append((cx, cy))

    # Exigir exactamente un ojo en cada hemisferio
    if len(ojos_izq) != 1 or len(ojos_der) != 1:
        return cara

    cx_izq, cy_izq = ojos_izq[0]
    cx_der, cy_der = ojos_der[0]

    # 3. Distancia horizontal razonable entre ojos (25 % a 65 % del ancho facial)
    dist_h = cx_der - cx_izq
    if not (0.25 * ancho <= dist_h <= 0.65 * ancho):
        return cara

    # 4. Calcular angulo de inclinacion respecto a la horizontal
    dx = cx_der - cx_izq
    dy = cy_der - cy_izq
    angulo = float(np.degrees(np.arctan2(dy, dx)))

    if abs(angulo) > ANGULO_MAXIMO_ALINEACION:
        return cara

    # 5. Rotar preservando bordes con BORDER_REPLICATE
    centro = (ancho / 2.0, alto / 2.0)
    matriz_rot = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(cara, matriz_rot, (ancho, alto), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def normalizar_cara(gris, caja=None):
    """
    Normaliza un rostro garantizando salida fija de TAM_CARA:
    1. Extrae el recorte crudo segun 'caja' o toma 'gris' directo si caja=None.
    2. Si ALINEAR_OJOS=True, alinea la cara rotando antes de recortar margen.
    3. Aplica recorte ajustado descartando MARGEN_RECORTE_CARA (~8 %).
    4. Redimensiona a tamano fijo TAM_CARA con INTER_AREA.
    5. Aplica CLAHE global.
    6. Aplica filtro GaussianBlur (3x3).
    """
    if len(gris.shape) == 3:
        gris = cv2.cvtColor(gris, cv2.COLOR_BGR2GRAY)

    if caja is not None:
        x, y, w, h = caja
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gris.shape[1], x + w)
        y2 = min(gris.shape[0], y + h)
        cara = gris[y1:y2, x1:x2]
    else:
        cara = gris

    if cara.size == 0:
        return np.zeros(TAM_CARA, dtype=np.uint8)

    # Alineacion previa al recorte de margen
    if ALINEAR_OJOS and _clasificador_ojos is not None:
        cara = alinear_ojos(cara)

    # Descarte de margen exterior
    alto, ancho = cara.shape[:2]
    margen_x = int(ancho * MARGEN_RECORTE_CARA)
    margen_y = int(alto * MARGEN_RECORTE_CARA)
    recorte = cara[margen_y:max(margen_y + 1, alto - margen_y), margen_x:max(margen_x + 1, ancho - margen_x)]

    if recorte.size == 0:
        recorte = cara

    # Redimension a resolucion fija
    cara_redim = cv2.resize(recorte, TAM_CARA, interpolation=cv2.INTER_AREA)

    # CLAHE y desenfoque
    cara_clahe = _CLAHE.apply(cara_redim)
    return cv2.GaussianBlur(cara_clahe, (3, 3), 0)

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

    # Paso 2: Usamos cv2.blur con kernel (3, 3) para quitar ruido y grano de la camara
    roi_suavizada = cv2.blur(roi_cara_byn, (3, 3))
    alto_roi, ancho_roi = roi_suavizada.shape[:2]

    for indice in range(len(lista_caras)):
        patron = lista_caras[indice]

        # Paso 5: Elegimos la interpolacion según si agrandamos o reducimos:
        # INTER_AREA para reducir (mas nítida) e INTER_LINEAR para ampliar sin pixelar
        if ancho_roi < patron.shape[1] or alto_roi < patron.shape[0]:
            tipo_interpolacion = cv2.INTER_AREA
        else:
            tipo_interpolacion = cv2.INTER_LINEAR

        patron_redimensionado = cv2.resize(patron, (ancho_roi, alto_roi), interpolation=tipo_interpolacion)

        # Suavizamos tambien la plantilla para que la comparacion de pixeles sea justa
        patron_suavizado = cv2.blur(patron_redimensionado, (3, 3))

        # Busqueda por plantilla con TM_SQDIFF_NORMED
        res = cv2.matchTemplate(roi_suavizada, patron_suavizado, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)

        if min_val < mejor_score:
            mejor_score = min_val
            nombre_detectado = lista_nombres[indice]

    if mejor_score < umbral:
        return nombre_detectado, mejor_score
    return None, mejor_score
