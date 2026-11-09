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
    ANGULO_MAXIMO_ALINEACION,
    UMBRAL_MOVIMIENTO_ESTABLE,
    KNN_HISTORIAL,
    KNN_UMBRAL_DISTANCIA,
    KNN_DETECTAR_SOMBRAS,
    UMBRAL_MOVIMIENTO_KNN,
    UMBRAL_REENGANCHE_SESION
)
from depurador import (
    es_debug_activo,
    mostrar_debug_normalizacion,
    mostrar_debug_estabilidad,
    mostrar_debug_matching
)

# Instancias globales a nivel de modulo
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_clasificador_rostros = None
_clasificador_ojos = None
_substractor_knn = None

def obtener_substractor_knn(reiniciar=False):
    """Devuelve la instancia singleton del sustractor de fondo KNN o la reinicia."""
    global _substractor_knn
    if _substractor_knn is None or reiniciar:
        _substractor_knn = cv2.createBackgroundSubtractorKNN(
            history=KNN_HISTORIAL,
            dist2Threshold=KNN_UMBRAL_DISTANCIA,
            detectShadows=KNN_DETECTAR_SOMBRAS
        )
    return _substractor_knn

def reiniciar_substractor_knn():
    """Reinicia el modelo de sustraccion de fondo KNN."""
    global _substractor_knn
    _substractor_knn = None
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
    """Devuelve la instancia cacheada del clasificador Haar Cascade de rostros."""
    global _clasificador_rostros
    if _clasificador_rostros is None and os.path.exists(RUTA_CASCADE):
        _clasificador_rostros = cv2.CascadeClassifier(RUTA_CASCADE)
    return _clasificador_rostros

def detectar_posicion_ojos(cara_gris, clasificador_ojos=None):
    """
    Detecta los centros de ambos ojos en la mitad superior de la cara.
    Retorna ((cx_izq, cy_izq), (cx_der, cy_der)) si se detecta exactamente un ojo
    en cada hemisferio con separacion horizontal valida, o None en caso contrario.
    """
    clasificador = clasificador_ojos or _clasificador_ojos
    if clasificador is None or cara_gris is None or cara_gris.size == 0:
        return None

    alto, ancho = cara_gris.shape[:2]
    if alto < 40 or ancho < 40:
        return None

    y1_ojos = int(alto * 0.10)
    y2_ojos = int(alto * 0.60)
    roi_ojos = cara_gris[y1_ojos:y2_ojos, :]

    ojos = clasificador.detectMultiScale(
        roi_ojos,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(int(ancho * 0.12), int(alto * 0.10)),
        maxSize=(int(ancho * 0.45), int(alto * 0.35))
    )

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

    if len(ojos_izq) != 1 or len(ojos_der) != 1:
        return None

    cx_izq, cy_izq = ojos_izq[0]
    cx_der, cy_der = ojos_der[0]

    dist_h = cx_der - cx_izq
    if not (0.25 * ancho <= dist_h <= 0.65 * ancho):
        return None

    return (cx_izq, cy_izq), (cx_der, cy_der)

def calcular_angulo_ojos(cara_gris, clasificador_ojos=None):
    """
    Calcula el angulo de inclinacion entre ambos ojos en la mitad superior de la cara.
    Retorna el angulo en grados si se detecta exactamente un ojo en cada hemisferio
    con separacion horizontal valida, o None si no se cumplen las condiciones.
    """
    pos_ojos = detectar_posicion_ojos(cara_gris, clasificador_ojos)
    if pos_ojos is None:
        return None

    (cx_izq, cy_izq), (cx_der, cy_der) = pos_ojos
    dx = cx_der - cx_izq
    dy = cy_der - cy_izq
    angulo = float(np.degrees(np.arctan2(dy, dx)))

    if abs(angulo) > ANGULO_MAXIMO_ALINEACION:
        return None

    return angulo

def ajustar_brillo_histograma(frame, brillo_objetivo=BRILLO_OBJETIVO):
    """
    Calcula el nivel medio de iluminacion mediante histograma en escala de grises
    y aplica compensacion de brillo sobre el canal V del espacio HSV.
    """
    if len(frame.shape) == 3:
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gris = frame

    # Calculo del histograma en escala de grises
    hist = cv2.calcHist([frame_gris], [0], None, [256], [0, 256])
    total_pixeles = np.sum(hist)
    if total_pixeles == 0:
        return frame

    # Estimacion de brillo medio ponderado por frecuencias
    brillo_medio = int(np.sum(np.arange(256) * hist.flatten()) / total_pixeles)

    # Incremento de luminancia si la iluminacion media esta bajo el objetivo
    if brillo_medio < brillo_objetivo and len(frame.shape) == 3:
        incremento = brillo_objetivo - brillo_medio
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Limite superior a 255 para prevenir desbordamiento de rango
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

    # Normalizacion previa de iluminacion para optimizar la deteccion facial
    frame_ajustado = ajustar_brillo_histograma(frame)

    if len(frame_ajustado.shape) == 3:
        frame_byn = cv2.cvtColor(frame_ajustado, cv2.COLOR_BGR2GRAY)
    else:
        frame_byn = frame_ajustado

    caras = clasificador.detectMultiScale(frame_byn, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Descarte de detecciones por debajo del umbral de area minima
    caras_filtradas = []
    for (x, y, ancho, alto) in caras:
        if (ancho * alto) >= area_min:
            caras_filtradas.append((x, y, ancho, alto))

    return caras_filtradas, frame_byn

def verificar_rostro_estatico(caja_actual, caja_previa, frame_gris, frames_estables=0):
    """
    Determina si un rostro detectado se mantiene quieto y frontal mediante Sustraccion de Fondo (KNN).
    Evalua relacion de aspecto, desplazamiento entre fotogramas y ausencia de movimiento en la ROI.
    """
    x, y, w, h = caja_actual
    if w <= 0 or h <= 0:
        return False

    # 1. Proporcion frontal: un rostro humano frontal tiene relacion w/h entre 0.70 y 1.30
    aspect_ratio = float(w) / float(h)
    proporcion_ok = (0.70 <= aspect_ratio <= 1.30)

    # 2. Desplazamiento respecto al fotograma previo
    dist_desplazamiento = 0.0
    variacion_tam_ok = True
    if caja_previa is not None:
        px, py, pw, ph = caja_previa
        cx_actual = x + w / 2.0
        cy_actual = y + h / 2.0
        cx_prev = px + pw / 2.0
        cy_prev = py + ph / 2.0

        dist_desplazamiento = float(np.hypot(cx_actual - cx_prev, cy_actual - cy_prev))
        var_w = abs(w - pw) / float(max(w, pw))
        var_h = abs(h - ph) / float(max(h, ph))
        if var_w > 0.20 or var_h > 0.20:
            variacion_tam_ok = False

    desplazamiento_ok = (caja_previa is None or dist_desplazamiento <= UMBRAL_MOVIMIENTO_ESTABLE)

    # 3. Deteccion de quietud mediante Sustraccion de Fondo KNN
    substractor = obtener_substractor_knn()
    mascara_1er_plano = substractor.apply(frame_gris)

    y1, y2 = max(0, y), min(frame_gris.shape[0], y + h)
    x1, x2 = max(0, x), min(frame_gris.shape[1], x + w)
    roi_cara = frame_gris[y1:y2, x1:x2]
    roi_mascara = mascara_1er_plano[y1:y2, x1:x2]

    if roi_cara.size == 0 or roi_mascara.size == 0:
        return False

    # Extraccion del contorno de movimiento maximo en la ROI del rostro
    contornos, _ = cv2.findContours(roi_mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area_movimiento = 0.0
    contorno_max = None
    if len(contornos) > 0:
        contorno_max = max(contornos, key=cv2.contourArea)
        area_movimiento = abs(cv2.contourArea(contorno_max, True))

    quietud_knn_ok = (area_movimiento <= UMBRAL_MOVIMIENTO_KNN)

    es_estable = bool(proporcion_ok and desplazamiento_ok and variacion_tam_ok and quietud_knn_ok)

    if es_debug_activo():
        mostrar_debug_estabilidad(
            roi_cara,
            roi_mascara,
            area_movimiento,
            dist_desplazamiento,
            aspect_ratio,
            es_estable,
            frames_estables,
            contorno_max=contorno_max
        )

    return es_estable

def alinear_ojos(cara):
    """
    Nivela horizontalmente los ojos si se detectan ambos en la mitad superior de la cara.
    Aplica rotacion afin preservando bordes mediante BORDER_REPLICATE.
    """
    cara_gris = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY) if len(cara.shape) == 3 else cara
    angulo = calcular_angulo_ojos(cara_gris, _clasificador_ojos)
    if angulo is None:
        return cara

    alto, ancho = cara.shape[:2]
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

    debug_activo = es_debug_activo()
    etapas_debug = {}
    if debug_activo:
        etapas_debug["crudo"] = cara.copy()
        etapas_debug["brillo_medio"] = int(np.mean(cara))

    # Alineacion previa al recorte de margen
    angulo_alineacion = None
    if ALINEAR_OJOS and _clasificador_ojos is not None:
        angulo_alineacion = calcular_angulo_ojos(cara, _clasificador_ojos)
        cara = alinear_ojos(cara)

    if debug_activo:
        etapas_debug["alineada"] = cara.copy()
        etapas_debug["angulo"] = angulo_alineacion

    # Descarte de margen exterior
    alto, ancho = cara.shape[:2]
    margen_x = int(ancho * MARGEN_RECORTE_CARA)
    margen_y = int(alto * MARGEN_RECORTE_CARA)
    recorte = cara[margen_y:max(margen_y + 1, alto - margen_y), margen_x:max(margen_x + 1, ancho - margen_x)]

    if recorte.size == 0:
        recorte = cara

    if debug_activo:
        etapas_debug["margen"] = recorte.copy()

    # Redimension a resolucion fija
    cara_redim = cv2.resize(recorte, TAM_CARA, interpolation=cv2.INTER_AREA)

    # CLAHE y desenfoque
    cara_clahe = _CLAHE.apply(cara_redim)
    if debug_activo:
        etapas_debug["clahe"] = cara_clahe.copy()

    cara_final = cv2.GaussianBlur(cara_clahe, (3, 3), 0)
    if debug_activo:
        etapas_debug["final"] = cara_final.copy()
        mostrar_debug_normalizacion(etapas_debug)

    return cara_final

def cargar_caras_confianza():
    """Carga en memoria los nombres y plantillas de rostros autorizados normalizadas."""
    lista_nombres = []
    lista_caras = []

    if not os.path.exists(CARPETA_CONFIANZA):
        return lista_nombres, lista_caras

    sufijos_conocidos = (
        "_frente_normal",
        "_frente_sonriendo",
        "_perfil_derecho",
        "_perfil_izquierdo"
    )

    for archivo in os.listdir(CARPETA_CONFIANZA):
        if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            nombre_sin_ext, _ = os.path.splitext(archivo)
            nombre_base = nombre_sin_ext
            for sufijo in sufijos_conocidos:
                if nombre_base.endswith(sufijo):
                    nombre_base = nombre_base[:-len(sufijo)]
                    break

            img = cv2.imread(os.path.join(CARPETA_CONFIANZA, archivo), 0)
            if img is not None:
                if (img.shape[1], img.shape[0]) != TAM_CARA:
                    img = normalizar_cara(img)
                lista_nombres.append(nombre_base)
                lista_caras.append(img)

    return lista_nombres, lista_caras

def identificar_rostro(roi_cara_byn, lista_nombres, lista_caras, umbral=UMBRAL_COINCIDENCIA):
    """
    Compara el rostro recortado contra las plantillas guardadas usando matchTemplate.
    Aplica suavizado con blur y redimensiona con la interpolacion adecuada si difiere el tamano.
    """
    mejor_score = 1.0
    nombre_detectado = None
    mejor_patron = None

    if not lista_caras or roi_cara_byn is None or roi_cara_byn.size == 0:
        return None, mejor_score

    # Redimension preventiva si la ROI no tiene la resolucion normalizada
    if (roi_cara_byn.shape[1], roi_cara_byn.shape[0]) != TAM_CARA:
        roi_cara_byn = cv2.resize(roi_cara_byn, TAM_CARA, interpolation=cv2.INTER_AREA)

    roi_suavizada = cv2.blur(roi_cara_byn, (3, 3))
    alto_roi, ancho_roi = roi_suavizada.shape[:2]

    for indice in range(len(lista_caras)):
        patron = lista_caras[indice]

        if (patron.shape[1], patron.shape[0]) != (ancho_roi, alto_roi):
            if ancho_roi < patron.shape[1] or alto_roi < patron.shape[0]:
                tipo_interpolacion = cv2.INTER_AREA
            else:
                tipo_interpolacion = cv2.INTER_LINEAR
            patron_redimensionado = cv2.resize(patron, (ancho_roi, alto_roi), interpolation=tipo_interpolacion)
        else:
            patron_redimensionado = patron

        patron_suavizado = cv2.blur(patron_redimensionado, (3, 3))

        res = cv2.matchTemplate(roi_suavizada, patron_suavizado, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)

        if min_val < mejor_score:
            mejor_score = min_val
            nombre_detectado = lista_nombres[indice]
            mejor_patron = patron

    if es_debug_activo():
        mostrar_debug_matching(roi_cara_byn, mejor_patron, nombre_detectado, mejor_score, umbral)

    if mejor_score < umbral:
        return nombre_detectado, mejor_score
    return None, mejor_score

def obtener_plantillas_usuario(nombre_usuario, lista_nombres, lista_caras):
    """Filtra y devuelve la lista de plantillas normalizadas asociadas a un usuario."""
    return [lista_caras[i] for i in range(len(lista_nombres)) if lista_nombres[i] == nombre_usuario]

def verificar_usuario_especifico(roi_cara_byn, plantillas_usuario, umbral=UMBRAL_REENGANCHE_SESION):
    """
    Compara un rostro normalizado contra las plantillas exclusivas del usuario activo.
    Retorna una tupla (coincide, mejor_score, mejor_similitud).
    """
    if not plantillas_usuario or roi_cara_byn is None or roi_cara_byn.size == 0:
        return False, 1.0, 0.0

    if (roi_cara_byn.shape[1], roi_cara_byn.shape[0]) != TAM_CARA:
        roi_cara_byn = cv2.resize(roi_cara_byn, TAM_CARA, interpolation=cv2.INTER_AREA)

    roi_suavizada = cv2.blur(roi_cara_byn, (3, 3))
    alto_roi, ancho_roi = roi_suavizada.shape[:2]
    mejor_score = 1.0
    mejor_patron = None

    for patron in plantillas_usuario:
        if (patron.shape[1], patron.shape[0]) != (ancho_roi, alto_roi):
            if ancho_roi < patron.shape[1] or alto_roi < patron.shape[0]:
                tipo_interpolacion = cv2.INTER_AREA
            else:
                tipo_interpolacion = cv2.INTER_LINEAR
            patron_redimensionado = cv2.resize(patron, (ancho_roi, alto_roi), interpolation=tipo_interpolacion)
        else:
            patron_redimensionado = patron

        patron_suavizado = cv2.blur(patron_redimensionado, (3, 3))
        res = cv2.matchTemplate(roi_suavizada, patron_suavizado, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)

        if min_val < mejor_score:
            mejor_score = float(min_val)
            mejor_patron = patron

    similitud = max(0.0, min(100.0, (1.0 - mejor_score) * 100.0))
    coincide = mejor_score < umbral

    if es_debug_activo():
        mostrar_debug_matching(roi_cara_byn, mejor_patron, "ACTIVO", mejor_score, umbral)

    return coincide, mejor_score, similitud
