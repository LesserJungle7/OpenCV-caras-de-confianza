import collections
import os
import random
import time
import numpy as np
import cv2

from config import (
    ALINEAR_OJOS,
    ANGULO_MAXIMO_ALINEACION,
    RUTA_CASCADE_EYE,
    GESTOS_DISPONIBLES,
    GESTOS_DESCRIPCION,
    TIEMPO_MAXIMO_GESTO,
    FRAMES_CONFIRMACION_GESTO,
    FRAMES_CALIBRACION,
    FACTOR_K_CIERRE_OJO,
    VENTANA_TEMPORAL_GESTO,
    MIN_FRAMES_CONFIRMACION_GESTO,
    DIF_MAXIMA_ROJO_AZUL_DIENTES,
    DIF_MAXIMA_ROJO_VERDE_DIENTES,
    ANCHO_MINIMO_FRANJA_DIENTES,
    PORCENTAJE_MINIMO_DIENTES,
    COLOR_INACTIVO,
    COLOR_ACTIVO,
    COLOR_RETO_ACTIVO,
    COLOR_REGION_ACTIVA,
    COLOR_TEXTO_HUD,
    FUENTE
)

_clasificador_ojos = None

def obtener_clasificador_ojos():
    """Instancia y cachea el clasificador de ojos."""
    global _clasificador_ojos
    if _clasificador_ojos is None and os.path.exists(RUTA_CASCADE_EYE):
        _clasificador_ojos = cv2.CascadeClassifier(RUTA_CASCADE_EYE)
    return _clasificador_ojos

def extraer_rois(frame, x, y, w, h, angulo=0.0):
    """
    Segmenta las regiones geometricas de cejas, ojos y boca a partir de la cara.
    Permite aplicar una correccion de rotacion neutra para nivelar horizontalmente los ojos.
    Retorna un diccionario con las imagenes recortadas y sus coordenadas absolutas.
    """
    alto_f, ancho_f = frame.shape[:2]

    def recortar(img_fuente, x1, y1, x2, y2):
        alto_src, ancho_src = img_fuente.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho_src, x2), min(alto_src, y2)
        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1, 0, 0)
        return img_fuente[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    # Coordenadas proporcionales desacopladas con zona buffer libre (0.29h - 0.35h)
    cejas_y1, cejas_y2 = int(y + 0.14 * h), int(y + 0.29 * h)
    cejas_x1, cejas_x2 = int(x + 0.15 * w), int(x + 0.85 * w)

    # Franja vertical optimizada de ojos centrada en el globo ocular (0.32h - 0.45h)
    ojo_rel_y1, ojo_rel_y2 = int(0.32 * h), int(0.45 * h)
    oi_rel_x1, oi_rel_x2 = int(0.17 * w), int(0.45 * w)
    od_rel_x1, od_rel_x2 = int(0.55 * w), int(0.83 * w)

    boca_y1, boca_y2 = int(y + 0.65 * h), int(y + 0.90 * h)
    boca_x1, boca_x2 = int(x + 0.22 * w), int(x + 0.78 * w)

    img_cejas, rect_cejas = recortar(frame, cejas_x1, cejas_y1, cejas_x2, cejas_y2)
    img_boca, rect_boca = recortar(frame, boca_x1, boca_y1, boca_x2, boca_y2)

    # Nivelacion opcional del rostro para la extraccion de ojos
    x1_c, y1_c = max(0, x), max(0, y)
    x2_c, y2_c = min(ancho_f, x + w), min(alto_f, y + h)
    cara_recorte = frame[y1_c:y2_c, x1_c:x2_c]

    if abs(angulo) > 0.5 and cara_recorte.shape[0] == h and cara_recorte.shape[1] == w:
        mat_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angulo, 1.0)
        cara_nivelada = cv2.warpAffine(cara_recorte, mat_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
        img_oi, _ = recortar(cara_nivelada, oi_rel_x1, ojo_rel_y1, oi_rel_x2, ojo_rel_y2)
        img_od, _ = recortar(cara_nivelada, od_rel_x1, ojo_rel_y1, od_rel_x2, ojo_rel_y2)
    else:
        img_oi, _ = recortar(frame, x + oi_rel_x1, y + ojo_rel_y1, x + oi_rel_x2, y + ojo_rel_y2)
        img_od, _ = recortar(frame, x + od_rel_x1, y + ojo_rel_y1, x + od_rel_x2, y + ojo_rel_y2)

    rect_oi = (x + oi_rel_x1, y + ojo_rel_y1, oi_rel_x2 - oi_rel_x1, ojo_rel_y2 - ojo_rel_y1)
    rect_od = (x + od_rel_x1, y + ojo_rel_y1, od_rel_x2 - od_rel_x1, ojo_rel_y2 - ojo_rel_y1)

    return {
        "cejas": {"img": img_cejas, "rect": rect_cejas},
        "ojo_izq": {"img": img_oi, "rect": rect_oi},
        "ojo_der": {"img": img_od, "rect": rect_od},
        "boca": {"img": img_boca, "rect": rect_boca},
        "cara": {"rect": (x, y, w, h)}
    }

def dibujar_rois(frame_mostrar, rois, regiones_activas=None, estados_texto=None, gesto_confirmando=False, estado_boca_texto="SERIA"):
    """
    Dibuja los rectangulos delimitadores y etiquetas de estado sobre las ROIs faciales.
    Usa rojo para estados neutros/inactivos y verde para estados activos/detectados.
    """
    if regiones_activas is None:
        regiones_activas = {}
    if estados_texto is None:
        estados_texto = {
            "cejas": "LEVANTADAS" if regiones_activas.get("cejas") else "REPOSO",
            "ojo_izq": "CERRADO" if regiones_activas.get("ojo_izq") else "ABIERTO",
            "ojo_der": "CERRADO" if regiones_activas.get("ojo_der") else "ABIERTO",
            "boca": "SONRISA" if regiones_activas.get("boca") else estado_boca_texto
        }

    fx, fy, fw, fh = rois["cara"]["rect"]
    color_cara = COLOR_ACTIVO if gesto_confirmando else (0, 255, 0)
    grosor_cara = 3 if gesto_confirmando else 2
    cv2.rectangle(frame_mostrar, (fx, fy), (fx + fw, fy + fh), color_cara, grosor_cara)

    # 1. Cejas
    cx, cy, cw, ch = rois["cejas"]["rect"]
    if cw > 0 and ch > 0:
        activa_c = bool(regiones_activas.get("cejas", False))
        color_c = COLOR_ACTIVO if activa_c else COLOR_INACTIVO
        cv2.rectangle(frame_mostrar, (cx, cy), (cx + cw, cy + ch), color_c, 2 if activa_c else 1)
        texto_cejas = f"Cejas: {estados_texto.get('cejas', 'REPOSO')}"
        cv2.putText(frame_mostrar, texto_cejas, (cx, cy - 6), FUENTE, 0.4, color_c, 1)

    # 2. Ojo izquierdo
    oix, oiy, oiw, oih = rois["ojo_izq"]["rect"]
    if oiw > 0 and oih > 0:
        activa_oi = bool(regiones_activas.get("ojo_izq", False))
        color_oi = COLOR_ACTIVO if activa_oi else COLOR_INACTIVO
        cv2.rectangle(frame_mostrar, (oix, oiy), (oix + oiw, oiy + oih), color_oi, 2 if activa_oi else 1)
        texto_oi = f"Ojo Izq: {estados_texto.get('ojo_izq', 'ABIERTO')}"
        cv2.putText(frame_mostrar, texto_oi, (oix, oiy - 6), FUENTE, 0.4, color_oi, 1)

    # 3. Ojo derecho
    odx, ody, odw, odh = rois["ojo_der"]["rect"]
    if odw > 0 and odh > 0:
        activa_od = bool(regiones_activas.get("ojo_der", False))
        color_od = COLOR_ACTIVO if activa_od else COLOR_INACTIVO
        cv2.rectangle(frame_mostrar, (odx, ody), (odx + odw, ody + odh), color_od, 2 if activa_od else 1)
        texto_od = f"Ojo Der: {estados_texto.get('ojo_der', 'ABIERTO')}"
        cv2.putText(frame_mostrar, texto_od, (odx, ody - 6), FUENTE, 0.4, color_od, 1)

    # 4. Boca
    bx, by, bw, bh = rois["boca"]["rect"]
    if bw > 0 and bh > 0:
        activa_b = bool(regiones_activas.get("boca", False))
        color_b = COLOR_ACTIVO if activa_b else COLOR_INACTIVO
        cv2.rectangle(frame_mostrar, (bx, by), (bx + bw, by + bh), color_b, 2 if activa_b else 1)
        texto_boca = f"Boca: {estados_texto.get('boca', 'SERIA')}"
        cv2.putText(frame_mostrar, texto_boca, (bx, by + bh + 14), FUENTE, 0.4, color_b, 1)

def dibujar_hud(frame_mostrar, paso_actual, total_pasos, nombre_gesto, tiempo_restante, frames_ok, total_frames_ok, calibrando=False):
    """Renderiza el panel superior HUD con las instrucciones del reto y progreso."""
    ancho_f = frame_mostrar.shape[1]

    overlay = frame_mostrar.copy()
    cv2.rectangle(overlay, (0, 0), (ancho_f, 95), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame_mostrar, 0.25, 0, frame_mostrar)

    if calibrando:
        texto_principal = "CALIBRANDO ROSTRO NEUTRO... MIRA A LA CAMARA"
        cv2.putText(frame_mostrar, texto_principal, (20, 45), FUENTE, 0.65, (0, 255, 255), 2)
        progreso_calib = int((frames_ok / total_frames_ok) * (ancho_f - 40))
        cv2.rectangle(frame_mostrar, (20, 65), (20 + progreso_calib, 75), (0, 255, 255), -1)
        return

    # Encabezado del reto
    descripcion = GESTOS_DESCRIPCION.get(nombre_gesto, nombre_gesto)
    titulo = f"CONTRASENA FACIAL - RETO [{paso_actual}/{total_pasos}]"
    cv2.putText(frame_mostrar, titulo, (20, 30), FUENTE, 0.65, (0, 255, 255), 2)

    # Instruccion del gesto actual
    cv2.putText(frame_mostrar, f">> {descripcion.upper()}", (20, 60), FUENTE, 0.7, (255, 255, 255), 2)

    # Temporizador
    color_timer = (0, 255, 0) if tiempo_restante > 2.5 else (0, 0, 255)
    texto_timer = f"{tiempo_restante:.1f}s"
    cv2.putText(frame_mostrar, texto_timer, (ancho_f - 110, 45), FUENTE, 0.8, color_timer, 2)

    # Barra de confirmacion
    ancho_barra = ancho_f - 40
    cv2.rectangle(frame_mostrar, (20, 75), (20 + ancho_barra, 85), (60, 60, 60), -1)
    if frames_ok > 0:
        progreso_px = int((frames_ok / total_frames_ok) * ancho_barra)
        cv2.rectangle(frame_mostrar, (20, 75), (20 + progreso_px, 85), COLOR_RETO_ACTIVO, -1)


class EvaluadorGestos:
    """Gestiona la calibracion de rostro neutro y la validacion de gestos por ROIs."""

    def __init__(self, lista_gestos=None):
        self.clasificador_ojos = obtener_clasificador_ojos()
        self.lista_retos = lista_gestos or random.sample(GESTOS_DISPONIBLES, 3)
        self.paso_actual = 0
        self.frames_confirmados = 0
        self.tiempo_inicio_gesto = None

        # Calibracion adaptativa neutra
        self.calibrado = False
        self.frames_calibracion_cont = 0
        self.hist_cejas = []
        self.hist_brillo_piel = []
        self.hist_dientes_neutro = []

        # Historiales de aspecto y alineacion neutra
        self.hist_aspect_oi = []
        self.hist_aspect_od = []
        self.angulos_calibracion = []
        self.angulo_calibrado = 0.0
        self.neutro_aspect_oi = 0.35
        self.neutro_aspect_od = 0.35

        # Ventanas deslizantes para confirmacion temporal de gestos de ojos
        self.ventana_ojo_izq = collections.deque(maxlen=VENTANA_TEMPORAL_GESTO)
        self.ventana_ojo_der = collections.deque(maxlen=VENTANA_TEMPORAL_GESTO)

        self.neutro_dist_cejas = 25.0
        self.neutro_brillo_piel = 110.0
        self.neutro_dientes_pct = 0.0

        self.ultimo_estado_boca = "SERIA"

    def calibrar(self, rois, cara_gris=None):
        """Calcula los promedios neutros de cejas, piel, ojos y boca durante los primeros fotogramas."""
        dist_c = self._medir_distancia_cejas(rois)

        img_boca = rois["boca"]["img"]
        if img_boca is not None and img_boca.size > 0:
            gris_boca = cv2.cvtColor(img_boca, cv2.COLOR_BGR2GRAY)
            self.hist_brillo_piel.append(float(np.mean(gris_boca)))
            _, pct_d = self._detectar_dientes_crudo(img_boca, brillo_ref=110.0)
            self.hist_dientes_neutro.append(pct_d)

        # Muestreo adaptativo de la relacion de aspecto neutra por cada ojo
        img_oi = rois["ojo_izq"]["img"]
        img_od = rois["ojo_der"]["img"]
        if img_oi is not None and img_oi.size > 0:
            asp_oi = self._analizar_ojo_aspecto(img_oi)
            if asp_oi > 0:
                self.hist_aspect_oi.append(asp_oi)
        if img_od is not None and img_od.size > 0:
            asp_od = self._analizar_ojo_aspecto(img_od)
            if asp_od > 0:
                self.hist_aspect_od.append(asp_od)

        # Deteccion de angulo neutro si la clasificacion de ojos esta activa
        if ALINEAR_OJOS and self.clasificador_ojos is not None and cara_gris is not None:
            alto_cara, ancho_cara = cara_gris.shape[:2]
            y1_det = int(alto_cara * 0.10)
            y2_det = int(alto_cara * 0.60)
            roi_ojos_det = cara_gris[y1_det:y2_det, :]

            ojos_haar = self.clasificador_ojos.detectMultiScale(
                roi_ojos_det,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(int(ancho_cara * 0.12), int(alto_cara * 0.10)),
                maxSize=(int(ancho_cara * 0.45), int(alto_cara * 0.35))
            )
            izq_c, der_c = [], []
            for (ox, oy, ow_d, oh_d) in ojos_haar:
                cx = ox + ow_d / 2.0
                cy = y1_det + oy + oh_d / 2.0
                if cx < ancho_cara / 2.0:
                    izq_c.append((cx, cy))
                else:
                    der_c.append((cx, cy))

            if len(izq_c) == 1 and len(der_c) == 1:
                dx = der_c[0][0] - izq_c[0][0]
                dy = der_c[0][1] - izq_c[0][1]
                if (0.25 * ancho_cara <= dx <= 0.65 * ancho_cara) and (abs(dy) <= 0.15 * alto_cara):
                    ang = float(np.degrees(np.arctan2(dy, dx)))
                    if abs(ang) <= ANGULO_MAXIMO_ALINEACION:
                        self.angulos_calibracion.append(ang)

        if dist_c is not None:
            self.hist_cejas.append(dist_c)

        self.frames_calibracion_cont += 1
        if self.frames_calibracion_cont >= FRAMES_CALIBRACION:
            self.neutro_dist_cejas = float(np.mean(self.hist_cejas)) if self.hist_cejas else 25.0
            self.neutro_brillo_piel = float(np.mean(self.hist_brillo_piel)) if self.hist_brillo_piel else 110.0
            self.neutro_dientes_pct = float(np.mean(self.hist_dientes_neutro)) if self.hist_dientes_neutro else 0.0

            self.neutro_aspect_oi = max(0.15, float(np.median(self.hist_aspect_oi))) if self.hist_aspect_oi else 0.35
            self.neutro_aspect_od = max(0.15, float(np.median(self.hist_aspect_od))) if self.hist_aspect_od else 0.35

            if self.angulos_calibracion:
                self.angulo_calibrado = float(np.median(self.angulos_calibracion))
            else:
                self.angulo_calibrado = 0.0

            self.calibrado = True
            self.tiempo_inicio_gesto = time.time()
            return True
        return False

    def _medir_distancia_cejas(self, rois):
        """Calcula la distancia vertical entre el centroide oscuro de cejas y el nivel de ojos."""
        img_cejas = rois["cejas"]["img"]
        if img_cejas is None or img_cejas.size == 0:
            return None

        gris = cv2.cvtColor(img_cejas, cv2.COLOR_BGR2GRAY)
        umbral_val = np.percentile(gris, 25)
        puntos_oscuros = np.where(gris <= umbral_val)

        if len(puntos_oscuros[0]) == 0:
            return None

        y_centroide = np.mean(puntos_oscuros[0])
        alto_cejas = img_cejas.shape[0]
        return float(alto_cejas - y_centroide)

    def _analizar_ojo_aspecto(self, img_ojo):
        """
        Calcula la relacion de aspecto vertical/horizontal (bh / bw) del contorno oscuro
        principal en la region central del ojo, aplicando umbralizacion dinamica local.
        """
        if img_ojo is None or img_ojo.size == 0:
            return 0.0

        gris = cv2.cvtColor(img_ojo, cv2.COLOR_BGR2GRAY)
        h, w = gris.shape
        zona = gris[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
        if zona.size == 0:
            return 0.0

        p10 = float(np.percentile(zona, 10))
        p50 = float(np.percentile(zona, 50))
        umbral = (p10 + p50) / 2.0

        mascara = (zona < umbral).astype(np.uint8) * 255
        kernel = np.ones((2, 2), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contornos) > 0:
            c_mayor = max(contornos, key=cv2.contourArea)
            _, _, bw, bh = cv2.boundingRect(c_mayor)
            return float(bh / max(1.0, float(bw)))

        return 0.0

    def _evaluar_ojo_individual(self, img_ojo, neutro_aspect):
        """
        Evalua si el ojo esta cerrado comparando su relacion de aspecto actual
        con su linea base neutra mediante la Regla K adaptativa.
        Retorna una tupla (esta_cerrado, aspect_actual).
        """
        if img_ojo is None or img_ojo.size == 0:
            return True, 0.0

        aspect_actual = self._analizar_ojo_aspecto(img_ojo)
        umbral_cierre = FACTOR_K_CIERRE_OJO * neutro_aspect
        esta_cerrado = bool(aspect_actual < umbral_cierre)

        return esta_cerrado, aspect_actual

    def _evaluar_ojos(self, rois):
        """Determina de forma diferencial el estado de ambos ojos usando sus lineas base independientes."""
        izq_cerrado, aspect_oi = self._evaluar_ojo_individual(
            rois["ojo_izq"]["img"],
            self.neutro_aspect_oi
        )
        der_cerrado, aspect_od = self._evaluar_ojo_individual(
            rois["ojo_der"]["img"],
            self.neutro_aspect_od
        )

        return izq_cerrado, der_cerrado, aspect_oi, aspect_od

    def _detectar_dientes_crudo(self, img_boca, brillo_ref):
        """
        Segmenta pixeles correspondientes a esmalte dental aplicando filtros de brillo
        y neutralidad cromatica (|R - B| y |R - G| acotados).
        Retorna una tupla (tiene_franja_horizontal, porcentaje_area).
        """
        alto, ancho = img_boca.shape[:2]
        y1_c, y2_c = int(alto * 0.20), int(alto * 0.80)
        x1_c, x2_c = int(ancho * 0.20), int(ancho * 0.80)
        zona = img_boca[y1_c:y2_c, x1_c:x2_c]
        if zona.size == 0:
            return False, 0.0

        B = zona[:, :, 0].astype(np.int16)
        G = zona[:, :, 1].astype(np.int16)
        R = zona[:, :, 2].astype(np.int16)

        brillo = (R + G + B) // 3
        dif_rb = np.abs(R - B)
        dif_rg = np.abs(R - G)

        umbral_brillo = max(145.0, min(175.0, brillo_ref + 20.0))

        # Filtro de alto brillo y equilibrio cromatico neutro
        mascara = (brillo > umbral_brillo) & (dif_rb < DIF_MAXIMA_ROJO_AZUL_DIENTES) & (dif_rg < DIF_MAXIMA_ROJO_VERDE_DIENTES)
        mascara = mascara.astype(np.uint8) * 255

        kernel = np.ones((2, 2), np.uint8)
        mascara_limpia = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        total_px = float(zona.shape[0] * zona.shape[1])
        pct_dientes = np.count_nonzero(mascara_limpia) / total_px

        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tiene_franja = False
        if len(contornos) > 0:
            c_mayor = max(contornos, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c_mayor)
            if bw >= ANCHO_MINIMO_FRANJA_DIENTES and bh >= 3:
                tiene_franja = True

        return tiene_franja, pct_dientes

    def _evaluar_sonrisa(self, rois):
        """
        Determina si la boca se encuentra en estado de sonrisa comparando el porcentaje
        dental detectado contra la linea base neutra de reposo.
        """
        img_boca = rois["boca"]["img"]
        if img_boca is None or img_boca.size == 0:
            return False, 0.0

        tiene_franja, pct_dientes = self._detectar_dientes_crudo(img_boca, self.neutro_brillo_piel)

        # Exige superar el porcentaje minimo y sobrepasar la linea base de labios en reposo
        umbral_dinamico = max(PORCENTAJE_MINIMO_DIENTES, self.neutro_dientes_pct + 0.035)
        es_sonrisa = bool(pct_dientes >= umbral_dinamico and tiene_franja)
        return es_sonrisa, float(pct_dientes)

    def evaluar_gesto_activo(self, gesto, rois):
        """
        Comprueba si el gesto objetivo esta activo en el fotograma.
        Retorna (gesto_cumplido, regiones_activas_dict).
        """
        regiones = {"cejas": False, "ojo_izq": False, "ojo_der": False, "boca": False}

        # 1. Evaluacion de cejas
        dist_cejas = self._medir_distancia_cejas(rois)
        if dist_cejas is not None and self.neutro_dist_cejas > 0:
            if dist_cejas > (self.neutro_dist_cejas * 1.16):
                regiones["cejas"] = True

        # 2. Evaluacion de ojos
        izq_c, der_c, _, _ = self._evaluar_ojos(rois)
        if izq_c:
            regiones["ojo_izq"] = True
        if der_c:
            regiones["ojo_der"] = True

        # Actualizar historiales deslizantes para confirmacion temporal
        self.ventana_ojo_izq.append(1 if izq_c else 0)
        self.ventana_ojo_der.append(1 if der_c else 0)

        # 3. Evaluacion de sonrisa (dientes visibles)
        es_sonrisa, pct_dientes = self._evaluar_sonrisa(rois)
        self.ultimo_estado_boca = "SONRISA" if es_sonrisa else "SERIA"
        regiones["boca"] = bool(es_sonrisa)

        # Estados de texto descriptivos para cada region
        self.estados_texto = {
            "cejas": "LEVANTADAS" if regiones["cejas"] else "REPOSO",
            "ojo_izq": "CERRADO" if izq_c else "ABIERTO",
            "ojo_der": "CERRADO" if der_c else "ABIERTO",
            "boca": "SONRISA" if es_sonrisa else "SERIA"
        }

        # 4. Validar segun el gesto objetivo solicitado
        cumplido = False
        if gesto == "levantar_cejas":
            cumplido = bool(regiones["cejas"])

        elif gesto == "sonreir":
            cumplido = bool(es_sonrisa)

        elif gesto == "cerrar_ambos_ojos":
            val_oi = (len(self.ventana_ojo_izq) >= VENTANA_TEMPORAL_GESTO and
                      sum(self.ventana_ojo_izq) >= MIN_FRAMES_CONFIRMACION_GESTO)
            val_od = (len(self.ventana_ojo_der) >= VENTANA_TEMPORAL_GESTO and
                      sum(self.ventana_ojo_der) >= MIN_FRAMES_CONFIRMACION_GESTO)
            cumplido = bool(val_oi and val_od)

        elif gesto == "cerrar_ojo_izq":
            val_oi = (len(self.ventana_ojo_izq) >= VENTANA_TEMPORAL_GESTO and
                      sum(self.ventana_ojo_izq) >= MIN_FRAMES_CONFIRMACION_GESTO)
            val_od_abierto = (sum(self.ventana_ojo_der) <= (VENTANA_TEMPORAL_GESTO - MIN_FRAMES_CONFIRMACION_GESTO))
            cumplido = bool(val_oi and val_od_abierto)

        elif gesto == "cerrar_ojo_der":
            val_od = (len(self.ventana_ojo_der) >= VENTANA_TEMPORAL_GESTO and
                      sum(self.ventana_ojo_der) >= MIN_FRAMES_CONFIRMACION_GESTO)
            val_oi_abierto = (sum(self.ventana_ojo_izq) <= (VENTANA_TEMPORAL_GESTO - MIN_FRAMES_CONFIRMACION_GESTO))
            cumplido = bool(val_od and val_oi_abierto)

        return bool(cumplido), regiones

    def procesar_fotograma(self, frame, cara_rect):
        """
        Ciclo principal de validacion: extrae ROIs, evalua reto activo y dibuja HUD.
        Retorna (estado, frame_anotado).
        """
        x, y, w, h = cara_rect
        rois = extraer_rois(frame, x, y, w, h, self.angulo_calibrado)
        frame_anotado = frame.copy()

        # 1. Fase de calibracion inicial
        if not self.calibrado:
            x1_c, y1_c = max(0, x), max(0, y)
            x2_c, y2_c = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            cara_recorte = frame[y1_c:y2_c, x1_c:x2_c]
            cara_gris = cv2.cvtColor(cara_recorte, cv2.COLOR_BGR2GRAY) if cara_recorte.size > 0 else None
            self.calibrar(rois, cara_gris=cara_gris)
            estados_calib = {
                "cejas": "CALIBRANDO",
                "ojo_izq": "CALIBRANDO",
                "ojo_der": "CALIBRANDO",
                "boca": "CALIBRANDO"
            }
            dibujar_rois(frame_anotado, rois, regiones_activas={}, estados_texto=estados_calib, gesto_confirmando=False)
            dibujar_hud(
                frame_anotado,
                paso_actual=1,
                total_pasos=len(self.lista_retos),
                nombre_gesto="",
                tiempo_restante=TIEMPO_MAXIMO_GESTO,
                frames_ok=self.frames_calibracion_cont,
                total_frames_ok=FRAMES_CALIBRACION,
                calibrando=True
            )
            return "CALIBRANDO", frame_anotado

        # 2. Control de tiempo limite del reto actual
        if self.tiempo_inicio_gesto is None:
            self.tiempo_inicio_gesto = time.time()

        tiempo_transcurrido = time.time() - self.tiempo_inicio_gesto
        tiempo_restante = max(0.0, TIEMPO_MAXIMO_GESTO - tiempo_transcurrido)

        if tiempo_restante <= 0.0:
            return "TIMEOUT", frame_anotado

        # 3. Evaluacion del gesto en curso
        gesto_objetivo = self.lista_retos[self.paso_actual]
        gesto_detectado, regiones_activas = self.evaluar_gesto_activo(gesto_objetivo, rois)

        es_gesto_ojos = gesto_objetivo in ("cerrar_ambos_ojos", "cerrar_ojo_izq", "cerrar_ojo_der")
        total_necesarios = MIN_FRAMES_CONFIRMACION_GESTO if es_gesto_ojos else FRAMES_CONFIRMACION_GESTO

        if es_gesto_ojos:
            if gesto_objetivo == "cerrar_ambos_ojos":
                frames_progreso = min(sum(self.ventana_ojo_izq), sum(self.ventana_ojo_der))
            elif gesto_objetivo == "cerrar_ojo_izq":
                frames_progreso = sum(self.ventana_ojo_izq) if sum(self.ventana_ojo_der) <= 2 else 0
            else:
                frames_progreso = sum(self.ventana_ojo_der) if sum(self.ventana_ojo_izq) <= 2 else 0
            self.frames_confirmados = frames_progreso
        else:
            if gesto_detectado:
                self.frames_confirmados += 1
            else:
                self.frames_confirmados = max(0, self.frames_confirmados - 1)

        # 4. Comprobar superacion del paso
        paso_superado = gesto_detectado if es_gesto_ojos else (self.frames_confirmados >= FRAMES_CONFIRMACION_GESTO)
        if paso_superado:
            self.paso_actual += 1
            self.frames_confirmados = 0
            self.tiempo_inicio_gesto = time.time()
            self.ventana_ojo_izq.clear()
            self.ventana_ojo_der.clear()

            if self.paso_actual >= len(self.lista_retos):
                return "EXITO", frame_anotado

        # 5. Renderizado visual con feedback inmediato por region activa
        dibujar_rois(
            frame_anotado,
            rois,
            regiones_activas=regiones_activas,
            estados_texto=getattr(self, "estados_texto", None),
            estado_boca_texto=self.ultimo_estado_boca,
            gesto_confirmando=(self.frames_confirmados > 1)
        )
        dibujar_hud(
            frame_anotado,
            paso_actual=self.paso_actual + 1,
            total_pasos=len(self.lista_retos),
            nombre_gesto=self.lista_retos[self.paso_actual],
            tiempo_restante=tiempo_restante,
            frames_ok=self.frames_confirmados,
            total_frames_ok=total_necesarios,
            calibrando=False
        )

        return "EN_PROCESO", frame_anotado
