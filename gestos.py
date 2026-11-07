import os
import random
import time
import numpy as np
import cv2

from config import (
    RUTA_CASCADE_EYE,
    GESTOS_DISPONIBLES,
    GESTOS_DESCRIPCION,
    TIEMPO_MAXIMO_GESTO,
    FRAMES_CONFIRMACION_GESTO,
    FRAMES_CALIBRACION,
    FACTOR_APERTURA_STD,
    FACTOR_APERTURA_RANGO,
    ALTURA_MINIMA_IRIS_ABIERTO,
    ASPECTO_MINIMO_IRIS_ABIERTO,
    DIF_MAXIMA_ROJO_AZUL_DIENTES,
    DIF_MAXIMA_ROJO_VERDE_DIENTES,
    ANCHO_MINIMO_FRANJA_DIENTES,
    PORCENTAJE_MINIMO_DIENTES,
    COLOR_INACTIVO,
    COLOR_ACTIVO,
    COLOR_RETO_ACTIVO,
    COLOR_REGION_ACTIVA,
    COLOR_TEXTO_HUD,
    FUENTE,
    AUDIO_SUCCESS
)
from sistema import reproducir_audio

_clasificador_ojos = None

def obtener_clasificador_ojos():
    """Instancia y cachea el clasificador de ojos."""
    global _clasificador_ojos
    if _clasificador_ojos is None and os.path.exists(RUTA_CASCADE_EYE):
        _clasificador_ojos = cv2.CascadeClassifier(RUTA_CASCADE_EYE)
    return _clasificador_ojos

def extraer_rois(frame, x, y, w, h):
    """
    Segmenta las regiones geometricas de cejas, ojos y boca a partir de la cara.
    Retorna un diccionario con las imagenes recortadas y sus coordenadas absolutas.
    """
    alto_f, ancho_f = frame.shape[:2]

    def recortar(x1, y1, x2, y2):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho_f, x2), min(alto_f, y2)
        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1, 0, 0)
        return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    # Coordenadas proporcionales desacopladas con zona buffer libre (0.29h - 0.35h)
    cejas_y1, cejas_y2 = int(y + 0.14 * h), int(y + 0.29 * h)
    cejas_x1, cejas_x2 = int(x + 0.15 * w), int(x + 0.85 * w)

    # Ojos reubicados estrictamente en el globo ocular (evita vello de ceja)
    ojo_y1, ojo_y2 = int(y + 0.35 * h), int(y + 0.48 * h)
    oi_x1, oi_x2 = int(x + 0.17 * w), int(x + 0.45 * w)  # Izquierda en cámara
    od_x1, od_x2 = int(x + 0.55 * w), int(x + 0.83 * w)  # Derecha en cámara

    boca_y1, boca_y2 = int(y + 0.65 * h), int(y + 0.90 * h)
    boca_x1, boca_x2 = int(x + 0.22 * w), int(x + 0.78 * w)

    img_cejas, rect_cejas = recortar(cejas_x1, cejas_y1, cejas_x2, cejas_y2)
    img_oi, rect_oi = recortar(oi_x1, ojo_y1, oi_x2, ojo_y2)
    img_od, rect_od = recortar(od_x1, ojo_y1, od_x2, ojo_y2)
    img_boca, rect_boca = recortar(boca_x1, boca_y1, boca_x2, boca_y2)

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

        # Historiales independientes por cada ojo
        self.hist_std_oi = []
        self.hist_std_od = []
        self.hist_rango_oi = []
        self.hist_rango_od = []
        self.hist_alt_oi = []
        self.hist_alt_od = []

        self.neutro_dist_cejas = 25.0
        self.neutro_brillo_piel = 110.0
        self.neutro_dientes_pct = 0.0

        # Lineas base neutras por cada ojo
        self.neutro_std_oi = 28.0
        self.neutro_std_od = 28.0
        self.neutro_rango_oi = 75.0
        self.neutro_rango_od = 75.0
        self.neutro_alt_oi = 12.0
        self.neutro_alt_od = 12.0

        self.ultimo_estado_boca = "SERIA"

    def calibrar(self, rois):
        """Calcula los promedios neutros de cejas, piel, ojos y boca durante los primeros fotogramas."""
        dist_c = self._medir_distancia_cejas(rois)

        img_boca = rois["boca"]["img"]
        if img_boca is not None and img_boca.size > 0:
            gris_boca = cv2.cvtColor(img_boca, cv2.COLOR_BGR2GRAY)
            self.hist_brillo_piel.append(float(np.mean(gris_boca)))
            _, pct_d = self._detectar_dientes_crudo(img_boca, brillo_ref=110.0)
            self.hist_dientes_neutro.append(pct_d)

        # Muestreo adaptativo autonomo por cada ojo (independiente de la boca)
        img_oi = rois["ojo_izq"]["img"]
        img_od = rois["ojo_der"]["img"]
        if img_oi is not None and img_oi.size > 0:
            m_oi = self._analizar_ojo_local(img_oi)
            self.hist_std_oi.append(m_oi["std"])
            self.hist_rango_oi.append(m_oi["rango"])
            self.hist_alt_oi.append(m_oi["alt_iris"])
        if img_od is not None and img_od.size > 0:
            m_od = self._analizar_ojo_local(img_od)
            self.hist_std_od.append(m_od["std"])
            self.hist_rango_od.append(m_od["rango"])
            self.hist_alt_od.append(m_od["alt_iris"])

        if dist_c is not None:
            self.hist_cejas.append(dist_c)

        self.frames_calibracion_cont += 1
        if self.frames_calibracion_cont >= FRAMES_CALIBRACION:
            self.neutro_dist_cejas = float(np.mean(self.hist_cejas)) if self.hist_cejas else 25.0
            self.neutro_brillo_piel = float(np.mean(self.hist_brillo_piel)) if self.hist_brillo_piel else 110.0
            self.neutro_dientes_pct = float(np.mean(self.hist_dientes_neutro)) if self.hist_dientes_neutro else 0.0

            if self.hist_std_oi:
                self.neutro_std_oi = max(14.0, float(np.mean(self.hist_std_oi)))
                self.neutro_rango_oi = max(35.0, float(np.mean(self.hist_rango_oi)))
                self.neutro_alt_oi = max(8.0, float(np.mean(self.hist_alt_oi)))
            if self.hist_std_od:
                self.neutro_std_od = max(14.0, float(np.mean(self.hist_std_od)))
                self.neutro_rango_od = max(35.0, float(np.mean(self.hist_rango_od)))
                self.neutro_alt_od = max(8.0, float(np.mean(self.hist_alt_od)))

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

    def _analizar_ojo_local(self, img_ojo):
        """
        Analiza las caracteristicas fotometricas y geometricas del ojo de manera totalmente
        autonoma (sin depender de la iluminacion de la barbilla o boca).
        """
        if img_ojo is None or img_ojo.size == 0:
            return {"rango": 0.0, "std": 0.0, "alt_iris": 0, "aspect_iris": 0.0}

        gris = cv2.cvtColor(img_ojo, cv2.COLOR_BGR2GRAY)
        h, w = gris.shape
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        zona = gris[y1:y2, x1:x2]
        if zona.size == 0:
            return {"rango": 0.0, "std": 0.0, "alt_iris": 0, "aspect_iris": 0.0}

        p10 = float(np.percentile(zona, 10))
        p50 = float(np.percentile(zona, 50))
        p90 = float(np.percentile(zona, 90))
        std_val = float(np.std(zona))
        rango_dinamico = float(p90 - p10)

        # Umbral adaptativo local al ojo: separa iris de esclerotica y parpado
        umbral_iris = (p10 + p50) / 2.0
        masc_iris = (zona < umbral_iris).astype(np.uint8) * 255

        kernel = np.ones((2, 2), np.uint8)
        masc_iris = cv2.morphologyEx(masc_iris, cv2.MORPH_OPEN, kernel)

        contornos, _ = cv2.findContours(masc_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alt_iris = 0
        aspect_iris = 0.0

        if len(contornos) > 0:
            c_mayor = max(contornos, key=cv2.contourArea)
            _, _, bw, bh = cv2.boundingRect(c_mayor)
            alt_iris = bh
            aspect_iris = bh / max(1.0, float(bw))

        return {
            "rango": rango_dinamico,
            "std": std_val,
            "alt_iris": int(alt_iris),
            "aspect_iris": float(aspect_iris)
        }

    def _evaluar_ojo_individual(self, img_ojo, neutro_std, neutro_rango, neutro_alt):
        """
        Evalua el estado de apertura de un ojo comparando su desviacion estandar,
        rango dinamico y altura de iris contra los valores neutros de calibracion.
        Retorna una tupla (esta_cerrado, score_cierre).
        """
        if img_ojo is None or img_ojo.size == 0:
            return True, 1.0

        m = self._analizar_ojo_local(img_ojo)
        alt_iris = m["alt_iris"]
        aspect_iris = m["aspect_iris"]
        std_val = m["std"]
        rango = m["rango"]

        # Deteccion de pestanas horizontales en cierre forzado
        es_linea_pestanas = (alt_iris <= 4) or (aspect_iris < ASPECTO_MINIMO_IRIS_ABIERTO)

        # Caida de contraste caracteristica del parpado cerrado
        umbral_std_min = max(10.0, neutro_std * FACTOR_APERTURA_STD)
        umbral_rango_min = max(25.0, neutro_rango * FACTOR_APERTURA_RANGO)
        es_parpado_liso = (std_val < umbral_std_min) or (rango < umbral_rango_min)

        # Altura vertical minima del iris visible
        umbral_alt_min = max(ALTURA_MINIMA_IRIS_ABIERTO, int(neutro_alt * 0.40))

        ojo_abierto = (alt_iris >= umbral_alt_min) and (not es_linea_pestanas) and (not es_parpado_liso)

        if (std_val >= neutro_std * 0.85) and (alt_iris >= 5) and (aspect_iris >= 0.25):
            ojo_abierto = True

        score_cierre = 0.0 if ojo_abierto else 1.0
        return bool(not ojo_abierto), float(score_cierre)

    def _evaluar_ojos(self, rois):
        """Determina de forma diferencial el estado de ambos ojos usando sus lineas base independientes."""
        izq_cerrado, score_oi = self._evaluar_ojo_individual(
            rois["ojo_izq"]["img"],
            self.neutro_std_oi,
            self.neutro_rango_oi,
            self.neutro_alt_oi
        )
        der_cerrado, score_od = self._evaluar_ojo_individual(
            rois["ojo_der"]["img"],
            self.neutro_std_od,
            self.neutro_rango_od,
            self.neutro_alt_od
        )

        return izq_cerrado, der_cerrado, float(score_oi), float(score_od)

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
        izq_c, der_c, score_oi, score_od = self._evaluar_ojos(rois)
        if izq_c:
            regiones["ojo_izq"] = True
        if der_c:
            regiones["ojo_der"] = True

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
            cumplido = bool(izq_c and der_c)

        elif gesto == "cerrar_ojo_izq":
            cumplido = bool(izq_c and not der_c)

        elif gesto == "cerrar_ojo_der":
            cumplido = bool(der_c and not izq_c)

        return bool(cumplido), regiones

    def procesar_fotograma(self, frame, cara_rect):
        """
        Ciclo principal de validacion: extrae ROIs, evalua reto activo y dibuja HUD.
        Retorna (estado, frame_anotado).
        """
        x, y, w, h = cara_rect
        rois = extraer_rois(frame, x, y, w, h)
        frame_anotado = frame.copy()

        # 1. Fase de calibracion inicial
        if not self.calibrado:
            self.calibrar(rois)
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

        if gesto_detectado:
            self.frames_confirmados += 1
        else:
            self.frames_confirmados = max(0, self.frames_confirmados - 1)

        # 4. Comprobar superacion del paso
        if self.frames_confirmados >= FRAMES_CONFIRMACION_GESTO:
            reproducir_audio(AUDIO_SUCCESS)
            self.paso_actual += 1
            self.frames_confirmados = 0
            self.tiempo_inicio_gesto = time.time()

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
            total_frames_ok=FRAMES_CONFIRMACION_GESTO,
            calibrando=False
        )

        return "EN_PROCESO", frame_anotado
