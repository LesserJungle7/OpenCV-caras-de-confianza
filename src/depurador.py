"""
Modulo de depuracion visual, telemetria biometrica y gestion de ventanas auxiliares.
Controla el ciclo de vida de inspeccion por fases operativas y provee mecanismos
de pausa interactiva para analisis de procesado de imagen en tiempo real.
"""

import cv2
import numpy as np

from config import (
    MODO_DEBUG_POR_DEFECTO,
    VENTANA_PRINCIPAL,
    VENTANA_DEBUG_NORMALIZACION,
    VENTANA_DEBUG_ESTABILIDAD,
    VENTANA_DEBUG_MATCHING,
    VENTANA_DEBUG_BOCA,
    VENTANA_DEBUG_OJOS,
    VENTANA_DEBUG_CEJAS,
    VENTANAS_DEBUG,
    TAM_CARA,
    UMBRAL_COINCIDENCIA,
    UMBRAL_MOVIMIENTO_ESTABLE,
    UMBRAL_MOVIMIENTO_KNN
)

# Estado global del modo depurador y control de ciclo de vida por fases
_modo_debug = MODO_DEBUG_POR_DEFECTO
_ventanas_abiertas = set()
_fase_actual = "menu"

# Mapeo de fases operativas y ventanas de depuracion permitidas
FASES_VENTANAS = {
    "menu": set(),
    "creacion_perfil": {VENTANA_DEBUG_NORMALIZACION},
    "guardia_login": {VENTANA_DEBUG_ESTABILIDAD, VENTANA_DEBUG_NORMALIZACION, VENTANA_DEBUG_MATCHING},
    "reto_gestos": {VENTANA_DEBUG_BOCA, VENTANA_DEBUG_OJOS, VENTANA_DEBUG_CEJAS},
    "retorno_ausencia": {VENTANA_DEBUG_ESTABILIDAD, VENTANA_DEBUG_NORMALIZACION, VENTANA_DEBUG_MATCHING},
    "sesion_activa": set(),
}

def es_debug_activo():
    """Devuelve True si el modo debug esta actualmente activado."""
    return _modo_debug

def activar_debug():
    """Activa el modo debug de forma explicita."""
    global _modo_debug
    _modo_debug = True
    print("[DEBUG] Modo depurador activado (ON). Visualizando etapas de procesamiento.")

def desactivar_debug():
    """Desactiva el modo debug y cierra todas las ventanas auxiliares creadas."""
    global _modo_debug
    _modo_debug = False
    cerrar_ventanas_debug()
    print("[DEBUG] Modo depurador desactivado (OFF). Ventanas auxiliares cerradas.")

def toggle_debug():
    """Alterna el estado del modo debug al pulsar la tecla 'd'."""
    if _modo_debug:
        desactivar_debug()
    else:
        activar_debug()
    return _modo_debug

def cambiar_fase(nueva_fase):
    """
    Actualiza la fase operativa y destruye de inmediato las ventanas de depuracion
    que no pertenezcan a la nueva fase.
    """
    global _fase_actual, _ventanas_abiertas
    _fase_actual = nueva_fase
    cerradas = False
    if nueva_fase in FASES_VENTANAS:
        permitidas = FASES_VENTANAS[nueva_fase]
        a_cerrar = [v for v in _ventanas_abiertas if v not in permitidas]
        for v in a_cerrar:
            try:
                cv2.destroyWindow(v)
                cerradas = True
            except Exception:
                pass
            _ventanas_abiertas.discard(v)
    if cerradas:
        cv2.waitKey(1)

def cerrar_debug_fase(fase):
    """
    Cierra todas las ventanas de depuracion asociadas a una fase especifica.
    """
    global _ventanas_abiertas
    cerradas = False
    if fase in FASES_VENTANAS:
        for v in FASES_VENTANAS[fase]:
            if v in _ventanas_abiertas:
                try:
                    cv2.destroyWindow(v)
                    cerradas = True
                except Exception:
                    pass
                _ventanas_abiertas.discard(v)
    if cerradas:
        cv2.waitKey(1)

def cerrar_ventanas_debug():
    """Cierra limpiamente todas las ventanas de inspeccion de OpenCV."""
    global _ventanas_abiertas
    for ventana in VENTANAS_DEBUG:
        try:
            cv2.destroyWindow(ventana)
        except Exception:
            pass
    _ventanas_abiertas.clear()
    cv2.waitKey(1)

def _es_ventana_permitida(nombre_ventana):
    """Verifica si el modo debug esta activo y si la ventana corresponde a la fase operativa actual."""
    if not _modo_debug:
        return False
    if _fase_actual in FASES_VENTANAS:
        return nombre_ventana in FASES_VENTANAS[_fase_actual]
    return True

def pausar_inspeccion_debug(frame, mensaje, tecla_continuar='s'):
    """
    Si el modo depurador esta activo, renderiza una franja informativa sobre 'frame',
    actualiza la ventana principal de la camara y entra en bucle de espera
    hasta que el usuario confirme con 's'/'S', alterne debug con 'd'/'D' o cancele con 'q'/'Q'/ESC.

    Retorna True si el usuario confirma continuar ('s'), False si cancela.
    Si el modo depurador NO esta activo, retorna True inmediatamente sin pausar.
    """
    if not _modo_debug or frame is None or frame.size == 0:
        return True

    h, w = frame.shape[:2]
    frame_pausa = frame.copy()

    # Franja inferior destacada para notificar de la pausa de inspeccion
    alto_franja = 54
    cv2.rectangle(frame_pausa, (0, h - alto_franja), (w, h), (15, 15, 30), -1)
    cv2.rectangle(frame_pausa, (0, h - alto_franja), (w, h), (0, 255, 255), 2)
    cv2.putText(frame_pausa, mensaje, (15, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 255), 1)
    cv2.putText(frame_pausa, f"[{tecla_continuar.upper()}] Continuar  |  [D] Toggle Debug  |  [Q] Cancelar",
                (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.imshow(VENTANA_PRINCIPAL, frame_pausa)

    while True:
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord(tecla_continuar.lower()), ord(tecla_continuar.upper())):
            return True
        elif tecla in (ord('d'), ord('D')):
            toggle_debug()
            if not _modo_debug:
                return True
        elif tecla in (ord('q'), ord('Q'), 27):
            return False

def _asegurar_bgr(imagen):
    """Convierte una imagen de 1 canal a 3 canales BGR para estandarizar visualizaciones."""
    if imagen is None or imagen.size == 0:
        return np.zeros((120, 120, 3), dtype=np.uint8)
    if len(imagen.shape) == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    return imagen

def _crear_tile(imagen, etiqueta, subtitulo="", tam=(160, 160), color_borde=(60, 60, 60)):
    """Crea un recuadro individual con titulo superior y subtitulo inferior."""
    base = _asegurar_bgr(imagen)
    redim = cv2.resize(base, tam, interpolation=cv2.INTER_LINEAR)
    
    alto, ancho = redim.shape[:2]
    canvas = np.zeros((alto + 42, ancho, 3), dtype=np.uint8)
    
    # Franja superior de titulo
    cv2.rectangle(canvas, (0, 0), (ancho, 22), (25, 25, 30), -1)
    cv2.putText(canvas, etiqueta, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    
    # Imagen centrada
    canvas[22:22 + alto, 0:ancho] = redim
    cv2.rectangle(canvas, (0, 22), (ancho - 1, 22 + alto - 1), color_borde, 1)
    
    # Franja inferior de subtitulo
    cv2.rectangle(canvas, (0, 22 + alto), (ancho, alto + 42), (20, 20, 25), -1)
    if subtitulo:
        cv2.putText(canvas, subtitulo, (6, alto + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
    return canvas

# Pipeline de normalizacion facial
def mostrar_debug_normalizacion(etapas):
    """
    Renderiza un mosaico con las etapas consecutivas de normalizacion facial:
    1. Recorte crudo / Ajuste de iluminacion
    2. Nivelacion afín por vector ocular
    3. Recorte de margen de bordes (8%)
    4. Ecualizacion adaptativa CLAHE
    5. Filtro Gaussiano final (120x120)
    """
    if not _es_ventana_permitida(VENTANA_DEBUG_NORMALIZACION) or not etapas:
        return

    tile_crudo = _crear_tile(
        etapas.get("crudo"),
        "1. Recorte Crudo",
        f"Luz: {etapas.get('brillo_medio', 'N/D')}"
    )
    
    ang = etapas.get("angulo")
    sub_ang = f"Ang: {ang:.1f} deg" if ang is not None else "Ang: 0.0 (Neutro)"
    tile_alineada = _crear_tile(
        etapas.get("alineada"),
        "2. Alineacion Ocular",
        sub_ang
    )
    
    tile_margen = _crear_tile(
        etapas.get("margen"),
        "3. Margen (-8%)",
        "Bordes excluidos"
    )
    
    tile_clahe = _crear_tile(
        etapas.get("clahe"),
        "4. Filtro CLAHE",
        "Contraste adapt."
    )
    
    tile_final = _crear_tile(
        etapas.get("final"),
        "5. Gaussiano Final",
        f"Tam: {TAM_CARA[0]}x{TAM_CARA[1]}",
        color_borde=(0, 255, 0)
    )

    mosaico = np.hstack([tile_crudo, tile_alineada, tile_margen, tile_clahe, tile_final])
    cv2.imshow(VENTANA_DEBUG_NORMALIZACION, mosaico)
    _ventanas_abiertas.add(VENTANA_DEBUG_NORMALIZACION)

# Estabilizacion y quietud facial (sustractor KNN)
def mostrar_debug_estabilidad(roi_gris, roi_mascara_knn, area_movimiento, desplazamiento, aspect, es_estable, frames_estables=0, umbral_frames=12, contorno_max=None):
    """
    Muestra la inspeccion de quietud (Sustraccion de Fondo KNN) y estabilidad posicional.
    Visualiza el rostro capturado, la mascara de primer plano con el contorno de movimiento
    y el panel de telemetria con histeresis de estabilizacion.
    """
    if not _es_ventana_permitida(VENTANA_DEBUG_ESTABILIDAD) or roi_gris is None or roi_gris.size == 0:
        return

    tam = (180, 180)
    roi_bgr = _asegurar_bgr(cv2.resize(roi_gris, tam))

    # Mascara de primer plano KNN con contorno de movimiento resaltado
    if roi_mascara_knn is not None and roi_mascara_knn.size > 0:
        mascara_redim = cv2.resize(roi_mascara_knn, tam, interpolation=cv2.INTER_NEAREST)
        mascara_bgr = cv2.cvtColor(mascara_redim, cv2.COLOR_GRAY2BGR)
        if contorno_max is not None and len(contorno_max) > 0:
            h_orig, w_orig = roi_mascara_knn.shape[:2]
            if w_orig > 0 and h_orig > 0:
                sx = tam[0] / float(w_orig)
                sy = tam[1] / float(h_orig)
                contorno_escalado = (contorno_max * np.array([sx, sy])).astype(np.int32)
                cv2.drawContours(mascara_bgr, [contorno_escalado], 0, (0, 255, 255), 2)
    else:
        mascara_bgr = np.zeros((tam[1], tam[0], 3), dtype=np.uint8)

    # Panel lateral de telemetria
    panel_info = np.zeros((tam[1], 260, 3), dtype=np.uint8)
    cv2.rectangle(panel_info, (0, 0), (259, tam[1] - 1), (30, 30, 35), -1)

    col_knn = (0, 255, 0) if area_movimiento <= UMBRAL_MOVIMIENTO_KNN else (0, 0, 255)
    col_desp = (0, 255, 0) if desplazamiento <= UMBRAL_MOVIMIENTO_ESTABLE else (0, 0, 255)
    col_asp = (0, 255, 0) if (0.70 <= aspect <= 1.30) else (0, 0, 255)
    col_est = (0, 255, 0) if es_estable else (0, 165, 255)

    cv2.putText(panel_info, "TELEMETRIA ESTABILIDAD (KNN)", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 255, 255), 1)
    cv2.putText(panel_info, f"Mov. KNN: {area_movimiento:.0f}px (max {int(UMBRAL_MOVIMIENTO_KNN)})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_knn, 1)
    cv2.putText(panel_info, f"Desplazamiento: {desplazamiento:.1f}px (max {int(UMBRAL_MOVIMIENTO_ESTABLE)})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_desp, 1)
    cv2.putText(panel_info, f"Aspecto w/h: {aspect:.2f} [0.7-1.3]", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_asp, 1)
    cv2.putText(panel_info, f"Frames OK: {frames_estables}/{umbral_frames}", (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    estado_txt = "ESTABLE (+1)" if es_estable else "MOVIMIENTO KNN (-2)"
    cv2.putText(panel_info, f"Estado: {estado_txt}", (10, 158), cv2.FONT_HERSHEY_DUPLEX, 0.40, col_est, 1)

    mosaico = np.hstack([
        _crear_tile(roi_bgr, "Rostro Capturado", "ROI facial", tam=tam),
        _crear_tile(mascara_bgr, "Mascara KNN", "Primer plano / Contorno", tam=tam),
        _crear_tile(panel_info, "Evaluacion Metricas", "Histeresis dinamica", tam=(260, tam[1]))
    ])

    cv2.imshow(VENTANA_DEBUG_ESTABILIDAD, mosaico)
    _ventanas_abiertas.add(VENTANA_DEBUG_ESTABILIDAD)

# Reconocimiento facial y template matching
def mostrar_debug_matching(cara_actual, mejor_patron, nombre_detectado, score, umbral=UMBRAL_COINCIDENCIA):
    """
    Renderiza la comparacion lado a lado entre la cara capturada, la plantilla
    del usuario y el mapa de calor de diferencias cuadraticas normalizadas.
    """
    if not _es_ventana_permitida(VENTANA_DEBUG_MATCHING) or cara_actual is None or cara_actual.size == 0:
        return

    tam = (170, 170)
    actual_bgr = _asegurar_bgr(cv2.resize(cara_actual, tam))

    if mejor_patron is not None and mejor_patron.size > 0:
        patron_bgr = _asegurar_bgr(cv2.resize(mejor_patron, tam))
        # Diferencia absoluta punto a punto proyectada como mapa de calor
        diff = cv2.absdiff(
            cv2.resize(cara_actual, (120, 120)),
            cv2.resize(mejor_patron, (120, 120))
        )
        mapa_calor = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        mapa_calor = cv2.resize(mapa_calor, tam)
    else:
        patron_bgr = np.zeros((tam[1], tam[0], 3), dtype=np.uint8)
        mapa_calor = np.zeros((tam[1], tam[0], 3), dtype=np.uint8)

    # Panel de datos numéricos
    panel_info = np.zeros((tam[1], 240, 3), dtype=np.uint8)
    cv2.rectangle(panel_info, (0, 0), (239, tam[1] - 1), (25, 25, 30), -1)

    coincide = score < umbral and nombre_detectado is not None
    col_res = (0, 255, 0) if coincide else (0, 0, 255)
    similitud = max(0.0, min(100.0, (1.0 - score) * 100.0))

    cv2.putText(panel_info, "METRICAS SQDIFF", (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)
    usr_str = nombre_detectado.upper() if nombre_detectado else "DESCONOCIDO"
    cv2.putText(panel_info, f"Candidato: {usr_str}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(panel_info, f"Score SQDIFF: {score:.4f}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_res, 1)
    cv2.putText(panel_info, f"Umbral max: {umbral:.4f}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(panel_info, f"Similitud: {similitud:.1f}%", (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_res, 1)
    
    estado_str = "VALIDADO" if coincide else "DENEGADO / INTRUSO"
    cv2.putText(panel_info, f"Resultado: {estado_str}", (10, 154), cv2.FONT_HERSHEY_DUPLEX, 0.38, col_res, 1)

    mosaico = np.hstack([
        _crear_tile(actual_bgr, "Cara Evaluada", "Normalizada 120x120", tam=tam),
        _crear_tile(patron_bgr, "Plantilla Autorizada", f"Ref: {usr_str}", tam=tam),
        _crear_tile(mapa_calor, "Mapa Discrepancia", "Jet Heatmap", tam=tam),
        _crear_tile(panel_info, "Decision Algoritmica", "TM_SQDIFF_NORMED", tam=(240, tam[1]))
    ])

    cv2.imshow(VENTANA_DEBUG_MATCHING, mosaico)
    _ventanas_abiertas.add(VENTANA_DEBUG_MATCHING)

# Prueba de vida (boca, ojos y cejas)
def mostrar_debug_gestos_boca(img_boca_canon, mascara_dientes, pct_dientes, umbral_dientes, tiene_franja, es_sonrisa):
    """Visualiza la segmentacion de la boca, mascara de esmalte y contorno dental."""
    if not _es_ventana_permitida(VENTANA_DEBUG_BOCA) or img_boca_canon is None:
        return

    tam = (180, 110)
    boca_bgr = _asegurar_bgr(cv2.resize(img_boca_canon, tam))
    
    masc_col = cv2.applyColorMap(cv2.resize(mascara_dientes, tam), cv2.COLORMAP_BONE)
    
    # Overlay del contorno
    overlay_contorno = boca_bgr.copy()
    col_franja = (0, 255, 0) if tiene_franja else (0, 0, 255)
    cv2.rectangle(overlay_contorno, (int(tam[0] * 0.20), int(tam[1] * 0.28)), 
                  (int(tam[0] * 0.80), int(tam[1] * 0.58)), col_franja, 1)

    panel_info = np.zeros((tam[1] + 42, 220, 3), dtype=np.uint8)
    cv2.rectangle(panel_info, (0, 0), (219, tam[1] + 41), (25, 25, 30), -1)

    col_sonr = (0, 255, 0) if es_sonrisa else (0, 0, 255)
    cv2.putText(panel_info, "ANALISIS BUCAL", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)
    cv2.putText(panel_info, f"Esmalte: {pct_dientes * 100:.1f}%", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_sonr, 1)
    cv2.putText(panel_info, f"Umbral min: {umbral_dientes * 100:.1f}%", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(panel_info, f"Franja continua: {'SI' if tiene_franja else 'NO'}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_franja, 1)
    cv2.putText(panel_info, f"Estado: {'SONRISA' if es_sonrisa else 'SERIA'}", (10, 125), cv2.FONT_HERSHEY_DUPLEX, 0.40, col_sonr, 1)

    mosaico = np.hstack([
        _crear_tile(boca_bgr, "Boca Canonica (100x60)", "Apertura labial", tam=tam),
        _crear_tile(masc_col, "Mascara Esmalte", "BGR + YCrCb", tam=tam),
        _crear_tile(overlay_contorno, "Zona de Dientes", "Alineacion dental", tam=tam),
        panel_info
    ])

    cv2.imshow(VENTANA_DEBUG_BOCA, mosaico)
    _ventanas_abiertas.add(VENTANA_DEBUG_BOCA)

def mostrar_debug_gestos_ojos(img_oi, img_od, asp_oi, asp_od, umbral_oi, umbral_od, hist_oi, hist_od, izq_cerrado, der_cerrado):
    """Visualiza la inspeccion ocular y aplicacion de la Regla K adaptativa."""
    if not _es_ventana_permitida(VENTANA_DEBUG_OJOS):
        return

    tam = (150, 110)
    oi_bgr = _asegurar_bgr(cv2.resize(img_oi, tam)) if img_oi is not None else np.zeros((tam[1], tam[0], 3), dtype=np.uint8)
    od_bgr = _asegurar_bgr(cv2.resize(img_od, tam)) if img_od is not None else np.zeros((tam[1], tam[0], 3), dtype=np.uint8)

    panel_info = np.zeros((tam[1] + 42, 280, 3), dtype=np.uint8)
    cv2.rectangle(panel_info, (0, 0), (279, tam[1] + 41), (25, 25, 30), -1)

    col_oi = (0, 255, 0) if izq_cerrado else (0, 165, 255)
    col_od = (0, 255, 0) if der_cerrado else (0, 165, 255)

    cv2.putText(panel_info, "REGLA K - OJOS", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)
    cv2.putText(panel_info, f"Ojo Izq (bh/bw): {asp_oi:.3f} (K: {umbral_oi:.3f})", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_oi, 1)
    cv2.putText(panel_info, f"Estado Izq: {'CERRADO' if izq_cerrado else 'ABIERTO'}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.38, col_oi, 1)
    cv2.putText(panel_info, f"Ojo Der (bh/bw): {asp_od:.3f} (K: {umbral_od:.3f})", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_od, 1)
    cv2.putText(panel_info, f"Estado Der: {'CERRADO' if der_cerrado else 'ABIERTO'}", (10, 118), cv2.FONT_HERSHEY_DUPLEX, 0.38, col_od, 1)
    
    suma_oi = sum(hist_oi) if hist_oi else 0
    suma_od = sum(hist_od) if hist_od else 0
    cv2.putText(panel_info, f"Buffer temporal: Izq={suma_oi}/8 | Der={suma_od}/8", (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)

    mosaico = np.hstack([
        _crear_tile(oi_bgr, "Ojo Izquierdo", "Region ocular", tam=tam),
        _crear_tile(od_bgr, "Ojo Derecho", "Region ocular", tam=tam),
        panel_info
    ])

    cv2.imshow(VENTANA_DEBUG_OJOS, mosaico)
    _ventanas_abiertas.add(VENTANA_DEBUG_OJOS)

def mostrar_debug_gestos_cejas(img_cejas, dist_norm, neutro_norm, cejas_levantadas):
    """Visualiza el centroide y desplazamiento vertical antropometrico de las cejas."""
    if not _es_ventana_permitida(VENTANA_DEBUG_CEJAS) or img_cejas is None:
        return

    tam = (200, 100)
    cejas_bgr = _asegurar_bgr(cv2.resize(img_cejas, tam))

    panel_info = np.zeros((tam[1] + 42, 240, 3), dtype=np.uint8)
    cv2.rectangle(panel_info, (0, 0), (239, tam[1] + 41), (25, 25, 30), -1)

    col_cej = (0, 255, 0) if cejas_levantadas else (0, 165, 255)
    umbral_activacion = neutro_norm * 1.16 if neutro_norm else 0.40

    cv2.putText(panel_info, "EVALUACION CEJAS", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)
    dist_txt = f"{dist_norm:.3f}" if dist_norm is not None else "N/D"
    cv2.putText(panel_info, f"Distancia vert / D: {dist_txt}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_cej, 1)
    cv2.putText(panel_info, f"Umbral (x1.16): {umbral_activacion:.3f}", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
    
    estado_cej = "LEVANTADAS" if cejas_levantadas else "REPOSO"
    cv2.putText(panel_info, f"Estado: {estado_cej}", (10, 110), cv2.FONT_HERSHEY_DUPLEX, 0.40, col_cej, 1)

    mosaico = np.hstack([
        _crear_tile(cejas_bgr, "Region Cejas", "Percentil 25", tam=tam),
        panel_info
    ])

    cv2.imshow(VENTANA_DEBUG_CEJAS, mosaico)
    _ventanas_abiertas.add(VENTANA_DEBUG_CEJAS)
