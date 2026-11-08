import os
import time
import numpy as np
import cv2

from config import (
    CARPETA_CONFIANZA,
    CARPETA_INTRUSOS,
    ARCHIVO_CONFIG,
    RUTA_CASCADE,
    UMBRAL_COINCIDENCIA,
    TIEMPO_AUSENCIA_GUARDIA,
    TIEMPO_MAXIMO_GESTO,
    FRAMES_ESTABILIDAD_GUARDIA,
    TIEMPO_DISPARO_REENGANCHE,
    FRAMES_CONFIRMACION_RETORNO,
    UMBRAL_REENGANCHE_SESION,
    COLOR_RECUADRO,
    COLOR_TEXTO,
    FUENTE,
    GROSOR,
    AUDIO_ERROR
)
from vision import (
    cargar_caras_confianza,
    identificar_rostro,
    obtener_clasificador,
    detectar_caras,
    normalizar_cara,
    verificar_rostro_estatico,
    verificar_usuario_especifico,
    obtener_plantillas_usuario
)
from sistema import suspender_equipo, guardar_evidencia_intruso, reproducir_audio
from usuarios import (
    cargar_usuarios,
    inicializar_entorno,
    obtener_usuario,
    obtener_estadisticas_usuario,
    registrar_inicio_sesion,
    registrar_fin_sesion
)
from gestos import EvaluadorGestos

def formatear_duracion(segundos):
    """Convierte una duracion en segundos a formato legible HHh MMm SSs o MMm SSs."""
    segundos = int(max(0, segundos))
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    if horas > 0:
        return f"{horas:02d}h {minutos:02d}m {segs:02d}s"
    return f"{minutos:02d}m {segs:02d}s"

def dibujar_interfaz_usuario(frame, usuario, rol, stats, duracion_actual, mostrar_detalles, tiempo_ausente=0.0, similitud_sesion=None):
    """
    Renderiza la informacion del usuario en la esquina superior izquierda.
    Muestra 'Nombre de usuario: USUARIO' y 'm - Mostrar más detalles'.
    Al pulsar 'm', despliega un panel HUD con las estadísticas detalladas:
    rol, tiempo de sesion, inicios de sesion, similitud facial media (Template Matching)
    e intentos de intrusion registrados desde la ultima sesion.
    """
    overlay = frame.copy()
    x1, y1 = 15, 12
    ancho_card = 415
    alto_card = 62

    # Fondo semi-transparente para la tarjeta principal
    cv2.rectangle(overlay, (x1, y1), (x1 + ancho_card, y1 + alto_card), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x1 + ancho_card, y1 + alto_card), (0, 255, 255), 1)

    # Si se pulsa 'm', se expande el panel de estadisticas hacia abajo
    if mostrar_detalles:
        alto_panel = 248
        cv2.rectangle(overlay, (x1, y1 + alto_card + 4), (x1 + ancho_card, y1 + alto_card + 4 + alto_panel), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x1, y1 + alto_card + 4), (x1 + ancho_card, y1 + alto_card + 4 + alto_panel), (0, 255, 0), 1)

    # Efecto traslucido
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Texto esquina superior izquierda
    cv2.putText(frame, f"Nombre de usuario: {usuario.upper()}", (x1 + 12, y1 + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 255), 1)

    texto_guia = "m - Ocultar detalles" if mostrar_detalles else "m - Mostrar mas detalles"
    color_guia = (0, 255, 0) if mostrar_detalles else (190, 190, 190)
    cv2.putText(frame, texto_guia, (x1 + 12, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_guia, 1)

    # Panel de estadisticas expandido
    if mostrar_detalles:
        base_y = y1 + alto_card + 26
        cv2.putText(frame, "--- ESTADISTICAS DE USUARIO ---", (x1 + 15, base_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 0), 1)

        inicios = stats.get("inicios_sesion", 1)
        tiempo_acum = stats.get("tiempo_acumulado_segundos", 0.0) + duracion_actual
        tiempo_medio = tiempo_acum / max(1, inicios)
        ultimo_acc = stats.get("ultimo_acceso") or "Primer inicio"
        intrusos = stats.get("intrusos_ultima_sesion", 0)

        sim_media = similitud_sesion if similitud_sesion is not None else stats.get("similitud_media", 0.0)

        items = [
            ("Rol asignado:", str(rol), (0, 255, 255)),
            ("Tiempo de sesion:", formatear_duracion(duracion_actual), (0, 255, 255)),
            ("Inicios de sesion:", f"{inicios} veces", (0, 255, 255)),
            ("Similitud facial media:", f"{sim_media:.1f}%", (0, 255, 0) if sim_media >= 80 else (0, 165, 255)),
            ("Intrusos ult. sesion:", f"{intrusos} detectados" if intrusos > 0 else "0 (Ninguno)", (0, 0, 255) if intrusos > 0 else (0, 255, 0)),
            ("Ultimo acceso:", str(ultimo_acc), (0, 255, 255)),
            ("Tiempo acumulado:", formatear_duracion(tiempo_acum), (0, 255, 255)),
            ("Promedio por sesion:", formatear_duracion(tiempo_medio), (0, 255, 255)),
        ]

        if tiempo_ausente > 0:
            items.append(("Estado presencia:", f"AUSENTE ({int(tiempo_ausente)}s)", (0, 0, 255)))
        else:
            items.append(("Estado presencia:", "PRESENTE", (0, 255, 0)))

        y_offset = base_y + 24
        for etiqueta, valor, col_val in items:
            cv2.putText(frame, etiqueta, (x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            cv2.putText(frame, valor, (x1 + 220, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.42, col_val, 1)
            y_offset += 24

def dibujar_barra_progreso(frame, x, y, ancho, alto, porcentaje, color_barra):
    """Dibuja una barra horizontal de progreso con fondo oscuro y borde."""
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (50, 50, 50), -1)
    if porcentaje > 0:
        ancho_lleno = int(ancho * (min(100, max(0, porcentaje)) / 100.0))
        cv2.rectangle(frame, (x, y), (x + ancho_lleno, y + alto), color_barra, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (100, 100, 100), 1)

def actualizar_estabilidad_facial(caja_actual, ultima_caja, frame_byn, frames_estables):
    """
    Determina si el rostro se mantiene estatico y actualiza el contador acumulador
    aplicando penalizacion de histeresis (-2) en caso de movimiento.
    Retorna (es_estable, nuevo_contador_frames).
    """
    es_estable = verificar_rostro_estatico(caja_actual, ultima_caja, frame_byn)
    if es_estable:
        nuevo_contador = frames_estables + 1
    else:
        nuevo_contador = max(0, frames_estables - 2)
    return es_estable, nuevo_contador

def dibujar_feedback_estabilizacion(frame, caja, progreso, texto_titulo="ESTABILIZANDO ROSTRO", pos_banner=(20, 40)):
    """Renderiza el recuadro, etiqueta y barra de progreso de estabilizacion facial."""
    x, y, ancho, alto = caja
    color_estabilidad = (0, 255, 255) if progreso > 0 else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color_estabilidad, GROSOR)
    cv2.putText(frame, f"{texto_titulo}: {progreso}%", pos_banner, FUENTE, 0.65, color_estabilidad, GROSOR)
    cv2.putText(frame, "MANTENTE QUIETO Y MIRA A CAMARA", (x, max(20, y - 10)), FUENTE, 0.50, color_estabilidad, 2)
    dibujar_barra_progreso(frame, x, y + alto + 10, ancho, 6, progreso, color_estabilidad)

def verificar_password_texto(usuario, datos_sistema=None):
    """Solicita y valida la contraseña escrita del usuario detectado."""
    print(f"\n[AVISO] Usuario de confianza detectado: {usuario.upper()}")
    password_ingresada = input(f"Introduce tu contraseña, {usuario}: ").strip()

    info_usuario = datos_sistema.get("usuarios", {}).get(usuario) if datos_sistema else obtener_usuario(usuario)

    if info_usuario and password_ingresada == info_usuario.get("password"):
        print("[OK] Contraseña de texto correcta. Iniciando reto de contraseña facial...")
        return True
    else:
        print("[ERROR] Contraseña incorrecta.")
        return False

def mostrar_permisos_usuario(usuario, datos_sistema=None):
    """Muestra por terminal el mensaje de bienvenida y el rol asignado."""
    info_usuario = datos_sistema.get("usuarios", {}).get(usuario) if datos_sistema else obtener_usuario(usuario)

    rol = info_usuario.get("rol", "Invitado") if info_usuario else "Invitado"
    print(f"\n[ACCESO CONCEDIDO] Bienvenido/a {usuario}.")
    print(f"[PERMISOS ACTIVADOS] Nivel de acceso: {rol}")
    if rol == "Total":
        print(" -> Tienes acceso a configuración, terminal y archivos restringidos.")
    elif rol == "Operador":
        print(" -> Tienes acceso a ejecución de aplicaciones estándar.")
    else:
        print(" -> Modo invitado: Acceso restringido solo lectura.")

def validar_contrasena_facial(camara, clasificador):
    """
    Ejecuta el reto de 3 gestos aleatorios en la ventana de la camara.
    Retorna True si el usuario supera la prueba de vida, o False en caso de timeout.
    """
    evaluador = EvaluadorGestos()
    print("[PRUEBA DE VIDA] Sigue las instrucciones que aparecen en la ventana de la camara.")

    while True:
        ret, frame = camara.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)

        caras, _ = detectar_caras(frame, clasificador)
        if len(caras) > 0:
            estado_reto, frame_hud = evaluador.procesar_fotograma(frame, caras[0])
            if estado_reto == "EXITO":
                cv2.putText(frame_hud, "ACCESO AUTORIZADO", (frame.shape[1] // 4, frame.shape[0] // 2),
                            FUENTE, 1.0, (0, 255, 0), 3)
                cv2.imshow("Autenticador Facial", frame_hud)
                cv2.waitKey(600)
                return True
            elif estado_reto == "TIMEOUT":
                reproducir_audio(AUDIO_ERROR)
                cv2.putText(frame_hud, "TIEMPO AGOTADO - BLOQUEANDO", (frame.shape[1] // 6, frame.shape[0] // 2),
                            FUENTE, 0.8, (0, 0, 255), 2)
                cv2.imshow("Autenticador Facial", frame_hud)
                cv2.waitKey(800)
                return False
        else:
            frame_hud = frame.copy()
            if evaluador.calibrado and evaluador.tiempo_inicio_gesto is not None:
                tiempo_restante = max(0.0, TIEMPO_MAXIMO_GESTO - (time.time() - evaluador.tiempo_inicio_gesto))
                if tiempo_restante <= 0.0:
                    reproducir_audio(AUDIO_ERROR)
                    cv2.putText(frame_hud, "TIEMPO AGOTADO - BLOQUEANDO", (frame.shape[1] // 6, frame.shape[0] // 2),
                                FUENTE, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Autenticador Facial", frame_hud)
                    cv2.waitKey(800)
                    return False

            cv2.putText(frame_hud, "SITUATE FRENTE A LA CAMARA", (20, 60),
                        FUENTE, 0.7, (0, 0, 255), 2)

        cv2.imshow("Autenticador Facial", frame_hud)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

def ejecutar_guardia():
    """Inicia el ciclo principal de vigilancia y autenticacion continua."""
    inicializar_entorno()
    clasificador = obtener_clasificador()
    lista_nombres, lista_caras = cargar_caras_confianza()

    if not os.path.exists(ARCHIVO_CONFIG):
        print("[ERROR] No se encuentra el archivo de datos/usuarios.json")
        return

    datos_sistema = cargar_usuarios()

    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("[ERROR CRÍTICO] No es posible abrir la cámara del ordenador.")
        return

    print("[SISTEMA] Servicio de autenticación activo en segundo plano.")
    print(f"[SISTEMA] Tiempo de gracia por ausencia: {TIEMPO_AUSENCIA_GUARDIA} segundos.")
    
    estado_sesion = False
    usuario_autenticado = None
    plantillas_usuario_activo = []
    tiempo_inicio_sesion = None
    tiempo_inicio_ausencia = None
    ultimo_tiempo_ausencia = 0.0
    verificando_retorno = False
    buffer_scores_retorno = []
    tiempo_confirmacion_retorno = 0.0
    mostrar_detalles = False
    stats_usuario = {}
    historial_similitud_sesion = []
    contador_frames_sesion = 0
    frames_estables = 0
    ultima_caja = None
    frames_estables_retorno = 0
    ultima_caja_retorno = None

    def resetear_estado_sesion():
        """Restablece todas las variables de sesion y guardia a su estado inicial."""
        nonlocal estado_sesion, usuario_autenticado, plantillas_usuario_activo
        nonlocal tiempo_inicio_sesion, tiempo_inicio_ausencia, ultimo_tiempo_ausencia
        nonlocal verificando_retorno, buffer_scores_retorno, tiempo_confirmacion_retorno
        nonlocal mostrar_detalles, stats_usuario, historial_similitud_sesion
        nonlocal contador_frames_sesion, frames_estables, ultima_caja
        nonlocal frames_estables_retorno, ultima_caja_retorno

        estado_sesion = False
        usuario_autenticado = None
        plantillas_usuario_activo = []
        tiempo_inicio_sesion = None
        tiempo_inicio_ausencia = None
        ultimo_tiempo_ausencia = 0.0
        verificando_retorno = False
        buffer_scores_retorno = []
        tiempo_confirmacion_retorno = 0.0
        mostrar_detalles = False
        stats_usuario = {}
        historial_similitud_sesion = []
        contador_frames_sesion = 0
        frames_estables = 0
        ultima_caja = None
        frames_estables_retorno = 0
        ultima_caja_retorno = None

    try:
        while True:
            ret, frame = camara.read()
            if not ret:
                print("[ERROR] No es posible obtener la imagen de la cámara.")
                break
            frame = cv2.flip(frame, 1)

            frame_mostrar = frame.copy()
            caras, frame_byn = detectar_caras(frame, clasificador)

            # -------------------------------------------------------------
            # ESTADO 1: SESIÓN INACTIVA (MODO GUARDIA)
            # -------------------------------------------------------------
            if not estado_sesion:
                if len(caras) == 0:
                    frames_estables = 0
                    ultima_caja = None
                    cv2.putText(frame_mostrar, "MODO GUARDIA: VIGILANDO", (20, 40), FUENTE, 0.7, (0, 255, 0), GROSOR)
                else:
                    caja_actual = caras[0]
                    x, y, ancho, alto = caja_actual

                    _, frames_estables = actualizar_estabilidad_facial(
                        caja_actual, ultima_caja, frame_byn, frames_estables
                    )
                    ultima_caja = caja_actual
                    progreso = min(100, int((frames_estables / FRAMES_ESTABILIDAD_GUARDIA) * 100))

                    if frames_estables < FRAMES_ESTABILIDAD_GUARDIA:
                        dibujar_feedback_estabilizacion(frame_mostrar, caja_actual, progreso, "ESTABILIZANDO ROSTRO")
                    else:
                        # Fase de verificacion: rostro confirmado estatico y frontal
                        cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), COLOR_RECUADRO, GROSOR)
                        roi_cara = normalizar_cara(frame_byn, caja=caja_actual)
                        usuario, score = identificar_rostro(roi_cara, lista_nombres, lista_caras)

                        if usuario is None:
                            # Recarga en caliente ante nuevo usuario registrado durante la ejecucion
                            lista_nombres, lista_caras = cargar_caras_confianza()
                            datos_sistema = cargar_usuarios()
                            usuario, score = identificar_rostro(roi_cara, lista_nombres, lista_caras)
                        elif usuario not in datos_sistema.get("usuarios", {}):
                            datos_sistema = cargar_usuarios()

                        if usuario is not None and usuario in datos_sistema.get("usuarios", {}):
                            cv2.putText(frame_mostrar, f"Usuario: {usuario.upper()}", (x, y - 10), FUENTE, 0.6, COLOR_TEXTO, 2)
                            cv2.imshow("Autenticador Facial", frame_mostrar)
                            cv2.waitKey(1)

                            # Validacion de contrasena de texto y posterior reto facial
                            if verificar_password_texto(usuario, datos_sistema):
                                if validar_contrasena_facial(camara, clasificador):
                                    mostrar_permisos_usuario(usuario, datos_sistema)
                                    similitud_login = max(0.0, min(100.0, (1.0 - score) * 100.0))
                                    registrar_inicio_sesion(usuario, similitud_inicial=similitud_login)
                                    stats_usuario = obtener_estadisticas_usuario(usuario)
                                    estado_sesion = True
                                    usuario_autenticado = usuario
                                    plantillas_usuario_activo = obtener_plantillas_usuario(usuario, lista_nombres, lista_caras)
                                    tiempo_inicio_sesion = time.time()
                                    tiempo_inicio_ausencia = None
                                    ultimo_tiempo_ausencia = 0.0
                                    verificando_retorno = False
                                    buffer_scores_retorno = []
                                    tiempo_confirmacion_retorno = 0.0
                                    mostrar_detalles = False
                                    historial_similitud_sesion = [similitud_login]
                                    contador_frames_sesion = 0
                                    frames_estables = 0
                                    ultima_caja = None
                                    frames_estables_retorno = 0
                                    ultima_caja_retorno = None
                                else:
                                    print("\n[ALERTA INTRUSO] Prueba de gestos no superada o tiempo agotado.")
                                    guardar_evidencia_intruso(camara, frame)
                                    time.sleep(1.0)
                                    suspender_equipo()
                                    time.sleep(5)
                                    frames_estables = 0
                                    ultima_caja = None
                            else:
                                frames_estables = 0
                                ultima_caja = None
                        else:
                            cv2.imshow("Autenticador Facial", frame_mostrar)
                            cv2.waitKey(1)
                            print("\n[ALERTA INTRUSO] Persona no autorizada detectada frente al equipo.")
                            guardar_evidencia_intruso(camara, frame)
                            suspender_equipo()
                            time.sleep(5)
                            frames_estables = 0
                            ultima_caja = None

            # -------------------------------------------------------------
            # ESTADO 2: SESIÓN ACTIVA (EL USUARIO YA ACCEDIÓ)
            # -------------------------------------------------------------
            else:
                duracion_actual = time.time() - tiempo_inicio_sesion if tiempo_inicio_sesion else 0.0

                if len(caras) == 0:
                    if tiempo_inicio_ausencia is None:
                        tiempo_inicio_ausencia = time.time()

                    tiempo_ausente = time.time() - tiempo_inicio_ausencia
                    ultimo_tiempo_ausencia = tiempo_ausente

                    # Reiniciar acumuladores de retorno si el rostro se retira
                    frames_estables_retorno = 0
                    ultima_caja_retorno = None
                    buffer_scores_retorno = []

                    tiempo_restante_guardia = max(0.0, TIEMPO_AUSENCIA_GUARDIA - tiempo_ausente)
                    porcentaje_guardia = (tiempo_restante_guardia / TIEMPO_AUSENCIA_GUARDIA) * 100.0

                    # Banner de cuenta regresiva de la guardia de ausencia (10s)
                    ancho_f, alto_f = frame_mostrar.shape[1], frame_mostrar.shape[0]
                    y_banner = alto_f - 75
                    cv2.rectangle(frame_mostrar, (15, y_banner), (ancho_f - 15, alto_f - 15), (20, 20, 60), -1)
                    cv2.rectangle(frame_mostrar, (15, y_banner), (ancho_f - 15, alto_f - 15), (0, 165, 255), 2)
                    cv2.putText(frame_mostrar, f"GUARDIA DE AUSENCIA: TIEMPO RESTANTE {tiempo_restante_guardia:.1f}s",
                                (25, y_banner + 26), FUENTE, 0.60, (0, 165, 255), 2)
                    cv2.putText(frame_mostrar, "Vuelve frente a la camara antes de 10s para reanudar sesion",
                                (25, y_banner + 46), FUENTE, 0.42, (200, 200, 200), 1)
                    dibujar_barra_progreso(frame_mostrar, 25, y_banner + 52, ancho_f - 50, 4, porcentaje_guardia, (0, 165, 255))

                    if tiempo_ausente >= TIEMPO_AUSENCIA_GUARDIA:
                        print(f"\n[BLOQUEO] Usuario '{usuario_autenticado}' ausente durante {int(tiempo_ausente)} segundos. Reactivando guardia...")
                        registrar_fin_sesion(usuario_autenticado, duracion_actual)
                        resetear_estado_sesion()
                else:
                    caja_actual = caras[0]
                    x, y, ancho, alto = caja_actual

                    # Si hubo ausencia previa que supero el tiempo minimo de disparo (1.0s)
                    if tiempo_inicio_ausencia is not None and ultimo_tiempo_ausencia >= TIEMPO_DISPARO_REENGANCHE:
                        verificando_retorno = True
                        frames_estables_retorno = 0
                        ultima_caja_retorno = None
                        buffer_scores_retorno = []

                    tiempo_inicio_ausencia = None
                    tiempo_ausente = 0.0
                    ultimo_tiempo_ausencia = 0.0

                    # ---------------------------------------------------------
                    # CASO A: FASE DE VERIFICACION DE RETORNO (ANTI-USURPACION)
                    # ---------------------------------------------------------
                    if verificando_retorno:
                        # Subfase 1: Calibracion y estabilizacion facial
                        if frames_estables_retorno < FRAMES_ESTABILIDAD_GUARDIA:
                            _, frames_estables_retorno = actualizar_estabilidad_facial(
                                caja_actual, ultima_caja_retorno, frame_byn, frames_estables_retorno
                            )
                            ultima_caja_retorno = caja_actual
                            progreso_est = min(100, int((frames_estables_retorno / FRAMES_ESTABILIDAD_GUARDIA) * 100))
                            dibujar_feedback_estabilizacion(
                                frame_mostrar, caja_actual, progreso_est, "CALIBRANDO RETORNO",
                                pos_banner=(20, frame_mostrar.shape[0] - 25)
                            )
                        else:
                            # Subfase 2: Verificacion de identidad con rostro estabilizado
                            roi_cara = normalizar_cara(frame_byn, caja=caja_actual)
                            coincide_frame, score_frame, _ = verificar_usuario_especifico(
                                roi_cara, plantillas_usuario_activo, UMBRAL_REENGANCHE_SESION
                            )
                            buffer_scores_retorno.append(score_frame)

                            progreso_ret = min(100, int((len(buffer_scores_retorno) / FRAMES_CONFIRMACION_RETORNO) * 100))
                            color_verif = (0, 255, 255)
                            cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), color_verif, GROSOR)
                            cv2.putText(frame_mostrar, f"VERIFICANDO IDENTIDAD: {progreso_ret}%",
                                        (20, frame_mostrar.shape[0] - 25), FUENTE, 0.65, color_verif, GROSOR)
                            cv2.putText(frame_mostrar, f"VALIDANDO A {usuario_autenticado.upper()}",
                                        (x, max(20, y - 10)), FUENTE, 0.50, color_verif, 2)
                            dibujar_barra_progreso(frame_mostrar, x, y + alto + 10, ancho, 6, progreso_ret, color_verif)

                            if len(buffer_scores_retorno) >= FRAMES_CONFIRMACION_RETORNO:
                                score_medio = float(np.mean(buffer_scores_retorno))
                                if score_medio < UMBRAL_REENGANCHE_SESION:
                                    verificando_retorno = False
                                    frames_estables_retorno = 0
                                    ultima_caja_retorno = None
                                    buffer_scores_retorno = []
                                    tiempo_confirmacion_retorno = time.time()
                                else:
                                    cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), (0, 0, 255), GROSOR)
                                    cv2.putText(frame_mostrar, "ALERTA: USURPACION DE SESION", (20, 80), FUENTE, 0.70, (0, 0, 255), 2)
                                    cv2.imshow("Autenticador Facial", frame_mostrar)
                                    cv2.waitKey(1)
                                    print(f"\n[ALERTA INTRUSO] Intento de usurpación en la sesión de '{usuario_autenticado}'. Score: {score_medio:.4f}")
                                    guardar_evidencia_intruso(camara, frame)
                                    registrar_fin_sesion(usuario_autenticado, duracion_actual)
                                    suspender_equipo()
                                    time.sleep(5)
                                    resetear_estado_sesion()

                    # ---------------------------------------------------------
                    # CASO B: OPERACION NORMAL DE SESION ACTIVA
                    # ---------------------------------------------------------
                    else:
                        cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), (0, 255, 0), GROSOR)

                        if (time.time() - tiempo_confirmacion_retorno) < 1.8:
                            cv2.putText(frame_mostrar, f"RETORNO CONFIRMADO: {usuario_autenticado.upper()}", (x, max(20, y - 10)), FUENTE, 0.50, (0, 255, 0), 2)

                        # Monitoreo continuo de similitud facial cada 15 frames
                        contador_frames_sesion += 1
                        if contador_frames_sesion % 15 == 0:
                            roi_cara = normalizar_cara(frame_byn, caja=caras[0])
                            _, score_actual, sim_act = verificar_usuario_especifico(
                                roi_cara, plantillas_usuario_activo, UMBRAL_REENGANCHE_SESION
                            )
                            historial_similitud_sesion.append(sim_act)

                if estado_sesion:
                    info_usr = datos_sistema.get("usuarios", {}).get(usuario_autenticado, {})
                    rol_usr = info_usr.get("rol", "Invitado")
                    sim_sesion_media = float(np.mean(historial_similitud_sesion)) if historial_similitud_sesion else None
                    dibujar_interfaz_usuario(
                        frame_mostrar,
                        usuario=usuario_autenticado,
                        rol=rol_usr,
                        stats=stats_usuario,
                        duracion_actual=duracion_actual,
                        mostrar_detalles=mostrar_detalles,
                        tiempo_ausente=tiempo_ausente if len(caras) == 0 else 0.0,
                        similitud_sesion=sim_sesion_media
                    )

            cv2.imshow("Autenticador Facial", frame_mostrar)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'):
                if estado_sesion and usuario_autenticado and tiempo_inicio_sesion:
                    duracion_actual = time.time() - tiempo_inicio_sesion
                    registrar_fin_sesion(usuario_autenticado, duracion_actual)
                print("Cerrando autenticador...")
                break
            elif tecla == ord('m') or tecla == ord('M'):
                if estado_sesion:
                    mostrar_detalles = not mostrar_detalles

    except KeyboardInterrupt:
        if estado_sesion and usuario_autenticado and tiempo_inicio_sesion:
            duracion_actual = time.time() - tiempo_inicio_sesion
            registrar_fin_sesion(usuario_autenticado, duracion_actual)
        print("\n[INFO] Autenticador detenido por el usuario.")
    finally:
        camara.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ejecutar_guardia()
