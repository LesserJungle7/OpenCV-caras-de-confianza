import os
import sys
import time
import numpy as np
import cv2

# Asegura que la raiz del proyecto y la carpeta src esten en sys.path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_BASE_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    obtener_plantillas_usuario,
    reiniciar_substractor_knn
)
from sistema import suspender_equipo, guardar_evidencia_intruso, reproducir_audio
from usuarios import (
    cargar_usuarios,
    inicializar_entorno,
    obtener_usuario,
    obtener_estadisticas_usuario,
    registrar_inicio_sesion,
    registrar_fin_sesion,
    cambiar_password_usuario,
    listar_usuarios,
    eliminar_usuario
)
from gestos import EvaluadorGestos
from depurador import (
    es_debug_activo,
    toggle_debug,
    cerrar_ventanas_debug,
    cambiar_fase,
    cerrar_debug_fase,
    pausar_inspeccion_debug
)
from interfaz import (
    formatear_duracion,
    dibujar_barra_progreso,
    dibujar_barra_menu_principal,
    dibujar_hud_sesion,
    capturar_texto_pantalla,
    seleccionar_opcion_pantalla,
    seleccionar_usuario_pantalla,
    mostrar_mensaje_pantalla,
    mostrar_estadisticas_usuario_modal
)

def dibujar_interfaz_usuario(frame, usuario, rol, stats, duracion_actual, mostrar_detalles, tiempo_ausente=0.0, similitud_sesion=None, mostrar_opciones=False):
    """Renderiza el HUD de usuario activo con accesos a detalles (m) y opciones (o)."""
    dibujar_hud_sesion(frame, usuario, rol, stats, duracion_actual, mostrar_detalles, mostrar_opciones, tiempo_ausente, similitud_sesion)

def actualizar_estabilidad_facial(caja_actual, ultima_caja, frame_byn, frames_estables):
    """
    Determina si el rostro se mantiene estatico y actualiza el contador acumulador
    aplicando penalizacion de histeresis (-2) en caso de movimiento.
    Retorna (es_estable, nuevo_contador_frames).
    """
    es_estable = verificar_rostro_estatico(caja_actual, ultima_caja, frame_byn, frames_estables)
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

def verificar_password_interfaz(camara, usuario, datos_sistema=None):
    """Solicita y valida la contraseña del usuario a traves de la interfaz visual en pantalla."""
    password_ingresada = capturar_texto_pantalla(
        camara,
        titulo=f"AUTENTICACION: {usuario.upper()}",
        prompt=f"Introduce tu contraseña, {usuario}:",
        es_password=True
    )

    if password_ingresada is None:
        return None

    info_usuario = datos_sistema.get("usuarios", {}).get(usuario) if datos_sistema else obtener_usuario(usuario)

    if info_usuario and password_ingresada == info_usuario.get("password"):
        mostrar_mensaje_pantalla(
            camara,
            titulo="CONTRASEÑA CORRECTA",
            lineas_mensaje=[f"Identidad de {usuario.upper()} confirmada.", "Iniciando reto de contraseña facial..."],
            color=(0, 255, 0),
            duracion_ms=1200
        )
        return True
    else:
        mostrar_mensaje_pantalla(
            camara,
            titulo="CONTRASEÑA INCORRECTA",
            lineas_mensaje=["La contraseña introducida no es correcta."],
            color=(0, 0, 255),
            duracion_ms=1800
        )
        return False

def verificar_password_texto(usuario, datos_sistema=None):
    """Compatibilidad con versiones de consola."""
    info_usuario = datos_sistema.get("usuarios", {}).get(usuario) if datos_sistema else obtener_usuario(usuario)
    return info_usuario is not None

def mostrar_permisos_usuario(usuario, datos_sistema=None):
    """Muestra por terminal el mensaje de bienvenida y el rol asignado."""
    info_usuario = datos_sistema.get("usuarios", {}).get(usuario) if datos_sistema else obtener_usuario(usuario)

    rol = info_usuario.get("rol", "Invitado") if info_usuario else "Invitado"
    print(f"\n[ACCESO CONCEDIDO] Bienvenido/a {usuario}.")
    print(f"[PERMISOS ACTIVADOS] Nivel de acceso: {rol}")
    if rol == "Total":
        print(" -> Tienes acceso a configuracion, terminal y archivos restringidos.")
    elif rol == "Operador":
        print(" -> Tienes acceso a ejecucion de aplicaciones estandar.")
    else:
        print(" -> Modo invitado: Acceso restringido solo lectura.")

def validar_contrasena_facial(camara, clasificador):
    """
    Ejecuta el reto de 3 gestos aleatorios en la ventana de la camara.
    Retorna True si el usuario supera la prueba de vida, o False en caso de timeout o cancelacion.
    """
    cambiar_fase("reto_gestos")
    try:
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
                    if not es_debug_activo():
                        reproducir_audio(AUDIO_ERROR)
                        cv2.putText(frame_hud, "TIEMPO AGOTADO - BLOQUEANDO", (frame.shape[1] // 6, frame.shape[0] // 2),
                                    FUENTE, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Autenticador Facial", frame_hud)
                        cv2.waitKey(800)
                        return False
            else:
                frame_hud = frame.copy()
                if evaluador.calibrado and evaluador.tiempo_inicio_gesto is not None and not es_debug_activo():
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

            # Banner inferior para salir / cancelar reto y conmutar debug
            h_hud, w_hud = frame_hud.shape[:2]
            cv2.rectangle(frame_hud, (0, h_hud - 34), (w_hud, h_hud), (15, 15, 22), -1)
            cv2.putText(frame_hud, "[Q] Cancelar reto  |  [D] Modo Debug", (20, h_hud - 11), FUENTE, 0.44, (0, 255, 255), 1)

            # Badge de modo debug
            if es_debug_activo():
                cv2.rectangle(frame_hud, (w_hud - 145, 12), (w_hud - 15, 38), (0, 60, 140), -1)
                cv2.rectangle(frame_hud, (w_hud - 145, 12), (w_hud - 15, 38), (0, 255, 255), 1)
                cv2.putText(frame_hud, "[DEBUG ON]", (w_hud - 135, 30), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)

            cv2.imshow("Autenticador Facial", frame_hud)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord('d'), ord('D')):
                toggle_debug()
            elif tecla == ord('q') or tecla == ord('Q') or tecla == 27:
                return False
    finally:
        cerrar_debug_fase("reto_gestos")

def ejecutar_guardia(camara=None, clasificador=None):
    """Inicia el ciclo principal de vigilancia y autenticacion continua."""
    inicializar_entorno()
    camara_propia = False
    if clasificador is None:
        clasificador = obtener_clasificador()
    lista_nombres, lista_caras = cargar_caras_confianza()

    if not os.path.exists(ARCHIVO_CONFIG):
        print("[ERROR] No se encuentra el archivo de datos/usuarios.json")
        return

    datos_sistema = cargar_usuarios()

    if camara is None:
        camara = cv2.VideoCapture(0)
        camara_propia = True

    if not camara.isOpened():
        print("[ERROR CRITICO] No es posible abrir la camara del ordenador.")
        return

    print("[SISTEMA] Servicio de autenticacion activo.")
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
    mostrar_opciones = False
    stats_usuario = {}
    historial_similitud_sesion = []
    contador_frames_sesion = 0
    frames_estables = 0
    ultima_caja = None
    frames_estables_retorno = 0
    ultima_caja_retorno = None

    def limpiar_sesion_invitado(usuario):
        """Si el usuario es Invitado, elimina su perfil de usuarios.json y borra sus fotografias."""
        nonlocal datos_sistema, lista_nombres, lista_caras
        if not usuario:
            return
        info = datos_sistema.get("usuarios", {}).get(usuario, {})
        if info.get("rol") == "Invitado" or usuario == "invitado":
            eliminar_usuario(usuario)
            datos_sistema = cargar_usuarios()
            lista_nombres, lista_caras = cargar_caras_confianza()
            print(f"[INVITADO] Sesion concluida. Perfil y fotografias de '{usuario}' eliminados del sistema.")

    def resetear_estado_sesion():
        """Restablece todas las variables de sesion y guardia a su estado inicial y olvida imagenes de RAM."""
        nonlocal estado_sesion, usuario_autenticado, plantillas_usuario_activo
        nonlocal tiempo_inicio_sesion, tiempo_inicio_ausencia, ultimo_tiempo_ausencia
        nonlocal verificando_retorno, buffer_scores_retorno, tiempo_confirmacion_retorno
        nonlocal mostrar_detalles, mostrar_opciones, stats_usuario, historial_similitud_sesion
        nonlocal contador_frames_sesion, frames_estables, ultima_caja
        nonlocal frames_estables_retorno, ultima_caja_retorno

        if usuario_autenticado:
            limpiar_sesion_invitado(usuario_autenticado)

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
        mostrar_opciones = False
        stats_usuario = {}
        historial_similitud_sesion = []
        contador_frames_sesion = 0
        frames_estables = 0
        ultima_caja = None
        frames_estables_retorno = 0
        ultima_caja_retorno = None
        cerrar_ventanas_debug()
        cambiar_fase("guardia_login")

    cambiar_fase("guardia_login")
    reiniciar_substractor_knn()
    try:
        while True:
            ret, frame = camara.read()
            if not ret:
                print("[ERROR] No es posible obtener la imagen de la camara.")
                break
            frame = cv2.flip(frame, 1)

            frame_mostrar = frame.copy()
            caras, frame_byn = detectar_caras(frame, clasificador)
            h_f, w_f = frame_mostrar.shape[:2]

            # -------------------------------------------------------------
            # ESTADO 1: SESION INACTIVA (MODO GUARDIA)
            # -------------------------------------------------------------
            if not estado_sesion:
                # Barra inferior permanente en modo guardia
                cv2.rectangle(frame_mostrar, (0, h_f - 34), (w_f, h_f), (15, 15, 22), -1)
                cv2.putText(frame_mostrar, "[Q] Menu principal  |  [I] Invitado  |  [D] Modo Debug",
                            (20, h_f - 11), FUENTE, 0.44, (0, 255, 255), 1)

                # Badge de modo debug
                if es_debug_activo():
                    cv2.rectangle(frame_mostrar, (w_f - 145, 12), (w_f - 15, 38), (0, 60, 140), -1)
                    cv2.rectangle(frame_mostrar, (w_f - 145, 12), (w_f - 15, 38), (0, 255, 255), 1)
                    cv2.putText(frame_mostrar, "[DEBUG ON]", (w_f - 135, 30), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)

                if len(caras) == 0:
                    frames_estables = 0
                    ultima_caja = None
                    reiniciar_substractor_knn()
                    cv2.putText(frame_mostrar, "MODO GUARDIA: VIGILANDO", (20, 36), FUENTE, 0.7, (0, 255, 0), GROSOR)
                    cv2.putText(frame_mostrar, "[Q] Volver al menu  |  [I] Acceso Invitado", (20, 68), FUENTE, 0.46, (200, 200, 200), 1)
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
                        cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), COLOR_RECUADRO, GROSOR)
                        roi_cara = normalizar_cara(frame_byn, caja=caja_actual)
                        usuario, score = identificar_rostro(roi_cara, lista_nombres, lista_caras)

                        if usuario is None:
                            lista_nombres, lista_caras = cargar_caras_confianza()
                            datos_sistema = cargar_usuarios()
                            usuario, score = identificar_rostro(roi_cara, lista_nombres, lista_caras)
                        elif usuario not in datos_sistema.get("usuarios", {}):
                            datos_sistema = cargar_usuarios()

                        if usuario is not None and usuario in datos_sistema.get("usuarios", {}):
                            info_usr = datos_sistema["usuarios"][usuario]
                            rol_usr = info_usr.get("rol", "Operador")
                            cv2.putText(frame_mostrar, f"Usuario: {usuario.upper()} ({rol_usr})", (x, y - 10), FUENTE, 0.6, COLOR_TEXTO, 2)
                            cv2.imshow("Autenticador Facial", frame_mostrar)
                            cv2.waitKey(1)

                            similitud_login = max(0.0, min(100.0, (1.0 - score) * 100.0))
                            if es_debug_activo():
                                msg_pausa = f"[DEBUG] Rostro identificado: {usuario.upper()} ({similitud_login:.1f}%). Pulsa [S] para continuar"
                                if not pausar_inspeccion_debug(frame_mostrar, msg_pausa, tecla_continuar='s'):
                                    frames_estables = 0
                                    ultima_caja = None
                                    continue

                            # Verificacion diferenciada segun rol
                            cerrar_debug_fase("guardia_login")
                            if rol_usr == "Invitado":
                                print(f"\n[INVITADO DETECTADO] Acceso directo para '{usuario}'. Iniciando reto facial...")
                                if validar_contrasena_facial(camara, clasificador):
                                    mostrar_permisos_usuario(usuario, datos_sistema)
                                    similitud_login = max(0.0, min(100.0, (1.0 - score) * 100.0))
                                    registrar_inicio_sesion(usuario, similitud_inicial=similitud_login)
                                    stats_usuario = obtener_estadisticas_usuario(usuario)
                                    estado_sesion = True
                                    usuario_autenticado = usuario
                                    plantillas_usuario_activo = [roi_cara]
                                    tiempo_inicio_sesion = time.time()
                                    tiempo_inicio_ausencia = None
                                    ultimo_tiempo_ausencia = 0.0
                                    verificando_retorno = False
                                    buffer_scores_retorno = []
                                    tiempo_confirmacion_retorno = 0.0
                                    mostrar_detalles = False
                                    mostrar_opciones = False
                                    historial_similitud_sesion = [similitud_login]
                                    contador_frames_sesion = 0
                                    frames_estables = 0
                                    ultima_caja = None
                                    cambiar_fase("sesion_activa")
                                else:
                                    print("\n[ALERTA] Reto facial cancelado o tiempo agotado.")
                                    frames_estables = 0
                                    ultima_caja = None
                                    cambiar_fase("guardia_login")
                            else:
                                # Autenticacion mediante la interfaz visual en pantalla
                                resultado_pass = verificar_password_interfaz(camara, usuario, datos_sistema)
                                if resultado_pass is True:
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
                                        mostrar_opciones = False
                                        historial_similitud_sesion = [similitud_login]
                                        contador_frames_sesion = 0
                                        frames_estables = 0
                                        ultima_caja = None
                                        frames_estables_retorno = 0
                                        ultima_caja_retorno = None
                                        cambiar_fase("sesion_activa")
                                    else:
                                        print("\n[ALERTA INTRUSO] Prueba de gestos no superada o tiempo agotado.")
                                        guardar_evidencia_intruso(camara, frame)
                                        time.sleep(1.0)
                                        suspender_equipo()
                                        time.sleep(5)
                                        frames_estables = 0
                                        ultima_caja = None
                                        cambiar_fase("guardia_login")
                                else:
                                    frames_estables = 0
                                    ultima_caja = None
                                    cambiar_fase("guardia_login")
                        else:
                            sim_intr = max(0.0, min(100.0, (1.0 - score) * 100.0))
                            if es_debug_activo():
                                msg_intruso = f"[DEBUG] Intruso no autorizado (Similitud: {sim_intr:.1f}%). Pulsa [S] para bloquear"
                                pausar_inspeccion_debug(frame_mostrar, msg_intruso, tecla_continuar='s')
                            cerrar_debug_fase("guardia_login")
                            cv2.imshow("Autenticador Facial", frame_mostrar)
                            cv2.waitKey(1)
                            print("\n[ALERTA INTRUSO] Persona no autorizada detectada frente al equipo.")
                            guardar_evidencia_intruso(camara, frame)
                            suspender_equipo()
                            time.sleep(5)
                            frames_estables = 0
                            ultima_caja = None
                            cambiar_fase("guardia_login")

            # -------------------------------------------------------------
            # ESTADO 2: SESION ACTIVA (EL USUARIO YA ACCEDIO)
            # -------------------------------------------------------------
            else:
                duracion_actual = time.time() - tiempo_inicio_sesion if tiempo_inicio_sesion else 0.0

                if len(caras) == 0:
                    if tiempo_inicio_ausencia is None:
                        tiempo_inicio_ausencia = time.time()

                    tiempo_ausente = time.time() - tiempo_inicio_ausencia
                    ultimo_tiempo_ausencia = tiempo_ausente

                    frames_estables_retorno = 0
                    ultima_caja_retorno = None
                    buffer_scores_retorno = []

                    tiempo_restante_guardia = max(0.0, TIEMPO_AUSENCIA_GUARDIA - tiempo_ausente)
                    porcentaje_guardia = (tiempo_restante_guardia / TIEMPO_AUSENCIA_GUARDIA) * 100.0

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
                        if usuario_autenticado in datos_sistema.get("usuarios", {}):
                            registrar_fin_sesion(usuario_autenticado, duracion_actual)
                        resetear_estado_sesion()
                else:
                    caja_actual = caras[0]
                    x, y, ancho, alto = caja_actual

                    if tiempo_inicio_ausencia is not None and ultimo_tiempo_ausencia >= TIEMPO_DISPARO_REENGANCHE:
                        verificando_retorno = True
                        cambiar_fase("retorno_ausencia")
                        frames_estables_retorno = 0
                        ultima_caja_retorno = None
                        buffer_scores_retorno = []
                        reiniciar_substractor_knn()

                    tiempo_inicio_ausencia = None
                    tiempo_ausente = 0.0
                    ultimo_tiempo_ausencia = 0.0

                    # ---------------------------------------------------------
                    # CASO A: FASE DE VERIFICACION DE RETORNO (ANTI-USURPACION)
                    # ---------------------------------------------------------
                    if verificando_retorno:
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
                                    sim_ret = max(0.0, min(100.0, (1.0 - score_medio) * 100.0))
                                    if es_debug_activo():
                                        msg_ret = f"[DEBUG] Retorno verificado: {usuario_autenticado.upper()} ({sim_ret:.1f}%). Pulsa [S] para reanudar sesion"
                                        pausar_inspeccion_debug(frame_mostrar, msg_ret, tecla_continuar='s')

                                    verificando_retorno = False
                                    cerrar_debug_fase("retorno_ausencia")
                                    cambiar_fase("sesion_activa")
                                    frames_estables_retorno = 0
                                    ultima_caja_retorno = None
                                    buffer_scores_retorno = []
                                    tiempo_confirmacion_retorno = time.time()
                                else:
                                    sim_usurp = max(0.0, min(100.0, (1.0 - score_medio) * 100.0))
                                    if es_debug_activo():
                                        msg_usurp = f"[DEBUG] USURPACION DETECTADA ({sim_usurp:.1f}% vs umbral). Pulsa [S] para continuar bloqueo"
                                        pausar_inspeccion_debug(frame_mostrar, msg_usurp, tecla_continuar='s')

                                    cerrar_debug_fase("retorno_ausencia")
                                    cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), (0, 0, 255), GROSOR)
                                    cv2.putText(frame_mostrar, "ALERTA: USURPACION DE SESION", (20, 80), FUENTE, 0.70, (0, 0, 255), 2)
                                    cv2.imshow("Autenticador Facial", frame_mostrar)
                                    cv2.waitKey(1)
                                    print(f"\n[ALERTA INTRUSO] Intento de usurpacion en la sesion de '{usuario_autenticado}'. Score: {score_medio:.4f}")
                                    guardar_evidencia_intruso(camara, frame)
                                    if usuario_autenticado in datos_sistema.get("usuarios", {}):
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

                        contador_frames_sesion += 1
                        if contador_frames_sesion % 15 == 0 and len(plantillas_usuario_activo) > 0:
                            roi_cara = normalizar_cara(frame_byn, caja=caras[0])
                            _, score_actual, sim_act = verificar_usuario_especifico(
                                roi_cara, plantillas_usuario_activo, UMBRAL_REENGANCHE_SESION
                            )
                            historial_similitud_sesion.append(sim_act)

                if estado_sesion:
                    info_usr = datos_sistema.get("usuarios", {}).get(usuario_autenticado, {})
                    rol_usr = info_usr.get("rol", "Invitado") if usuario_autenticado in datos_sistema.get("usuarios", {}) else "Invitado"
                    sim_sesion_media = float(np.mean(historial_similitud_sesion)) if historial_similitud_sesion else None
                    dibujar_hud_sesion(
                        frame_mostrar,
                        usuario=usuario_autenticado,
                        rol=rol_usr,
                        stats=stats_usuario,
                        duracion_actual=duracion_actual,
                        mostrar_detalles=mostrar_detalles,
                        mostrar_opciones=mostrar_opciones,
                        tiempo_ausente=tiempo_ausente if len(caras) == 0 else 0.0,
                        similitud_sesion=sim_sesion_media
                    )

            cv2.imshow("Autenticador Facial", frame_mostrar)
            tecla = cv2.waitKey(1) & 0xFF

            # Tecla Q o ESC para volver a la pantalla anterior
            if tecla == ord('q') or tecla == ord('Q') or tecla == 27:
                if estado_sesion and mostrar_opciones:
                    # Si el panel de opciones estaba abierto, Q/ESC lo oculta y vuelve a la sesion
                    mostrar_opciones = False
                elif estado_sesion:
                    # Si la sesion esta activa, Q/ESC cierra la sesion y vuelve al menu principal
                    if usuario_autenticado and tiempo_inicio_sesion:
                        duracion_actual = time.time() - tiempo_inicio_sesion
                        if usuario_autenticado in datos_sistema.get("usuarios", {}):
                            registrar_fin_sesion(usuario_autenticado, duracion_actual)
                    resetear_estado_sesion()
                    print("Cerrando sesion y regresando al menu principal...")
                    break
                else:
                    # En modo guardia, Q/ESC regresa al menu principal
                    print("Regresando al menu principal...")
                    break

            elif not estado_sesion and (tecla == ord('i') or tecla == ord('I')):
                # Inicio rapido de sesion de Invitado
                if len(caras) > 0:
                    roi_cara = normalizar_cara(frame_byn, caja=caras[0])
                    print("\n[INVITADO] Iniciando sesion como invitado temporal...")
                    cerrar_debug_fase("guardia_login")
                    if validar_contrasena_facial(camara, clasificador):
                        estado_sesion = True
                        cambiar_fase("sesion_activa")
                        usuario_autenticado = "invitado"
                        plantillas_usuario_activo = [roi_cara]
                        tiempo_inicio_sesion = time.time()
                        tiempo_inicio_ausencia = None
                        ultimo_tiempo_ausencia = 0.0
                        verificando_retorno = False
                        buffer_scores_retorno = []
                        tiempo_confirmacion_retorno = 0.0
                        mostrar_detalles = False
                        mostrar_opciones = False
                        stats_usuario = {
                            "inicios_sesion": 1,
                            "ultimo_acceso": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "tiempo_acumulado_segundos": 0.0,
                            "similitud_media": 100.0,
                            "intrusos_ultima_sesion": 0
                        }
                        historial_similitud_sesion = [100.0]
                        contador_frames_sesion = 0
                        frames_estables = 0
                        ultima_caja = None
                    else:
                        cambiar_fase("guardia_login")

            elif tecla == ord('m') or tecla == ord('M'):
                if estado_sesion:
                    mostrar_detalles = not mostrar_detalles

            elif tecla == ord('o') or tecla == ord('O'):
                if estado_sesion:
                    mostrar_opciones = not mostrar_opciones

            elif tecla in (ord('d'), ord('D')):
                toggle_debug()

            elif estado_sesion and mostrar_opciones:
                info_usr = datos_sistema.get("usuarios", {}).get(usuario_autenticado, {})
                rol_actual = info_usr.get("rol", "Invitado") if usuario_autenticado in datos_sistema.get("usuarios", {}) else "Invitado"

                if rol_actual == "Total":
                    if tecla == ord('1'):
                        nueva_pass = capturar_texto_pantalla(
                            camara,
                            titulo="CAMBIAR CONTRASEÑA",
                            prompt=f"Nueva contraseña para {usuario_autenticado}:",
                            es_password=True
                        )
                        if nueva_pass:
                            cambiar_password_usuario(usuario_autenticado, nueva_pass)
                            datos_sistema = cargar_usuarios()
                            mostrar_mensaje_pantalla(
                                camara,
                                titulo="CONTRASEÑA ACTUALIZADA",
                                lineas_mensaje=[f"Tu contraseña ha sido actualizada con éxito."],
                                color=(0, 255, 0),
                                duracion_ms=1800
                            )
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                    elif tecla == ord('2'):
                        seleccion = seleccionar_usuario_pantalla(
                            camara,
                            listar_usuarios(),
                            titulo="CAMBIAR CLAVE DE OTRO USUARIO"
                        )
                        if seleccion:
                            otro_nombre, otro_rol = seleccion
                            nueva_pass = capturar_texto_pantalla(
                                camara,
                                titulo=f"CAMBIAR CLAVE: {otro_nombre.upper()}",
                                prompt=f"Nueva clave para {otro_nombre}:",
                                es_password=True
                            )
                            if nueva_pass:
                                cambiar_password_usuario(otro_nombre, nueva_pass)
                                datos_sistema = cargar_usuarios()
                                mostrar_mensaje_pantalla(
                                    camara,
                                    titulo="CLAVE ACTUALIZADA",
                                    lineas_mensaje=[f"Contraseña de {otro_nombre.upper()} modificada."],
                                    color=(0, 255, 0),
                                    duracion_ms=1800
                                )
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                    elif tecla == ord('3'):
                        seleccion = seleccionar_usuario_pantalla(
                            camara,
                            listar_usuarios(),
                            titulo="CONSULTAR ESTADISTICAS"
                        )
                        if seleccion:
                            otro_nombre, otro_rol = seleccion
                            st = obtener_estadisticas_usuario(otro_nombre)
                            mostrar_estadisticas_usuario_modal(camara, otro_nombre, otro_rol, st)
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                    elif tecla == ord('4'):
                        conf = seleccionar_opcion_pantalla(
                            camara,
                            titulo="BORRAR MI USUARIO",
                            opciones=[("1", "Si, borrar mi usuario"), ("2", "Cancelar")],
                            subtitulo="Confirmar eliminacion de cuenta"
                        )
                        if conf == "1":
                            eliminar_usuario(usuario_autenticado)
                            mostrar_mensaje_pantalla(
                                camara,
                                titulo="USUARIO ELIMINADO",
                                lineas_mensaje=[f"Tu usuario {usuario_autenticado.upper()} ha sido eliminado.", "Cerrando aplicacion..."],
                                color=(0, 0, 255),
                                duracion_ms=2200
                            )
                            resetear_estado_sesion()
                            return "SALIR"
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                    elif tecla == ord('5'):
                        otros_usuarios = [(u, r) for (u, r) in listar_usuarios() if u != usuario_autenticado]
                        if not otros_usuarios:
                            mostrar_mensaje_pantalla(
                                camara,
                                titulo="SIN OTROS USUARIOS",
                                lineas_mensaje=["No hay otros usuarios registrados."],
                                color=(0, 165, 255),
                                duracion_ms=1800
                            )
                        else:
                            seleccion = seleccionar_usuario_pantalla(
                                camara,
                                otros_usuarios,
                                titulo="BORRAR OTRO USUARIO"
                            )
                            if seleccion:
                                otro_nombre, otro_rol = seleccion
                                conf = seleccionar_opcion_pantalla(
                                    camara,
                                    titulo=f"BORRAR: {otro_nombre.upper()}",
                                    opciones=[("1", "Si, borrar usuario"), ("2", "Cancelar")],
                                    subtitulo="Confirmar eliminacion"
                                )
                                if conf == "1":
                                    eliminar_usuario(otro_nombre)
                                    datos_sistema = cargar_usuarios()
                                    lista_nombres, lista_caras = cargar_caras_confianza()
                                    mostrar_mensaje_pantalla(
                                        camara,
                                        titulo="USUARIO ELIMINADO",
                                        lineas_mensaje=[f"El usuario {otro_nombre.upper()} fue eliminado."],
                                        color=(0, 255, 0),
                                        duracion_ms=1800
                                    )
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                elif rol_actual == "Operador":
                    if tecla == ord('1'):
                        nueva_pass = capturar_texto_pantalla(
                            camara,
                            titulo="CAMBIAR CONTRASEÑA",
                            prompt=f"Nueva contraseña para {usuario_autenticado}:",
                            es_password=True
                        )
                        if nueva_pass:
                            cambiar_password_usuario(usuario_autenticado, nueva_pass)
                            datos_sistema = cargar_usuarios()
                            mostrar_mensaje_pantalla(
                                camara,
                                titulo="CONTRASEÑA ACTUALIZADA",
                                lineas_mensaje=[f"Tu contraseña ha sido actualizada con éxito."],
                                color=(0, 255, 0),
                                duracion_ms=1800
                            )
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                    elif tecla == ord('2'):
                        conf = seleccionar_opcion_pantalla(
                            camara,
                            titulo="BORRAR MI USUARIO",
                            opciones=[("1", "Si, borrar mi usuario"), ("2", "Cancelar")],
                            subtitulo="Confirmar eliminacion de cuenta"
                        )
                        if conf == "1":
                            eliminar_usuario(usuario_autenticado)
                            mostrar_mensaje_pantalla(
                                camara,
                                titulo="USUARIO ELIMINADO",
                                lineas_mensaje=[f"Tu usuario {usuario_autenticado.upper()} ha sido eliminado.", "Cerrando aplicacion..."],
                                color=(0, 0, 255),
                                duracion_ms=2200
                            )
                            resetear_estado_sesion()
                            return "SALIR"
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

                elif rol_actual == "Invitado":
                    if tecla in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                        mostrar_mensaje_pantalla(
                            camara,
                            titulo="ACCESO RESTRINGIDO",
                            lineas_mensaje=[
                                "El rol Invitado no dispone de opciones configurables.",
                                "Puedes consultar tus estadisticas pulsando [m]."
                            ],
                            color=(0, 165, 255),
                            duracion_ms=2000
                        )
                        tiempo_inicio_ausencia = None
                        mostrar_opciones = False

    except KeyboardInterrupt:
        if estado_sesion and usuario_autenticado:
            if tiempo_inicio_sesion:
                duracion_actual = time.time() - tiempo_inicio_sesion
                if usuario_autenticado in datos_sistema.get("usuarios", {}):
                    registrar_fin_sesion(usuario_autenticado, duracion_actual)
            limpiar_sesion_invitado(usuario_autenticado)
        print("\n[INFO] Autenticador detenido por el usuario.")
    finally:
        cerrar_ventanas_debug()
        cambiar_fase("menu")
        if camara_propia:
            camara.release()
            cv2.destroyAllWindows()

def ejecutar_aplicacion(camara=None, clasificador=None):
    """
    Inicia la aplicacion mostrando de inmediato el flujo de la camara
    y un menu superior para seleccionar entre Crear perfil o Autenticador.
    """
    inicializar_entorno()
    cambiar_fase("menu")
    camara_propia = False
    if clasificador is None:
        clasificador = obtener_clasificador()
    if camara is None:
        camara = cv2.VideoCapture(0)
        camara_propia = True

    if not camara.isOpened():
        print("[ERROR CRITICO] No es posible abrir la camara.")
        return

    # Importacion diferida para evitar dependencias circulares
    from creador_perfiles import crear_perfil_interactivo

    try:
        while True:
            ret, frame = camara.read()
            if not ret:
                print("[ERROR] No se puede capturar imagen de la camara.")
                break
            frame = cv2.flip(frame, 1)

            frame_mostrar = frame.copy()
            caras, _ = detectar_caras(frame, clasificador)
            for (x, y, w, h) in caras:
                cv2.rectangle(frame_mostrar, (x, y), (x + w, y + h), COLOR_RECUADRO, 1)

            dibujar_barra_menu_principal(frame_mostrar)
            cv2.imshow("Autenticador Facial", frame_mostrar)

            tecla = cv2.waitKey(20) & 0xFF
            if tecla == ord('1'):
                crear_perfil_interactivo(camara, clasificador)
                cambiar_fase("menu")
            elif tecla == ord('2'):
                res_guardia = ejecutar_guardia(camara, clasificador)
                cambiar_fase("menu")
                if res_guardia == "SALIR":
                    break
            elif tecla in (ord('d'), ord('D')):
                toggle_debug()
            elif tecla == ord('q') or tecla == ord('Q') or tecla == 27:
                print("Saliendo de la aplicacion...")
                break
    finally:
        cerrar_ventanas_debug()
        cambiar_fase("menu")
        if camara_propia:
            camara.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    ejecutar_aplicacion()
