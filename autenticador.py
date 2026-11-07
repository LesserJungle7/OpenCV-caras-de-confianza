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
    COLOR_RECUADRO,
    COLOR_TEXTO,
    FUENTE,
    GROSOR
)
from vision import (
    cargar_caras_confianza,
    identificar_rostro,
    obtener_clasificador,
    detectar_caras
)
from sistema import suspender_equipo, guardar_evidencia_intruso
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

def verificar_password_texto(usuario, datos_sistema=None):
    """Solicita y valida la contraseña escrita del usuario detectado."""
    print(f"\n[AVISO] Usuario de confianza detectado: {usuario.upper()}")
    password_ingresada = input(f"Introduce tu contraseña, {usuario}: ").strip()

    info_usuario = obtener_usuario(usuario)
    if info_usuario is None and datos_sistema:
        info_usuario = datos_sistema.get("usuarios", {}).get(usuario, {})

    if info_usuario and password_ingresada == info_usuario.get("password"):
        print("[OK] Contraseña de texto correcta. Iniciando reto de contraseña facial...")
        return True
    else:
        print("[ERROR] Contraseña incorrecta.")
        return False

def mostrar_permisos_usuario(usuario, datos_sistema=None):
    """Muestra por terminal el mensaje de bienvenida y el rol asignado."""
    info_usuario = obtener_usuario(usuario)
    if info_usuario is None and datos_sistema:
        info_usuario = datos_sistema.get("usuarios", {}).get(usuario, {})

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
                return False
        else:
            frame_hud = frame.copy()
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
    tiempo_inicio_sesion = None
    tiempo_inicio_ausencia = None
    mostrar_detalles = False
    stats_usuario = {}
    historial_similitud_sesion = []
    contador_frames_sesion = 0

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
                cv2.putText(frame_mostrar, "MODO GUARDIA: VIGILANDO", (20, 40), FUENTE, 0.7, (0, 255, 0), GROSOR)

                if len(caras) > 0:
                    x, y, ancho, alto = caras[0]
                    cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), COLOR_RECUADRO, GROSOR)
                    roi_cara = frame_byn[y:y+alto, x:x+ancho]
                    usuario, score = identificar_rostro(roi_cara, lista_nombres, lista_caras)

                    datos_sistema = cargar_usuarios()
                    if usuario is not None and usuario in datos_sistema.get("usuarios", {}):
                        cv2.putText(frame_mostrar, f"Usuario: {usuario.upper()}", (x, y - 10), FUENTE, 0.6, COLOR_TEXTO, 2)
                        cv2.imshow("Autenticador Facial", frame_mostrar)
                        cv2.waitKey(1)

                        # Validacion de contrasena de texto y posterior reto facial
                        if verificar_password_texto(usuario, datos_sistema):
                            if validar_contrasena_facial(camara, clasificador):
                                mostrar_permisos_usuario(usuario, datos_sistema)
                                # Similitud obtenida mediante Template Matching en escala de grises
                                similitud_login = max(0.0, min(100.0, (1.0 - score) * 100.0))
                                registrar_inicio_sesion(usuario, similitud_inicial=similitud_login)
                                stats_usuario = obtener_estadisticas_usuario(usuario)
                                estado_sesion = True
                                usuario_autenticado = usuario
                                tiempo_inicio_sesion = time.time()
                                tiempo_inicio_ausencia = None
                                mostrar_detalles = False
                                historial_similitud_sesion = [similitud_login]
                                contador_frames_sesion = 0
                            else:
                                print("\n[ALERTA INTRUSO] Prueba de gestos no superada o tiempo agotado.")
                                guardar_evidencia_intruso(camara, frame)
                                suspender_equipo()
                                time.sleep(5)
                    else:
                        cv2.imshow("Autenticador Facial", frame_mostrar)
                        cv2.waitKey(1)
                        print("\n[ALERTA INTRUSO] Persona no autorizada detectada frente al equipo.")
                        guardar_evidencia_intruso(camara, frame)
                        suspender_equipo()
                        time.sleep(5)

            # -------------------------------------------------------------
            # ESTADO 2: SESIÓN ACTIVA (EL USUARIO YA ACCEDIÓ)
            # -------------------------------------------------------------
            else:
                duracion_actual = time.time() - tiempo_inicio_sesion if tiempo_inicio_sesion else 0.0

                if len(caras) == 0:
                    if tiempo_inicio_ausencia is None:
                        tiempo_inicio_ausencia = time.time()

                    tiempo_ausente = time.time() - tiempo_inicio_ausencia
                    if tiempo_ausente >= TIEMPO_AUSENCIA_GUARDIA:
                        print(f"\n[BLOQUEO] Usuario '{usuario_autenticado}' ausente durante {int(tiempo_ausente)} segundos. Reactivando guardia...")
                        registrar_fin_sesion(usuario_autenticado, duracion_actual)
                        estado_sesion = False
                        usuario_autenticado = None
                        tiempo_inicio_ausencia = None
                        tiempo_inicio_sesion = None
                        mostrar_detalles = False
                        historial_similitud_sesion = []
                        contador_frames_sesion = 0
                else:
                    tiempo_inicio_ausencia = None
                    tiempo_ausente = 0.0
                    x, y, ancho, alto = caras[0]
                    cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), (0, 255, 0), GROSOR)

                    # Monitoreo continuo de similitud facial por Template Matching cada 15 frames
                    contador_frames_sesion += 1
                    if contador_frames_sesion % 15 == 0:
                        roi_cara = frame_byn[y:y+alto, x:x+ancho]
                        _, score_actual = identificar_rostro(roi_cara, lista_nombres, lista_caras)
                        if score_actual is not None:
                            sim_act = max(0.0, min(100.0, (1.0 - score_actual) * 100.0))
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
