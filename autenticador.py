import os
import time
import cv2

from config import (
    CARPETA_CONFIANZA,
    CARPETA_INTRUSOS,
    ARCHIVO_CONFIG,
    RUTA_CASCADE,
    UMBRAL_COINCIDENCIA,
    TIEMPO_AUSENCIA_GUARDIA,
    TIEMPO_MAXIMO_GESTO,
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
    detectar_caras
)
from sistema import suspender_equipo, guardar_evidencia_intruso, reproducir_audio
from usuarios import cargar_usuarios, inicializar_entorno, obtener_usuario
from gestos import EvaluadorGestos

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
    tiempo_inicio_ausencia = None

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

                    if usuario is not None and usuario not in datos_sistema.get("usuarios", {}):
                        datos_sistema = cargar_usuarios()

                    if usuario is not None and usuario in datos_sistema.get("usuarios", {}):
                        cv2.putText(frame_mostrar, f"Usuario: {usuario.upper()}", (x, y - 10), FUENTE, 0.6, COLOR_TEXTO, 2)
                        cv2.imshow("Autenticador Facial", frame_mostrar)
                        cv2.waitKey(1)

                        # Validacion de contrasena de texto y posterior reto facial
                        if verificar_password_texto(usuario, datos_sistema):
                            if validar_contrasena_facial(camara, clasificador):
                                mostrar_permisos_usuario(usuario, datos_sistema)
                                estado_sesion = True
                                usuario_autenticado = usuario
                                tiempo_inicio_ausencia = None
                            else:
                                print("\n[ALERTA INTRUSO] Prueba de gestos no superada o tiempo agotado.")
                                guardar_evidencia_intruso(camara, frame)
                                time.sleep(1.0)
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
                texto_sesion = f"SESION ACTIVA: {usuario_autenticado.upper()}"
                cv2.putText(frame_mostrar, texto_sesion, (20, 40), FUENTE, 0.7, (0, 255, 255), GROSOR)

                if len(caras) == 0:
                    if tiempo_inicio_ausencia is None:
                        tiempo_inicio_ausencia = time.time()

                    tiempo_ausente = time.time() - tiempo_inicio_ausencia
                    if tiempo_ausente >= TIEMPO_AUSENCIA_GUARDIA:
                        print(f"\n[BLOQUEO] Usuario '{usuario_autenticado}' ausente durante {int(tiempo_ausente)} segundos. Reactivando guardia...")
                        estado_sesion = False
                        usuario_autenticado = None
                        tiempo_inicio_ausencia = None
                else:
                    tiempo_inicio_ausencia = None
                    x, y, ancho, alto = caras[0]
                    cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), (0, 255, 0), GROSOR)

            cv2.imshow("Autenticador Facial", frame_mostrar)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Cerrando autenticador...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Autenticador detenido por el usuario.")
    finally:
        camara.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ejecutar_guardia()
