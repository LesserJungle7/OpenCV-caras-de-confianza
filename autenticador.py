import os
import time
import cv2

from config import (
    CARPETA_CONFIANZA,
    CARPETA_INTRUSOS,
    ARCHIVO_CONFIG,
    RUTA_CASCADE,
    UMBRAL_COINCIDENCIA
)
from vision import (
    cargar_caras_confianza,
    identificar_rostro,
    obtener_clasificador,
    detectar_caras
)
from sistema import suspender_equipo, guardar_evidencia_intruso
from usuarios import cargar_usuarios

def procesar_login_usuario(usuario, datos_sistema):
    """Verifica credenciales del usuario detectado y muestra permisos de acceso."""
    print(f"\n[AVISO] Usuario de confianza detectado: {usuario.upper()}")
    password_ingresada = input(f"Introduce tu contraseña, {usuario}: ").strip()

    info_usuario = datos_sistema.get("usuarios", {}).get(usuario, {})
    if password_ingresada == info_usuario.get("password"):
        rol = info_usuario.get("rol", "Invitado")
        print(f"[ACCESO CONCEDIDO] Bienvenido/a {usuario}.")
        print(f"[PERMISOS ACTIVADOS] Nivel de acceso: {rol}")
        if rol == "Total":
            print(" -> Tienes acceso a configuración, terminal y archivos restringidos.")
        elif rol == "Operador":
            print(" -> Tienes acceso a ejecución de aplicaciones estándar.")
        else:
            print(" -> Modo invitado: Acceso restringido solo lectura.")
        return True
    else:
        print("[ERROR] Contraseña incorrecta.")
        return False

def ejecutar_guardia():
    """Inicia el ciclo principal de vigilancia y autenticacion continua."""
    os.makedirs(CARPETA_INTRUSOS, exist_ok=True)
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
    
    estado_sesion = False
    usuario_autenticado = None
    frames_sin_persona = 0

    while True:
        ret, frame = camara.read()
        if not ret:
            print("[ERROR] No es posible obtener la imagen de la cámara.")
            break

        caras, frame_byn = detectar_caras(frame, clasificador)

        # Modo vigilancia (sesion inactiva)
        if not estado_sesion:
            if len(caras) > 0:
                x, y, ancho, alto = caras[0]
                roi_cara = frame_byn[y:y+alto, x:x+ancho]
                usuario, score = identificar_rostro(roi_cara, lista_nombres, lista_caras)

                if usuario is not None and usuario in datos_sistema.get("usuarios", {}):
                    if procesar_login_usuario(usuario, datos_sistema):
                        estado_sesion = True
                        usuario_autenticado = usuario
                        frames_sin_persona = 0
                else:
                    print("\n[ALERTA INTRUSO] Persona no autorizada detectada frente al equipo.")
                    guardar_evidencia_intruso(camara, frame)
                    suspender_equipo()
                    time.sleep(5)

        # Modo sesion activa (control de presencia)
        else:
            if len(caras) == 0:
                frames_sin_persona += 1
                if frames_sin_persona > 30:
                    print(f"\n[BLOQUEO] Usuario '{usuario_autenticado}' ausente. Reactivando guardia...")
                    estado_sesion = False
                    usuario_autenticado = None
                    frames_sin_persona = 0
            else:
                frames_sin_persona = 0

        if cv2.waitKey(30) == ord('q'):
            print("Cerrando autenticador...")
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ejecutar_guardia()
