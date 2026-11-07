import os
import sys
import time
import cv2
from config import CARPETA_INTRUSOS

def suspender_equipo():
    """Bloquea o suspende la estacion de trabajo segun el sistema operativo."""
    print("[SEGURIDAD] Bloqueando estacion de trabajo...")
    if sys.platform.startswith("win"):
        os.system("rundll32.exe user32.dll,LockWorkStation")
    elif sys.platform.startswith("linux"):
        os.system("systemctl suspend")
    elif sys.platform == "darwin":
        os.system("pmset sleepnow")

def guardar_evidencia_intruso(camara, frame_actual):
    """Guarda dos capturas consecutivas del intruso detectado."""
    os.makedirs(CARPETA_INTRUSOS, exist_ok=True)
    timestamp = int(time.time())
    cv2.imwrite(os.path.join(CARPETA_INTRUSOS, f"intruso_{timestamp}_1.jpg"), frame_actual)
    time.sleep(0.3)
    ret, frame_segundo = camara.read()
    if ret:
        cv2.imwrite(os.path.join(CARPETA_INTRUSOS, f"intruso_{timestamp}_2.jpg"), frame_segundo)

def contar_intrusos_desde(timestamp_inicio=None):
    """
    Cuenta los eventos de intrusion registrados en CARPETA_INTRUSOS
    que ocurrieron con posterioridad a `timestamp_inicio`.
    Cada evento de intrusion guarda 2 fotografias con el mismo timestamp.
    """
    if not os.path.exists(CARPETA_INTRUSOS):
        return 0

    timestamps_eventos = set()
    for archivo in os.listdir(CARPETA_INTRUSOS):
        if archivo.startswith("intruso_") and archivo.endswith(".jpg"):
            partes = archivo.split("_")
            if len(partes) >= 2:
                try:
                    ts = int(partes[1])
                    if timestamp_inicio is None or ts > timestamp_inicio:
                        timestamps_eventos.add(ts)
                except ValueError:
                    continue

    return len(timestamps_eventos)
