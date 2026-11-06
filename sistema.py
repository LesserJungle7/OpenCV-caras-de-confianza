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
    cv2.imwrite(f"{CARPETA_INTRUSOS}/intruso_{timestamp}_1.jpg", frame_actual)
    time.sleep(0.3)
    ret, frame_segundo = camara.read()
    if ret:
        cv2.imwrite(f"{CARPETA_INTRUSOS}/intruso_{timestamp}_2.jpg", frame_segundo)
