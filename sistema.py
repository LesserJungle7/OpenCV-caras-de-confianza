import os
import sys
import time
import threading
import ctypes
import cv2
from config import CARPETA_INTRUSOS

def reproducir_audio(ruta_audio, bloquear=False):
    """
    Reproduce un archivo de audio (MP3, WAV) de forma asíncrona o síncrona.
    Utiliza MCI en Windows para reproducir audio nativamente sin librerías externas.
    """
    if not os.path.isabs(ruta_audio):
        ruta_directa = os.path.join(os.path.dirname(os.path.abspath(__file__)), ruta_audio)
        if os.path.exists(ruta_directa):
            ruta_audio = ruta_directa
        else:
            ruta_audio = os.path.abspath(ruta_audio)

    if not os.path.exists(ruta_audio):
        print(f"[AVISO AUDIO] Archivo de sonido no encontrado: {ruta_audio}")
        return None

    def _ejecutar():
        if sys.platform.startswith("win"):
            alias = f"snd_{threading.get_ident()}_{int(time.time() * 1000)}"
            mci = ctypes.windll.winmm.mciSendStringW
            try:
                res_open = mci(f'open "{ruta_audio}" type mpegvideo alias {alias}', None, 0, 0)
                if res_open == 0:
                    mci(f'play {alias} wait', None, 0, 0)
            except Exception as e:
                print(f"[ERROR AUDIO] No se pudo reproducir {ruta_audio}: {e}")
            finally:
                mci(f'close {alias}', None, 0, 0)
        elif sys.platform == "darwin":
            os.system(f'afplay "{ruta_audio}"')
        else:
            os.system(f'paplay "{ruta_audio}" 2>/dev/null || aplay "{ruta_audio}" 2>/dev/null || ffplay -nodisp -autoexit "{ruta_audio}" 2>/dev/null')

    if bloquear:
        _ejecutar()
        return None
    else:
        hilo = threading.Thread(target=_ejecutar, daemon=True)
        hilo.start()
        return hilo

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
