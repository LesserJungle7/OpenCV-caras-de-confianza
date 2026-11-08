import os
import json
import time
from config import ARCHIVO_CONFIG, CARPETA_CONFIANZA, CARPETA_DATOS, CARPETA_INTRUSOS
from sistema import contar_intrusos_desde

def inicializar_entorno():
    """Crea directorios y archivo de configuracion si no existen."""
    os.makedirs(CARPETA_CONFIANZA, exist_ok=True)
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    os.makedirs(CARPETA_INTRUSOS, exist_ok=True)
    if not os.path.exists(ARCHIVO_CONFIG):
        datos_iniciales = {
            "admin_pass": "admin123",
            "usuarios": {}
        }
        guardar_usuarios(datos_iniciales)

def cargar_usuarios():
    """Carga los datos de usuarios desde el archivo JSON."""
    if not os.path.exists(ARCHIVO_CONFIG):
        return {"admin_pass": "admin123", "usuarios": {}}
    with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)

def guardar_usuarios(datos):
    """Escribe los datos actualizados en el archivo JSON."""
    with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=4)

def verificar_admin(password_ingresada):
    """Valida la contrasena de administrador."""
    datos = cargar_usuarios()
    return password_ingresada == datos.get("admin_pass", "admin123")

def actualizar_admin_pass(nueva_pass):
    """Modifica la contrasena de administrador."""
    datos = cargar_usuarios()
    datos["admin_pass"] = nueva_pass
    guardar_usuarios(datos)

def obtener_usuario(nombre):
    """Devuelve los datos de un usuario por su nombre."""
    datos = cargar_usuarios()
    return datos.get("usuarios", {}).get(nombre)

def registrar_usuario(nombre, password, rol):
    """Agrega o actualiza un usuario en la configuracion preservando estadisticas existentes."""
    datos = cargar_usuarios()
    if "usuarios" not in datos:
        datos["usuarios"] = {}
    estadisticas_previas = datos["usuarios"].get(nombre, {}).get("estadisticas", {
        "inicios_sesion": 0,
        "ultimo_acceso": None,
        "tiempo_acumulado_segundos": 0.0
    })
    datos["usuarios"][nombre] = {
        "password": password,
        "rol": rol,
        "estadisticas": estadisticas_previas
    }
    guardar_usuarios(datos)

def obtener_estadisticas_usuario(nombre):
    """Devuelve el diccionario de estadisticas del usuario garantizando campos por defecto."""
    usuario = obtener_usuario(nombre)
    if not usuario:
        return {
            "inicios_sesion": 0,
            "ultimo_acceso": "Sin registros",
            "ultimo_cierre_timestamp": None,
            "tiempo_acumulado_segundos": 0.0,
            "similitud_acumulada": 0.0,
            "muestras_similitud": 0,
            "similitud_media": 0.0,
            "intrusos_ultima_sesion": 0
        }
    stats = usuario.get("estadisticas", {})
    return {
        "inicios_sesion": stats.get("inicios_sesion", 0),
        "ultimo_acceso": stats.get("ultimo_acceso") or "Primer inicio",
        "ultimo_cierre_timestamp": stats.get("ultimo_cierre_timestamp"),
        "tiempo_acumulado_segundos": stats.get("tiempo_acumulado_segundos", 0.0),
        "similitud_acumulada": stats.get("similitud_acumulada", 0.0),
        "muestras_similitud": stats.get("muestras_similitud", 0),
        "similitud_media": stats.get("similitud_media", 0.0),
        "intrusos_ultima_sesion": stats.get("intrusos_ultima_sesion", 0)
    }

def registrar_inicio_sesion(nombre, similitud_inicial=None):
    """
    Incrementa el contador de inicios de sesion, actualiza la fecha de acceso,
    calcula los intentos de intrusion desde la sesion anterior y registra la similitud.
    """
    datos = cargar_usuarios()
    if "usuarios" not in datos or nombre not in datos["usuarios"]:
        return
    
    usuario = datos["usuarios"][nombre]
    if "estadisticas" not in usuario:
        usuario["estadisticas"] = {
            "inicios_sesion": 0,
            "ultimo_acceso": None,
            "ultimo_cierre_timestamp": None,
            "tiempo_acumulado_segundos": 0.0,
            "similitud_acumulada": 0.0,
            "muestras_similitud": 0,
            "similitud_media": 0.0,
            "intrusos_ultima_sesion": 0
        }
    
    stats = usuario["estadisticas"]
    ultimo_cierre = stats.get("ultimo_cierre_timestamp")
    
    # Contar intentos de intrusion ocurridos desde la sesion previa
    if ultimo_cierre is not None:
        stats["intrusos_ultima_sesion"] = contar_intrusos_desde(ultimo_cierre)
    else:
        stats["intrusos_ultima_sesion"] = 0

    stats["inicios_sesion"] = stats.get("inicios_sesion", 0) + 1
    stats["ultimo_acceso"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if similitud_inicial is not None:
        similitud_val = float(similitud_inicial)
        stats["similitud_acumulada"] = stats.get("similitud_acumulada", 0.0) + similitud_val
        stats["muestras_similitud"] = stats.get("muestras_similitud", 0) + 1
        stats["similitud_media"] = round(stats["similitud_acumulada"] / stats["muestras_similitud"], 1)

    guardar_usuarios(datos)

def registrar_fin_sesion(nombre, duracion_segundos):
    """Suma la duracion de la sesion al tiempo acumulado y guarda el timestamp de cierre."""
    datos = cargar_usuarios()
    if "usuarios" not in datos or nombre not in datos["usuarios"]:
        return
    
    usuario = datos["usuarios"][nombre]
    if "estadisticas" not in usuario:
        usuario["estadisticas"] = {
            "inicios_sesion": 1,
            "ultimo_acceso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ultimo_cierre_timestamp": int(time.time()),
            "tiempo_acumulado_segundos": 0.0,
            "similitud_acumulada": 0.0,
            "muestras_similitud": 0,
            "similitud_media": 0.0,
            "intrusos_ultima_sesion": 0
        }
    
    stats = usuario["estadisticas"]
    tiempo_actual = stats.get("tiempo_acumulado_segundos", 0.0)
    stats["tiempo_acumulado_segundos"] = round(tiempo_actual + max(0.0, float(duracion_segundos)), 2)
    stats["ultimo_cierre_timestamp"] = int(time.time())
    guardar_usuarios(datos)
