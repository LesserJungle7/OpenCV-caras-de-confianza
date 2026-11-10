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

def cambiar_password_usuario(nombre, nueva_pass):
    """Actualiza la contraseña de un usuario registrado o del administrador."""
    datos = cargar_usuarios()
    modificado = False
    if nombre == "admin":
        datos["admin_pass"] = nueva_pass
        modificado = True
    if "usuarios" in datos and nombre in datos["usuarios"]:
        datos["usuarios"][nombre]["password"] = nueva_pass
        modificado = True
    if modificado:
        guardar_usuarios(datos)
        return True
    return False

def listar_usuarios():
    """Devuelve una lista de tuplas (nombre, rol) de los usuarios registrados."""
    datos = cargar_usuarios()
    usuarios_dict = datos.get("usuarios", {})
    return [(nom, info.get("rol", "Operador")) for nom, info in usuarios_dict.items()]

def eliminar_usuario(nombre):
    """Elimina al usuario del archivo JSON de configuracion y borra sus imagenes de confianza."""
    datos = cargar_usuarios()
    eliminado = False
    if "usuarios" in datos and nombre in datos["usuarios"]:
        del datos["usuarios"][nombre]
        guardar_usuarios(datos)
        eliminado = True

    # Borrar imagenes de confianza asociadas al usuario
    if os.path.exists(CARPETA_CONFIANZA):
        for archivo in os.listdir(CARPETA_CONFIANZA):
            prefijo = f"{nombre}_"
            if archivo.lower().startswith(prefijo.lower()) and archivo.lower().endswith((".jpg", ".png", ".jpeg")):
                ruta_img = os.path.join(CARPETA_CONFIANZA, archivo)
                try:
                    os.remove(ruta_img)
                    eliminado = True
                except OSError:
                    pass

    return eliminado

def _estadisticas_por_defecto():
    """Genera la estructura de estadisticas base para un usuario."""
    return {
        "inicios_sesion": 0,
        "ultimo_acceso": None,
        "ultimo_cierre_timestamp": None,
        "tiempo_acumulado_segundos": 0.0,
        "similitud_acumulada": 0.0,
        "muestras_similitud": 0,
        "similitud_media": 0.0,
        "intrusos_bloqueo_previo": 0,
        "intrusos_ultima_sesion": 0,
        "usurpaciones_detectadas": 0,
        "retornos_exitosos": 0,
        "ultimo_incidente_seguridad": None
    }

def registrar_usuario(nombre, password, rol):
    """Agrega o actualiza un usuario en la configuracion preservando estadisticas existentes."""
    datos = cargar_usuarios()
    if "usuarios" not in datos:
        datos["usuarios"] = {}
    estadisticas_previas = datos["usuarios"].get(nombre, {}).get("estadisticas", _estadisticas_por_defecto())
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
        stats_defecto = _estadisticas_por_defecto()
        stats_defecto["ultimo_acceso"] = "Sin registros"
        return stats_defecto
    stats = usuario.get("estadisticas", {})
    return {
        "inicios_sesion": stats.get("inicios_sesion", 0),
        "ultimo_acceso": stats.get("ultimo_acceso") or "Primer inicio",
        "ultimo_cierre_timestamp": stats.get("ultimo_cierre_timestamp"),
        "tiempo_acumulado_segundos": stats.get("tiempo_acumulado_segundos", 0.0),
        "similitud_acumulada": stats.get("similitud_acumulada", 0.0),
        "muestras_similitud": stats.get("muestras_similitud", 0),
        "similitud_media": stats.get("similitud_media", 0.0),
        "intrusos_bloqueo_previo": stats.get("intrusos_bloqueo_previo", stats.get("intrusos_ultima_sesion", 0)),
        "intrusos_ultima_sesion": stats.get("intrusos_ultima_sesion", 0),
        "usurpaciones_detectadas": stats.get("usurpaciones_detectadas", 0),
        "retornos_exitosos": stats.get("retornos_exitosos", 0),
        "ultimo_incidente_seguridad": stats.get("ultimo_incidente_seguridad")
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
        usuario["estadisticas"] = _estadisticas_por_defecto()
    
    stats = usuario["estadisticas"]
    ultimo_cierre = stats.get("ultimo_cierre_timestamp")
    
    # Contar intentos de intrusion ocurridos desde la sesion previa
    if ultimo_cierre is not None:
        intrusos_detectados = contar_intrusos_desde(ultimo_cierre)
    else:
        intrusos_detectados = 0
    stats["intrusos_bloqueo_previo"] = intrusos_detectados
    stats["intrusos_ultima_sesion"] = intrusos_detectados

    stats["inicios_sesion"] = stats.get("inicios_sesion", 0) + 1
    stats["ultimo_acceso"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if similitud_inicial is not None:
        similitud_val = float(similitud_inicial)
        stats["similitud_acumulada"] = stats.get("similitud_acumulada", 0.0) + similitud_val
        stats["muestras_similitud"] = stats.get("muestras_similitud", 0) + 1
        stats["similitud_media"] = round(stats["similitud_acumulada"] / stats["muestras_similitud"], 1)

    guardar_usuarios(datos)

def registrar_evento_seguridad(nombre, tipo_evento="usurpacion"):
    """Registra un incidente de seguridad asociado a la sesion del usuario."""
    datos = cargar_usuarios()
    if "usuarios" not in datos or nombre not in datos["usuarios"]:
        return
    stats = datos["usuarios"][nombre].setdefault("estadisticas", _estadisticas_por_defecto())
    if tipo_evento in ("usurpacion", "usurpacion_retorno"):
        stats["usurpaciones_detectadas"] = stats.get("usurpaciones_detectadas", 0) + 1
    stats["ultimo_incidente_seguridad"] = f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {tipo_evento}"
    guardar_usuarios(datos)

def registrar_retorno_exitoso(nombre):
    """Incrementa el contador de retornos de ausencia verificados exitosamente."""
    datos = cargar_usuarios()
    if "usuarios" not in datos or nombre not in datos["usuarios"]:
        return
    stats = datos["usuarios"][nombre].setdefault("estadisticas", _estadisticas_por_defecto())
    stats["retornos_exitosos"] = stats.get("retornos_exitosos", 0) + 1
    guardar_usuarios(datos)

def registrar_fin_sesion(nombre, duracion_segundos, similitudes_sesion=None):
    """
    Suma la duracion de la sesion al tiempo acumulado, consolida el promedio biometrico
    continuo de la sesion y guarda el timestamp de cierre.
    """
    datos = cargar_usuarios()
    if "usuarios" not in datos or nombre not in datos["usuarios"]:
        return
    
    usuario = datos["usuarios"][nombre]
    if "estadisticas" not in usuario:
        usuario["estadisticas"] = _estadisticas_por_defecto()
        usuario["estadisticas"]["inicios_sesion"] = 1
        usuario["estadisticas"]["ultimo_acceso"] = time.strftime("%Y-%m-%d %H:%M:%S")
        usuario["estadisticas"]["ultimo_cierre_timestamp"] = int(time.time())
    
    stats = usuario["estadisticas"]
    tiempo_actual = stats.get("tiempo_acumulado_segundos", 0.0)
    stats["tiempo_acumulado_segundos"] = round(tiempo_actual + max(0.0, float(duracion_segundos)), 2)
    stats["ultimo_cierre_timestamp"] = int(time.time())

    # Consolidar promedio de similitud de los fotogramas evaluados durante la sesion
    if similitudes_sesion and len(similitudes_sesion) > 0:
        media_sesion = float(sum(similitudes_sesion) / len(similitudes_sesion))
        stats["similitud_acumulada"] = stats.get("similitud_acumulada", 0.0) + media_sesion
        stats["muestras_similitud"] = stats.get("muestras_similitud", 0) + 1
        stats["similitud_media"] = round(stats["similitud_acumulada"] / stats["muestras_similitud"], 1)

    guardar_usuarios(datos)
