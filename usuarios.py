import os
import json
from config import ARCHIVO_CONFIG, CARPETA_CONFIANZA, CARPETA_DATOS, CARPETA_INTRUSOS

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
    """Agrega o actualiza un usuario en la configuracion."""
    datos = cargar_usuarios()
    if "usuarios" not in datos:
        datos["usuarios"] = {}
    datos["usuarios"][nombre] = {
        "password": password,
        "rol": rol
    }
    guardar_usuarios(datos)
