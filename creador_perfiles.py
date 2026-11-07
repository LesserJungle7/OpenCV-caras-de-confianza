import os
import cv2

from config import (
    CARPETA_CONFIANZA,
    COLOR_TEXTO,
    COLOR_RECUADRO,
    GROSOR,
    FUENTE,
    ESCALA
)
from vision import obtener_clasificador, detectar_caras
from usuarios import (
    inicializar_entorno,
    verificar_admin,
    actualizar_admin_pass,
    registrar_usuario
)

def capturar_poses_usuario(nombre_usuario):
    """Guía la captura de las 4 poses faciales y las guarda en disco."""
    clasificador = obtener_clasificador()
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("[ERROR] No es posible abrir la cámara")
        return False

    poses = [
        ("frente_normal", "Posa de FRENTE (Normal) - Pulsa ESPACIO"),
        ("frente_sonriendo", "Posa de FRENTE (Sonriendo) - Pulsa ESPACIO"),
        ("perfil_derecho", "Gira levemente a tu DERECHA - Pulsa ESPACIO"),
        ("perfil_izquierdo", "Gira levemente a tu IZQUIERDA - Pulsa ESPACIO")
    ]

    print("\n--- INICIO DE CAPTURA FACIAL ---")
    print("Sitúate frente a la cámara y pulsa ESPACIO cuando estés listo para cada pose.")

    for clave_pose, instruccion in poses:
        while True:
            ret, frame = camara.read()
            if not ret:
                print("[ERROR] No es posible obtener la imagen de la cámara")
                camara.release()
                cv2.destroyAllWindows()
                return False
            frame = cv2.flip(frame, 1)

            frame_mostrar = frame.copy()
            caras, frame_byn = detectar_caras(frame, clasificador)

            for (x, y, ancho, alto) in caras:
                cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), COLOR_RECUADRO, GROSOR)

            cv2.putText(frame_mostrar, instruccion, (20, 40), FUENTE, ESCALA, COLOR_TEXTO, GROSOR)
            cv2.imshow("Registro Facial", frame_mostrar)

            tecla = cv2.waitKey(1)
            if tecla == ord(' '):
                if len(caras) > 0:
                    x, y, ancho, alto = caras[0]
                    recorte = frame_byn[y:y+alto, x:x+ancho]
                else:
                    recorte = frame_byn

                nombre_archivo = f"{CARPETA_CONFIANZA}/{nombre_usuario}_{clave_pose}.jpg"
                cv2.imwrite(nombre_archivo, recorte)
                print(f"[OK] Fotograma capturado: {clave_pose}")
                break
            elif tecla == ord('q'):
                print("[CANCELADO] Registro cancelado por el usuario")
                camara.release()
                cv2.destroyAllWindows()
                return False

    camara.release()
    cv2.destroyAllWindows()
    return True

def menu_principal():
    """Gestiona el menu de administracion para alta de usuarios y claves."""
    inicializar_entorno()

    print("====================================")
    print("  CREADOR DE PERFILES DE SEGURIDAD  ")
    print("====================================")
    pass_admin = input("Introduce la contraseña de Administrador: ").strip()

    if not verificar_admin(pass_admin):
        print("[ACCESO DENEGADO] Contraseña de administrador errónea.")
        return

    while True:
        print("\n--- MENÚ ADMINISTRADOR ---")
        print("1. Registrar nuevo usuario de confianza")
        print("2. Cambiar contraseña de administrador")
        print("3. Salir")
        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            nombre = input("\nNombre del nuevo usuario: ").strip().lower()
            if not nombre:
                print("El nombre no puede estar vacío.")
                continue

            password_user = input(f"Introduce la contraseña para '{nombre}': ").strip()
            print("Selecciona el perfil/rol de acceso:")
            print("  1. Total (Acceso completo)")
            print("  2. Operador (Acceso estándar)")
            print("  3. Invitado (Solo lectura)")
            rol_op = input("Rol (1/2/3): ").strip()
            roles = {"1": "Total", "2": "Operador", "3": "Invitado"}
            rol = roles.get(rol_op, "Operador")

            if capturar_poses_usuario(nombre):
                registrar_usuario(nombre, password_user, rol)
                print(f"[ÉXITO] Usuario '{nombre}' creado con rol '{rol}'.")

        elif opcion == "2":
            nueva_pass = input("Introduce la nueva contraseña de administrador: ").strip()
            if nueva_pass:
                actualizar_admin_pass(nueva_pass)
                print("[ÉXITO] Contraseña de administrador actualizada.")
            else:
                print("La contraseña no puede estar vacía.")

        elif opcion == "3":
            print("Saliendo del creador de perfiles...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    menu_principal()
