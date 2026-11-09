import os
import sys
import cv2

# Asegura que la raiz del proyecto y la carpeta src esten en sys.path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_BASE_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    CARPETA_CONFIANZA,
    COLOR_TEXTO,
    COLOR_RECUADRO,
    GROSOR,
    FUENTE,
    ESCALA
)
from vision import obtener_clasificador, detectar_caras, normalizar_cara
from depurador import (
    es_debug_activo,
    toggle_debug,
    cambiar_fase,
    cerrar_debug_fase,
    pausar_inspeccion_debug
)
from usuarios import (
    inicializar_entorno,
    verificar_admin,
    actualizar_admin_pass,
    registrar_usuario
)
from interfaz import (
    capturar_texto_pantalla,
    seleccionar_opcion_pantalla,
    mostrar_mensaje_pantalla
)

def capturar_poses_usuario(nombre_usuario, camara=None, clasificador=None):
    """Guia la captura de 2 poses faciales frontales normalizadas y las guarda en disco."""
    cambiar_fase("creacion_perfil")
    try:
        camara_propia = False
        if clasificador is None:
            clasificador = obtener_clasificador()
        if camara is None:
            camara = cv2.VideoCapture(0)
            camara_propia = True

        if not camara.isOpened():
            print("[ERROR] No es posible abrir la camara")
            return False

        poses = [
            ("frente_normal", "Posa de FRENTE (Normal) - Pulsa ESPACIO"),
            ("frente_sonriendo", "Posa de FRENTE (Sonriendo) - Pulsa ESPACIO")
        ]

        for clave_pose, instruccion in poses:
            while True:
                ret, frame = camara.read()
                if not ret:
                    if camara_propia:
                        camara.release()
                        cv2.destroyAllWindows()
                    return False
                frame = cv2.flip(frame, 1)

                frame_mostrar = frame.copy()
                caras, frame_byn = detectar_caras(frame, clasificador)

                for (x, y, ancho, alto) in caras:
                    cv2.rectangle(frame_mostrar, (x, y), (x + ancho, y + alto), COLOR_RECUADRO, GROSOR)

                # Si el modo debug esta activo, procesa y visualiza la normalizacion en vivo
                if len(caras) > 0 and es_debug_activo():
                    normalizar_cara(frame_byn, caja=caras[0])

                # Cartel superior de instruccion de la pose
                cv2.putText(frame_mostrar, instruccion, (20, 40), FUENTE, ESCALA, COLOR_TEXTO, GROSOR)

                # Badge de modo debug
                h_f, w_f = frame_mostrar.shape[:2]
                if es_debug_activo():
                    cv2.rectangle(frame_mostrar, (w_f - 145, 12), (w_f - 15, 38), (0, 60, 140), -1)
                    cv2.rectangle(frame_mostrar, (w_f - 145, 12), (w_f - 15, 38), (0, 255, 255), 1)
                    cv2.putText(frame_mostrar, "[DEBUG ON]", (w_f - 135, 30), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)

                # Barra inferior visible con boton de captura, debug y boton Q para volver
                cv2.rectangle(frame_mostrar, (0, h_f - 36), (w_f, h_f), (15, 15, 22), -1)
                cv2.putText(frame_mostrar, "[ESPACIO] Capturar fotograma  |  [D] Modo Debug  |  [Q] Volver",
                            (20, h_f - 12), FUENTE, 0.44, (0, 255, 255), 1)

                cv2.imshow("Autenticador Facial", frame_mostrar)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord(' '):
                    if len(caras) == 0:
                        continue

                    recorte = normalizar_cara(frame_byn, caja=caras[0])

                    # En modo debug, pausa e inspecciona el fotograma normalizado antes de guardar
                    if es_debug_activo():
                        msg_inspeccion = f"[DEBUG] Pose '{clave_pose}' normalizada. Pulsa [S] para guardar y continuar"
                        confirmado = pausar_inspeccion_debug(frame_mostrar, msg_inspeccion, tecla_continuar='s')
                        if not confirmado:
                            continue

                    nombre_archivo = f"{CARPETA_CONFIANZA}/{nombre_usuario}_{clave_pose}.jpg"
                    cv2.imwrite(nombre_archivo, recorte)
                    break
                elif tecla in (ord('d'), ord('D')):
                    toggle_debug()
                elif tecla == ord('q') or tecla == ord('Q') or tecla == 27:
                    if camara_propia:
                        camara.release()
                        cv2.destroyAllWindows()
                    return False

        if camara_propia:
            camara.release()
            cv2.destroyAllWindows()
        return True
    finally:
        cerrar_debug_fase("creacion_perfil")
        cambiar_fase("menu")

def crear_perfil_interactivo(camara=None, clasificador=None):
    """
    Flujo interactivo completo del creador de perfiles sobre la ventana de camara.
    Permite verificar la contraseña de administrador, registrar usuarios con diferentes
    roles y capturar sus plantillas faciales sin recurrir a la terminal.
    """
    inicializar_entorno()
    cambiar_fase("menu")
    camara_propia = False
    if clasificador is None:
        clasificador = obtener_clasificador()
    if camara is None:
        camara = cv2.VideoCapture(0)
        camara_propia = True

    if not camara.isOpened():
        print("[ERROR] No es posible abrir la camara.")
        return

    # Validacion interactiva de clave de administrador
    pass_admin = capturar_texto_pantalla(
        camara,
        titulo="ADMINISTRACION DE PERFILES",
        prompt="Introduce la contraseña de administrador:",
        es_password=True
    )

    if pass_admin is None:
        if camara_propia:
            camara.release()
            cv2.destroyAllWindows()
        return

    if not verificar_admin(pass_admin):
        mostrar_mensaje_pantalla(
            camara,
            titulo="ACCESO DENEGADO",
            lineas_mensaje=["Contraseña de administrador incorrecta."],
            color=(0, 0, 255),
            duracion_ms=2000
        )
        if camara_propia:
            camara.release()
            cv2.destroyAllWindows()
        return

    # Menu interactivo del administrador
    while True:
        opciones = [
            ("1", "Registrar nuevo usuario de confianza"),
            ("2", "Cambiar clave de administrador"),
            ("3", "Volver al menu principal [3 / Q]")
        ]
        opcion = seleccionar_opcion_pantalla(
            camara,
            titulo="MENU DE ADMINISTRACION",
            opciones=opciones,
            subtitulo="Pulsa 1, 2, 3 o Q para volver"
        )

        # Opcion 3 y tecla Q permiten volver al menu principal
        if opcion is None or opcion == "3" or opcion in ("q", "Q"):
            break

        elif opcion == "1":
            # Nombre de usuario
            nombre = capturar_texto_pantalla(
                camara,
                titulo="REGISTRO DE USUARIO",
                prompt="Introduce el nombre del nuevo usuario:"
            )
            if not nombre:
                continue
            nombre = nombre.strip().lower()

            # Seleccion de perfil/rol
            roles_opciones = [
                ("1", "Total (Acceso completo)"),
                ("2", "Operador (Acceso estandar)"),
                ("3", "Invitado (Sin contraseña escrita)")
            ]
            rol_sel = seleccionar_opcion_pantalla(
                camara,
                titulo=f"ROL PARA: {nombre.upper()}",
                opciones=roles_opciones,
                subtitulo="Selecciona el nivel de acceso"
            )
            if not rol_sel or rol_sel in ("q", "Q"):
                continue
            roles_map = {"1": "Total", "2": "Operador", "3": "Invitado"}
            rol = roles_map.get(rol_sel, "Operador")

            # Contraseña (los invitados no requieren contraseña manual escrita)
            if rol == "Invitado":
                password_user = ""
                mostrar_mensaje_pantalla(
                    camara,
                    titulo="MODO INVITADO",
                    lineas_mensaje=[
                        f"El usuario {nombre.upper()} no requiere clave escrita.",
                        "Solo se verificara presencia y reto facial."
                    ],
                    color=(0, 255, 255),
                    duracion_ms=1800
                )
            else:
                password_user = capturar_texto_pantalla(
                    camara,
                    titulo=f"CONTRASEÑA: {nombre.upper()}",
                    prompt=f"Introduce la contraseña para '{nombre}':",
                    es_password=True
                )
                if password_user is None:
                    continue

            # Captura interactiva de poses
            exito = capturar_poses_usuario(nombre, camara=camara, clasificador=clasificador)
            if exito:
                registrar_usuario(nombre, password_user, rol)
                mostrar_mensaje_pantalla(
                    camara,
                    titulo="USUARIO REGISTRADO",
                    lineas_mensaje=[
                        f"Usuario: {nombre.upper()}",
                        f"Nivel de acceso: {rol}",
                        "Perfil guardado correctamente en el sistema."
                    ],
                    color=(0, 255, 0),
                    duracion_ms=2200
                )

        elif opcion == "2":
            nueva_pass = capturar_texto_pantalla(
                camara,
                titulo="CLAVE ADMINISTRADOR",
                prompt="Introduce la nueva contraseña de admin:",
                es_password=True
            )
            if nueva_pass:
                actualizar_admin_pass(nueva_pass)
                mostrar_mensaje_pantalla(
                    camara,
                    titulo="ADMIN ACTUALIZADO",
                    lineas_mensaje=["Contraseña de administrador actualizada con éxito."],
                    color=(0, 255, 0),
                    duracion_ms=1800
                )

    if camara_propia:
        camara.release()
        cv2.destroyAllWindows()

def menu_principal():
    """Gestiona el menu de administracion para alta de usuarios y claves desde consola."""
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
        print("3. Salir (o pulsa q)")
        opcion = input("Selecciona una opción: ").strip().lower()

        if opcion == "1":
            nombre = input("\nNombre del nuevo usuario: ").strip().lower()
            if not nombre:
                print("El nombre no puede estar vacío.")
                continue

            print("Selecciona el perfil/rol de acceso:")
            print("  1. Total (Acceso completo)")
            print("  2. Operador (Acceso estándar)")
            print("  3. Invitado (Solo lectura)")
            rol_op = input("Rol (1/2/3): ").strip()
            roles = {"1": "Total", "2": "Operador", "3": "Invitado"}
            rol = roles.get(rol_op, "Operador")

            if rol == "Invitado":
                password_user = ""
            else:
                password_user = input(f"Introduce la contraseña para '{nombre}': ").strip()

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

        elif opcion in ("3", "q"):
            print("Saliendo del creador de perfiles...")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    crear_perfil_interactivo()
