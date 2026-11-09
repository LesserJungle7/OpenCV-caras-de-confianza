import time
import numpy as np
import cv2

from config import FUENTE, GROSOR, COLOR_TEXTO, COLOR_RECUADRO
from depurador import es_debug_activo

def formatear_duracion(segundos):
    """Convierte una duracion en segundos a formato legible HHh MMm SSs o MMm SSs."""
    segundos = int(max(0, segundos))
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segs = segundos % 60
    if horas > 0:
        return f"{horas:02d}h {minutos:02d}m {segs:02d}s"
    return f"{minutos:02d}m {segs:02d}s"

def dibujar_tarjeta_hud(frame, x, y, ancho, alto, color_borde=(0, 255, 255), color_fondo=(20, 20, 20), opacidad=0.82):
    """Renderiza una tarjeta rectangular traslucida con borde."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), color_fondo, -1)
    cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), color_borde, 1)
    cv2.addWeighted(overlay, opacidad, frame, 1.0 - opacidad, 0, frame)

def dibujar_barra_progreso(frame, x, y, ancho, alto, porcentaje, color_barra):
    """Dibuja una barra horizontal de progreso con fondo oscuro y borde."""
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (50, 50, 50), -1)
    if porcentaje > 0:
        ancho_lleno = int(ancho * (min(100, max(0, porcentaje)) / 100.0))
        cv2.rectangle(frame, (x, y), (x + ancho_lleno, y + alto), color_barra, -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (100, 100, 100), 1)

def dibujar_barra_menu_principal(frame):
    """Dibuja la barra de navegacion superior en el menu principal de inicio."""
    h, w = frame.shape[:2]
    alto_menu = 66
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, alto_menu), (15, 15, 22), -1)
    cv2.line(overlay, (0, alto_menu), (w, alto_menu), (0, 255, 255), 2)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    cv2.putText(frame, "SISTEMA DE SEGURIDAD FACIAL", (20, 25), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 255), 1)

    btn1_x = 20
    btn2_x = 220
    btn_q_x = 440
    y_btns = 50

    cv2.putText(frame, "[1] Crear perfil", (btn1_x, y_btns), cv2.FONT_HERSHEY_DUPLEX, 0.54, (0, 255, 0), 1)
    cv2.putText(frame, "|", (btn2_x - 25, y_btns), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1)
    cv2.putText(frame, "[2] Autenticador", (btn2_x, y_btns), cv2.FONT_HERSHEY_DUPLEX, 0.54, (0, 255, 0), 1)
    cv2.putText(frame, "|", (btn_q_x - 25, y_btns), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1)
    cv2.putText(frame, "[Q] Salir", (btn_q_x, y_btns), cv2.FONT_HERSHEY_DUPLEX, 0.54, (0, 165, 255), 1)

    # Badge de modo debug si esta activo
    if es_debug_activo():
        cv2.rectangle(frame, (w - 145, 12), (w - 15, 38), (0, 60, 140), -1)
        cv2.rectangle(frame, (w - 145, 12), (w - 15, 38), (0, 255, 255), 1)
        cv2.putText(frame, "[DEBUG ON]", (w - 135, 30), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)

    # Pie de pagina con ayuda visible
    y_inf = h - 34
    overlay_inf = frame.copy()
    cv2.rectangle(overlay_inf, (0, y_inf), (w, h), (15, 15, 22), -1)
    cv2.addWeighted(overlay_inf, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "[1] Crear perfil  |  [2] Autenticador  |  [D] Modo Debug  |  [Q] Salir de la aplicacion",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

def dibujar_hud_sesion(frame, usuario, rol, stats, duracion_actual, mostrar_detalles, mostrar_opciones, tiempo_ausente=0.0, similitud_sesion=None):
    """
    Renderiza la interfaz de usuario en sesion activa:
    - Tarjeta izquierda: 'Nombre de usuario' y 'm - Mostrar mas detalles'.
    - Tarjeta derecha (simetrica a la misma altura): 'Rol' y 'o - Opciones'.
    - Barra inferior permanente con atajos claros incluyendo [Q] para volver.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Tarjeta izquierda
    x_izq, y_izq = 15, 12
    ancho_izq = 345
    alto_card = 62
    cv2.rectangle(overlay, (x_izq, y_izq), (x_izq + ancho_izq, y_izq + alto_card), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x_izq, y_izq), (x_izq + ancho_izq, y_izq + alto_card), (0, 255, 255), 1)

    # 2. Tarjeta derecha
    ancho_der = 230
    x_der = w - ancho_der - 15
    y_der = 12
    cv2.rectangle(overlay, (x_der, y_der), (x_der + ancho_der, y_der + alto_card), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x_der, y_der), (x_der + ancho_der, y_der + alto_card), (0, 255, 0), 1)

    # Panel izquierdo desplegado (Estadisticas)
    if mostrar_detalles:
        alto_panel_izq = 246
        cv2.rectangle(overlay, (x_izq, y_izq + alto_card + 4), (x_izq + ancho_izq + 60, y_izq + alto_card + 4 + alto_panel_izq), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x_izq, y_izq + alto_card + 4), (x_izq + ancho_izq + 60, y_izq + alto_card + 4 + alto_panel_izq), (0, 255, 255), 1)

    # Panel derecho desplegado (Opciones)
    if mostrar_opciones:
        ancho_panel_der = 350
        x_panel_der = w - ancho_panel_der - 15
        alto_panel_der = 248
        cv2.rectangle(overlay, (x_panel_der, y_der + alto_card + 4), (w - 15, y_der + alto_card + 4 + alto_panel_der), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x_panel_der, y_der + alto_card + 4), (w - 15, y_der + alto_card + 4 + alto_panel_der), (0, 255, 0), 1)

    # Barra inferior fija para volver / atajos
    y_inf = h - 34
    cv2.rectangle(overlay, (0, y_inf), (w, h), (15, 15, 22), -1)
    cv2.line(overlay, (0, y_inf), (w, y_inf), (60, 60, 70), 1)

    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    # Badge central de modo debug si esta activo
    if es_debug_activo():
        cv2.rectangle(frame, (w // 2 - 60, 14), (w // 2 + 60, 42), (0, 60, 140), -1)
        cv2.rectangle(frame, (w // 2 - 60, 14), (w // 2 + 60, 42), (0, 255, 255), 1)
        cv2.putText(frame, "[DEBUG ON]", (w // 2 - 50, 32), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 255, 255), 1)

    # Textos tarjeta izquierda
    cv2.putText(frame, f"Nombre de usuario: {usuario.upper()}", (x_izq + 10, y_izq + 25), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 1)
    texto_m = "m - Ocultar detalles" if mostrar_detalles else "m - Mostrar mas detalles"
    col_m = (0, 255, 0) if mostrar_detalles else (180, 180, 180)
    cv2.putText(frame, texto_m, (x_izq + 10, y_izq + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_m, 1)

    # Textos tarjeta derecha
    cv2.putText(frame, f"Rol: {rol.upper()}", (x_der + 12, y_der + 25), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 0), 1)
    texto_o = "o - Ocultar opciones" if mostrar_opciones else "o - Opciones"
    col_o = (0, 255, 255) if mostrar_opciones else (180, 180, 180)
    cv2.putText(frame, texto_o, (x_der + 12, y_der + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col_o, 1)

    # Texto de la barra inferior con boton Q para volver y D para depurar
    cv2.putText(frame, "[Q] Volver al menu  |  [m] Estadisticas  |  [o] Opciones  |  [D] Debug",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)

    # Despliegue panel de estadisticas
    if mostrar_detalles:
        base_y = y_izq + alto_card + 26
        cv2.putText(frame, "--- ESTADISTICAS DE USUARIO ---", (x_izq + 15, base_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 255), 1)

        inicios = stats.get("inicios_sesion", 1)
        tiempo_acum = stats.get("tiempo_acumulado_segundos", 0.0) + duracion_actual
        tiempo_medio = tiempo_acum / max(1, inicios)
        ultimo_acc = stats.get("ultimo_acceso") or "Primer inicio"
        intrusos = stats.get("intrusos_ultima_sesion", 0)
        sim_media = similitud_sesion if similitud_sesion is not None else stats.get("similitud_media", 0.0)

        items = [
            ("Rol asignado:", str(rol), (0, 255, 255)),
            ("Tiempo de sesion:", formatear_duracion(duracion_actual), (0, 255, 255)),
            ("Inicios de sesion:", f"{inicios} veces", (0, 255, 255)),
            ("Similitud facial media:", f"{sim_media:.1f}%", (0, 255, 0) if sim_media >= 80 else (0, 165, 255)),
            ("Intrusos ult. sesion:", f"{intrusos} detectados" if intrusos > 0 else "0 (Ninguno)", (0, 0, 255) if intrusos > 0 else (0, 255, 0)),
            ("Ultimo acceso:", str(ultimo_acc), (0, 255, 255)),
            ("Tiempo acumulado:", formatear_duracion(tiempo_acum), (0, 255, 255)),
            ("Promedio por sesion:", formatear_duracion(tiempo_medio), (0, 255, 255)),
        ]

        if tiempo_ausente > 0:
            items.append(("Estado presencia:", f"AUSENTE ({int(tiempo_ausente)}s)", (0, 0, 255)))
        else:
            items.append(("Estado presencia:", "PRESENTE", (0, 255, 0)))

        y_offset = base_y + 22
        for etiqueta, valor, col_val in items:
            cv2.putText(frame, etiqueta, (x_izq + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
            cv2.putText(frame, valor, (x_izq + 215, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.40, col_val, 1)
            y_offset += 21

    # Despliegue panel de opciones
    if mostrar_opciones:
        x_p = w - 350 - 15
        y_p = y_der + alto_card + 4
        cv2.putText(frame, f"--- OPCIONES [{rol.upper()}] ---", (x_p + 15, y_p + 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 255, 0), 1)

        if rol == "Total":
            opciones = [
                "[1] Cambiar mi contraseña",
                "[2] Cambiar contraseña de otro",
                "[3] Estadisticas de otros usuarios",
                "[4] Borrar mi usuario",
                "[5] Borrar otro usuario"
            ]
        elif rol == "Operador":
            opciones = [
                "[1] Cambiar mi contraseña",
                "[2] Borrar mi usuario"
            ]
        else:
            opciones = [
                "Rol Invitado: Sin opciones disp.",
                "Consulta tus estadisticas con [m]"
            ]

        y_opt = y_p + 58
        for opt in opciones:
            cv2.putText(frame, opt, (x_p + 15, y_opt), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (240, 240, 240), 1)
            y_opt += 28

        cv2.putText(frame, "[Q / ESC] Volver atras  |  [o] Ocultar", (x_p + 15, y_p + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

def capturar_texto_pantalla(camara, titulo, prompt, es_password=False, max_caracteres=24, valor_inicial=""):
    """
    Despliega un modal interactivo en pantalla y captura texto mediante teclado.
    Retorna la cadena ingresada o None si el usuario cancela con ESC.
    """
    texto = valor_inicial
    mensaje_error = ""

    while True:
        ret, frame = camara.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        ancho_modal = 480
        alto_modal = 195
        x1 = (w - ancho_modal) // 2
        y1 = (h - alto_modal) // 2
        x2 = x1 + ancho_modal
        y2 = y1 + alto_modal

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

        # Encabezado
        cv2.putText(frame, titulo, (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 255), 1)
        cv2.line(frame, (x1 + 20, y1 + 45), (x2 - 20, y1 + 45), (100, 100, 100), 1)

        # Instruccion
        cv2.putText(frame, prompt, (x1 + 20, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # Campo editable
        campo_y1 = y1 + 92
        campo_y2 = campo_y1 + 38
        cv2.rectangle(frame, (x1 + 20, campo_y1), (x2 - 20, campo_y2), (40, 40, 45), -1)
        cv2.rectangle(frame, (x1 + 20, campo_y1), (x2 - 20, campo_y2), (0, 255, 0), 1)

        display_text = ("*" * len(texto)) if es_password else texto
        cursor = "_" if (int(time.time() * 2) % 2 == 0) else ""
        cv2.putText(frame, f"{display_text}{cursor}", (x1 + 30, campo_y1 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1)

        if mensaje_error:
            cv2.putText(frame, mensaje_error, (x1 + 20, y1 + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "[ENTER] Confirmar   |   [ESC] Volver atras", (x1 + 20, y1 + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        cv2.imshow("Autenticador Facial", frame)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == 27:  # ESC vuelve a la pantalla previa
            return None
        elif tecla == 13 or tecla == 10:  # ENTER
            return texto.strip()
        elif tecla == 8:  # Backspace
            texto = texto[:-1]
            mensaje_error = ""
        elif 32 <= tecla <= 126:
            if len(texto) < max_caracteres:
                texto += chr(tecla)
                mensaje_error = ""

def seleccionar_opcion_pantalla(camara, titulo, opciones, subtitulo="Selecciona una opcion con el teclado"):
    """
    Presenta un menu de opciones enumeradas en pantalla.
    Retorna la clave seleccionada o None si cancela o pulsa Q / ESC.
    """
    while True:
        ret, frame = camara.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        ancho_modal = 500
        alto_modal = 75 + len(opciones) * 36 + 45
        x1 = (w - ancho_modal) // 2
        y1 = (h - alto_modal) // 2
        x2 = x1 + ancho_modal
        y2 = y1 + alto_modal

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        cv2.putText(frame, titulo, (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 255), 1)
        cv2.line(frame, (x1 + 20, y1 + 45), (x2 - 20, y1 + 45), (100, 100, 100), 1)

        y_line = y1 + 75
        teclas_validas = []
        for clave, etiqueta in opciones:
            teclas_validas.append(clave.lower())
            cv2.putText(frame, f"[{clave}] {etiqueta}", (x1 + 25, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
            y_line += 34

        cv2.putText(frame, "[Q / ESC] Volver a la pantalla anterior", (x1 + 20, y2 - 16), cv2.FONT_HERSHEY_DUPLEX, 0.43, (0, 255, 255), 1)

        cv2.imshow("Autenticador Facial", frame)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == 27 or tecla == ord('q') or tecla == ord('Q'):
            return "q"
        char = chr(tecla).lower() if 32 <= tecla <= 126 else ""
        if char in teclas_validas:
            return char

def seleccionar_usuario_pantalla(camara, lista_usuarios, titulo="SELECCIONAR USUARIO"):
    """
    Muestra modal para seleccionar un usuario de la lista.
    Retorna la tupla (nombre, rol) o None si cancela o pulsa Q / ESC.
    """
    if not lista_usuarios:
        mostrar_mensaje_pantalla(camara, "SIN USUARIOS", ["No hay usuarios registrados en el sistema."], color=(0, 165, 255), duracion_ms=1800)
        return None

    while True:
        ret, frame = camara.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        ancho_modal = 500
        alto_modal = 70 + min(len(lista_usuarios), 8) * 32 + 48
        x1 = (w - ancho_modal) // 2
        y1 = (h - alto_modal) // 2
        x2 = x1 + ancho_modal
        y2 = y1 + alto_modal

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        cv2.putText(frame, titulo, (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 255), 1)
        cv2.line(frame, (x1 + 20, y1 + 45), (x2 - 20, y1 + 45), (100, 100, 100), 1)

        y_line = y1 + 75
        for i, (usr, rol) in enumerate(lista_usuarios[:8]):
            cv2.putText(frame, f"[{i + 1}] {usr.upper()}", (x1 + 25, y_line), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 0), 1)
            cv2.putText(frame, f"Rol: {rol}", (x1 + 260, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
            y_line += 32

        cv2.putText(frame, "Pulsa el numero  |  [Q / ESC] Volver", (x1 + 20, y2 - 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.43, (0, 255, 255), 1)

        cv2.imshow("Autenticador Facial", frame)
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == 27 or tecla == ord('q') or tecla == ord('Q'):
            return None
        if ord('1') <= tecla <= ord('8'):
            idx = tecla - ord('1')
            if idx < len(lista_usuarios):
                return lista_usuarios[idx]

def mostrar_estadisticas_usuario_modal(camara, nombre, rol, stats):
    """Muestra un modal con las estadisticas completas del usuario seleccionado."""
    while True:
        ret, frame = camara.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        ancho_modal = 500
        alto_modal = 280
        x1 = (w - ancho_modal) // 2
        y1 = (h - alto_modal) // 2
        x2 = x1 + ancho_modal
        y2 = y1 + alto_modal

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        cv2.putText(frame, f"ESTADISTICAS: {nombre.upper()}", (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.56, (0, 255, 255), 1)
        cv2.line(frame, (x1 + 20, y1 + 45), (x2 - 20, y1 + 45), (100, 100, 100), 1)

        inicios = stats.get("inicios_sesion", 0)
        tiempo_acum = stats.get("tiempo_acumulado_segundos", 0.0)
        tiempo_medio = tiempo_acum / max(1, inicios)
        ultimo_acc = stats.get("ultimo_acceso") or "Sin registros"
        intrusos = stats.get("intrusos_ultima_sesion", 0)
        sim_media = stats.get("similitud_media", 0.0)

        items = [
            ("Rol asignado:", str(rol), (0, 255, 255)),
            ("Inicios de sesion:", f"{inicios} veces", (0, 255, 255)),
            ("Tiempo acumulado total:", formatear_duracion(tiempo_acum), (0, 255, 255)),
            ("Promedio por sesion:", formatear_duracion(tiempo_medio), (0, 255, 255)),
            ("Similitud facial media:", f"{sim_media:.1f}%", (0, 255, 0) if sim_media >= 80 else (0, 165, 255)),
            ("Intrusos ult. sesion:", f"{intrusos} detectados" if intrusos > 0 else "0 (Ninguno)", (0, 0, 255) if intrusos > 0 else (0, 255, 0)),
            ("Ultima conexion:", str(ultimo_acc), (0, 255, 255)),
        ]

        y_offset = y1 + 75
        for etiqueta, valor, col_val in items:
            cv2.putText(frame, etiqueta, (x1 + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            cv2.putText(frame, valor, (x1 + 240, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.42, col_val, 1)
            y_offset += 24

        cv2.putText(frame, "[Q / ESC / Cualquier tecla] Volver", (x1 + 20, y2 - 16), cv2.FONT_HERSHEY_DUPLEX, 0.43, (0, 255, 255), 1)

        cv2.imshow("Autenticador Facial", frame)
        tecla = cv2.waitKey(25) & 0xFF
        if tecla != 255:
            break

def mostrar_mensaje_pantalla(camara, titulo, lineas_mensaje, color=(0, 255, 0), duracion_ms=1800):
    """Muestra un modal informativo en pantalla durante un tiempo determinado."""
    t_inicio = time.time()
    duracion_s = duracion_ms / 1000.0

    while (time.time() - t_inicio) < duracion_s:
        ret, frame = camara.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        ancho_modal = 480
        alto_modal = 70 + len(lineas_mensaje) * 30 + 35
        x1 = (w - ancho_modal) // 2
        y1 = (h - alto_modal) // 2
        x2 = x1 + ancho_modal
        y2 = y1 + alto_modal

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        cv2.putText(frame, titulo, (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.58, color, 1)
        cv2.line(frame, (x1 + 20, y1 + 45), (x2 - 20, y1 + 45), (100, 100, 100), 1)

        y_line = y1 + 75
        for linea in lineas_mensaje:
            cv2.putText(frame, linea, (x1 + 25, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1)
            y_line += 28

        cv2.imshow("Autenticador Facial", frame)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == 27 or tecla == ord('q') or tecla == ord('Q'):
            break
