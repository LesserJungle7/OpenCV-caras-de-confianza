# OpenCV - Caras de confianza

Sistema inteligente de control de acceso y vigilancia biométrica continua basado en visión artificial con **OpenCV** y **Python**. El proyecto implementa reconocimiento facial con normalización y alineación de ojos, retos dinámicos de prueba de vida (liveness detection mediante gestos faciales), control de acceso basado en roles (**Total**, **Operador** e **Invitado**), estadísticas en tiempo real y defensa proactiva contra intrusos y usurpaciones de sesión.

---

## Índice

1. [Características Principales](#características-principales)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos e Instalación](#requisitos-e-instalación)
4. [Guía de Uso](#guía-de-uso)
5. [Ejemplos de Flujos de Ejecución](#ejemplos-de-flujos-de-ejecución)
6. [Configuración del Sistema](#configuración-del-sistema)
7. [Licencia](#licencia)

---

## Características Principales

- **Interfaz visual interactiva sobre la cámara**: Toda la interacción (menús, modales de texto, contraseñas, confirmaciones y captura de poses) se realiza directamente sobre el flujo de vídeo de la webcam, prescindiendo del uso de la consola.
- **Menú de inicio rápido**: Al arrancar, el sistema muestra la cámara en vivo con opciones superiores para acceder al creador de perfiles `[1]`, al autenticador `[2]` o salir `[Q]`.
- **Registro facial interactivo (2 poses)**: Asistente visual guiado para capturar dos poses frontales estandarizadas (`frente_normal` y `frente_sonriendo`), aplicando alineación geométrica por ojos, recorte de márgenes y ecualización de luminosidad (CLAHE).
- **Control de acceso basado en roles**:
  - **Total (Administrador)**: Acceso completo, cambio de contraseña propia y de terceros, consulta de estadísticas globales de otros usuarios, y borrado de cuentas (propia o de otros).
  - **Operador**: Acceso estándar de trabajo, modificación de contraseña propia y autoeliminación de cuenta.
  - **Invitado**: Acceso temporal sin contraseña escrita. El sistema memoriza su rostro únicamente durante la sesión activa en memoria RAM; al concluir la sesión, **se eliminan automáticamente sus datos de `usuarios.json` y sus fotos de la carpeta de confianza**.
- **Contraseña facial (Prueba de vida / Liveness Detection)**: Reto secuencial de 3 gestos aleatorios con temporizador individual (levantar cejas, guiño/cierre ocular con factor K y sonrisa con discriminación colorimétrica de esmalte dental). Evita la suplantación mediante fotografías estáticas.
- **Feedback auditivo nativo**: Efectos sonoros inmediatos al validar cada gesto correctamente (`success.mp3`) o al agotar el tiempo/detectar intrusión (`error.mp3`).
- **HUD de estadísticas en tiempo real (`m`)**: Despliega un panel con el rol, tiempo de sesión, inicios de sesión acumulados, similitud facial media (Template Matching SQDIFF normalizado) e intentos de intrusión detectados desde la última sesión.
- **Panel de opciones contextuales (`o`)**: Panel simétrico al de estadísticas que permite gestionar credenciales y usuarios según los permisos del rol activo.
- **Guardia de ausencia y retorno anti-usurpación**: Si el usuario se ausenta más de 10 segundos, la sesión se bloquea. Si regresa dentro del margen de gracia, se valida su identidad frente a sus plantillas en RAM antes de reanudar el acceso.
- **Defensa ante intrusos**: Ante un rostro no reconocido o fallo en el reto de vida, captura evidencias fotográficas dobles en `recursos/imagenes/intrusos/` y bloquea la estación de trabajo a nivel de sistema operativo.
- **Navegación unificada con `Q`**: En todas las pantallas y modales se indica y permite regresar al nivel anterior mediante la tecla `Q` (o `ESC`), saliendo de la aplicación si se pulsa en la pantalla principal.

---

## Estructura del Proyecto

El proyecto está organizado siguiendo el estándar modular de Python (`src-layout`), separando el código fuente de los recursos multimedia y datos persistentes:

```text
OpenCV-caras-de-confianza/
│
├── main.py                     # Punto de entrada principal de la aplicación
├── config.py                   # Configuración global, umbrales y resolución de rutas
├── requirements.txt            # Dependencias de Python del proyecto
├── README.md                   # Documentación técnica y manual de usuario
├── .gitignore                  # Exclusiones de Git
├── LICENSE                     # Licencia del proyecto (MIT)
│
├── recursos/                   # Directorio contenedor de recursos y datos persistentes
│   ├── audios/                 # Archivos de audio de feedback (error.mp3, success.mp3)
│   ├── datos/                  # Base de datos local (usuarios.json)
│   ├── haarcascades/           # Clasificadores XML de OpenCV (rostro y ojos)
│   └── imagenes/
│       ├── confianza/          # Plantillas de rostros autorizados ({usuario}_{pose}.jpg)
│       └── intrusos/           # Fotografías de evidencia de accesos no autorizados
│
└── src/                        # Código fuente modularizado del sistema
    ├── __init__.py             # Inicializador de paquete y configuración de sys.path
    ├── autenticador.py         # Módulo de guardia continua, autenticación y sesión
    ├── creador_perfiles.py     # Asistente interactivo de registro facial y usuarios
    ├── depurador.py            # Modulo de depuracion visual, telemetria y pausas
    ├── gestos.py               # Evaluador de retos faciales y prueba de vida
    ├── interfaz.py             # Renderizado del HUD, paneles traslúcidos y modales
    ├── sistema.py              # Bloqueo del SO, evidencias y reproducción de audio
    ├── usuarios.py             # Gestión de credenciales, roles, estadísticas y borrado
    └── vision.py               # Detección facial, alineación de ojos y normalización
```

### Descripción de ficheros clave

- **`main.py`**: Archivo ejecutable raíz. Configura el entorno e inicia el ciclo visual principal de la cámara.
- **`config.py`**: Define las rutas relativas hacia `recursos/`, parámetros de normalización (120x120 px), umbrales de coincidencia, temporizadores de guardia (10s) y estilos gráficos.
- **`src/autenticador.py`**: Coordina la máquina de estados del sistema (Guardia inactiva, Verificación de contraseña, Reto de gestos, Sesión activa, Verificación de retorno y Opciones).
- **`src/creador_perfiles.py`**: Gestiona el alta interactiva de usuarios mediante modales en pantalla, captura guiada de poses y guardado de plantillas normalizadas.
- **`src/interfaz.py`**: Implementa los componentes visuales: barra superior de menú, tarjeta HUD de usuario (`m`), tarjeta de rol y opciones (`o`), modales de texto con teclado en pantalla y selectores numéricos.
- **`src/vision.py`**: Contiene la lógica de visión artificial: detección Haar Cascade, alineación de inclinación mediante vector ocular, ecualización CLAHE y comparación por Template Matching (`cv2.TM_SQDIFF_NORMED`).
- **`src/gestos.py`**: Algoritmos de segmentación de regiones de interés (ROIs) faciales para verificar apertura bucal/esmalte, posición de cejas y parpadeo/guiño (Regla K).
- **`src/usuarios.py`**: Módulo de persistencia JSON que administra usuarios, contraseñas, roles, registros de accesos y la función `eliminar_usuario` (borrado de JSON y fotos asociadas).
- **`src/sistema.py`**: Integración con el sistema operativo para bloquear la pantalla (`rundll32` en Windows, `systemctl` en Linux, `pmset` en macOS) y reproducir audio sin dependencias externas pesadas.

---

## Requisitos e Instalación

### Requisitos previos
- **Python 3.8** o superior (recomendado Python 3.10 / 3.11 / 3.12 / 3.13).
- **Webcam** funcional conectada al equipo.
- Sistema Operativo: **Windows**, **Linux** o **macOS**.

### Pasos de instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/LesserJungle7/OpenCV-caras-de-confianza.git
   cd OpenCV-caras-de-confianza
   ```

2. **Crear y activar un entorno virtual** (recomendado):
   ```bash
   # En Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # En Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Guía de Uso

Para iniciar la aplicación, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
python main.py
```

Al iniciarse, se abrirá la ventana de la cámara con la barra superior de opciones:
- Pulsa **`1`**: Abre el **Creador de perfiles** interactivo.
- Pulsa **`2`**: Inicia el **Autenticador** en modo guardia.
- Pulsa **`Q`** o **`ESC`**: Cierra la aplicación.

---

## Ejemplos de Flujos de Ejecución

### Flujo 1: Registro de un nuevo usuario de confianza (Rol Total u Operador)
1. En la pantalla principal, pulsa **`1`**.
2. Aparecerá el modal en pantalla solicitando la contraseña de administrador (por defecto: `admin123`). Escríbela y pulsa `ENTER`.
3. En el menú de administración, pulsa **`1`** (*Registrar nuevo usuario de confianza*).
4. Introduce el nombre de usuario (ej. `carlos`) y pulsa `ENTER`.
5. Selecciona el rol deseado pulsando **`1`** (*Total*), **`2`** (*Operador*) o **`3`** (*Invitado*).
6. Si seleccionaste `Total` u `Operador`, introduce la contraseña deseada para el usuario en el modal interactivo.
7. Comienza la captura de poses guiada:
   - Sitúate frente a la cámara mirando de frente en posición neutra y pulsa `ESPACIO`.
   - Sonríe mirando a la cámara y pulsa `ESPACIO`.
8. El sistema normaliza los rostros, guarda las imágenes en `recursos/imagenes/confianza/` y registra al usuario en `recursos/datos/usuarios.json`.
9. Pulsa **`3`** o **`Q`** para volver al menú principal.

---

### Flujo 2: Autenticación, reto facial y sesión activa
1. En la pantalla principal, pulsa **`2`** para entrar en el modo guardia (`MODO GUARDIA: VIGILANDO`).
2. Colócate frente a la cámara. El sistema detectará tu rostro y mostrará la barra de estabilización facial.
3. Una vez estabilizado, el sistema te identificará y solicitará tu contraseña mediante un modal en pantalla:
   - Introduce tu contraseña y pulsa `ENTER`.
4. Si la contraseña es correcta, dará comienzo el **reto de contraseña facial**:
   - Sigue las instrucciones en pantalla completando los 3 gestos aleatorios indicados (ej. levantar cejas, cerrar ojo izquierdo, sonreír).
   - Con cada gesto completado escucharás el sonido de confirmación (`success.mp3`).
5. Tras superar los gestos, se abrirá la **sesión activa**:
   - En la esquina superior izquierda verás tu nombre de usuario y la opción `m - Mostrar más detalles`.
   - En la esquina superior derecha verás tu rol asignado y la opción `o - Opciones`.
   - Puedes pulsar **`m`** para consultar tus estadísticas detalladas de sesión.
6. Pulsa **`Q`** para cerrar la sesión y volver al menú principal.

---

### Flujo 3: Acceso como Invitado y borrado automático
1. En el modo guardia, sitúate frente a la cámara:
   - Si tu usuario fue registrado previamente con rol **Invitado**, el sistema te reconocerá y **no te pedirá contraseña escrita**, pasando directamente al reto facial.
   - También puedes pulsar **`I`** en modo guardia para acceder como invitado temporal.
2. Completa los 3 gestos faciales requeridos.
3. El sistema almacena tu plantilla facial **únicamente en memoria RAM** durante la sesión.
4. **Ciclo de olvido y borrado automático**:
   - Si te ausentas durante más de 10 segundos o si pulsas **`Q`** para cerrar sesión, el sistema concluye la sesión.
   - Automáticamente, **el usuario es eliminado de `usuarios.json` y sus dos fotografías son borradas de la carpeta `recursos/imagenes/confianza/`**, dejando el sistema limpio.

---

### Flujo 4: Gestión y borrado de usuarios desde el menú de opciones (`o`)
1. Inicia sesión con un usuario con rol **Total**.
2. Pulsa **`o`** para desplegar el panel de opciones a la derecha de la pantalla:
   - Pulsa **`1`**: Cambiar tu propia contraseña.
   - Pulsa **`2`**: Cambiar la contraseña de otro usuario registrado.
   - Pulsa **`3`**: Ver las estadísticas completas de otro usuario.
   - Pulsa **`4`**: **Borrar tu propio usuario**. Tras confirmar con `1`, el sistema elimina tu cuenta de `usuarios.json`, borra tus fotos y **cierra la aplicación**.
   - Pulsa **`5`**: **Borrar otro usuario**. Selecciona de la lista al usuario que deseas eliminar y confirma con `1`. El usuario y sus fotos se eliminarán del disco y tu sesión continuará activa.
3. Si inicias sesión con un usuario **Operador**, podrás cambiar tu contraseña (`1`) o borrar tu propio usuario cerrando la aplicación (`2`).

---

### Flujo 5: Detección de intrusos y guardia de ausencia
- **Persona no autorizada**: Si un rostro desconocido permanece frente a la cámara en modo guardia o si se intenta usurpar la sesión activa de otro usuario, el sistema emitirá el audio de alerta (`error.mp3`), guardará dos fotografías consecutivas de evidencia en `recursos/imagenes/intrusos/` y suspenderá/bloqueará inmediatamente el equipo.
- **Ausencia prolongada**: Si el usuario autorizado se retira del encuadre durante una sesión activa, un banner inferior mostrará la cuenta atrás de 10 segundos. Si regresa antes de que expire, el sistema verifica sus facciones y reanuda la sesión; si transcurren los 10 segundos, la sesión se cierra automáticamente.

---

## Configuración del Sistema

Los parámetros del sistema pueden ajustarse en [`config.py`](config.py):

| Parámetro | Valor por defecto | Descripción |
| :--- | :--- | :--- |
| `TIEMPO_AUSENCIA_GUARDIA` | `10` | Segundos de ausencia tolerados antes de bloquear la sesión |
| `UMBRAL_COINCIDENCIA` | `0.07` | Umbral SQDIFF normalizado para identificación facial (menor valor = mayor exigencia) |
| `UMBRAL_REENGANCHE_SESION` | `0.07` | Umbral SQDIFF para validar el retorno tras una ausencia breve |
| `FRAMES_ESTABILIDAD_GUARDIA` | `12` | Fotogramas consecutivos con rostro estático requeridos antes de procesar |
| `TAM_CARA` | `(120, 120)` | Resolución estándar en píxeles para el recorte y normalización facial |
| `ALINEAR_OJOS` | `True` | Activa/desactiva la alineación rotacional basada en centros oculares |
| `TIEMPO_MAXIMO_GESTO` | `30.0` | Tiempo límite en segundos para completar cada gesto del reto de vida |
| `FACTOR_K_CIERRE_OJO` | `0.78` | Factor de compresión vertical respecto al estado neutro para detectar ojo cerrado |

---

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Consulta el archivo `LICENSE` para más información.
