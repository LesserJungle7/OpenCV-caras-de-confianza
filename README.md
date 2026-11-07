# OpenCV Trusted Faces

Terminal-based facial recognition and user access control system developed with OpenCV and Python. The project provides biometric profile registration and continuous monitoring to secure workstations against unauthorized access.

## Features

- Facial Profile Registration: Administrative assistant to capture multi-angle facial reference templates (front normal, smiling, left and right profiles).
- Role-Based Access: Assigns permission levels (Total, Operador, Invitado) stored locally in JSON format.
- Continuous Guard Mode: Actively monitors user presence in front of the camera.
- Intruder Defense: Automatically captures evidence frames and locks the workstation when an unauthorized face is detected.
- Absence Auto-Lock: Deactivates session and reactivates guard mode when the authenticated user is absent for more than 10 seconds.

## Prerequisites

- Python 3.8 or higher
- Webcam device
- Operating System: Windows, Linux, or macOS

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LesserJungle7/OpenCV-caras-de-confianza.git
   cd OpenCV-caras-de-confianza
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify that the Haar Cascade classifier exists at:
   `haarcascades/haarcascade_frontalface_default.xml`

## Project Structure

- `autenticador.py`: Main entry point for continuous surveillance and authentication.
- `creador_perfiles.py`: Administrative CLI utility to register users and facial reference templates.
- `config.py`: Centralized configuration (directories, threshold values, visual styles).
- `vision.py`: Face detection and template matching routines using OpenCV.
- `usuarios.py`: Data persistence and credential management (`datos/usuarios.json`).
- `sistema.py`: Platform-specific screen locking and intruder evidence capture.

## Usage Guide

### 1. User and Profile Registration

Run the profile management utility:
```bash
python creador_perfiles.py
```

1. Enter the administrator password (default: `admin123`).
2. Select Option 1 to register a new user.
3. Provide username, password, and select an access role:
   - 1: Total (full access to configuration, terminal, and restricted files)
   - 2: Operador (standard application execution privileges)
   - 3: Invitado (read-only restricted access)
4. Follow the on-screen instructions to capture the 4 required poses. Press SPACE to capture each frame or 'q' to abort.

### 2. Guard and Authentication Mode

Launch the authentication service:
```bash
python autenticador.py
```

- When a registered face is detected, the terminal prompts for the user password.
- Upon successful authentication, the session becomes active and permissions are granted.
- If an unrecognized face is detected, photographic evidence is stored in `imagenes/intrusos/` and the workstation is locked.
- If the authenticated user is absent for more than 10 seconds (configurable via `TIEMPO_AUSENCIA_GUARDIA`), the session locks and returns to guard mode.
- Press 'q' while focusing on the camera feed window to terminate the application.

## Configuration

Operational settings can be customized in `config.py`:
- `UMBRAL_COINCIDENCIA`: Matching threshold for template comparison (default: `0.22`). Lower values require closer resemblance.
- `TIEMPO_AUSENCIA_GUARDIA`: Inactivity time in seconds without detecting a user before reactivating guard mode (default: `10`).
- `CARPETA_CONFIANZA`: Storage directory for trusted face images (`imagenes/confianza`).
- `CARPETA_INTRUSOS`: Storage directory for intruder evidence (`imagenes/intrusos`).
- `ARCHIVO_CONFIG`: Path to the user registry file (`datos/usuarios.json`).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
