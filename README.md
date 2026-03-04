# IDS con Machine Learning

**Autor:** Juan Antonio Carlos Balsalobre

---

## Características

- Captura y análisis de tráfico de red en tiempo real
- Análisis de archivos PCAP mediante Zeek
- Dashboard web interactivo con Streamlit
- API REST con FastAPI
- Historial de sesiones con exportación CSV/JSON
- Detección de ataques: PortScan, FTP-Patator, SSH-Patator, DoS, DDoS, entre otros

---

## Requisitos previos

### Opción A — Docker (recomendado)
- [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/)
- Git con soporte LFS: `sudo apt-get install git-lfs`

### Opción B — Instalación manual
- Python 3.12
- [Zeek](https://zeek.org/get-zeek/) instalado y disponible en el PATH
- `tcpdump` y `libpcap-dev`
- Git con soporte LFS: `sudo apt-get install git-lfs`

---

## Instalación

### 1. Clonar el repositorio

> Los modelos de ML se almacenan con Git LFS. Asegúrate de tener `git-lfs` instalado antes de clonar.

```bash
git lfs install
git clone https://github.com/juanan-00/tfg-ids.git
cd tfg-ids
```

---

### Opción A — Docker

```bash
docker-compose up --build
```

Esto levanta la API y el frontend automáticamente. Una vez iniciado:

| Servicio | URL |
|----------|-----|
| API REST | http://localhost:8000 |
| Documentación API | http://localhost:8000/docs |
| Dashboard (Streamlit) | http://localhost:8501 |

> La captura de tráfico en tiempo real requiere permisos de red. El `docker-compose.yml` ya incluye las capacidades necesarias (`NET_RAW`, `NET_ADMIN`).

---

### Opción B — Instalación manual

#### 1. Instalar dependencias del sistema

```bash
sudo apt-get install tcpdump libpcap-dev
```

Para instalar Zeek en Ubuntu/Debian:
```bash
echo 'deb http://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/ /' | sudo tee /etc/apt/sources.list.d/zeek.list
curl -fsSL https://download.opensuse.org/repositories/security:zeek/xUbuntu_24.04/Release.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/zeek.gpg
sudo apt-get update && sudo apt-get install zeek
```

#### 2. Crear entorno virtual e instalar dependencias Python

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-docker.txt
```

#### 3. Iniciar la API

```bash
sudo python run.py --api
```

> Se necesita `sudo` para la captura de paquetes con Scapy.

#### 4. Iniciar el frontend (en otra terminal)

```bash
source venv/bin/activate
streamlit run frontend.py
```

| Servicio | URL |
|----------|-----|
| API REST | http://localhost:8000 |
| Documentación API | http://localhost:8000/docs |
| Dashboard (Streamlit) | http://localhost:8501 |

---

## Uso

### Monitoreo en tiempo real

1. Accede al dashboard en http://localhost:8501
2. Selecciona la interfaz de red en el panel lateral
3. Pulsa **Iniciar captura** para comenzar el análisis
4. Las predicciones y alertas se actualizan automáticamente cada 2 segundos

### Análisis de archivos PCAP

1. Ve a la sección **Análisis de PCAP** en el dashboard
2. Sube un archivo `.pcap` o `.pcapng`
3. El sistema procesará el archivo con Zeek y devolverá las predicciones

### API REST

La API expone los siguientes endpoints principales:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Estado de la API |
| GET | `/api/v1/interfaces` | Interfaces de red disponibles |
| POST | `/start_escaner` | Iniciar captura de tráfico |
| POST | `/stop_escaner` | Detener captura |
| GET | `/api/v1/last` | Último batch de predicciones |
| GET | `/api/v1/threats` | Amenazas detectadas |
| GET | `/api/v1/sesiones` | Historial de sesiones |
| POST | `/analizar_pcap` | Analizar archivo PCAP |

Documentación interactiva completa disponible en http://localhost:8000/docs

---

## Estructura del proyecto

```
tfg-ids/
├── src/
│   ├── api/
│   │   └── api.py              # API REST (FastAPI)
│   ├── modelos/
│   │   ├── ModeloRandomForest.py
│   │   ├── ModeloDecisionTree.py
│   │   ├── ModeloKNN.py
│   │   ├── ModeloNaiveBayes.py
│   │   └── modeloSVM.py
│   ├── database.py             # Base de datos SQLite
│   ├── escanear_trafico_red.py # Captura de tráfico
│   └── procesar_predicciones.py
├── models/
│   ├── modelo_rf_cic.joblib    # Modelo CIC-IDS2017 (tráfico en tiempo real)
│   └── modelo_rf_unsw.joblib   # Modelo UNSW-NB15 (análisis PCAP)
├── frontend.py                 # Dashboard Streamlit
├── run.py                      # Punto de entrada
├── Dockerfile
├── docker-compose.yml
└── requirements-docker.txt
```

---

## Modelos de Machine Learning

| Modelo | Dataset | Uso | Algoritmo |
|--------|---------|-----|-----------|
| `modelo_rf_cic.joblib` | CIC-IDS2017 | Tráfico en tiempo real | Random Forest |
| `modelo_rf_unsw.joblib` | UNSW-NB15 | Análisis de archivos PCAP | Random Forest |

Los modelos se almacenan con **Git LFS** por su tamaño. Si los archivos no se descargan correctamente al clonar, ejecuta:

```bash
git lfs pull
```
