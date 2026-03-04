FROM python:3.12-slim

# Dependencias del sistema: zeek, tcpdump, permisos de red
RUN apt-get update && apt-get install -y --no-install-recommends \
    tcpdump \
    libpcap-dev \
    gcc \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Instalar Zeek desde repositorio oficial
RUN echo 'deb http://download.opensuse.org/repositories/security:/zeek/Debian_12/ /' \
    > /etc/apt/sources.list.d/zeek.list && \
    curl -fsSL https://download.opensuse.org/repositories/security:zeek/Debian_12/Release.key \
    | gpg --dearmor > /etc/apt/trusted.gpg.d/zeek.gpg && \
    apt-get update && apt-get install -y --no-install-recommends zeek && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/zeek/bin:$PATH"

WORKDIR /app

# Instalar dependencias Python
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar codigo fuente
COPY src/ src/
COPY models/ models/
COPY frontend.py .
COPY style.css .
COPY run.py .

# Crear directorios necesarios
RUN mkdir -p logs/captura_trafico/stream logs/temp logs/zeek_temp

# Puertos: 8000 (API FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Iniciar API y frontend
CMD ["sh", "-c", "uvicorn src.api.api:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0 --server.headless true && wait"]
