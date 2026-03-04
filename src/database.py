"""Módulo centralizado de acceso a base de datos para el sistema IDS."""
import threading
import logging
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DB_URL = "sqlite:///ids.db"

_engine = None
_lock = threading.Lock()


def get_engine():
    """Devuelve un engine singleton thread-safe."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = create_engine(
                    DB_URL,
                    connect_args={"timeout": 30, "check_same_thread": False},
                    echo=False,
                )
    return _engine


def init_tables():
    """Crea las tablas si no existen. Se llama una vez al iniciar."""
    engine = get_engine()
    with engine.begin() as conn:
        # Tabla de amenazas detectadas
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS THREATS(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                srcip INTEGER,
                sport INTEGER,
                dstip INTEGER,
                dsport INTEGER,
                dur INTEGER,
                sbytes INTEGER,
                dbytes INTEGER,
                spkts INTEGER,
                dpkts INTEGER,
                proto TEXT,
                service TEXT,
                rate TEXT,
                prediccion TEXT,
                confianza INTEGER,
                tiempo TEXT,
                sesion_id INTEGER
            )
        """))

        # Tabla de sesiones de escaneo
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS SESIONES(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_inicio TEXT NOT NULL,
                fecha_fin TEXT,
                total_flujos INTEGER DEFAULT 0,
                total_amenazas INTEGER DEFAULT 0,
                tipos_ataques TEXT,
                estado TEXT DEFAULT 'activa',
                interfaz TEXT
            )
        """))

        # Migración: añadir sesion_id a THREATS si no existe (BD existentes)
        _migrate_threats_table(conn)

    logger.info("Tablas de base de datos inicializadas")


def _migrate_threats_table(conn):
    """Añade columnas faltantes a tablas existentes (para bases de datos antiguas)."""
    # Migrar THREATS: añadir sesion_id si no existe
    result = conn.execute(text("PRAGMA table_info(THREATS)"))
    columnas_threats = [row[1] for row in result.fetchall()]
    if 'sesion_id' not in columnas_threats:
        conn.execute(text("ALTER TABLE THREATS ADD COLUMN sesion_id INTEGER"))
        logger.info("Columna sesion_id añadida a tabla THREATS")

    # Migrar SESIONES: añadir tipos_ataques si no existe
    result = conn.execute(text("PRAGMA table_info(SESIONES)"))
    columnas_sesiones = [row[1] for row in result.fetchall()]
    if 'tipos_ataques' not in columnas_sesiones:
        conn.execute(text("ALTER TABLE SESIONES ADD COLUMN tipos_ataques TEXT"))
        logger.info("Columna tipos_ataques añadida a tabla SESIONES")
