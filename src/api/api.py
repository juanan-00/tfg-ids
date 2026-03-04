import os
import glob
import tempfile
import shutil
import socket
import fcntl
import struct
import pandas as pd
from collections import deque
from datetime import datetime
import logging

from fastapi import FastAPI, UploadFile, File, Query, WebSocket, WebSocketDisconnect
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.responses import StreamingResponse
from sqlalchemy import text
import io

from src.escanear_trafico_red import EscanearTraficoRed
from src.transformar_datos_multilog import TransformarDatos
from src.modelos.ModeloRandomForest import ModeloRandomForest
from src.procesar_predicciones import extract_cic_features_from_columns
from cicflowmeter.sniffer import create_sniffer as create_sniffer_cic
from src.database import get_engine, init_tables
from src.ws_manager import ws_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api-loger")

app = FastAPI(title="IDS API", version="1.0.0")

# Buffer circular de logs para el frontend (máximo 500 entradas)
log_buffer = deque(maxlen=500)

def log_with_buffer(msg):
    """Callback que loguea y guarda en el buffer para el frontend."""
    logger.info(msg)
    log_buffer.append({"timestamp": datetime.now().isoformat(), "message": msg})

DIR_DATOS = "data"
LOGS_DIR = "logs/captura_trafico"
ZEEK_LOGS_DIR = "logs/zeek_stream"

os.makedirs(ZEEK_LOGS_DIR, exist_ok=True)

# Una sola instancia del escaner
escanner = EscanearTraficoRed(
    interface=EscanearTraficoRed.get_interface_default() or "eth0",
    zeek_logs_dir=ZEEK_LOGS_DIR,
    pcap_file="stream.pcap"
)

# ID de la sesión de escaneo activa
_sesion_actual_id = None


@app.on_event("startup")
async def on_startup():
    """Inicializa BD y guarda referencia al event loop para broadcast WebSocket."""
    import asyncio
    init_tables()
    ws_manager._loop = asyncio.get_event_loop()


@app.get("/")
async def root():
    """Endpoint raiz con informacion de la API"""
    return {
        "message": "IDS ML API",
        "version": "1.0.0",
        "status": "running",
        "interface": escanner.interface,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "start_scan": "/start_escaner",
            "stop_scan": "/stop_escaner",
            "analyze_pcap": "/analizar_pcap",
            "last_predictions": "/api/v1/last",
            "threats": "/api/v1/threats",
            "interfaces": "/api/v1/interfaces",
            "sessions": "/api/v1/sesiones",
            "export": "/api/v1/export",
            "ws_alerts": "/ws/alertas",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "interfaz": escanner.interface,
        "scanning": escanner.escanear_activo,
        "timestamp": datetime.now().ctime()
    }


@app.get("/api/v1/last")
async def obtener_ultimo_csv():
    try:
        csv_files = glob.glob(f"{LOGS_DIR}/stream/last_pred.csv")
        if not csv_files:
            return {
                "filename": None,
                "path": None,
                "rows": 0,
                "columns": [],
                "data": []
            }

        last_csv = csv_files[0]
        logger.info(last_csv)

        if os.path.getsize(last_csv) == 0:
            return {
                "filename": os.path.basename(last_csv),
                "path": last_csv,
                "rows": 0,
                "columns": [],
                "data": []
            }

        df = pd.read_csv(last_csv)
        if df.empty:
            return {
                "filename": os.path.basename(last_csv),
                "path": last_csv,
                "rows": 0,
                "columns": df.columns.tolist(),
                "data": []
            }
        df = df.fillna(0)

        return {
            "filename": os.path.basename(last_csv),
            "path": last_csv,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "data": df.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error obteniendo ultimo csv: {e}")
        return {
            "filename": None,
            "path": None,
            "rows": 0,
            "columns": [],
            "data": []
        }


@app.get("/api/v1/threats")
async def obtener_threats():
    """Devuelve las amenazas del último batch"""
    try:
        threats_path = f"{LOGS_DIR}/stream/last_threats.csv"
        if not os.path.exists(threats_path) or os.path.getsize(threats_path) == 0:
            return {"rows": 0, "data": []}
        df = pd.read_csv(threats_path, on_bad_lines='skip')
        df = df.fillna(0)
        return {
            "rows": len(df),
            "data": df.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error obteniendo threats: {e}")
        return {"rows": 0, "data": []}


@app.post("/start_escaner")
async def escanear_trafico_red(tipo_modelo: str = "cic-ids2017"):
    global _sesion_actual_id
    logger.info(f"Inicio de escaneo de trafico con modelo: {tipo_modelo}")

    # Crear nueva sesión en BD
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO SESIONES (fecha_inicio, estado, interfaz) VALUES (:fi, 'activa', :iface)"
        ), {"fi": datetime.now().isoformat(), "iface": escanner.interface})
        result = conn.execute(text("SELECT last_insert_rowid()"))
        _sesion_actual_id = result.scalar()

    escanner.continous_capture_cic(
        log_callback=log_with_buffer,
        sesion_id=_sesion_actual_id,
        ws_manager=ws_manager
    )

    return {
        "status": "escaneo trafico red inicializado",
        "modelo": tipo_modelo,
        "sesion_id": _sesion_actual_id
    }


@app.post("/analizar_pcap")
async def analizar_pcap_file(
    file: UploadFile = File(...),
    tipo_modelo: str = Query("unsw-nb15", pattern="^(unsw-nb15|cic-ids2017)$")
):
    """Endpoint para analizar archivos PCAP y convertirlos a CSV.

    - tipo_modelo=unsw-nb15: usa Zeek para extraer features y modelo UNSW-NB15
    - tipo_modelo=cic-ids2017: usa CICFlowMeter para extraer features y modelo CIC-IDS2017
    """
    temp_pcap_path = None
    temp_cic_csv = None
    tipo_modelo = "cic-ids2017"
    try:
        logger.info(f"Analizando archivo pcap: {file.filename} con modelo {tipo_modelo}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pcap')
        temp_pcap_path = temp_file.name
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        nombre_base = os.path.splitext(file.filename)[0]

        if tipo_modelo == "cic-ids2017":
            df = _analizar_pcap_cic(temp_pcap_path, nombre_base, timestamp)
        else:
            df = _analizar_pcap_unsw(temp_pcap_path, nombre_base, timestamp)

        df = df.fillna(0)
        logger.info(f"Predicciones {tipo_modelo} completadas: {len(df)} filas")

        return {
            "status": "success",
            "filename": file.filename,
            "modelo": tipo_modelo,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "data": df.to_dict('records')
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al analizar PCAP: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar PCAP: {str(e)}")
    finally:
        if temp_pcap_path and os.path.exists(temp_pcap_path):
            try:
                os.remove(temp_pcap_path)
            except Exception:
                pass


def _analizar_pcap_unsw(pcap_path: str, nombre_base: str, timestamp: str) -> pd.DataFrame:
    """Pipeline UNSW-NB15: Zeek → CSV → predicción."""
    os.makedirs("logs/zeek_temp", exist_ok=True)
    temp_csv_path = f"logs/temp/{nombre_base}.csv"
    zeek_logs_dir = f"logs/zeek_temp/zeek_{timestamp}"
    os.makedirs(zeek_logs_dir, exist_ok=True)

    escanner.zeek_logs_dir = zeek_logs_dir
    escanner.procesar_pcap_file(pcap_path)
    logger.info(f"Procesamiento con Zeek completado en: {zeek_logs_dir}")

    transformer = TransformarDatos()
    transformer.logs_dir = zeek_logs_dir
    transformer.output_path = temp_csv_path

    csv_file = transformer.generate_csv_zeek(zeek_logs_dir)
    logger.info(f"CSV generado: {csv_file}")

    if not os.path.exists(csv_file):
        raise HTTPException(status_code=500, detail="No se pudo generar el archivo CSV con Zeek")

    df = pd.read_csv(csv_file)
    modelo_obj = MODELOS_DISPONIBLES["unsw-nb15"]
    features = modelo_obj.features_seleccionadas
    batch_df = df[features].copy()
    batch_df = modelo_obj.preprocesar_datos_unsw(batch_df, train=False)
    y_pred, y_proba = modelo_obj.prediccion_real2(batch_df, modelo_obj.modelo)
    df['prediccion'] = y_pred

    return df


def _analizar_pcap_cic(pcap_path: str, nombre_base: str, timestamp: str) -> pd.DataFrame:
    """Pipeline CIC-IDS2017: CICFlowMeter → CSV → predicción."""
    os.makedirs("logs/temp", exist_ok=True)
    cic_csv_path = f"logs/temp/{nombre_base}_cic_{timestamp}.csv"

    # Ejecutar CICFlowMeter sobre el PCAP
    sniffer = create_sniffer_cic(
        input_file=pcap_path,
        input_interface=None,
        output_mode="csv",
        output=cic_csv_path,
        verbose=False
    )
    sniffer.start()
    sniffer.join(timeout=120)
    logger.info(f"CICFlowMeter completado: {cic_csv_path}")

    if not os.path.exists(cic_csv_path) or os.path.getsize(cic_csv_path) == 0:
        raise HTTPException(status_code=500, detail="CICFlowMeter no generó resultados para este PCAP")

    df_raw = pd.read_csv(cic_csv_path, on_bad_lines='skip')
    df = extract_cic_features_from_columns(df_raw)

    # Conservar metadatos de conexión
    meta_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp']
    for col in meta_cols:
        if col in df_raw.columns:
            df[col] = df_raw[col].values

    modelo_obj = MODELOS_DISPONIBLES["cic-ids2017"]
    features = modelo_obj.features_seleccionadas
    batch_df = df[features].copy()
    batch_df = modelo_obj.preprocesar_datos_cic(batch_df, train=False)
    y_pred, y_proba = modelo_obj.prediccion_real_cic(batch_df, modelo_obj.modelo)
    df['prediccion'] = y_pred

    # Limpiar CSV temporal
    try:
        os.remove(cic_csv_path)
    except Exception:
        pass

    return df


# ── Modelos precargados para /predict ──────────────────────────────────
modelo_cic = ModeloRandomForest.cargar_modelo("models/modelo_rf_cic.joblib")
modelo_unsw = ModeloRandomForest.cargar_modelo("models/modelo_rf_unsw.joblib")

MODELOS_DISPONIBLES = {
    "cic-ids2017": modelo_cic,
    "unsw-nb15": modelo_unsw,
}





@app.post("/stop_escaner")
async def stop_trafico_red():
    global _sesion_actual_id
    if escanner.cicflowmeter_running:
        escanner.stop_captura_cic()
        logger.info("Escaneo de tráfico finalizado")

        # Cerrar sesión en BD
        if _sesion_actual_id is not None:
            import json
            engine = get_engine()
            with engine.begin() as conn:
                # Contar flujos y amenazas
                row = conn.execute(text(
                    "SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN UPPER(prediccion) != 'BENIGN' AND prediccion IS NOT NULL THEN 1 ELSE 0 END) as amenazas "
                    "FROM THREATS WHERE sesion_id = :sid"
                ), {"sid": _sesion_actual_id}).fetchone()
                total_flujos = row[0] if row else 0
                total_amenazas = row[1] if row else 0

                # Obtener tipos de ataques con conteo
                tipos_rows = conn.execute(text(
                    "SELECT prediccion, COUNT(*) as cantidad "
                    "FROM THREATS WHERE sesion_id = :sid AND prediccion IS NOT NULL "
                    "GROUP BY prediccion ORDER BY cantidad DESC"
                ), {"sid": _sesion_actual_id}).fetchall()
                tipos_ataques = {r[0]: r[1] for r in tipos_rows}

                conn.execute(text(
                    "UPDATE SESIONES SET fecha_fin = :ff, total_flujos = :tf, "
                    "total_amenazas = :ta, tipos_ataques = :tipos, estado = 'completada' WHERE id = :sid"
                ), {
                    "ff": datetime.now().isoformat(),
                    "tf": total_flujos,
                    "ta": total_amenazas,
                    "tipos": json.dumps(tipos_ataques),
                    "sid": _sesion_actual_id
                })
            _sesion_actual_id = None

        return {"status": "trafico real finalizado"}
    else:
        logger.info("No hay escaneo activo")
        return {"status": "no hay escaneo activo"}


@app.get("/api/v1/logs")
async def obtener_logs(last: int = Query(50, ge=1, le=500)):
    """Devuelve las últimas N entradas del log del sistema."""
    logs = list(log_buffer)[-last:]
    return {
        "total": len(log_buffer),
        "returned": len(logs),
        "logs": logs
    }


def _get_iface_ip(iface_name: str) -> str:
    """Obtiene la dirección IP de una interfaz de red usando ioctl"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', iface_name[:15].encode('utf-8'))
        )[20:24])
        s.close()
        return ip
    except Exception:
        return None


@app.get("/api/v1/interfaces")
async def listar_interfaces():
    """Lista las interfaces de red disponibles en el sistema"""
    interfaces = []
    try:
        net_dir = "/sys/class/net"
        for iface in sorted(os.listdir(net_dir)):
            info = {"name": iface}
            try:
                with open(f"{net_dir}/{iface}/operstate", "r") as f:
                    info["state"] = f.read().strip()
            except Exception:
                info["state"] = "unknown"
            info["ip"] = _get_iface_ip(iface)
            interfaces.append(info)
    except Exception as e:
        logger.error(f"Error listando interfaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "current": escanner.interface,
        "interfaces": interfaces
    }


@app.post("/api/v1/interfaces/{interface_name}")
async def cambiar_interface(interface_name: str):
    """Cambia la interfaz de red utilizada por el escáner"""
    net_dir = "/sys/class/net"
    if not os.path.exists(f"{net_dir}/{interface_name}"):
        raise HTTPException(status_code=404, detail=f"Interfaz '{interface_name}' no encontrada")

    escanner.interface = interface_name
    logger.info(f"Interfaz cambiada a: {interface_name}")
    return {
        "status": "ok",
        "interface": interface_name
    }




@app.get("/api/v1/sesiones")
async def listar_sesiones():
    """Lista todas las sesiones de escaneo."""
    import json
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, fecha_inicio, fecha_fin, total_flujos, total_amenazas, tipos_ataques, estado, interfaz "
            "FROM SESIONES ORDER BY id DESC"
        )).fetchall()
    sesiones = []
    for r in rows:
        tipos = {}
        if r[5]:
            try:
                tipos = json.loads(r[5])
            except Exception:
                pass
        sesiones.append({
            "id": r[0], "fecha_inicio": r[1], "fecha_fin": r[2],
            "total_flujos": r[3], "total_amenazas": r[4],
            "tipos_ataques": tipos,
            "estado": r[6], "interfaz": r[7]
        })
    return {"total": len(sesiones), "sesiones": sesiones}


@app.get("/api/v1/sesiones/{sesion_id}")
async def detalle_sesion(sesion_id: int):
    """Detalle de una sesión con estadísticas de ataques."""
    import json
    engine = get_engine()
    with engine.connect() as conn:
        sesion = conn.execute(text(
            "SELECT id, fecha_inicio, fecha_fin, total_flujos, total_amenazas, "
            "tipos_ataques, estado, interfaz FROM SESIONES WHERE id = :sid"
        ), {"sid": sesion_id}).fetchone()

        if not sesion:
            raise HTTPException(status_code=404, detail=f"Sesión {sesion_id} no encontrada")

        tipos = {}
        if sesion[5]:
            try:
                tipos = json.loads(sesion[5])
            except Exception:
                pass

        # Estadísticas de amenazas por tipo
        amenazas = conn.execute(text(
            "SELECT prediccion, COUNT(*) as total, "
            "MIN(tiempo) as primera_deteccion, MAX(tiempo) as ultima_deteccion "
            "FROM THREATS WHERE sesion_id = :sid "
            "GROUP BY prediccion ORDER BY total DESC"
        ), {"sid": sesion_id}).fetchall()

        # IPs involucradas en la sesión
        ips_rows = conn.execute(text(
            "SELECT DISTINCT srcip FROM THREATS WHERE sesion_id = :sid AND srcip IS NOT NULL "
            "UNION "
            "SELECT DISTINCT dstip FROM THREATS WHERE sesion_id = :sid AND dstip IS NOT NULL"
        ), {"sid": sesion_id}).fetchall()
        ips = [r[0] for r in ips_rows if r[0]]

    return {
        "sesion": {
            "id": sesion[0],
            "fecha_inicio": sesion[1],
            "fecha_fin": sesion[2],
            "total_flujos": sesion[3],
            "total_amenazas": sesion[4],
            "estado": sesion[6],
            "interfaz": sesion[7],
        },
        "distribucion": tipos,
        "ips": ips,
        "detalle_predicciones": [
            {
                "prediccion": a[0],
                "total": a[1],
                "primera_deteccion": a[2],
                "ultima_deteccion": a[3]
            } for a in amenazas
        ]
    }


@app.get("/api/v1/sesiones/{sesion_id}/trafico_por_ip")
async def trafico_por_ip(sesion_id: int):
    """Flujos agrupados por srcip, dstip y predicción para comparar tráfico."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT srcip, dstip, prediccion, COUNT(*) as total "
            "FROM THREATS WHERE sesion_id = :sid AND srcip IS NOT NULL "
            "GROUP BY srcip, dstip, prediccion ORDER BY total DESC"
        ), {"sid": sesion_id}).fetchall()

    return {
        "sesion_id": sesion_id,
        "trafico_por_ip": [
            {"srcip": r[0], "dstip": r[1], "prediccion": r[2], "total": r[3]}
            for r in rows
        ]
    }


@app.get("/api/v1/export")
async def exportar_datos(
    format: str = Query("csv", pattern="^(csv|json)$"),
    sesion_id: int = Query(None, description="Filtrar por ID de sesión")
):
    """Exporta predicciones como CSV o JSON desde la base de datos."""
    engine = get_engine()
    with engine.connect() as conn:
        if sesion_id is not None:
            result = conn.execute(
                text("SELECT * FROM THREATS WHERE sesion_id = :sid"),
                {"sid": sesion_id}
            )
        else:
            result = conn.execute(text("SELECT * FROM THREATS"))
        rows = result.fetchall()
        columns = list(result.keys())

    if not rows:
        raise HTTPException(status_code=404, detail="No hay datos para exportar")

    df = pd.DataFrame(rows, columns=columns)

    if format == "json":
        return {
            "total": len(df),
            "data": df.to_dict("records")
        }

    # CSV download
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ids_export_{timestamp}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.websocket("/ws/alertas")
async def websocket_alertas(websocket: WebSocket):
    """WebSocket para recibir alertas en tiempo real."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"tipo": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
