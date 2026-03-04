import logging
import argparse
import os
import uvicorn

from src.database import init_tables
from src.escanear_trafico_red import EscanearTraficoRed
from src.transformar_datos_multilog import TransformarDatos
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_fecha():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(
        prog="ids",
        description="Sistema de Deteccion de Intrusiones basado en ML"
    )
    parser.add_argument(
        "--api", "-a",
        action="store_true",
        help="Ejecutar con API REST (fastAPI)"
    )
    parser.add_argument(
        "--interface", "-i",
        default="enp6s0",
        help="Interfaz de red"
    )
    parser.add_argument(
        "--pcap", "-f",
        help="Archivo pcap"
    )
    parser.add_argument(
        "--argus", "-af",
        help="Archivo argus"
    )
    parser.add_argument(
        "--logs-dir", "-d",
        default="logs/zeek",
        help="Directorio donde se guardan los logs de zeek",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        required=False,
        help="Archivo configuracion"
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=20,
        required=False,
        help="Tiempo de captura en segundos"
    )
    parser.add_argument(
        "-n",
        required=False,
        help="Numero de paquetes a analizar",
    )
    parser.add_argument(
        "-w",
        required=False,
        help="Archivo de salida para el tráfico",
    )
    args = parser.parse_args()

    init_tables()

    if args.api:
        logger.info("Iniciando API REST en http://0.0.0.0:8000")
        logger.info("Docs: http://localhost:8000/docs")
        uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)
        return

    fecha = get_fecha()
    pcap_file = args.pcap if args.pcap else f"logs/captura_trafico/trafico_red_{fecha}.pcap"
    argus_file = args.argus if args.argus else f"logs/captura_trafico/file_{fecha}.argus"

    os.makedirs("logs/captura_trafico", exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    scanner = EscanearTraficoRed(
        interface=args.interface,
        pcap_file=pcap_file,
        argus_file=argus_file,
        zeek_logs_dir=args.logs_dir,
    )

    logger.info(f"Iniciando análisis de tráfico en interfaz {args.interface}")
    logger.info(f"  PCAP: {pcap_file}")
    logger.info(f"  Argus: {argus_file}")
    logger.info(f"  Logs Zeek: {args.logs_dir}")

    scanner.run_analysis(capture_time=args.time)
    transformer = TransformarDatos(
        logs_dir=args.logs_dir,
        output_path=f"data/trafico_red_{fecha}.csv"
    )
    transformer.generate_csv(argus_file, args.logs_dir)


if __name__ == "__main__":
    main()
