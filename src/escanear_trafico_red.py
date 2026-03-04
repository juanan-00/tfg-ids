import os
import sys
import time
import subprocess
import shutil
import logging
import threading
from datetime import datetime
from src.procesar_predicciones import ProcesarPrediccionesCIC
from cicflowmeter.sniffer import create_sniffer as create_sniffer_cic
import cicflowmeter.flow_session as cic_session


# Variable global para directorio actual
CURRENT_DIR = os.getcwd()

def get_fecha():
    """Devuelve la fecha actual para diferenciar los archivos"""
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

class EscanearTraficoRed: 
    """
    clase encargada de manejar herramientas para monitorizacion de red 
    """
    def __init__(self, interface="eth0", pcap_file="output.pcap", zeek_logs_dir="logs/zeek"):
        self.interface = interface
        self.zeek_logs_dir = zeek_logs_dir
        self.pcap_file = pcap_file

        self.zeek_process = None

        self.zeek_running = False
        self.cicflowmeter_running = False

        self.escanear_activo = False

        self.zeek_bin = self._find_zeek_binary()
    
    @staticmethod
    def get_interface_default():
        '''Detecta automaticamente la interfaz de red, lee del archivo route la primera dir ip'''
        try:
            with open('/proc/net/route', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        if len(parts) > 1 and parts[1] == '00000000':
                            interfaz = parts[0]
                            return interfaz
        except (IOError, IndexError):
            logging.warning("Método /proc/net/route fallo")

    
    def _find_zeek_binary(self):
        """
           localiza donde se encuentra zeek en el sistema
            return path donde se encuentra el ejecutable de zeek 
        """
        possible_paths = [
            "/usr/bin/zeek",
            "/usr/local/bin/zeek",
            "/opt/zeek/bin/zeek"
        ]

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        zeek_path = shutil.which("zeek")
        if zeek_path:
            return zeek_path
        print("error: zeek not located in system")        
        print("make sure zeek is installed and that its in path")
        sys.exit(1)

    def start_zeek(self):
        """inicia zeek para leer archivos .pcap"""

        if self.zeek_running:
            print("zeek ya se está ejecutando")
            return
        
        # crear directorio para logs
        os.makedirs(self.zeek_logs_dir, exist_ok=True)

        # Usar ruta absoluta al archivo PCAP
        pcap_absolute_path = os.path.join(CURRENT_DIR, self.pcap_file)
        zeek_logs_absolute = os.path.join(CURRENT_DIR, self.zeek_logs_dir)
        print(f"Dir actual: {CURRENT_DIR}") 
        print(f"Dir pcap absolute {pcap_absolute_path}")
        print(f"Dir zeek absolute {zeek_logs_absolute}")
        # define el comando de ejecucion con Log::default_logdir y múltiples protocolos
        cmd = [
            self.zeek_bin,
            "-C",
            "-r", pcap_absolute_path,
            f"Log::default_logdir={zeek_logs_absolute}",
            # Cargar scripts de análisis de protocolos
            "base/protocols/conn",      
            "base/protocols/http",      
            "base/protocols/dns",       
            "base/protocols/ssl",       
            "base/protocols/ssh",       
            "base/protocols/ftp",       
            "base/protocols/smtp",      
        ]
        print(f"iniciando zeek con análisis completo de protocolos...")
        print(f"iniciando zeek: {' '.join(cmd)}")
        print(f"Logs de Zeek se escribirán en: {zeek_logs_absolute}")

        try:
            self.zeek_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(2)
            if self.zeek_process.poll() is not None:
                out,err = self.zeek_process.communicate()
                print("Zeek finalizó su ejecución")
                print(f"STDOUT: {out}")
                print(f"STDERR: {err}")

                # Zeek puede completar exitosamente incluso con stderr (warnings)
                # Solo falla si el código de salida no es 0
                if self.zeek_process.returncode != 0:
                    print(f"ERROR: zeek falló con código de salida {self.zeek_process.returncode}")
                    return False
                else:
                    print("Zeek completó el análisis exitosamente.")
                    self.zeek_running = False
                    return True
            
            self.zeek_running = True
            print(f"Zeek iniciado correctamente (PID: {self.zeek_process.pid})")
        except Exception as e:
            print(f"error al ejecutar zeek: {e}")
            return True 


    def procesar_pcap_file(self, pcap_path: str = None) -> bool:
        """
        Procesa un PCAP local sin capturar tráfico en vivo.
        - Si no se pasa pcap_path, se usa ../logs/captura_trafico/<interfaz>.pcap
        - Ejecuta Zeek (si está configurado) y Argus sobre el PCAP
        """
        try:
            # 0) Resolver ruta del PCAP "por interfaz" si no se pasa por argumento
            if pcap_path is None:
                default_pcap = os.path.join("logs/captura_trafico", f"{self.interface}.pcap")
                chosen_pcap = default_pcap
            else:
                chosen_pcap = pcap_path

            if not os.path.exists(chosen_pcap) or os.path.getsize(chosen_pcap) == 0:
                print(f"ERROR: PCAP no existe o está vacío: {chosen_pcap}")
                return False

            # 1) Ajustar self.pcap_file (start_zeek usa CURRENT_DIR + self.pcap_file)
            if os.path.isabs(chosen_pcap):
                self.pcap_file = os.path.relpath(chosen_pcap, start=os.getcwd())
            else:
                self.pcap_file = chosen_pcap

            # 2) Zeek sobre PCAP (logs a self.zeek_logs_dir)
            print("Analizando PCAP con Zeek (offline)...")
            print(f"Directorio de logs de Zeek: {self.zeek_logs_dir}")

            # Resetear flag para permitir análisis offline aunque haya corrido antes
            self.zeek_running = False

            if not self.start_zeek():
                print("Aviso: Zeek no se inició correctamente.")
                return False
            else:
                if self.zeek_process and self.zeek_running:
                    print("Esperando a que Zeek complete el análisis...")
                    self.zeek_process.wait()
                    print("Zeek completó el análisis")
                self.zeek_running = False

            # Verificar que se generaron logs de Zeek
            zeek_logs_exist = any(f.endswith('.log') for f in os.listdir(self.zeek_logs_dir) if os.path.isfile(os.path.join(self.zeek_logs_dir, f)))

            print("Proceso de archivo pcap correcto:")
            print("---------------------------------")
            print(f"Archivo pcap procesado: {chosen_pcap}")
            print(f"archivo self.pcap_file: {self.pcap_file}")
            print(f"Zeek logs dir: {self.zeek_logs_dir}")
            print(f"Logs de Zeek generados: {zeek_logs_exist}")

            if zeek_logs_exist:
                logs = [f for f in os.listdir(self.zeek_logs_dir) if f.endswith('.log')]
                print(f"Archivos de log generados: {logs}")

            return True 

        except Exception as e:
            print(f"procesando pcap: {e}")
            return False


    def stop_zeek(self):
        """Detiene el proceso de Zeek si está ejecutándose"""
        if self.zeek_running and self.zeek_process:
            try:
                self.zeek_process.terminate()
                self.zeek_process.wait(timeout=5)
                self.zeek_running = False
                print("Zeek detenido correctamente")
            except Exception as e:
                print(f"Error al detener Zeek: {e}") 

    def continous_capture_cic(self, log_callback=None, poll_interval=15, sesion_id=None, ws_manager=None):
        """Captura continua con CICFlowMeter directo (sin tcpdump).
        CICFlowMeter sniffa la interfaz y escribe flows a CSV.
        Un thread de flush fuerza escritura cada 10s.
        Un thread de polling lee filas nuevas y ejecuta predicciones.
        """
        self.log_callback = log_callback or print
        os.makedirs("logs/captura_trafico/stream", exist_ok=True)

        # Limpiar archivos de sesiones anteriores
        for old_file in ["flows_cic.csv", "last_pred.csv", "last_threats.csv"]:
            old_path = f"logs/captura_trafico/stream/{old_file}"
            if os.path.exists(old_path):
                os.remove(old_path)

        csv_path = "logs/captura_trafico/stream/flows_cic.csv"
        self.escanear_activo = True

        # Parchear garbage_collect para que escriba flows inactivos >10s
        # Original: solo escribe flows inactivos >240s o con duración >90s
        def _aggressive_gc(session_self, latest_time):
            for k in list(session_self.flows.keys()):
                flow = session_self.flows.get(k)
                if not flow:
                    continue
                if latest_time is None or \
                   (latest_time - flow.latest_timestamp) > 10 or \
                   flow.duration > 10:
                    session_self.output_writer.write(flow.get_data(session_self.fields))
                    del session_self.flows[k]

        cic_session.FlowSession.garbage_collect = _aggressive_gc
        cic_session.GARBAGE_COLLECT_PACKETS = 20

        # 1. Iniciar sniffer CIC directo en la interfaz
        self.log_callback(f"Iniciando CICFlowMeter en {self.interface}...")
        self._cic_sniffer = create_sniffer_cic(
            input_file=None,
            input_interface=self.interface,
            output_mode="csv",
            output=csv_path,
            verbose=False
        )
        self._cic_sniffer.start()
        self.cicflowmeter_running = True
        self.log_callback("CICFlowMeter activo - capturando flujos de red")

        # 2. Thread de flush periódico cada 10s
        #    Accede a la sesión del sniffer y fuerza garbage_collect
        _log = self.log_callback

        def flush_loop():
            time.sleep(5)  # esperar a que el sniffer inicialice la sesión
            while self.escanear_activo:
                try:
                    session = getattr(self._cic_sniffer, 'session', None)
                    if session and hasattr(session, 'flows') and session.flows:
                        latest = max(f.latest_timestamp for f in session.flows.values() if f)
                        before = len(session.flows)
                        session.garbage_collect(latest + 11)
                        written = before - len(session.flows)
                        if written > 0:
                            _log(f"[CIC] Flush: {written} flows escritos al CSV")
                except Exception:
                    pass
                time.sleep(10)

        self._cic_flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._cic_flush_thread.start()

        # 3. Iniciar handler de predicciones
        self._cic_handler = ProcesarPrediccionesCIC(
            csv_path=csv_path,
            log_callback=log_callback,
            sesion_id=sesion_id,
            ws_manager=ws_manager
        )

        # 4. Thread de polling periódico
        def poll_loop():
            while self.escanear_activo and not self._cic_handler.shutdown_flag.is_set():
                self._cic_handler.poll()
                time.sleep(poll_interval)

        self._cic_poll_thread = threading.Thread(target=poll_loop, daemon=True)
        self._cic_poll_thread.start()
        self.log_callback(f"Polling de predicciones activo (cada {poll_interval}s)")
        return True

    def stop_captura_cic(self):
        """Detiene la captura CIC: sniffer + polling de predicciones."""
        self.escanear_activo = False

        if hasattr(self, '_cic_handler'):
            self._cic_handler.shutdown()

        if hasattr(self, '_cic_sniffer') and self.cicflowmeter_running:
            try:
                self._cic_sniffer.stop()
                self._cic_sniffer.join(timeout=10)
            except Exception as e:
                if hasattr(self, 'log_callback'):
                    self.log_callback(f"Error deteniendo CICFlowMeter: {e}")
            self.cicflowmeter_running = False

        if hasattr(self, '_cic_poll_thread'):
            self._cic_poll_thread.join(timeout=10)

        if hasattr(self, 'log_callback'):
            self.log_callback("Captura CIC detenida")



