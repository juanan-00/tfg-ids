import os
import time
import threading
import shutil
import pandas as pd
import logging
import sqlite3

from datetime import datetime
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Full, Empty
from src.modelos.ModeloRandomForest import ModeloRandomForest
from src.modelos.ModeloKNN import ModeloKNN
from sklearn.dummy import DummyClassifier
from sqlalchemy import create_engine
from sqlalchemy import text

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ruta_modelo_rf_unsw = "models/modelo_rf_unsw.joblib"
ruta_modelo_rf_cic = "models/modelo_rf_cic.joblib"

try:
    conn = sqlite3.connect("ids.db")
    cursor = conn.cursor()
    logger.info("conectado a base de datos")
except Exception as e:
    logger.error(f"Error al conectarte a la base de datos")

try:
    #cursor.execute("SELECT id FROM THREATS ORDER BY ROWID DESC LIMIT 1")  
    last_id = cursor.lastrowid
    print(last_id) 
except Exception as e:
    logger.warning(f"No se encuentra id: {e}")



mapa_features = {
    'flow_duration': 'FlowDuration',
    'bwd_pkt_len_max': 'BwdPacketLengthMax',
    'bwd_pkt_len_mean': 'BwdPacketLengthMean',
    'bwd_pkt_len_std': 'BwdPacketLengthStd',
    'bwd_pkt_len_min': 'BwdPacketLengthMin',
    'flow_iat_mean': 'FlowIATMean',
    'flow_iat_std': 'FlowIATStd',
    'flow_iat_max': 'FlowIATMax',
    'fwd_iat_tot': 'FwdIATTotal',
    'fwd_iat_mean': 'FwdIATMean',
    'fwd_iat_std': 'FwdIATStd',
    'fwd_iat_max': 'FwdIATMax',
    'bwd_iat_std': 'BwdIATStd',
    'bwd_iat_max': 'BwdIATMax',
    'pkt_len_min': 'MinPacketLength',
    'pkt_len_max': 'MaxPacketLength',
    'pkt_len_mean': 'PacketLengthMean',
    'pkt_len_std': 'PacketLengthStd',
    'fin_flag_cnt': 'FINFlagCount',
    'psh_flag_cnt': 'PSHFlagCount',
    'ack_flag_cnt': 'ACKFlagCount',
    'pkt_size_avg': 'AveragePacketSize'
}

# Mapea todas las columnas para que tenga el mismo nombre que se uso en entrenamiento 
COLUMN_TO_CIC_FEATURES_MAP = {
    "dst_port": "DestinationPort",
    "flow_duration": "FlowDuration",
    "tot_fwd_pkts": "TotalFwdPackets",
    "tot_bwd_pkts": "TotalBackwardPackets",
    "totlen_fwd_pkts": "TotalLengthofFwdPackets",
    "totlen_bwd_pkts": "TotalLengthofBwdPackets",
    "fwd_pkt_len_max": "FwdPacketLengthMax",
    "fwd_pkt_len_min": "FwdPacketLengthMin",
    "fwd_pkt_len_mean": "FwdPacketLengthMean",
    "fwd_pkt_len_std": "FwdPacketLengthStd",
    "bwd_pkt_len_max": "BwdPacketLengthMax",
    "bwd_pkt_len_min": "BwdPacketLengthMin",
    "bwd_pkt_len_mean": "BwdPacketLengthMean",
    "bwd_pkt_len_std": "BwdPacketLengthStd",
    "flow_byts_s": "FlowBytes/s",
    "flow_pkts_s": "FlowPackets/s",
    "flow_iat_mean": "FlowIATMean",
    "flow_iat_std": "FlowIATStd",
    "flow_iat_max": "FlowIATMax",
    "flow_iat_min": "FlowIATMin",
    "fwd_iat_tot": "FwdIATTotal",
    "fwd_iat_mean": "FwdIATMean",
    "fwd_iat_std": "FwdIATStd",
    "fwd_iat_max": "FwdIATMax",
    "fwd_iat_min": "FwdIATMin",
    "bwd_iat_tot": "BwdIATTotal",
    "bwd_iat_mean": "BwdIATMean",
    "bwd_iat_std": "BwdIATStd",
    "bwd_iat_max": "BwdIATMax",
    "bwd_iat_min": "BwdIATMin",
    "fwd_psh_flags": "FwdPSHFlags",
    "bwd_psh_flags": "BwdPSHFlags",
    "fwd_urg_flags": "FwdURGFlags",
    "bwd_urg_flags": "BwdURGFlags",
    "fwd_header_len": "FwdHeaderLength",
    "bwd_header_len": "BwdHeaderLength",
    "fwd_pkts_s": "FwdPackets/s",
    "bwd_pkts_s": "BwdPackets/s",
    "pkt_len_min": "MinPacketLength",
    "pkt_len_max": "MaxPacketLength",
    "pkt_len_mean": "PacketLengthMean",
    "pkt_len_std": "PacketLengthStd",
    "pkt_len_var": "PacketLengthVariance",
    "fin_flag_cnt": "FINFlagCount",
    "syn_flag_cnt": "SYNFlagCount",
    "rst_flag_cnt": "RSTFlagCount",
    "psh_flag_cnt": "PSHFlagCount",
    "ack_flag_cnt": "ACKFlagCount",
    "urg_flag_cnt": "URGFlagCount",
    "cwe_flag_count": "CWEFlagCount",
    "cwr_flag_count": "CWEFlagCount",
    "ece_flag_cnt": "ECEFlagCount",
    "down_up_ratio": "Down/UpRatio",
    "pkt_size_avg": "AveragePacketSize",
    "fwd_seg_size_avg": "AvgFwdSegmentSize",
    "bwd_seg_size_avg": "AvgBwdSegmentSize",
    "fwd_header_len": "FwdHeaderLength",
    "fwd_byts_b_avg": "FwdAvgBytes/Bulk",
    "fwd_pkts_b_avg": "FwdAvgPackets/Bulk",
    "fwd_blk_rate_avg": "FwdAvgBulkRate",
    "bwd_byts_b_avg": "BwdAvgBytes/Bulk",
    "bwd_pkts_b_avg": "BwdAvgPackets/Bulk",
    "bwd_blk_rate_avg": "BwdAvgBulkRate",
    "subflow_fwd_pkts": "SubflowFwdPackets",
    "subflow_fwd_byts": "SubflowFwdBytes",
    "subflow_bwd_pkts": "SubflowBwdPackets",
    "subflow_bwd_byts": "SubflowBwdBytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "ActiveMean",
    "active_std": "ActiveStd",
    "active_max": "ActiveMax",
    "active_min": "ActiveMin",
    "idle_mean": "IdleMean",
    "idle_std": "IdleStd",
    "idle_max": "IdleMax",
    "idle_min": "IdleMin"
}

def extract_cic_features_from_columns(input_df): 
    cic_features = {}
    
    for column_name, cic_feature in COLUMN_TO_CIC_FEATURES_MAP.items():
        if column_name in input_df.columns:
            cic_features[cic_feature] = input_df[column_name]
        else:
            cic_features[cic_feature] = None
    
    return pd.DataFrame(cic_features)

class ProcesarPrediccionesCIC:
    """
    Procesador de predicciones para CIC-IDS2017 usando CICFlowMeter directo.
    No depende de tcpdump ni de FileSystemEventHandler.
    Lee periódicamente el CSV generado por CICFlowMeter y ejecuta predicciones
    sobre las filas nuevas.
    """

    def __init__(self, csv_path, log_callback=None, sesion_id=None, ws_manager=None):
        self.csv_path = csv_path
        self.modelo_rf = ModeloRandomForest.cargar_modelo(ruta_modelo_rf_cic)
        self.log = log_callback or logger.info
        self.last_row = 0
        self.shutdown_flag = threading.Event()
        self.csv_actual = None
        self.sesion_id = sesion_id
        self.ws_manager = ws_manager
        self.log("ProcesarPrediccionesCIC inicializado")

    def poll(self):
        """Lee filas nuevas del CSV de CICFlowMeter y ejecuta predicciones."""
        if self.shutdown_flag.is_set():
            return

        if not os.path.exists(self.csv_path):
            self.log(f"CSV no existe aún: {self.csv_path}")
            return

        size = os.path.getsize(self.csv_path)
        if size == 0:
            self.log("CSV existe pero está vacío (cicflowmeter aún no ha escrito flows)")
            return

        try:
            df = pd.read_csv(self.csv_path, on_bad_lines='skip')
        except Exception as e:
            self.log(f"Error leyendo CSV de CICFlowMeter: {e}")
            return

        if len(df) <= self.last_row:
            return

        nuevas = df.iloc[self.last_row:]
        self.last_row = len(df)
        self.log(f"Procesando {len(nuevas)} flujos nuevos")

        try:
            batch_df = extract_cic_features_from_columns(nuevas)
            batch_res = batch_df.copy()

            # Conservar metadatos de conexión del CSV original de CICFlowMeter
            meta_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp']
            for col in meta_cols:
                if col in nuevas.columns:
                    batch_res[col] = nuevas[col].values

            features = self.modelo_rf.features_seleccionadas
            batch_df = batch_df[features]
            batch_df = self.modelo_rf.preprocesar_datos_cic(batch_df, train=False)

            y_pred, y_proba = self.modelo_rf.prediccion_real_cic(batch_df, self.modelo_rf.modelo)

            clases = self.modelo_rf.modelo.classes_.tolist()
            if y_proba is not None and len(y_proba.shape) > 1:
                for j, (pred, proba) in enumerate(zip(y_pred, y_proba)):
                    probs_str = " | ".join(f"{c}: {p:.2%}" for c, p in zip(clases, proba) if p > 0.01)
                    self.log(f"Flow {j+1} -> {pred} [{probs_str}]")

            batch_res['prediccion'] = y_pred
            if y_proba is not None and len(y_proba) > 0:
                if len(y_proba.shape) > 1:
                    batch_res["confianza"] = y_proba.max(axis=1)
                else:
                    batch_res["confianza"] = y_proba
            else:
                batch_res["confianza"] = 0.5

            # Detección de PortScan por agregación:
            # Si una IP origen tiene muchos flujos a puertos distintos → PortScan
            batch_res = self._detectar_portscan_agregado(batch_res, nuevas)

            batch_res['tiempo'] = datetime.now().isoformat()
            if self.sesion_id is not None:
                batch_res['sesion_id'] = self.sesion_id

            # Mapear columnas CIC a columnas de THREATS
            col_map = {
                'src_ip': 'srcip', 'dst_ip': 'dstip',
                'src_port': 'sport', 'dst_port': 'dsport',
                'protocol': 'proto'
            }
            cols_to_drop = []
            for cic_col, db_col in col_map.items():
                if cic_col in batch_res.columns:
                    batch_res[db_col] = batch_res[cic_col]
                    cols_to_drop.append(cic_col)
            if cols_to_drop:
                batch_res = batch_res.drop(columns=cols_to_drop)
            batch_res = batch_res.fillna(0)
            threats = batch_res[batch_res['prediccion'].str.upper() != 'BENIGN']
            batch_res.to_csv("logs/captura_trafico/stream/last_pred.csv", index=False)
            threats.to_csv("logs/captura_trafico/stream/last_threats.csv", index=False)

            self._insert_conexion(batch_res)
            self.csv_actual = batch_res

            self.log(f"Predicciones: {len(batch_res)} | Amenazas: {len(threats)}")

            # Broadcast por WebSocket
            if self.ws_manager:
                if not threats.empty:
                    self.ws_manager.broadcast_sync({
                        "tipo": "nuevas_amenazas",
                        "total_predicciones": len(batch_res),
                        "total_amenazas": len(threats),
                        "timestamp": datetime.now().isoformat(),
                        "amenazas": threats.head(20).to_dict("records")
                    })
                elif not batch_res.empty:
                    self.ws_manager.broadcast_sync({
                        "tipo": "nuevas_predicciones",
                        "total_predicciones": len(batch_res),
                        "total_amenazas": 0,
                        "timestamp": datetime.now().isoformat()
                    })


        except Exception as e:
            self.log(f"Error en predicción CIC: {e}")

    def _detectar_portscan_agregado(self, batch_res, nuevas):
        """
        Detecta PortScan por patrón agregado: si una IP origen contacta
        muchos puertos distintos en un batch, se reclasifica como PortScan.
        Un flujo individual de portscan parece tráfico normal, pero el
        patrón de MUCHOS flujos cortos a puertos diferentes es la firma real.
        """
        MIN_PUERTOS_DISTINTOS = 5  # mínimo puertos distintos para considerar portscan

        if 'src_ip' not in nuevas.columns or 'dst_port' not in nuevas.columns:
            return batch_res

        # Agrupar por IP origen y contar puertos destino distintos
        batch_analisis = nuevas[['src_ip', 'dst_port']].copy()
        batch_analisis.index = batch_res.index  # alinear índices

        por_ip = batch_analisis.groupby('src_ip')['dst_port'].nunique()
        ips_scanner = por_ip[por_ip >= MIN_PUERTOS_DISTINTOS].index.tolist()

        if ips_scanner:
            for ip in ips_scanner:
                mask = batch_analisis['src_ip'] == ip
                n_puertos = por_ip[ip]
                # Reclasificar flujos BENIGN de esa IP como PortScan
                mask_benign = mask & (batch_res['prediccion'] == 'BENIGN')
                n_reclasificados = mask_benign.sum()
                if n_reclasificados > 0:
                    batch_res.loc[mask_benign, 'prediccion'] = 'PortScan'
                    logger.info(
                        f"[PortScan Agregado] IP {ip}: {n_puertos} puertos distintos "
                        f"-> {n_reclasificados} flujos reclasificados como PortScan"
                    )

        return batch_res

    # Columnas de la tabla THREATS en la BD
    THREATS_COLUMNS = [
        'srcip', 'sport', 'dstip', 'dsport', 'dur', 'sbytes', 'dbytes',
        'spkts', 'dpkts', 'proto', 'service', 'rate',
        'prediccion', 'confianza', 'tiempo', 'sesion_id'
    ]

    def _insert_conexion(self, batch_res):
        """Inserta en la tabla THREATS los registros de la ultima conexión."""
        from src.database import get_engine
        try:
            # Filtrar solo columnas que existen en la tabla THREATS
            cols_to_insert = [c for c in self.THREATS_COLUMNS if c in batch_res.columns]
            df_insert = batch_res[cols_to_insert].copy()

            engine = get_engine()
            with engine.begin() as conn:
                df_insert.to_sql('THREATS', con=conn, if_exists='append', method="multi", index=False)
        except Exception as e:
            self.log(f"Error insertando en SQLite: {e}")

    def shutdown(self):
        self.shutdown_flag.set()


class ProcesarPrediccionesML(FileSystemEventHandler):

    """
    Handler que detecta y procesa archivos nuevos PCAP
    Procesa los datos recibidos y realizar predicciones
    Utiliza FileSystemHandler para procesar archivos en tiempo real
    """
    
    def __init__(self, scanner, transformer, log_callback, tipo_modelo="unsw-nb15"):
        print(f"{tipo_modelo}")
        if tipo_modelo == "unsw-nb15":
            self.modelo_rf = ModeloRandomForest.cargar_modelo(ruta_modelo_rf_unsw)
            self.tipo_modelo = "unsw-nb15"
        else:
            self.modelo_rf = ModeloRandomForest.cargar_modelo(ruta_modelo_rf_cic)        
            self.tipo_modelo = "cic-ids2017"

        if isinstance(self.modelo_rf, DummyClassifier):
            print("dymmy")
        else:
            print("no lo es")

        self.csv_actual = None
        
        #self.modelo_knn = ModeloKNN.cargar_modelo(ruta_modelo_knn)
        self.scanner = scanner
        self.transformer = transformer
        self.log = log_callback or logger.info

        # Flag en false para no repetir procesado de archivos
        self.processing = False
        # se establa como un conjunto porque solo puede ser ese archivo
        self.processed_files = set()
        self.shutdown_flag = threading.Event()

        self.log("Handler inicializado")

    # funcion principal que ejecuta el observer
    def on_created(self, event):
        """Callback cuando el observer detecta un nuevo archivo, se ejecuta automaticamente por el Observer"""

        if self.shutdown_flag.is_set(): 
            return
        # espera 2 segundos para que se acumula el trafico
        time.sleep(2)
            # Filtrar solo archivos pcap
        if not event.src_path.endswith('.pcap'):
            return
        # Ignorar archivos temporales
        if '.tmp' in event.src_path:
            return
        # vericar que no esté vacío
        if not os.path.exists(event.src_path):
            return
    
        pcap_path = event.src_path 
        pcap_name = os.path.basename(pcap_path)
        self.log(f"Archivo detectado: {pcap_name}")

        if pcap_path in self.processed_files:
            self.log(f"Archivo ya procesado, saltando: {pcap_name}")
            return 
        if self.shutdown_flag.is_set():
            self.log(f"proceso de cierre este archivo no se procesa: {pcap_name}")
            return

        if not self.scanner.escanear_activo:
            self.log("escaner inactivo, no se inicar el proceso")

        self.processing = True
        self.processed_files.add(pcap_path)

        if not os.path.exists(pcap_path) or os.path.getsize(pcap_path) == 0:
            self.log(f"Archivo inactivo o inexistente: {pcap_name}") 
            self.processing = False 
            return

        # procesar pcap 
        try:
            start_time = time.time()
            self.log(f"Procesando {pcap_name}")
            self.process_pcap(pcap_path)
            final = time.time() - start_time
            self.log(f"tiempo transcurrido: {final}")
        except Exception as e:
            self.log(f"error procesando pcap {e}")
        finally:
            self.processing = False
        """ 
        try:
            size = os.path.getsize(event.src_path)
            if size == 0:
                self.log(f"Pcap vacio ignorado: {os.path.basename(event.src_path)}")
                return
            try:
                os.chmod(event.src_path, 0o666)
            except Exception as e:
                self.log(f"No se pudieron dar permisos: {e}")
        except Exception as e:
            print(f"Error en proceso {e}")

        self.process_pcap(event.src_path)
        """

    # procesa archivo pcap y hace predicciones sobre el 
    def process_pcap(self, pcap_path):
        """
        Procesa un pcap individual: zeek + argus -> csv -> kafka.
        """
        start_time = time.time()
        pcap_name = os.path.basename(pcap_path)

        # Genera path para arcivos
        base = os.path.splitext(pcap_path)[0]
        print(f"{base}")
        argus_file = f"{base}.argus"
        zeek_dir = f"{base}_zeek"
        csv_path = f"{base}.csv"

        csv_path_cic =f"{base}_cic.csv"
        csv_path_pred = f"{base}_pred.csv"

        logger.debug(f"Directorio base: {base}")
        logger.debug(f"Directorio argus {argus_file}")
        logger.debug(f"Directorio zeek {zeek_dir}")
        logger.debug(f"Directorio csv {csv_path}")
        
        orig_pcap = self.scanner.pcap_file
        orig_argus = self.scanner.argus_output
        orig_zeek = self.scanner.zeek_logs_dir

        try:
            self.scanner.pcap_file = pcap_path
            self.scanner.argus_output = argus_file
            self.scanner.zeek_logs_dir = zeek_dir

            print(f"tipo modelo: {self.tipo_modelo}") 
            self.tipo_modelo = "cic-ids2017"
            # Ejecutar cicflowmeter  
            if self.tipo_modelo == "cic-ids2017":
                try:
                    #self.scanner.start_cicflowmeter(csv_path_cic)
                    path = "~/Documentos/tfg-ids-api/logs/captura_trafico/stream/last_pred_cic.csv"
                    self.scanner.start_cicflowmeter_directo(path)
                    #if self.scanner.cicflowmeter_process:
                        #self.scanner.cicflowmeter_process.wait(timeout=20)
                except Exception as e: 
                    self.log(f"Error a ejecutar cicflowmeter {e}")
                self.log("hola")
                time.sleep(6)
                batch_df = pd.read_csv(csv_path_cic)
                batch_df = extract_cic_features_from_columns(batch_df)
                batch_res = batch_df.copy()
                features = self.modelo_rf.features_seleccionadas
                batch_df = batch_df[features]
                print(batch_df.columns.tolist)
                print(f"features seleccionadas: {features}")
                batch_df = self.modelo_rf.preprocesar_datos_cic(batch_df, train=False)

                try:
                    #batch_df = self.modelo_rf.aplicar_transformador(batch_df)  
                    print(batch_df)
                except Exception as e:
                    print(f"Error: {e}")
                y_pred, y_proba = self.modelo_rf.prediccion_real_cic(batch_df, self.modelo_rf.modelo)
                print("Res predicción")

                print(y_pred)
                batch_res['prediccion'] = y_pred 
                print(batch_res)
                logger.info(batch_res)
                print(f"Archivo con todas las predicciones en {csv_path_pred}")
                batch_res.to_csv(csv_path_pred)

            ## MODELO UNSW-NB15 
            if self.tipo_modelo == "unsw-nb15":
                # Ejecutar zeek
                try:
                    self.scanner.start_zeek()
                    if self.scanner.zeek_process:
                        self.scanner.zeek_process.wait(timeout=20)
                    self.scanner.zeek_running = False
                except Exception as e:
                    self.log(f"Hubo un fallo al ejecutar zeek: {e}")

                #Ejecutar argus 
                try:
                    self.scanner.start_argus()
                    if self.scanner.argus_process:
                        self.scanner.argus_process.wait(timeout=20)
                    self.scanner.argus_running = False
                except Exception as e:
                    self.log(f"Hubo un fallo al ejecutar argus: {e}")

                try:
                    # Transformar datos a csv de zeek y argus
                    df = "zeek"
                    self.transformer.logs_dir = zeek_dir
                    self.transformer.output_path = csv_path
                    csv = self.transformer.generate_csv(argus_file, df, zeek_dir)
        
                    # leer archivo csv
                    batch_df = pd.read_csv(csv)
                    batch_res = batch_df.copy()

                    col_permitidas = [
                        "srcip",
                        "sport",
                        "dstip",
                        "dsport",
                        "dur",
                        "sbytes",
                        "dbytes",
                        "spkts",
                        "dpkts",
                        "proto",
                        "service",
                        "rate"
                    ]

                    drop_columns = [col for col in batch_df.columns if col not in col_permitidas]
                    print("Columnas a eliminar")
                    print(drop_columns)
                    batch_res.drop(columns=drop_columns, inplace=True) 
                    
                    print(batch_res.columns) 
                    # seleccionar atributos de entrenamiento
                    features = self.modelo_rf.features_seleccionadas
                    batch_df = batch_df[features]
                    print(features)
                    print(batch_df.columns) 
                    # preprocesar datos obtenidos
                    batch_df = self.modelo_rf.preprocesar_datos_unsw(batch_df, train=False)
                    # realizar prediccion y generar csv trafico etiquetado con predicciones
                    y_pred, y_proba = self.modelo_rf.prediccion_real2(batch_df, self.modelo_rf.modelo)
                    batch_res['prediccion'] = y_pred
                    
                    print(y_proba[:,1])
                    print(y_proba[:,0])
                     
                    if y_proba is not None and len(y_proba) > 0:
                        if len(y_proba.shape) > 1:
                            batch_res["confianza"] = y_proba.max(axis=1)
                        else:
                            batch_res["confianza"] = y_proba
                    else:
                        batch_res["confianza"] = 0.5
                    
                    batch_res['tiempo'] = datetime.now().isoformat()
                    threats = batch_res[batch_res['prediccion'].str.lower() != 'normal']
                    batch_res.drop(columns="confianza")
                    print("Columnas dataframe batch_res")
                    print(batch_res.columns)
                    

                    # Resultado de predicciones
                    batch_res.to_csv("logs/captura_trafico/stream/last_pred.csv")
                    threats.to_csv("logs/captura_trafico/stream/last_threats.csv") 
                    
                    print(batch_res)
                    self._insert_conexion(batch_res)
                    print(f"Archivo con todas las predicciones en {csv_path_pred}")
                    print(f"Amenezas encontradas: {len(threats)}")
                    print(threats)
                    self.csv_actual = batch_res
                except Exception as e:
                    self.log(f"Sucedio el siguiente error en el proceso de prediccion: {e}") 
             
            #batch_df['prob'] = y_proba
            #
            #print(y_proba)
            

            # Metricas
            t_transcurrido = time.time() - start_time
            size_mb = os.path.getsize(pcap_path) / 1024 / 1024
            self.log(f" completado: {pcap_name} ({size_mb:.2f}MB en {t_transcurrido:.1f}s)")

        except Exception as e:
            self.log(f"Error {e}")

        finally:
            # Limpiar archivos temporales para ahorrar espacio
            self._cleanup_files(pcap_path, argus_file, zeek_dir)

            # Restaurar configuración original del scanner
            self.scanner.pcap_file = orig_pcap
            self.scanner.argus_output = orig_argus
            self.scanner.zeek_logs_dir = orig_zeek

    def _insert_conexion(self, batch_res):
        """
            inserta en la tabla THREATS los registros de la ultima conn 
        """
        # no especifica el id, se añade automaticamente ? 
        engine = create_engine('sqlite:///ids.db', connect_args={'timeout': 15}, echo=False)
        with engine.begin() as conn:
            batch_res.to_sql('THREATS', con=engine, if_exists='append', method="multi", index=False)

    def shutdown(self):
       self.shutdown_flag.set() 
         
    def _cleanup_files(self, pcap_path, argus_file, zeek_dir):
        """Limpia los archivos utilizados durant el analisis"""
        self.log("Limpiando archivos...")
        if os.path.exists(pcap_path):
            os.remove(pcap_path)
        if os.path.exists(argus_file):
            os.remove(argus_file)
        if os.path.exists(zeek_dir):
            shutil.rmtree(zeek_dir)






