#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import glob
import logging
from datetime import datetime
import shutil
import subprocess
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

fecha = datetime.now().strftime("%Y%m%d_%H%M%S")

proto_map = {
    'tcp': 1, 'udp': 2, 'icmp': 3, 'igmp': 4,
    'ipv6': 5, 'ipv6-frag': 6, 'ipv6-route': 7, 'ipv6-opts': 8,
    'gre': 9, 'sctp': 10, 'rtp': 11,
    'unknown': 0, '-': 0, '': 0
}

service_map = {
    'http': 1, 'https': 2, 'ssl': 3, 'quic-ssl':3,
    'ftp': 4, 'ftp-data': 5,
    'dns': 6, 'dhcp': 7,
    'ssh': 8, 'telnet': 9,
    'smtp': 10, 'pop3': 11, 'imap': 12,
    'snmp': 13, 'radius': 14, 'irc': 15, 'ntp': 16,
    'unknown': 0, '-': 0
}

state_map = {
    'fin':1, 'int':2, 'con':3, 'eco':4,
    'req':5, 'rst': 6, 'par':7, 'urn':8,
    'no':9, 'unknown': 0, '-':0
}

def _series_or_zeros(df, col):
    """Devuelve df[col] si existe; si no, una Serie de ceros con el mismo índice (evita .fillna sobre int)."""
    return df[col] if col in df.columns else pd.Series(0, index=df.index)

class TransformarDatos:
    """
    Clase simplificada que transforma logs de Zeek y datos de Argus
    en un CSV orientado a UNSW-NB15 (versión SIN extras inventados).
    """
    def __init__(self, logs_dir: str = None, output_path: str = None):
        self.logs_dir = logs_dir
        self.output_path = output_path if output_path else 'logs/captura_trafico/network_flow.csv'


    def read_argus_file(self, argus_file):
        """
        Lee el archivo .argus usando ra y extrae características de flujo.
        """
        if not os.path.exists(argus_file):
            logger.error(f"Archivo .argus no encontrado: {argus_file}")
            return pd.DataFrame()

        # Buscar binario ra
        ra_bin = shutil.which("ra") or "/usr/local/bin/ra"
        if not ra_bin or not os.path.exists(ra_bin):
            logger.error("Comando 'ra' no encontrado")
            return pd.DataFrame()

        # Atributos que extrae argus del archivo .argus
        fields = (
            "stime, ltime, saddr, sport, daddr, dport,"
            "proto, state, dur, sbytes, dbytes, spkts, dpkts,"
            "sttl, dttl, rate, sintpkt, dintpkt, "
            "sjit, djit, swin, dwin, stcpb, dtcpb, tcprtt"
        )
        cmd = [
            ra_bin, 
            "-n", 
            "-r", argus_file, 
            "-s", fields, 
            "-c", ",",
            "-Z", "b"
        ]
           
        try:
            logger.info(f"Extrayendo características desde {argus_file}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"Error ejecutando ra: {result.stderr}")
                return pd.DataFrame()

            if not result.stdout.strip():
                logger.warning("Archivo .argus vacío")
                return pd.DataFrame()

            # Convertir a DataFrame
            argus_df = pd.read_csv(StringIO(result.stdout))
            return argus_df

        except Exception as e:
            logger.error(f"Error procesando .argus: {e}")
            return pd.DataFrame()

    def read_zeek_files(self, zeek_dir):
        """
        Lee los archivos .log de Zeek y los convierte en DataFrames.
        (Usaremos solo conn.log y dns.log para 'service').
        """
        log_files = glob.glob(os.path.join(zeek_dir, "*.log"))
        if not log_files:
            logger.error(f"No se encontraron archivos .log en {zeek_dir}")
            return {}

        dataframes_dict = {}
        logger.info(f"Zeek: {len(log_files)} archivos encontrados")

        for file_path in log_files:
            log_name = os.path.basename(file_path).split('.')[0]

            try:
                # Extraer encabezados
                headers = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#fields'):
                            headers = line.strip().replace('#fields\t', '').split('\t')
                            break

                if not headers:
                    continue

                # Leer datos
                df = pd.read_csv(
                    file_path,
                    sep='\t',
                    names=headers,
                    comment='#',
                    na_values=['-', '(empty)'],
                    dtype=str
                )

                if not df.empty:
                    dataframes_dict[log_name] = df
                    logger.info(f"Zeek {log_name}: {len(df)} registros")

            except Exception as e:
                logger.warning(f"Error procesando {file_path}: {e}")

        return dataframes_dict

    def safe_int_convert(self, series, default=0):
        """
        Convierte series a entero manejando valores hexadecimales y otros formatos
        """
        def convert_value(val):
            if pd.isna(val) or val == '-' or val == '':
                return default
            try:
                if isinstance(val, str) and val.startswith('0x'):
                    return int(val, 16)
                return int(float(val))
            except (ValueError, TypeError):
                return default

        return series.apply(convert_value)

    def extract_argus_features(self, argus_df):
        """
        Extrae características base desde Argus (sin inventar timings ni ct_*).
        """
        if argus_df.empty:
            return pd.DataFrame()

        features_df = pd.DataFrame(index=argus_df.index)

        # 1-4: IPs y puertos metadatos para identificar conexiones
        features_df['srcip'] = _series_or_zeros(argus_df, 'SrcAddr').replace(0, '-').fillna('-')
        features_df['sport'] = self.safe_int_convert(_series_or_zeros(argus_df, 'Sport'), 0)
        features_df['dstip'] = _series_or_zeros(argus_df, 'DstAddr').replace(0, '-').fillna('-')
        features_df['dsport'] = self.safe_int_convert(_series_or_zeros(argus_df, 'Dport'), 0)

        # 5-6: Protocolo y estado
        #features_df['proto'] = _series_or_zeros(argus_df, 'Proto').replace(0, '-').fillna('-').astype(str).str.lower()
        #raw_state = _series_or_zeros(argus_df, 'State').replace(0, '-').fillna('-').astype(str).str.lower()
        #features_df['state'] = raw_state
        #features_df['service'] = '-'  # Se rellenará con Zeek

        # 7: Duración
        features_df['dur'] = pd.to_numeric(_series_or_zeros(argus_df, 'Dur'), errors='coerce').fillna(0)

        # 8-9: tamaño bytes
        features_df['sbytes'] = self.safe_int_convert(_series_or_zeros(argus_df, 'SrcBytes'), 0)
        features_df['dbytes'] = self.safe_int_convert(_series_or_zeros(argus_df, 'DstBytes'), 0)

        # 10-11: Paquetes origen y destino
        features_df['spkts'] = self.safe_int_convert(_series_or_zeros(argus_df, 'SrcPkts'), 0)
        features_df['dpkts'] = self.safe_int_convert(_series_or_zeros(argus_df, 'DstPkts'), 0)

        # 12-13: Loads canónicos (bits/s). No usar 'rate' si no viene: lo recalculamos/ajustamos más abajo.
        features_df['sload'] = np.where(features_df['dur'] > 0, (features_df['sbytes'] * 8) / features_df['dur'], 0.0)
        features_df['dload'] = np.where(features_df['dur'] > 0, (features_df['dbytes'] * 8) / features_df['dur'], 0.0)

        # 14-15: Time to live
        features_df['sttl'] = self.safe_int_convert(_series_or_zeros(argus_df, 'sTtl'), 0)
        features_df['dttl'] = self.safe_int_convert(_series_or_zeros(argus_df, 'dTtl'), 0)

        # 16-17: Ventanas
        features_df['swin']  = self.safe_int_convert(_series_or_zeros(argus_df, 'SrcWin'), 0)
        features_df['dwin']  = self.safe_int_convert(_series_or_zeros(argus_df, 'DstWin'), 0)

        # 18-19: tcp base
        features_df['stcpb'] = self.safe_int_convert(_series_or_zeros(argus_df, 'SrcTCPBase'), 0)
        features_df['dtcpb'] = self.safe_int_convert(_series_or_zeros(argus_df, 'DstTCPBase'), 0)

        # 20-21: Inter-arrival
        features_df['sinpkt'] = pd.to_numeric(_series_or_zeros(argus_df, 'SIntPkt'), errors='coerce').fillna(0)
        features_df['dinpkt'] = pd.to_numeric(_series_or_zeros(argus_df, 'DIntPkt'), errors='coerce').fillna(0)

        """
        features_df['tbytes'] = features_df['sbytes'] + df['dbytes']
        df['tpkts'] = df['spkts'] + df['dpkts']
        df['bratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pratio'] = df['spkts'] / (df['dpkts'] + 1)
        """

        epsilon = 1e-8
        # 22 rate
        # Calcular rate si no está; si está, usarlo.
        features_df['rate'] = (features_df['sbytes'] + features_df['dbytes']) / (features_df['dur']+epsilon) 

        # Totales y ratios básicos (existentes)
        features_df['tbytes'] = features_df['sbytes'] + features_df['dbytes']
        features_df['tpkts'] = features_df['spkts'] + features_df['dpkts']
        features_df['bratio'] = features_df['sbytes'] / (features_df['dbytes'] + 1)
        features_df['pratio'] = features_df['spkts'] / (features_df['dpkts'] + 1)
        features_df['rate'] = features_df['sbytes'] + features_df['dbytes'] / (features_df['dur']+epsilon) 
        features_df['basymmetry'] = abs(features_df['sbytes'] - features_df['dbytes']) / (features_df['tbytes'] + 1)
        features_df['pasymmetry'] = abs(features_df['spkts'] - features_df['dpkts']) / (features_df['tpkts'] + 1) 
        # 23-24 jitter
        features_df["sjit"] = pd.to_numeric(_series_or_zeros(argus_df, 'SrcJitter'), errors='coerce').fillna(0)
        features_df["djit"] = pd.to_numeric(_series_or_zeros(argus_df, 'DstJitter'), errors='coerce').fillna(0)

        # 25 tcprtt
        features_df["tcprtt"] = pd.to_numeric(_series_or_zeros(argus_df, 'TcpRtt'), errors='coerce').fillna(0)

        return features_df

    def extract_zeek_features(self, zeek_dict):
        """
        Extrae SOLO 'service' desde Zeek (conn.log y dns.log). 
        """
        zeek_features = pd.DataFrame()

        # Service desde conn.log
        if 'conn' in zeek_dict:
            conn_df = zeek_dict['conn']
            conn_features = pd.DataFrame()

            conn_features['srcip'] = _series_or_zeros(conn_df, 'id.orig_h').replace(0,'-').fillna('-')
            conn_features['sport'] = self.safe_int_convert(conn_df.get('id.orig_p', pd.Series(0, index=conn_df.index)), 0)
            conn_features['dstip'] = conn_df.get('id.resp_h','-').fillna('-')
            conn_features['dsport'] = self.safe_int_convert(conn_df.get('id.resp_p', pd.Series(0, index=conn_df.index)), 0)

            print(list(conn_features.columns))
            conn_features['dur'] = pd.to_numeric(_series_or_zeros(conn_df, 'duration'), errors='coerce').fillna(0.0) 
            conn_features['sbytes'] = self.safe_int_convert(_series_or_zeros(conn_df, 'orig_ip_bytes'), 0)
            conn_features['dbytes'] = self.safe_int_convert(_series_or_zeros(conn_df, 'resp_ip_bytes'), 0)
            conn_features['spkts'] = self.safe_int_convert(_series_or_zeros(conn_df, 'orig_pkts'), 0)
            conn_features['dpkts'] = self.safe_int_convert(_series_or_zeros(conn_df, 'resp_pkts'), 0)
            
            proto = conn_df.get('proto', '-').fillna('-').astype(str).str.lower() 
            #print(f"Proto antes: {proto}")
            #proto = proto.map(proto_map).fillna(0).astype(int)
            #print(f"Proto despues: {proto}")
            conn_features['proto'] = proto 

            service = conn_df.get('service', '-').fillna('-').astype(str).str.lower() 
            #print(f"service antes: {service}")
            #service = service.map(service_map).fillna(0).astype(int)
            #print(f"service desp: {service}")
            conn_features['service'] = service 
            
            conn_features['rate'] = np.where( 
                conn_features['dur'] > 0,
                (conn_features['sbytes'] + conn_features['dbytes']) / conn_features['dur'], 
                0.0
            )
            conn_features['tbytes'] = conn_features['sbytes'] + conn_features['dbytes']
            conn_features['tpkts'] = conn_features['spkts'] + conn_features['dpkts']
            conn_features['basymmetry'] = abs(conn_features['sbytes'] - conn_features['dbytes']) / (conn_features['tbytes'] + 1)
            conn_features['pasymmetry'] = abs(conn_features['spkts'] - conn_features['dpkts']) / (conn_features['tpkts'] + 1)
            zeek_features = conn_features
            #print(conn_features)

        if 'dns' in zeek_dict:
            dns_df = zeek_dict['dns']
            dns_features = pd.DataFrame()

            dns_features['srcip'] = _series_or_zeros(dns_df, 'id.orig_h').replace(0, '-').fillna('-')
            dns_features['sport'] = self.safe_int_convert(_series_or_zeros(dns_df, 'id.orig_p'), 0)
            dns_features['dstip'] = _series_or_zeros(dns_df, 'id.resp_h').replace(0, '-').fillna('-')
            dns_features['dsport'] = self.safe_int_convert(_series_or_zeros(dns_df, 'id.resp_p'), 0)
            dns_features['service'] = 'dns'

            #zeek_features = self.merge_features(zeek_features, dns_features)

        #print(zeek_features)
        return zeek_features

    def merge_features(self, base_features, additional_features):
        """
        Fusiona características basándose en claves comunes (srcip, sport, dstip, dsport)
        """
        if additional_features.empty:
            return base_features

        if base_features.empty:
            return additional_features

        try:
            # Claves comunes para el merge
            key_cols = ['srcip', 'sport', 'dstip', 'dsport']

            # Verificar que las claves existen en ambos DataFrames
            common_keys = [col for col in key_cols if col in base_features.columns and col in additional_features.columns]

            if not common_keys:
                logger.warning("No hay columnas comunes para hacer merge")
                return base_features

            # Columnas adicionales (excluir las claves)
            additional_cols = [col for col in additional_features.columns if col not in common_keys]
            if not additional_cols:
                return base_features

            #Elimina columnas sobreescritas de argus y zeek
            overlap_cols = [col for col in additional_cols if col in base_features.columns]
            if overlap_cols:
                base_features = base_features.drop(columns=overlap_cols)

            # Merge left join
            merged_df = pd.merge(
                base_features,
                additional_features[common_keys + additional_cols],
                on=common_keys,
                how='left'
            )

            if 'service' in merged_df.columns:
                merged_df['service'] = merged_df['service'].fillna('-').astype('str').str.lower()

            return merged_df

        except Exception as e:
            logger.error(f"Error en merge_features: {e}")
            return base_features


    def generate_csv_zeek(self, zeek_dir):
        """
        Genera archivo csv solo utilizando zeek         
        """
        zeek_dir_abs = os.path.abspath(zeek_dir)
        zeek_files = glob.glob(os.path.join(zeek_dir_abs, "*.log"))
        logger.info(f"Buscando archivos de zeek en {zeek_dir_abs}")
        try:
            if zeek_dir and os.path.exists(zeek_dir):
                zeek_dict = self.read_zeek_files(zeek_dir)
                zeek_df = self.extract_zeek_features(zeek_dict)
                if not zeek_df.empty:
                    df = zeek_df
            print(df)
            df.to_csv(self.output_path,index=False)
        except Exception as e:
            logger.error(f"Ocurrio un error {e}")

        return self.output_path

    def generate_csv(self, argus_file, df, zeek_dir=None):
        """
        Función principal que genera el CSV (17 features reales, sin extras inventados).
        """

        # Asegúrate de que la ruta sea absoluta para evitar fallos de ubicación
        zeek_dir_abs = os.path.abspath(zeek_dir)
        zeek_files = glob.glob(os.path.join(zeek_dir_abs, "*.log"))
        
        logger.info(f"Buscando logs en: {zeek_dir_abs}")
        logger.info(f"Archivos encontrados: {zeek_files}")

        try:
            # 1. Leer datos de Argus
            argus_df = self.read_argus_file(argus_file)
            if argus_df.empty:
                logger.error("No se pueden procesar datos sin Argus")
                sys.exit(1)
            
            argus_df = self.extract_argus_features(argus_df)

            # 2. Leer logs de Zeek (solo service)
            if zeek_dir and os.path.exists(zeek_dir):
                zeek_dict = self.read_zeek_files(zeek_dir)
                zeek_df = self.extract_zeek_features(zeek_dict)
                if not zeek_df.empty:
                    
                    #final_dataset = self.merge_features(base_features, zeek_features)
                    if df == "argus":
                        final_dataset = argus_df
                    else:
                        final_dataset = zeek_df 
            else:
                logger.warning("Directorio Zeek no encontrado, usando solo Argus (service='-')")

            # 3. Guardar CSV
            final_dataset.to_csv(self.output_path, index=False)

            # 4. Estadísticas
            #self.show_stats(final_dataset)

            return self.output_path

        except Exception as e:
            logger.error(f"Error en generate_csv: {e}")
            sys.exit(1)


    def pipe_line(argus_file, df, zeek_dir):
        """
        Crea una instancia de TransformarDatos con parámetros predeterminados
        """
        transformer = TransformarDatos(
            logs_dir="logs/captura_trafico",
            output_path=f"data/trafico_red_{fecha}.csv"
        )
        csv_path = transformer.generate_csv(argus_file, df, zeek_dir) 
        return csv_path

    def show_stats(self, df):
        """
        Muestra estadísticas del dataset generado
        """
        logger.info("Estadísticas del dataset")
        logger.info("-------------------------")
        logger.info(f"Total flujos de conexiones: {len(df):,}")
        logger.info(f"Total atributos (columnas): {len(df.columns)}")

        if 'dbytes' in df.columns:
            total_bytes = df['dbytes'].sum()
            logger.info(f"Total dbytes: {int(total_bytes):,}")

        if 'service' in df.columns:
            svc = df['service'].value_counts().to_dict()
            logger.info(f"Services: {svc}")

        logger.info("Dataset (features) generado exitosamente")

if __name__ == "__main__":
    # Ejemplo de uso
    transformer = TransformarDatos(
        logs_dir="../logs/zeek",
        output_path="../data/prueba.csv"
    )
    # Generar CSV con archivo .argus + Zeek (opcional)
    transformer.generate_csv(
        argus_file="../logs/captura_trafico/file.argus",
        zeek_dir="../logs/zeek"
    )
