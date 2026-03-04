#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder,
    label_binarize, RobustScaler, MinMaxScaler
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, roc_auc_score
)
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment
import datetime  # Añadido para consistencia con DecisionTree

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('logger_modelo_nb')

columns_to_drop_cicids = [
    # Columnas redundantes o derivadas
    'TotalLengthofFwdPackets',
    'TotalLengthofBwdPackets',
    'FlowBytes/s',
    'PacketLengthVariance',
    'SubflowFwdPackets',
    'SubflowBwdPackets', 
    'SubflowFwdBytes',
    'SubflowBwdBytes',
    
    # Columnas siempre 0 o valores constantes
    'FwdAvgBytes/Bulk',
    'FwdAvgPackets/Bulk',
    'BwdAvgBytes/Bulk',
    'BwdAvgPackets/Bulk',
    'FwdAvgBulkRate',
    'BwdAvgBulkRate',
    'CWEFlagCount',
    'ECEFlagCount',
    'ActiveMax',
    'ActiveMin',
    'ActiveMean',
    'ActiveStd',
    'IdleMax',
    'IdleMin',
    'IdleMean',
    'IdleStd',
    'FwdPSHFlags',
    'BwdPSHFlags',
    'FwdURGFlags',
    'BwdURGFlags',
    'URGFlagCount',
    
    # Columnas con baja varianza
    'FwdHeaderLength',
    'BwdHeaderLength',
    'FwdHeaderLength.1',  # Duplicada
    'min_seg_size_forward',
    'AvgFwdSegmentSize',
    'AvgBwdSegmentSize'
]

class ModeloNaiveBayes:
    """
    Implementación de Naive Bayes con estructura idéntica a ModeloDecisionTree.
    Permite intercambiar modelos sin modificar el pipeline de procesamiento.
    """

    def __init__(self):
        self.var_objetivo = None
        self.modelo = None
        self.scaler = None
        self.scaler_categorico = None
        self.features_seleccionadas = []
        self.best_features = []
        self.feature_selector = None
        self.balanceado = False
        self.tipo_scaler_cat = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.label_encoders = {}


    def cargar_datos(self):
        """Funcion donde carga datos de entrenamiento y test"""

        logger.info("Cargando datos")

        training_data = pd.read_csv("../../datasets/UNSW_NB15_training-set.csv")
        testing_data = pd.read_csv("../../datasets/UNSW_NB15_testing-set.csv")
        features = pd.read_csv("../../datasets/NUSW-NB15_features.csv", encoding='cp1252')
        cols = list(features["Name"])
        col_normal = []
        for col in cols:
            col_norm = col.replace(" ", "").strip().lower()
            col_normal.append(col_norm)
        print(col_normal)
        data1 = pd.read_csv("../../datasets/UNSW-NB15_1.csv", header=None)
        data2 = pd.read_csv("../../datasets/UNSW-NB15_2.csv", header=None)
        data3 = pd.read_csv("../../datasets/UNSW-NB15_3.csv", header=None)
        data4 = pd.read_csv("../../datasets/UNSW-NB15_4.csv", header=None)
        
        data1.columns = col_normal 
        data2.columns = col_normal 
        data3.columns = col_normal
        data4.columns = col_normal
        
        training_set = pd.concat([data1,data2,data3,data4], ignore_index=True)
        training_set = self.create_argus_flow_features(training_set)

        num_ataques = 0
        for _, col in training_data.iterrows():
            if col['label'] == 1:
               num_ataques+=1

        print(num_ataques)
        #testing_data
        return training_set


    def cargar_datos_CIC(self):

        logger.info("Cargando datos CIC")
        cic_1 = pd.read_csv("../../datasets/CIC-1.csv")     
        cic_2 = pd.read_csv("../../datasets/CIC-2.csv")     
        cic_3 = pd.read_csv("../../datasets/CIC-3.csv")     
        cic_4 = pd.read_csv("../../datasets/CIC-4.csv")     
        cic_5 = pd.read_csv("../../datasets/CIC-5.csv")     
        cic_6 = pd.read_csv("../../datasets/CIC-6.csv")     
        cic_7 = pd.read_csv("../../datasets/CIC-7.csv")     

        cols = [col.replace(' ', '') for col in cic_1.columns]
        print(cols)
        
        cic = [cic_1, cic_2, cic_3, cic_4, cic_5, cic_6, cic_7]
        for df in cic:
            df.columns = cols

        training_dataset = pd.concat([cic_1, cic_2, cic_3, cic_4, cic_5, cic_6, cic_7], ignore_index=True)
        print(training_dataset)
        return training_dataset 

    def create_argus_flow_features(self, df):
        """Features básicos derivados de métricas de flujo de Argus"""
    
        epsilon = 1e-8
        
        # Totales y ratios básicos (existentes)
        df['tbytes'] = df['sbytes'] + df['dbytes']
        df['tpkts'] = df['spkts'] + df['dpkts']
        df['bratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['rate'] = (df['sbytes'] + df['dbytes']) / (df['dur']+epsilon) 
        # Tamaños promedio de paquetes (existentes)
        df['avg_pkt_size_src'] = df['sbytes'] / (df['spkts'] + 1)
        df['avg_pkt_size_dst'] = df['dbytes'] / (df['dpkts'] + 1)
        df['avg_pkt_size_flow'] = df['tbytes'] / (df['tpkts'] + 1)
        
        # Asimetría (existentes)
        df['basymmetry'] = abs(df['sbytes'] - df['dbytes']) / (df['tbytes'] + 1)
        df['pasymmetry'] = abs(df['spkts'] - df['dpkts']) / (df['tpkts'] + 1)
        
        # 1. Características de eficiencia y densidad
        df['bytes_per_second'] = df['tbytes'] / (df['dur'] + epsilon)
        df['packets_per_second'] = df['tpkts'] / (df['dur'] + epsilon)
        
        # 2. Ratios de normalización
        df['byte_intensity'] = df['tbytes'] / (df['dur'] * df['tpkts'] + epsilon)
        df['packet_size_variability'] = abs(df['avg_pkt_size_src'] - df['avg_pkt_size_dst']) / (df['avg_pkt_size_flow'] + epsilon)
        
        # 3. Características de balance con signo
        df['byte_balance_signed'] = (df['sbytes'] - df['dbytes']) / (df['tbytes'] + epsilon)
        df['packet_balance_signed'] = (df['spkts'] - df['dpkts']) / (df['tpkts'] + epsilon)
        
        # 4. Características de interacción no lineales
        df['flow_complexity'] = df['tpkts'] * df['avg_pkt_size_flow'] / (df['dur'] + epsilon)
        df['traffic_burstiness'] = df['packets_per_second'] * df['byte_balance_signed']
        
        # 5. Características de proporción compuesta
        df['efficiency_ratio'] = df['bytes_per_second'] / (df['packets_per_second'] + epsilon)
        df['size_consistency'] = df['avg_pkt_size_src'] / (df['avg_pkt_size_dst'] + epsilon)
        
        # 6. Características basadas en rate existente
        df['rate_byte_ratio'] = df['rate'] / (df['bytes_per_second'] + epsilon)
        df['rate_packet_ratio'] = df['rate'] / (df['packets_per_second'] + epsilon)
        
        # 7. Características de distribución
        df['src_dominance'] = (df['sbytes'] * df['spkts']) / (df['tbytes'] * df['tpkts'] + epsilon)
        df['flow_symmetry_index'] = 1 - (df['basymmetry'] + df['pasymmetry']) / 2
        
        # 8. Características de throughput normalizado
        df['normalized_throughput'] = df['tbytes'] * df['rate'] / (df['dur'] + epsilon)
        df['throughput_efficiency'] = df['tbytes'] / (df['dur'] * df['rate'] + epsilon)
        
        # 9. Características de relación temporal
        df['duration_density'] = df['tpkts'] / (df['dur'] + epsilon)
        df['byte_throughput_ratio'] = df['tbytes'] / (df['rate'] + epsilon)
        
        # 10. Características de variabilidad compuesta
        df['traffic_unbalance'] = abs(df['byte_balance_signed'] - df['packet_balance_signed'])
        df['size_packet_correlation'] = df['avg_pkt_size_flow'] * df['tpkts'] / (df['tbytes'] + epsilon)
    
        return df

    def cargar_datos_analisis(self, ruta):
        """Función que carga una muestra de datos de tráfico de red real"""
        test_data_real = pd.read_csv(ruta)
        return test_data_real

    def limpiar_datos(self, X, real_test=False):
        """Elimina features innecesarias para el modelo"""
        metadata_features = ['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'service']
        non_metadata_features = ['ct_state_ttl', 'sttl', 'dttl', 'ct_dst_ltm', 'ct_srv_dst', 'ct_dst_sct_ltm',
                                'ct_srv_ltm', 'ct_srv_src', 'state', 'ct_dst_srpot_ltm']
        if real_test:
            col_id = ['id']
        else:
            col_id = ['id', 'attack_cat', 'label']

        X = X.drop(columns=metadata_features, errors="ignore")
        X = X.drop(columns=non_metadata_features, errors="ignore")
        X = X.drop(columns=col_id, errors='ignore')
        return X

    def get_best_features(self):
        """Devuelve las mejores características seleccionadas"""
        return self.best_features

    def set_numeric_scaler(self, scaler):
        """Establece un scaler para datos numéricos"""
        self.scaler = scaler

    def get_numeric_scaler(self):
        return self.scaler

    def set_cat_scaler(self, scaler):
        """Establece un scaler para datos categóricos"""
        self.scaler_categorico = scaler

    def get_cat_scaler(self):
        return self.scaler_categorico

    def select_scaler(self, tipo):
        """Selecciona el tipo de scaler: standard, minmax, robust, etc"""
        if tipo == "standard":
            scaler = StandardScaler()
        elif tipo == "min_max":
            scaler = MinMaxScaler()
        elif tipo == "robust":
            scaler = RobustScaler()
        else:
            print("Scaler numérico no reconocido | standard, min_max, robust")
            return
        self.set_numeric_scaler(scaler)

    def set_tipo_scaler_cat(self, tipo_scaler_cat):
        self.tipo_scaler_cat = tipo_scaler_cat

    def get_tipo_scaler_cat(self):
        return self.tipo_scaler_cat

    def select_cat_scaler(self):
        """Selecciona tipo de scaler categórico"""
        if self.get_tipo_scaler_cat() == "label":
            scaler = LabelEncoder()
        elif self.get_tipo_scaler_cat() == "one_hot":
            scaler = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            print("Scaler categórico no reconocido | label, one_hot")
            return

        self.set_cat_scaler(scaler)


    def limpiar_datos_cic(self, X):
        """Funcion que se encarga de eliminar valores nulos y filas redundantes" que el modelo no puede procesar"""

        col_categoricas = [col for col in X.select_dtypes(include=[np.object_]).columns.tolist()]
        
        for col in X.columns:
            if col not in col_categoricas:
                X[col] = X[col].replace('-', np.nan)
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].replace([np.nan, np.inf, -np.inf], 0)

        return X

    def preprocesar_datos_cic(self, X, train=True, limpiar_datos=False):
        """Funcion encargada de preprocesar datos del dataset cic-ids-2017
            X: dataset de entrenamiento o test
            train: determina si se esta entrenando los datos, crear el scaler y determinar si hace fit o fit y transform
            return: X_train_procesado: datos ya procesados

            si limpiar datos es False no aplica el transformador y solo elimina nan, valores a inf o a 0
        """
        
        logger.info("Preprocesando datos de CIC-IDS2017") 
        X = X.copy()
        X = X.reset_index(drop = True)
        
        if limpiar_datos:
            X = self.limpiar_datos_cic(X)
        
        col_categoricas = [col for col in X.select_dtypes(include=[np.object_]).columns.tolist()]
        col_numericas = [col for col in X.select_dtypes(include=[np.number]).columns.tolist() if col not in col_categoricas]        
        
        if limpiar_datos is False:  
            scaler = self.get_numeric_scaler()
            if train:
                X_num = scaler.fit_transform(X[col_numericas])
            else:
                X_num = scaler.transform(X[col_numericas])
            
            X[col_numericas] = X_num 

        return X

    def preprocesar_datos_unsw(self, X, train=True):
        """Funcion que preprocesa los datos antes de entrenar el modelo
            X: dataset que quiera realizar preprocesado de datos con los escalers indicados
            train: determina si esta entrenando los datos, para hacer fit y transform o solo transform
            return: X_train_procesado: datos procesados
        """
        logger.info("Preprocesando datos")

        X = X.copy()
        X = X.reset_index(drop=True)
        #X = X.drop_duplicates()

        # Manejar valores nulos para que todos los valores sean numÃ©ricos para mejor rendimiento modelo
        col_categoricas = [col for col in ['proto','service','state'] if col in X.columns] 
        col_numericas_originales = X.select_dtypes(exclude=['object']).columns.tolist()

        #X['attack_cat'].fillna('normal',inplace = True)
        #X['attack_cat'] = X['attack_cat'].apply(lambda x: x.strip().lower()) 

        datos_categoricos = {}
        for col in col_categoricas:
            datos_categoricos[col] = X[col].copy()  

        # Solo procesar columnas que no son categóricas
        for col in X.columns:
            if col not in col_categoricas:
                X[col] = X[col].replace('-', np.nan)
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].replace([np.nan, np.inf, -np.inf], 0)

        col_numericas = [col for col in X.select_dtypes(include=[np.number]).columns.tolist() if col not in col_categoricas]
        print(col_categoricas)
        print(col_numericas) 
        #print(X.columns) 
        # Aplicar transformador a datos numericos 

        scaler = self.get_numeric_scaler()
        if train:
            X_num = scaler.fit_transform(X[col_numericas])
        else:
            X_num = scaler.transform(X[col_numericas])
        
        X_escalado = pd.DataFrame(
            data=X_num,
            columns=col_numericas,
            index=X.index
        )
        
        X[col_numericas] = X_num 
        
        # codifica valores de proto, service y state para entrenamiento
        self.crear_mapeos_agrupados()

        for col in col_categoricas:
            valores_originales = datos_categoricos[col]
            if col == 'proto':
                X['proto'] = valores_originales.map(self.proto_map).fillna(0).astype(int)
            elif col == 'service':
                service_clean = X['service'].astype(str).str.lower()
                X['service'] = service_clean.map(self.service_map).fillna(0).astype(int)
            elif col == 'state' and 'state' in col_categoricas:
                state_clean = X['state'].astype(str).str.lower()
                X['state'] = state_clean.map(self.state_map).fillna(0).astype(int)

        # Debug
        print(f"Service valores {X['service']}")
        print(f"Proto valores {X['proto']}") 
        #print(f"Status valores: {X['state']}")
        #X_procesado = pd.concat([X_escalado, X], axis=1)

        #print(X)
        return X
    
    def crear_mapeos_agrupados(self):
        """En base a los valores obtenidos, crea diccionarios para codificar los valores que pueden tomar"""
        self.proto_map = {
            'tcp': 1, 'udp': 2, 'icmp': 3,
            
            'igmp': 4,          # Separado de ICMP
            'ipv6': 5,          # IPv6 base
            'ipv6-frag': 6,     # Fragmentación IPv6
            'ipv6-route': 7,    # Routing IPv6
            'ipv6-opts': 8,     # Opciones IPv6
            'gre': 9,           # GRE tunneling
            'sctp': 10,         # SCTP
            'rtp': 11,          # RTP

            'unknown': 0, '-': 0
        }
        
        self.service_map = {
            'http': 1, 'https': 2, 'ssl': 3, 'quic-ssl':3,
            'ftp': 4, 'ftp-data': 5,
            'dns': 6, 'dhcp': 7,
            'ssh': 8, 'telnet': 9,
            'smtp': 10, 'pop3': 11, 'imap': 12,
            'snmp': 13, 'radius': 14, 'irc': 15, 'ntp': 16,
            'unknown': 0, '-': 0
        } 

        self.state_map = {
            'fin':1, 'int':2, 'con':3, 'eco':4,
            'req':5, 'rst': 6, 'par':7, 'urn':8,
            'no':9, 'unknown': 0, '-':0
        }



    def seleccionar_best_features(self, X, y):
        """Devuelve una lista con los atributos mÃ¡s importantes
           return: mejores_features
        """
        col_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[col_numericas]
        logger.info(f"Selecccinonando best features con {self.feature_selector}")
        # KBest
        if self.feature_selector == "kbest":
            logger.info("kbest")
            #f_classif | mutual_info_classif
            selector = SelectKBest(mutual_info_classif, k=min(20, X.shape[1]))
            mejores_features = selector.fit(X[col_numericas], y)
            mask_best = selector.get_support()
            kbest_features = X_num.columns[mask_best].tolist()
            mejores_features = kbest_features
            print("\nMejores features KBest")
            print("------------------------")
            print(kbest_features)

        # PCA
        if self.feature_selector == "pca":
            num = len(col_numericas)
            pca = PCA(n_components=num)
            X_pca = pca.fit(X_num)
            cargas = pd.DataFrame(
                np.abs(pca.components_.T),
                index = X_num.columns,
                columns = [f'PC{i+1}'for i in range(len(col_numericas))]
            )
            importancia_total = cargas.sum(axis=1)
            importancia_total = importancia_total.sort_values(ascending=False)
            top_n_pca = 15
            pca_features = importancia_total.head(top_n_pca).index.tolist()
            mejores_features = pca_features
            print("\nMejores features numericas PCA")
            print("-------------------------------")
            print(pca_features)

        # XGboost
        if self.feature_selector == "xgboost":
            col_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
            X_num = X[col_numericas].copy()
            X_num = X_num.replace([np.inf, -np.inf], np.nan)
            X_num = X_num.fillna(0)
            feature_names = X_num.columns.tolist()
            X_array = X_num.values
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            xgb_model = XGBClassifier(
                eval_metric='mlogloss',
                tree_method='hist'
            )
            xgb_model.fit(X_array, y_encoded)
            importancias = xgb_model.feature_importances_
            importancia_df = pd.DataFrame({
                'Feature': X_num.columns,
                'Importance': importancias
            }).sort_values('Importance', ascending=False)
            xgboost_features = importancia_df.head(20)['Feature'].tolist()
            mejores_features = xgboost_features
            print(f"\nTop {len(xgboost_features)} features XGBoost:")
            print(xgboost_features)

        self.best_features = mejores_features
        return mejores_features

    def balancear_datos(self, X, y):
        logger.info("Balanceando datos con SMOTE y RandomUnderSampler")
    
        # 1. Ajustar k_neighbors dinámicamente
        # SMOTE necesita al menos (k_neighbors + 1) muestras de una clase para funcionar.
        counts = Counter(y)
        min_samples = min(counts.values())
        k = min(5, min_samples - 1) if min_samples > 1 else 1
        
        # 2. Definir una estrategia proporcional
        # En lugar de un número fijo, podemos decirle que suba las minoritarias 
        # y baje la clase 'normal' que tiene +600k filas.
        # Antes del balanceo

        # Ejemplo: Subir clases pequeñas a 10,000 y bajar 'normal' a 50,000
        target_oversample = {cls: 10000 for cls, count in counts.items() if count < 10000}
        target_undersample = {cls: 50000 for cls, count in counts.items() if count > 50000}

        pipeline = Pipeline([
            ('oversample', SMOTE(sampling_strategy=target_oversample, random_state=42)),
            ('undersample', RandomUnderSampler(sampling_strategy=target_undersample, random_state=42))
        ])

        X_res, y_res = pipeline.fit_resample(X, y)
        
        logger.info(f"Distribución tras balanceo: {Counter(y_res)}")
        self.balanceado = True

        return X_res, y_res

    def busqueda_hiperparametros(self, X, y):
        """Búsqueda de hiperparámetros para Gaussian Naive Bayes"""
        param_grid = {
            'var_smoothing': np.logspace(-9, -3, 7)  # Más opciones para explorar
        }

        grid_search = GridSearchCV(
            estimator=GaussianNB(),
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        self.logger.info(f"Mejores parámetros encontrados: {best_params}")
        return best_params

    
    def entrenar_modelo(self, X, X_test, y, y_test):
        """Entrenamiento del modelo Gaussian Naive Bayes"""
        self.logger.info("Entrenamiento del modelo GaussianNB")
        print(f"Features utilizadas para entrenamiento: {len(self.features_seleccionadas)}")
        print(f"Tamaño conjunto entrenamiento: {X.shape}")
        print(f"Tamaño conjunto prueba: {X_test.shape}")

        print("\nDistribución de clases original:")
        distribucion_original = Counter(y)
        for clase, cantidad in sorted(distribucion_original.items()):
            print(f"  {clase}: {cantidad:6d}")

        # Configuración del modelo Naive Bayes
        gaussian_nb = GaussianNB(
            var_smoothing=1e-9
        )

        self.logger.info("Entrenando modelo...")
        t_ini = time.time() 
        gaussian_nb.fit(X, y)
        t_fin = time.time()
        print(f"tiempo entrenamiento: {t_fin - t_ini}")
        t_ini = time.time()
        y_pred = gaussian_nb.predict(X_test)
        t_fin = time.time()
        print(f"Tiempo test={t_fin - t_ini}")
        y_pred_proba = gaussian_nb.predict_proba(X_test)

        self.obtener_resultados_clasificacion(y_test, y_pred)
        fpr, tpr, roc_auc, clases = self.calcular_roc_auc(y_test, y_pred_proba)
        self.visualizar_roc_curve(fpr, tpr, roc_auc, clases)

        # Validación cruzada
        self.validacion_cruzada(gaussian_nb, X, y)

        self.modelo = gaussian_nb
        return gaussian_nb

    def prediccion_real(self, X, modelo):
        """Realiza una predicción de prueba en datos reales"""
        self.logger.info("Realizando una predicción de prueba")
        y_pred = modelo.predict(X)
        y_pred_proba = modelo.predict_proba(X)
        return y_pred, y_pred_proba

    @classmethod
    def cargar_modelo(cls, ruta):
        """Carga un modelo previamente guardado"""
        data = joblib.load(str(ruta))
        if isinstance(data, dict):
            instancia = cls()
            instancia.modelo = data['modelo']
            instancia.scaler = data.get('scaler')
            instancia.scaler_categorico = data.get('scaler_categorico')
            instancia.features_seleccionadas = data.get('features_seleccionadas', [])
            instancia.best_features = data.get('best_features')
            instancia.feature_selector = data.get('feature_selector')
            instancia.tipo_scaler_cat = data.get('tipo_scaler_cat')
            if 'label_encoders' in data:
                instancia.label_encoders = data['label_encoders']
            return instancia
        return data

    def guardar_modelo(self, ruta=None):
        """Serializa una instancia de la clase"""
        if ruta is None:
            ruta = self.path_modelo  # Nota: path_modelo no está definido, usar con cuidado

        data = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'scaler_categorico': self.scaler_categorico,
            'tipo_scaler_cat': self.tipo_scaler_cat,
            'features_seleccionadas': self.features_seleccionadas,
            'feature_selector': self.feature_selector,
            'best_features': self.best_features,
            'label_encoders': self.label_encoders
        }
        joblib.dump(data, ruta)
        self.logger.info(f"Modelo guardado en {ruta} con toda la configuración")

    def calcular_roc_auc(self, y_test, y_pred_proba):
        """Calcula ROC curve y AUC score para clasificación multiclase"""
        clases = np.unique(y_test)
        n_clases = len(clases)

        y_test_binarized = label_binarize(y_test, classes=clases)

        if n_clases == 2:
            y_test_binarized = y_test_binarized.ravel()
            y_pred_proba = y_pred_proba[:, 1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        if n_clases == 2:
            fpr[0], tpr[0], _ = roc_curve(y_test_binarized, y_pred_proba)
            roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            for i in range(n_clases):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        if n_clases > 2:
            roc_auc_promedio = np.mean(list(roc_auc.values()))
        else:
            roc_auc_promedio = roc_auc[0]

        print("Curva ROC Y Puntuación AUC")
        print("--------------------------")

        for i, clase in enumerate(clases):
            if n_clases == 2 and i == 1:
                break
            idx = 0 if n_clases == 2 else i
            print(f"Clase {clase}: AUC = {roc_auc[idx]:.3f}")

        print(f"AUC Promedio: {roc_auc_promedio:.3f}")

        return fpr, tpr, roc_auc, clases

    def obtener_resultados_clasificacion(self, y_test, y_pred, y_pred_proba=None, y_test_original=None):
        """
        Calcula e imprime por-clase las métricas solicitadas. Para multiclass imprime
        la tabla "por_clase" con las columnas:
          TP, TN, FP, FN, TPR, TNR, FPR, FNR, Support

        Muestra por pantalla la tabla formateada con 2 decimales y además
        devuelve (detalle_df, otros_df, cm) para uso posterior.
        """
        # Asegurar arrays numpy
        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)
        n = len(y_test_arr)

        # Etiquetas presentes (orden determinista)
        labels = np.unique(np.concatenate([y_test_arr, y_pred_arr]))
        labels = list(labels)
        n_clases = len(labels)

        # Métricas globales (impresas en pantalla)
        average_target = 'binary' if n_clases == 2 else 'weighted'
        accuracy = accuracy_score(y_test_arr, y_pred_arr)
        precision = precision_score(y_test_arr, y_pred_arr, average=average_target, zero_division=0)
        recall = recall_score(y_test_arr, y_pred_arr, average=average_target, zero_division=0)
        f1 = f1_score(y_test_arr, y_pred_arr, average=average_target, zero_division=0)

        print(f"{'='*60}")
        print("RESULTADOS CLASIFICACION")
        print(f"{'='*60}")
        print(f"Clases detectadas en datos: {labels}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        # AUC-ROC si se proporcionan probabilidades
        if y_pred_proba is not None:
            try:
                if n_clases == 2:
                    probs_pos = (y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
                    auc_score = roc_auc_score(y_test_arr, probs_pos)
                else:
                    auc_score = roc_auc_score(y_test_arr, y_pred_proba, multi_class='ovr', average='weighted')
                print(f"AUC-ROC:   {auc_score:.4f}")
            except Exception as e:
                print(f"No se pudo calcular AUC-ROC: {e}")

        # Matriz de confusión (asegurando orden de labels)
        cm = confusion_matrix(y_test_arr, y_pred_arr, labels=labels)

        print(f"{'='*60}")
        print("MATRIZ DE CONFUSION (filas=real, columnas=predicha)")
        print(f"{'='*60}")
        print(cm)

        # Cálculo por clase: TP, FN, FP, TN
        per_class_tp = np.diag(cm).astype(int)
        per_class_support = cm.sum(axis=1).astype(int)   # TP + FN
        per_class_fn = per_class_support - per_class_tp  # FN
        per_class_fp = cm.sum(axis=0).astype(int) - per_class_tp  # FP
        per_class_predicted = per_class_tp + per_class_fp        # TP + FP
        per_class_tn = np.array([n - tp - fn - fp for tp, fn, fp in zip(per_class_tp, per_class_fn, per_class_fp)], dtype=int)

        # Cabecera personalizada (mostramos TP, TN, FP, FN y tasas)
        header = f"{'Clase':>25} {'Soporte':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6} {'TPR':>8} {'FPR':>8} {'TNR':>8} {'FNR':>8}"
        print(header)
        print('-' * len(header))

        # Imprimir por clase con formato (2 decimales en tasas)
        for idx, label in enumerate(labels):
            tp_c = int(per_class_tp[idx])
            fn_c = int(per_class_fn[idx])
            fp_c = int(per_class_fp[idx])
            tn_c = int(per_class_tn[idx])
            support_c = int(per_class_support[idx])

            # Tasas defensivas (evitan división por cero)
            tpr_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0   # sensibilidad
            fnr_c = fn_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            fpr_c = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0
            tnr_c = tn_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0

            print(f"{str(label):25s} {support_c:8d} {tp_c:6d} {tn_c:6d} {fp_c:6d} {fn_c:6d} {tpr_c:8.2f} {fpr_c:8.2f} {tnr_c:8.2f} {fnr_c:8.2f}")

        # Construir DataFrame por clase (útil para exportar)
        detalle_rows = []
        for i, lab in enumerate(labels):
            TP = int(per_class_tp[i])
            FN = int(per_class_fn[i])
            FP = int(per_class_fp[i])
            TN = int(per_class_tn[i])
            Support = int(per_class_support[i])

            TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            FNR = FN / (TP + FN) if (TP + FN) > 0 else np.nan
            FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
            TNR = TN / (FP + TN) if (FP + TN) > 0 else np.nan

            detalle_rows.append({
                'Clase': lab,
                'Support': Support,
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TPR': TPR,
                'FPR': FPR,
                'TNR': TNR,
                'FNR': FNR
            })

        detalle_df = pd.DataFrame(detalle_rows).set_index('Clase')

        # Comprobaciones: aseguran que cuadra con la muestra
        assert int(detalle_df['TP'].sum() + detalle_df['FN'].sum()) == n, "suma TP+FN != tamaño de la muestra"
        assert int((detalle_df['TP'] + detalle_df['FP']).sum()) == n, "suma Predichas != tamaño de la muestra"

        # Imprimir la tabla por clase completa (DataFrame) con 2 decimales en tasas
        print(f"{'='*60}")
        print("TABLA POR CLASE (DataFrame):")
        print(f"{'='*60}")
        # Formatear columnas float a 2 decimales
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print(detalle_df[['Support','TP','TN','FP','FN','TPR','FPR','TNR','FNR']].to_string())

        # Imprimir classification_report tradicional
        print("CLASSIFICATION REPORT (scikit-learn):")
        print(classification_report(y_test_arr, y_pred_arr, labels=labels, zero_division=0, digits=4))

        # Visualizar matriz de confusión anotada
        plt.figure(figsize=(6 + n_clases, 4 + n_clases*0.4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Matriz de Confusión - Naive Bayes')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10)
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        plt.tight_layout()

        confusion_dir = os.path.join('..', '..', 'imagenes')
        os.makedirs(confusion_dir, exist_ok=True)
        confusion_file = os.path.join(confusion_dir, 'matriz_confusion_modelo_nb.png')
        plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Matriz de confusión guardada en: {confusion_file}")

        # otros_df vacío para compatibilidad
        otros_df = pd.DataFrame()

        return detalle_df, otros_df, cm

    def visualizar_roc_curve(self, fpr, tpr, roc_auc, clases):
        """Visualiza las curvas ROC calculadas"""
        plt.figure(figsize=(12, 8))

        n_clases = len(clases)
        colores = plt.cm.Set1(np.linspace(0, 1, n_clases))

        if n_clases == 2:
            plt.plot(fpr[0], tpr[0], color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc[0]:.3f})')
        else:
            for i, (clase, color) in enumerate(zip(clases, colores)):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve - {clase} (AUC = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Ratio Falsos Positivos')
        plt.ylabel('Ratio Verdaderos Positivos')
        plt.title('Curvas ROC - Gaussian Naive Bayes')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.savefig("../../imagenes/modelo_nb_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nGráfico ROC curve guardado en: modelo_nb_roc_curve.png")

    def validacion_cruzada(self, modelo, X, y, cv=5):
        """Validación cruzada para evaluar robustez"""
        scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')
        scores_f1 = cross_val_score(modelo, X, y, cv=cv, scoring='f1_weighted')

        # Para Naive Bayes, roc_auc_ovo puede ser problemático, usar roc_auc_ovr
        try:
            scores_roc = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc_ovr')
        except:
            # Si falla, intentar con ovo
            try:
                scores_roc = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc_ovo')
            except:
                scores_roc = np.array([np.nan])

        print(f"\n{'='*60}")
        print("VALIDACION CRUZADA")
        print(f"{'='*60}")
        print(f"Accuracy CV:  {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"F1-Score CV:  {scores_f1.mean():.3f} ± {scores_f1.std():.3f}")
        if not np.isnan(scores_roc).all():
            print(f"ROC-AUC CV:   {scores_roc.mean():.3f} ± {scores_roc.std():.3f}")
        else:
            print("ROC-AUC CV:   No disponible para esta configuración")

   
def main():
    modelo_nb = ModeloNaiveBayes()

    # Configuración inicial - Optimizada para Naive Bayes
    modelo_nb.set_tipo_scaler_cat("label")
    modelo_nb.feature_selector = "xgboost"
    modelo_nb.select_scaler("standard")
    modelo_nb.select_cat_scaler  

    dataset = "cic"
    if dataset == "cic":
        X_train = modelo_nb.cargar_datos_CIC()
        y_train = X_train['Label']
        X_train = X_train.drop('Label', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        X_train = modelo_nb.preprocesar_datos_cic(X_train, train=False, limpiar_datos=True)
        X_test = modelo_nb.preprocesar_datos_cic(X_test, train=False, limpiar_datos=True)
        best_features = modelo_nb.seleccionar_best_features(X_train, y_train)
        modelo_nb.features_seleccionadas = best_features 
        X_train = X_train[best_features]
        X_test = X_test[best_features]
        X_train = modelo_nb.preprocesar_datos_cic(X_train, train=True)
        X_test = modelo_nb.preprocesar_datos_cic(X_test, train=False)
        X_train, y_train = modelo_nb.balancear_datos(X_train, y_train)
        modelo = modelo_nb.entrenar_modelo(X_train, X_test, y_train, y_test)
        modelo_nb.guardar_modelo("../../models/naive_bayes.pkl")
        return 

    X_train = modelo_nb.cargar_datos()
    X_train['attack_cat'].fillna('normal', inplace=True)
    X_train['attack_cat'] = X_train['attack_cat'].apply(lambda x:x.strip().lower())
    X_train['attack_cat'] = X_train['attack_cat'].replace(
        'backdoors',
        'backdoor'
    ) 

    y_train = X_train['attack_cat'] 

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, random_state=42, test_size=0.3
    )

    X_train = modelo_nb.preprocesar_datos_unsw(X_train, train=True)
    X_test = modelo_nb.preprocesar_datos_unsw(X_test, train=False)

    mini_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'proto', 'service', 'tbytes', 'tpkts', 'basymmetry', 'pasymmetry']  

    X_train = X_train[mini_features]
    X_test = X_test[mini_features]
    X_train, y_train = modelo_nb.balancear_datos(X_train, y_train)
    modelo = modelo_nb.entrenar_modelo(X_train, X_test, y_train, y_test)


    # Búsqueda de hiperparámetros (opcional, comentar para ejecución rápida)
    # hiperparametros = modelo_nb.busqueda_hiperparametros(X_train, y_train)
    # print(hiperparametros)


    # Nota: Para Naive Bayes, el balanceo puede no ser ideal ya que asume independencia
    # y el sobremuestreo puede distorsionar las distribuciones de probabilidad
    #X_train, y_train = modelo_nb.balancear_datos(X_train, y_train, "smote", balanceado=False)

    


if __name__ == "__main__":
    main()
