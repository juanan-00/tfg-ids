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
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, roc_auc_score
)
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('logger_modelo_dt')

columnas_a_eliminar = [
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
    'PSHFlagCount',
    'ACKFlagCount',

    'FwdHeaderLength',
    'BwdHeaderLength',
    'FwdHeaderLength.1',
    'min_seg_size_forward',
    'AvgFwdSegmentSize',
    'AvgBwdSegmentSize'
]


class ModeloDecisionTree:
    def __init__(self, path_modelo="../../models/decision_tree.pkl"):
        self.var_objetivo = None
        self.modelo = None
        self.scaler = None
        self.scaler_categorico = None
        self.features_seleccionadas = []
        self.best_features = []
        self.feature_selector = None
        self.balanceado = False
        self.tipo_scaler_cat = None
        self.path_modelo = path_modelo
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
        #cic_2 = pd.read_csv("../../datasets/CIC-2.csv")
        cic_3 = pd.read_csv("../../datasets/CIC-3.csv")
        cic_4 = pd.read_csv("../../datasets/CIC-4.csv")
        cic_5 = pd.read_csv("../../datasets/CIC-5.csv")
        cic_6 = pd.read_csv("../../datasets/CIC-6.csv")
        cic_7 = pd.read_csv("../../datasets/CIC-7.csv")
        cic_8 = pd.read_csv("../../datasets/CIC-8.csv")

        cols = [col.replace(' ', '') for col in cic_1.columns]
        print(cols)

        cic = [cic_1, cic_3, cic_4, cic_5, cic_6, cic_7, cic_8]
        for df in cic:
            df.columns = cols

        training_dataset = pd.concat([cic_1, cic_3, cic_4, cic_5, cic_6, cic_7, cic_8], ignore_index=True)
        print(training_dataset)
        return training_dataset

    def cargar_datos_analisis(self, ruta):
        """Funcion que carga una muestra de datos de tráfico de red real"""
        test_data_real = pd.read_csv(ruta)
        return test_data_real

    def create_argus_flow_features(self, df):
        """Features básicos derivados de métricas de flujo de Argus"""

        epsilon = 1e-8

        # Totales y ratios básicos (existentes)
        df['tbytes'] = df['sbytes'] + df['dbytes']
        df['tpkts'] = df['spkts'] + df['dpkts']
        df['bratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['rate'] = df['sbytes'] + df['dbytes'] / (df['dur']+epsilon)
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
        return self.best_features

    def set_numeric_scaler(self, scaler):
        """Establece un scaler para datos numericos"""
        self.scaler = scaler

    def get_numeric_scaler(self):
        return self.scaler

    def set_cat_scaler(self, scaler):
        """Establece un scaler para datos categoricos"""
        self.scaler_categorico = scaler

    def get_cat_scaler(self):
        return self.scaler_categorico

    def select_scaler(self, tipo):
        """Selecciona el tipo de scaler, standard, minmax, robust, etc"""
        if tipo == "standard":
            scaler = StandardScaler()
        elif tipo == "min_max":
            scaler = MinMaxScaler()
        elif tipo == "robust":
            scaler = RobustScaler()
        else:
            print("Scaler numerico no reconocido | standard, min_max, robust")
        self.set_numeric_scaler(scaler)

    def set_tipo_scaler_cat(self, tipo_scaler_cat):
        self.tipo_scaler_cat = tipo_scaler_cat

    def get_tipo_scaler_cat(self):
        return self.tipo_scaler_cat

    def select_cat_scaler(self):
        """Selecciona tipo de scaler categorico"""
        if self.get_tipo_scaler_cat() == "label":
            scaler = LabelEncoder()
        elif self.get_tipo_scaler_cat() == "one_hot":
            scaler = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            print("Scaler categorico no reconocido | label, one_hot")

        self.set_cat_scaler(scaler)

    def limpiar_datos_cic(self, X):
        """Funcion que se encarga de eliminar valores nulos y filas redundantes"""

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
        """
        if limpiar_datos is False:
            scaler = self.get_numeric_scaler()
            if train:
                X_num = scaler.fit_transform(X[col_numericas])
            else:
                X_num = scaler.transform(X[col_numericas])

            X[col_numericas] = X_num
        """
        return X


    def preprocesar_datos(self, X, train=True):
        """Funcion que preprocesa los datos antes de entrenar el modelo
            X: dataset que quiera realizar preprocesado de datos con los escalers indicados
            train: determina si esta entrenando los datos, para hacer fit y transform o solo transform
            return: X_train_procesado: datos procesados
        """
        logger.info("Preprocesando datos")

        X = X.copy()
        X = X.reset_index(drop=True)

        # Manejar valores nulos para que todos los valores sean numéricos para mejor rendimiento modelo
        col_categoricas = [col for col in ['proto','service','state'] if col in X.columns]
        col_numericas_originales = X.select_dtypes(exclude=['object']).columns.tolist()

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
        print(col_numericas)

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

        # codifica valores de proto, service y state para entrenamiento
        if train:
            self.crear_mapeos_agrupados()

        # Mapear columnas categoricas
        for col in col_categoricas:
            valores_originales = datos_categoricos[col]
            if col == 'proto':
                X_escalado['proto'] = valores_originales.map(self.proto_map).fillna(0).astype(int)
                print(X_escalado['proto'])
            elif col == 'service':
                service_clean = X['service'].astype(str).str.lower()
                X_escalado['service'] = service_clean.map(self.service_map).fillna(0).astype(int)

            elif col == 'state':
                state_clean = X['state'].astype(str).str.lower()
                X_escalado['state'] = state_clean.map(self.state_map).fillna(0).astype(int)

        # Debug
        if train:
            if 'proto' in X.columns:
                print(f"Proto valores únicos: {sorted(X_escalado['proto'].unique())}")
                #X_procesado = pd.concat([X_escalado, X_cat], axis=1)
        print(X_escalado)
        return X_escalado

    def seleccionar_best_feature(self, X, y):
        """Devuelve una lista con los atributos más importantes"""
        col_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[col_numericas]

        logger.info(f"Seleccionando best features con: {self.feature_selector}")
        if self.feature_selector == "kbest":
            selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
            mejores_features = selector.fit(X[col_numericas], y)
            mask_best = selector.get_support()
            kbest_features = X_num.columns[mask_best].tolist()
            mejores_features = kbest_features
            print("\nMejores features KBest")
            print("------------------------")
            print(kbest_features)

        elif self.feature_selector == "xgboost":
            X_num = X_num.replace([np.inf, -np.inf], np.nan)
            X_num = X_num.fillna(0)
            xgb_model = DecisionTreeClassifier(random_state=42)
            xgb_model.fit(X_num, y)
            importancias = xgb_model.feature_importances_
            importancia_df = pd.DataFrame({
                'Feature': X_num.columns,
                'Importance': importancias
            }).sort_values('Importance', ascending=False)
            xgboost_features = importancia_df.head(20)['Feature'].tolist()
            mejores_features = xgboost_features
            print(f"\nTop {len(xgboost_features)} features Decision Tree:")
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
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [None, 'balanced'],
            'min_impurity_decrease': [0.0, 0.001, 0.01],
            'max_leaf_nodes': [50, 100, None]
        }

        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
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

    def crear_mapeos_agrupados(self):
        """En base a los valores obtenidos, crea diccionarios para codificar los valores que pueden tomar"""
        self.proto_map = {
            'tcp':1 , 'udp':2, 'icmp':3, 'igmp':3,
            'ipv6':4, 'ipv6-frag':4, 'ipv6-route':4, 'ipv6-opts':4, 'ipv6-no':4,
            'gre':5, 'sctp':5, 'rtp':5,
            'unknown':0, '-':0
        }

        self.service_map = {
            'http': 1, 'https': 1, 'ssl': 1,
            'ftp': 2, 'ftp-data': 2,
            'dns': 3, 'dhcp': 3,
            'ssh': 4, 'smtp': 5, 'pop3':6,
            'snmp': 7, 'radius': 8, 'irc':9,
            'unknown': 0, '-': 0
        }

        self.state_map = {
            'fin':1, 'int':2, 'con':3, 'eco':4,
            'req':5, 'rst': 6, 'par':7, 'urn':8,
            'no':9, 'unknown': 0, '-':0
        }

    def codificar_val_cat(self, series, mapping, default_others=0):
        clean = series.astype(str).str.lower().fillna('unknown')
        encoded = clean.map(mapping)
        encoded = encoded.fillna(default_others).astype(int)
        return encoded

    def entrenar_modelo(self, X, X_test, y, y_test):
        self.logger.info("Entrenamiento del modelo DecisionTreeClassifier")
        print(f"Features utilizadas para entrenamiento: {len(self.features_seleccionadas)}")
        print(f"Tamaño conjunto entrenamiento: {X.shape}")
        print(f"Tamaño conjunto prueba: {X_test.shape}")

        print("\nDistribucion de clases original:")
        distribucion_original = Counter(y)
        for clase, cantidad in sorted(distribucion_original.items()):
            print(f"  {clase}: {cantidad:6d}")

        decision_tree = DecisionTreeClassifier(
            criterion='gini',
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            #min_impurity_decrease=0.0,
            #ccp_alpha=0.0,
            max_leaf_nodes=None
        )

        self.logger.info("Entrenando modelo...")

        t_ini = time.time()
        decision_tree.fit(X, y)
        t_fin = time.time()
        print(f"tiempo entrenamiento = {t_fin - t_ini}")

        t_ini = time.time()
        y_pred = decision_tree.predict(X_test)
        y_pred_proba = decision_tree.predict_proba(X_test)
        t_fin = time.time()
        print(f"tiempo test = {t_fin - t_ini}")

        labels_cm = sorted(set(y_test) | set(y_pred))
        cm_plot = confusion_matrix(y_test, y_pred, labels=labels_cm)
        fig, ax = plt.subplots(figsize=(max(10, len(labels_cm)), max(8, len(labels_cm) * 0.7)))
        sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_cm, yticklabels=labels_cm, ax=ax)
        ax.set_title('Matriz de Confusión - Decision Tree')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicción')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        self.obtener_resultados_clasificacion(y_test, y_pred)
        fpr, tpr, roc_auc, clases = self.calcular_roc_auc(y_test, y_pred_proba)
        self.visualizar_roc_curve(fpr, tpr, roc_auc, clases)

        # Validacion cruzada
        self.validacion_cruzada(decision_tree, X, y)

        self.modelo = decision_tree
        return decision_tree

    def prediccion_real(self, X, modelo):
        self.logger.info("Realizando una prediccion de prueba")
        y_pred = modelo.predict(X)
        y_pred_proba = modelo.predict_proba(X)
        return y_pred, y_pred_proba

    @classmethod
    def cargar_modelo(cls, ruta):
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
            ruta = self.path_modelo

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
        self.logger.info(f"Modelo guardado en {ruta} con toda la configuracion")

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

        print("Curva ROC Y Puntuacion AUC")
        print("--------------------------")

        for i, clase in enumerate(clases):
            if n_clases == 2 and i == 1:
                break
            idx = 0 if n_clases == 2 else i
            print(f"Clase {clase}: AUC = {roc_auc[idx]:.3f}")

        print(f"AUC Promedio: {roc_auc_promedio:.3f}")

        return fpr, tpr, roc_auc, clases

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
        plt.ylabel('Ratio verdaderos positivos')
        plt.title('Curvas ROC - Decision Tree Classifier')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.savefig("../../imagenes/modelo_dt_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nGráfico ROC curve guardado en: modelo_dt_roc_curve.png")


    def plot_metricas_por_clase(self, y_test, y_pred, titulo="Métricas por clase"):
        """Gráfica de barras agrupadas: Precision, Recall y F1-Score por clase."""
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        clases = [k for k in report.keys()
                  if k not in ('accuracy', 'macro avg', 'weighted avg')]

        precision = [report[c]['precision'] for c in clases]
        recall    = [report[c]['recall']    for c in clases]
        f1        = [report[c]['f1-score']  for c in clases]

        x = np.arange(len(clases))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(clases) * 0.9), 6))
        ax.bar(x - width, precision, width, label='Precision', color='steelblue')
        ax.bar(x,         recall,    width, label='Recall',    color='seagreen')
        ax.bar(x + width, f1,        width, label='F1-Score',  color='tomato')

        ax.set_xticks(x)
        ax.set_xticklabels(clases, rotation=40, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Puntuación')
        ax.set_title(titulo)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

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

        print("Resultados Clasificacion")
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

        print("Matriz de confusion (filas=real, columnas=predicha)")
        print(cm)

        # Cálculo por clase: TP, FN, FP, TN
        per_class_tp = np.diag(cm).astype(int)
        per_class_support = cm.sum(axis=1).astype(int)   # TP + FN
        per_class_fn = per_class_support - per_class_tp  # FN
        per_class_fp = cm.sum(axis=0).astype(int) - per_class_tp  # FP
        per_class_predicted = per_class_tp + per_class_fp        # TP + FP
        per_class_tn = np.array([n - tp - fn - fp for tp, fn, fp in zip(per_class_tp, per_class_fn, per_class_fp)], dtype=int)

        # Cabecera personalizada (mostramos TP, TN, FP, FN y tasas'FNR':>8s }"
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

            # tasas defensivas (evitan división por cero)
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
        print("TABLA POR CLASE (DataFrame):")
        # Formatear columnas float a 2 decimales
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print(detalle_df[['Support','TP','TN','FP','FN','TPR','FPR','TNR','FNR']].to_string())

        # Imprimir classification_report tradicional
        print("CLASSIFICATION REPORT (scikit-learn):")
        print(classification_report(y_test_arr, y_pred_arr, labels=labels, zero_division=0, digits=4))

        self.plot_metricas_por_clase(y_test_arr, y_pred_arr)

        otros_df = pd.DataFrame()
        return detalle_df, otros_df, cm

    def validacion_cruzada(self, modelo, X, y, cv=5):
        """Validacion cruzada para evaluar robustez"""
        scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')
        scores_f1 = cross_val_score(modelo, X, y, cv=cv, scoring='f1_weighted')
        scores_roc = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc_ovo')

        print("Resultados validacion cruzada")
        print(f"Accuracy CV:  {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"F1-Score CV:  {scores_f1.mean():.3f} ± {scores_f1.std():.3f}")
        print(f"ROC-AUC CV:   {scores_roc.mean():.3f} ± {scores_roc.std():.3f}")

    # Métodos adicionales para compatibilidad con ModeloDecisionTree
    def convertir_a_binario(self, y):
        """Convierte etiquetas multiclase to binary (0=Normal, 1=Ataque)"""
        return (y != 'Normal').astype(int)

    def seleccionar_features_binarias(self, X_train, y_train_binario, col_features, top_k=30):
        """Selecciona las features mas importantes para clasificacion binaria"""

        # Seleccionar top features para clasificacion binaria
        selector = SelectKBest(f_classif, k=min(top_k, len(col_features)))
        X_selected = selector.fit_transform(X_train[col_features], y_train_binario)

        # Obtener nombres de features seleccionadas
        selected_features = [col_features[i] for i in selector.get_support(indices=True)]
        scores = selector.scores_[selector.get_support()]

        print(f"\n{'='*60}")
        print("TOP FEATURES PARA CLASIFICACION BINARIA")
        print(f"{'='*60}")

        for i, (feature, score) in enumerate(zip(selected_features[:10], scores[:10])):
            print(f"{i+1:2d}. {feature:35s} | Score: {score:.2f}")

        return selected_features


def main():
    modelo_dt = ModeloDecisionTree()

    modelo_dt.set_tipo_scaler_cat("label")
    modelo_dt.feature_selector = "kbest"

    modelo = "cic"
    if modelo == "cic":
        X_train_cic = modelo_dt.cargar_datos_CIC()
        X_train_cic = X_train_cic.drop(columnas_a_eliminar, axis=1)
        y_train = X_train_cic['Label']
        X_train_cic = X_train_cic.drop('Label', axis=1)
        modelo_dt.select_scaler("standard")
        modelo_dt.select_cat_scaler()
        #modelo = "unsw"
        features_train = X_train_cic.columns.tolist()
        modelo_dt.features_seleccionadas = features_train
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_cic, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        X_train = modelo_dt.preprocesar_datos_cic(X_train, train=False, limpiar_datos=True)
        X_test = modelo_dt.preprocesar_datos_cic(X_test, train=False, limpiar_datos=True)
        best_features = modelo_dt.seleccionar_best_feature(X_train, y_train)
        X_train = X_train[best_features]
        X_test = X_test[best_features]
        X_train = modelo_dt.preprocesar_datos_cic(X_train)
        X_test = modelo_dt.preprocesar_datos_cic(X_test, train=False)
        X_train, y_train = modelo_dt.balancear_datos(X_train, y_train)
        modelo = modelo_dt.entrenar_modelo(X_train, X_test, y_train, y_test)
        modelo_dt.guardar_modelo("../../decision_tree_cic.pkl")
        return

    X_train = modelo_dt.cargar_datos()
    X_train['attack_cat'].fillna('normal',inplace = True)
    X_train['attack_cat'] = X_train['attack_cat'].apply(lambda x: x.strip().lower())

    X_train['attack_cat']  = X_train['attack_cat'].replace(
        "backdoors",
        "backdoor"
    )

    y_train = X_train['attack_cat']
    col_eliminar = ['attack_cat', 'label']
    X_train = X_train.drop(col_eliminar, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    #hiperparametros = modelo_dt.busqueda_hiperparametros(X_train, y_train)
    #print(hiperparametros)
    features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts','sttl','dttl','sload', 'dload','swin', 'dwin', 'stcpb', 'dtcpb', 'sinpkt', 'dinpkt', 'rate', 'sjit', 'djit', 'tcprtt']
    robust_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload', 'rate']
    middle_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate', 'tbytes','tpkts','bratio','pratio','proto','service']
    middle_features1 = ['dur', 'sbytes', 'dbytes', 'spkts', 'tpkts', 'tbytes']
    mini_features = ['dur', 'sbytes', 'dbytes', 'rate']
    #best_features = modelo_dt.seleccionar_best_feature(X_train, y_train)
    modelo_dt.select_scaler("standard")
    mini_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'proto', 'service', 'tbytes', 'tpkts', 'basymmetry', 'pasymmetry']

    X_train = X_train[mini_features]
    X_test = X_test[mini_features]
    X_train = modelo_dt.preprocesar_datos(X_train, train=True)
    X_test = modelo_dt.preprocesar_datos(X_test, train=False)
    #modelo_dt.best_features = best_features
    X_train, y_train = modelo_dt.balancear_datos(X_train, y_train)
    modelo = modelo_dt.entrenar_modelo(X_train, X_test, y_train, y_test)
    modelo_dt.guardar_modelo("../../models/decision_tree.pkl")

if __name__ == "__main__":
    main()
