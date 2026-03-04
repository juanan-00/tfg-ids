
import os
import datetime
import numpy as np
import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import (
    label_binarize, StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler, OneHotEncoder
)
from sklearn.metrics import (
    classification_report, confusion_matrix, multilabel_confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('logger_modelo_knn')


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

class ModeloKNN:

    def __init__(self):
        self.var_objetivo = None
        self.modelo = None

        # Scalers
        self.scaler = None
        self.scaler_categorico = None
        self.tipo_scaler_cat = None
        self.label_encoders = {}

        # Features
        self.features_seleccionadas = []
        self.best_features = []
        self.feature_selector = None

        # Balanceo
        self.balanceado = False

        self.service_map = {}
        self.proto_map = {}
        self.state_map = {}

    # -----------------------
    # Carga y limpieza
    # -----------------------
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

    def cargar_datos_analisis(self, ruta):
        return pd.read_csv(ruta)

    def limpiar_datos(self, X, real_test=False):
        """
        Igual que DT: quitamos metadatos y columnas no usadas por ambos.
        OJO: si vas a codificar categóricas, no elimines 'service'/'state' aquí.
        """
        metadata_features = ['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'service']
        non_metadata_features = [
            'ct_state_ttl', 'sttl', 'dttl', 'ct_dst_ltm', 'ct_srv_dst',
            'ct_dst_sct_ltm', 'ct_srv_ltm','ct_srv_src', 'state', 'ct_dst_srpot_ltm'
        ]
        col_id = ['id'] if real_test else ['id', 'attack_cat', 'label']

        X = X.drop(columns=metadata_features, errors="ignore")
        X = X.drop(columns=non_metadata_features, errors="ignore")
        X = X.drop(columns=col_id, errors='ignore')
        return X


    def create_argus_flow_features(self,df):
        """Features básicos derivados de métricas de flujo de Argus"""

        # Totales y ratios básicos
        df['tbytes'] = df['sbytes'] + df['dbytes']
        df['tpkts'] = df['spkts'] + df['dpkts']
        df['bratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pratio'] = df['spkts'] / (df['dpkts'] + 1)

        # Tamaños promedio de paquetes
        df['avg_pkt_size_src'] = df['sbytes'] / (df['spkts'] + 1)
        df['avg_pkt_size_dst'] = df['dbytes'] / (df['dpkts'] + 1)
        df['avg_pkt_size_flow'] = df['tbytes'] / (df['tpkts'] + 1)

        # Asimetría
        df['basymmetry'] = abs(df['sbytes'] - df['dbytes']) / (df['tbytes'] + 1)
        df['pasymmetry'] = abs(df['spkts'] - df['dpkts']) / (df['tpkts'] + 1)

        return df


    def get_best_features(self):
        return self.best_features

    def select_scaler(self, tipo="min_max"):
        if tipo == "standard":
            scaler = StandardScaler()
        elif tipo == "robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()  # default recomendado para KNN (distancias)
        self.scaler = scaler

    def set_tipo_scaler_cat(self, tipo_scaler_cat):
        self.tipo_scaler_cat = tipo_scaler_cat

    def get_tipo_scaler_cat(self):
        return self.tipo_scaler_cat

    def select_cat_scaler(self):
        if self.tipo_scaler_cat == "one_hot":
            self.scaler_categorico = OneHotEncoder(sparse=False, handle_unknown='ignore')
        elif self.tipo_scaler_cat == "label":
            self.scaler_categorico = None  # se crean LabelEncoders por columna en preprocesado
        else:
            self.scaler_categorico = None

    def get_numeric_scaler(self):
        return self.scaler

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

    def seleccionar_best_features(self, X, y):
        """Devuelve una lista con los atributos más importantes
           return: mejores_features
        """
        col_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[col_numericas]
        logger.info(f"Selecccinonando best features con {self.feature_selector}")
        # KBest
        if self.feature_selector == "kbest":
            logger.info("kbest")
            #f_classif | mutual_info_classif
            selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
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

    # -----------------------
    # Balanceo
    # -----------------------
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
            ('oversample', SMOTE(sampling_strategy=target_oversample, random_state=42, k_neighbors=5)),
            ('undersample', RandomUnderSampler(sampling_strategy=target_undersample, random_state=42))
        ])

        X_res, y_res = pipeline.fit_resample(X, y)

        logger.info(f"Distribución tras balanceo: {Counter(y_res)}")
        self.balanceado = True

        return X_res, y_res

    def obtener_valores_categoricos(self, X):
        """Obtiene una lista con todos los valores categoricos del dataset"""
        proto_val = []
        service_val = []
        state_val = []

        for _,row in X.iterrows():
            if row['proto'] not in proto_val:
                proto_val.append(row['proto'])
            if row['service'] not in service_val:
                service_val.append(row['service'])
            if row['state'] not in state_val:
                state_val.append(row['state'])

        return {
            'proto':proto_val,
            'service': service_val,
            'state': state_val
        }

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


    def busqueda_hiperparametros(self, X, y):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'leaf_size': [20, 30, 40],
            'p': [1, 2]
        }
        grid = GridSearchCV(
            estimator=KNeighborsClassifier(n_jobs=-1),
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X, y)
        logger.info(f"Mejores parámetros KNN: {grid.best_params_}")
        return grid.best_params_

    # -----------------------
    # Entrenamiento
    # -----------------------
    def entrenar_modelo(self, X, X_test, y, y_test):
        logger.info("Entrenamiento KNN (referencia DT)")
        print(f"Tamaño conjunto entrenamiento: {X.shape}")
        print(f"Tamaño conjunto prueba: {X_test.shape}")

        print("\nDistribución de clases (train):")
        for c, n in sorted(Counter(y).items()):
            print(f"  {c}: {n:6d}")

        # (Opcional) GridSearch – comentar si alarga demasiado
        # best = self.busqueda_hiperparametros(X, y)
        # knn = KNeighborsClassifier(**best, n_jobs=-1)

        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights = 'distance',
            metric='minkowski',
            leaf_size=30,
            p=2,
            n_jobs=-1
        )

        logger.info("Entrenando modelo...")

        t_ini = time.time()
        knn.fit(X, y)
        t_fin = time.time()
        print(f"Tiempo entrenamiento: {t_fin - t_fin}")
        t_ini = time.time()
        y_pred = knn.predict(X_test)
        t_fin = time.time()
        print(f"Tiempo test: {t_fin - t_ini}")
        y_pred_proba = knn.predict_proba(X_test)

        labels_cm = sorted(set(y_test) | set(y_pred))
        cm_plot = confusion_matrix(y_test, y_pred, labels=labels_cm)
        fig, ax = plt.subplots(figsize=(max(10, len(labels_cm)), max(8, len(labels_cm) * 0.7)))
        sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_cm, yticklabels=labels_cm, ax=ax)
        ax.set_title('Matriz de Confusión - KNN')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicción')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        df_por_clase, cm_df, rep_df = self.obtener_resultados_clasificacion(
            y_test, y_pred, y_pred_proba=y_pred_proba
        )
        fpr, tpr, roc_auc, clases = self.calcular_roc_auc(y_test, y_pred_proba)
        self.visualizar_roc_curve(fpr, tpr, roc_auc, clases)

        # Validación cruzada (como en DT)
        self.validacion_cruzada(knn, X, y)

        self.modelo = knn
        return knn

    # Prediccinoes en entorno real de monitorizacion
    def prediccion_real(self, X, modelo=None):
        m = self.modelo if modelo is None else modelo
        y_pred = m.predict(X)
        y_pred_proba = m.predict_proba(X)
        return y_pred, y_pred_proba

    # Persistencia del modelo
    @classmethod
    def cargar_modelo(cls, ruta):
        data = joblib.load(str(ruta))
        if isinstance(data, dict):
            inst = cls()
            inst.modelo = data['modelo']
            inst.scaler = data.get('scaler')
            inst.scaler_categorico = data.get('scaler_categorico')
            inst.features_seleccionadas = data.get('features_seleccionadas', [])
            inst.best_features = data.get('best_features', [])
            inst.feature_selector = data.get('feature_selector')
            inst.tipo_scaler_cat = data.get('tipo_scaler_cat')
            inst.label_encoders = data.get('label_encoders', {})
            return inst
        return data

    def guardar_modelo(self, ruta):
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
        logger.info(f"Modelo KNN guardado en {ruta}")


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

        # Cálculo por clase: VP, FN, FP, VN
        per_class_vp = np.diag(cm).astype(int)
        per_class_support = cm.sum(axis=1).astype(int)
        per_class_fn = per_class_support - per_class_vp
        per_class_fp = cm.sum(axis=0).astype(int) - per_class_vp
        per_class_predicted = per_class_vp + per_class_fp
        per_class_vn = np.array([n - vp - fn - fp for vp, fn, fp in zip(per_class_vp, per_class_fn, per_class_fp)], dtype=int)

        # Cabecera personalizada (mostramos VP, VN, FP, FN y tasas)
        header = f"{'Clase':>25} {'Soporte':>8} {'VP':>6} {'VN':>6} {'FP':>6} {'FN':>6} {'RVP':>8} {'RFP':>8} {'RVN':>8} {'RFN':>8}"
        print(header)
        print('-' * len(header))

        # Imprimir por clase con formato (2 decimales en tasas)
        for idx, label in enumerate(labels):
            vp_c = int(per_class_vp[idx])
            fn_c = int(per_class_fn[idx])
            fp_c = int(per_class_fp[idx])
            vn_c = int(per_class_vn[idx])
            support_c = int(per_class_support[idx])

            # tasas defensivas (evitan división por cero)
            rvp_c = vp_c / (vp_c + fn_c) if (vp_c + fn_c) > 0 else 0.0   # sensibilidad
            rfn_c = fn_c / (vp_c + fn_c) if (vp_c + fn_c) > 0 else 0.0
            rfp_c = fp_c / (fp_c + vn_c) if (fp_c + vn_c) > 0 else 0.0
            rvn_c = vn_c / (fp_c + vn_c) if (fp_c + vn_c) > 0 else 0.0

            print(f"{str(label):25s} {support_c:8d} {vp_c:6d} {vn_c:6d} {fp_c:6d} {fn_c:6d} {rvp_c:8.2f} {rfp_c:8.2f} {rvn_c:8.2f} {rfn_c:8.2f}")

        # Construir DataFrame por clase (útil para exportar)
        detalle_rows = []
        for i, lab in enumerate(labels):
            VP = int(per_class_vp[i])
            FN = int(per_class_fn[i])
            FP = int(per_class_fp[i])
            VN = int(per_class_vn[i])
            Support = int(per_class_support[i])

            RVP = VP / (VP + FN) if (VP + FN) > 0 else np.nan
            RFN = FN / (VP + FN) if (VP + FN) > 0 else np.nan
            RFP = FP / (FP + VN) if (FP + VN) > 0 else np.nan
            RVN = VN / (FP + VN) if (FP + VN) > 0 else np.nan

            detalle_rows.append({
                'Clase': lab,
                'Support': Support,
                'VP': VP,
                'VN': VN,
                'FP': FP,
                'FN': FN,
                'RVP': RVP,
                'RFP': RFP,
                'RVN': RVN,
                'RFN': RFN
            })

        detalle_df = pd.DataFrame(detalle_rows).set_index('Clase')
        # Comprobaciones: aseguran que cuadra con la muestra
        assert int(detalle_df['VP'].sum() + detalle_df['FN'].sum()) == n, "suma VP+FN != tamaño de la muestra"
        assert int((detalle_df['VP'] + detalle_df['FP']).sum()) == n, "suma Predichas != tamaño de la muestra"

        # Imprimir la tabla por clase completa (DataFrame) con 2 decimales en tasas
        print("TABLA POR CLASE (DataFrame):")
        # Formatear columnas float a 2 decimales
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print(detalle_df[['Support','VP','VN','FP','FN','RVP','RFP','RVN','RFN']].to_string())

        # Imprimir classification_report tradicional
        print("CLASSIFICATION REPORT (scikit-learn):")
        print(classification_report(y_test_arr, y_pred_arr, labels=labels, zero_division=0, digits=4))

        self.plot_metricas_por_clase(y_test_arr, y_pred_arr)

        otros_df = pd.DataFrame()
        return detalle_df, otros_df, cm

    def calcular_roc_auc(self, y_test, y_pred_proba):
        clases = np.unique(y_test)
        n_clases = len(clases)
        y_bin = label_binarize(y_test, classes=clases)

        if n_clases == 2:
            y_bin = y_bin.ravel()
            y_pred_vec = y_pred_proba[:, 1]
        else:
            y_pred_vec = y_pred_proba

        fpr, tpr, roc_auc = {}, {}, {}
        if n_clases == 2:
            fpr[0], tpr[0], _ = roc_curve(y_bin, y_pred_vec)
            roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            for i in range(n_clases):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_vec[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        auc_prom = float(np.mean(list(roc_auc.values())))
        print("\nCurva ROC y AUC por clase")
        for i, c in enumerate(clases):
            idx = 0 if (n_clases == 2) else i
            print(f"Clase {c}: AUC={roc_auc[idx]:.3f}")
        print(f"AUC promedio: {auc_prom:.3f}")
        return fpr, tpr, roc_auc, clases

    def visualizar_roc_curve(self, fpr, tpr, roc_auc, clases, outpath="../../imagenes/modelo_knn_roc.png"):
        plt.figure(figsize=(10, 7))
        if len(clases) == 2:
            plt.plot(fpr[0], tpr[0], label=f"Clase {clases[1]} vs resto (AUC={roc_auc[0]:.3f})")
        else:
            for i, c in enumerate(clases):
                plt.plot(fpr[i], tpr[i], label=f"{c} (AUC={roc_auc[i]:.3f})")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC - KNN (One-vs-Rest)")
        plt.legend(loc="lower right")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"ROC guardada en: {outpath}")

    def diagnostico_overfitting(self, modelo, X_train, y_train, X_test, y_test, cv=5, delta=0.03):
        # Predicciones/
        y_pred_tr = modelo.predict(X_train)
        y_pred_te = modelo.predict(X_test)

        # Métricas básicas
        acc_tr = accuracy_score(y_train, y_pred_tr)
        acc_te = accuracy_score(y_test,  y_pred_te)
        f1w_tr = f1_score(y_train, y_pred_tr, average='weighted', zero_division=0)
        f1w_te = f1_score(y_test,  y_pred_te, average='weighted', zero_division=0)

        # Brechas
        gap_acc = acc_tr - acc_te
        gap_f1w = f1w_tr - f1w_te

        # CV sobre train
        cv_acc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_f1w = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)

        report = pd.Series({
            'train_accuracy': float(acc_tr),
            'test_accuracy':  float(acc_te),
            'gap_accuracy':   float(gap_acc),
            'train_f1w':      float(f1w_tr),
            'test_f1w':       float(f1w_te),
            'gap_f1w':        float(gap_f1w),
            'cv_accuracy_mean': float(cv_acc.mean()),
            'cv_accuracy_std':  float(cv_acc.std()),
            'cv_f1w_mean':      float(cv_f1w.mean()),
            'cv_f1w_std':       float(cv_f1w.std()),
            'overfitting_flag': bool((gap_acc > delta) or (gap_f1w > delta))
        })
        print("\n[Diagnóstico Overfitting]")
        print(report.to_string())
        return report

    def validacion_cruzada(self, modelo, X, y, cv=5):
        scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        scores_f1 = cross_val_score(modelo, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print(f"Accuracy CV:  {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"F1-Score CV:  {scores_f1.mean():.3f} ± {scores_f1.std():.3f}")


def main():
    modelo_knn = ModeloKNN()

    modelo_knn.set_tipo_scaler_cat("label")
    modelo_knn.feature_selector = "kbest"
    modelo_knn.select_scaler("standard")
    modelo_knn.select_cat_scaler()

    dataset = "cic"
    if dataset == "cic":
        X_train_CIC = modelo_knn.cargar_datos_CIC()
        X_train_CIC = X_train_CIC.drop(columns_to_drop_cicids, axis=1)
        y_train = X_train_CIC['Label']
        X_train_CIC = X_train_CIC.drop('Label', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_CIC, y_train, test_size=0.3, random_state=42
        )
        X_train = modelo_knn.preprocesar_datos_cic(X_train, train=False, limpiar_datos=True)
        X_test  = modelo_knn.preprocesar_datos_cic(X_test,  train=False, limpiar_datos=True)

        best_features = modelo_knn.seleccionar_best_features(X_train, y_train)
        X_train = X_train[best_features]
        X_test = X_test[best_features]

        X_train = modelo_knn.preprocesar_datos_cic(X_train, train=True, limpiar_datos=False)
        X_test = modelo_knn.preprocesar_datos_cic(X_test, train=False, limpiar_datos=False)
        modelo_knn.features_seleccionadas = best_features
        ruta = "../../models/modelo_knn.joblib"
        modelo_knn.guardar_modelo(ruta)
        # 6) Balanceo (recomendado p/KNN)
        X_train, y_train = modelo_knn.balancear_datos(X_train, y_train)
        modelo = modelo_knn.entrenar_modelo(X_train, X_test, y_train, y_test)
        report = modelo_knn.diagnostico_overfitting(X_train, y_train, X_test, y_test)
        print(report)
        return

    X_train = modelo_knn.cargar_datos()
    X_train['attack_cat'].fillna('normal',inplace = True)
    X_train['attack_cat'] = X_train['attack_cat'].apply(lambda x: x.strip().lower())

    X_train['attack_cat'] = X_train['attack_cat'].replace(
        'backdoors',
        'backdoor'
    )

    y_train = X_train['attack_cat']
    col_eliminar = ['attack_cat', 'label']
    X_train = X_train.drop(col_eliminar, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    mini_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'proto', 'service', 'tbytes', 'tpkts', 'basymmetry', 'pasymmetry']

    X_train = X_train[mini_features]
    X_test = X_test[mini_features]

    X_train = modelo_knn.preprocesar_datos_unsw(X_train, train=True)
    X_test = modelo_knn.preprocesar_datos_unsw(X_test, train=False)
    X_train, y_train = modelo_knn.balancear_datos(X_train, y_train)
    modelo = modelo_knn.entrenar_modelo(X_train, X_test, y_train, y_test)
    report = modelo_knn.diagnostico_overfitting(X_train, y_train, X_test, y_test)

    ruta_modelo = "~/Documentos/tfg-ids-api/models/modelo_knn_unsw-nb15.pkl"

    modelo_knn.guardar_modelo(ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")


if __name__ == "__main__":
    main()
