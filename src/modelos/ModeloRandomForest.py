import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize, RobustScaler,MinMaxScaler
from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_score, recall_score, f1_score, accuracy_score,
        roc_curve, auc
)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logger = logging.getLogger('logger_modelo_rc')
logger.setLevel(logging.INFO)
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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

class ModeloRandomForest:

    def __init__(self):
        self.var_objetivo = None

        # Determina el tipo pero no lo guarda
        self.modelo = None
        self.scaler = None
        self.scaler_categorico = None
        self.features_seleccionadas = []
        self.best_features = []
        self.feature_selector = None
        self.balanceado = False

        # Mapeo valores categoricos
        self.proto_map = {}
        self.state_map = {}
        self.service_map = {}

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
        training_set = self.crear_features_adicional(training_set)
        #training_data = self.create_argus_flow_features(training_data)
        #testing_data = self.create_argus_flow_features(testing_data)

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
        # cic - 2 se corresonde con ataque web con menores datos
        cic_2 = pd.read_csv("../../datasets/CIC-2.csv")
        cic_3 = pd.read_csv("../../datasets/CIC-3.csv")
        cic_4 = pd.read_csv("../../datasets/CIC-4.csv")
        cic_5 = pd.read_csv("../../datasets/CIC-5.csv")
        cic_6 = pd.read_csv("../../datasets/CIC-6.csv")
        cic_7 = pd.read_csv("../../datasets/CIC-7.csv")
        cic_8 = pd.read_csv("../../datasets/CIC-8.csv")

        cols = [col.replace(' ', '') for col in cic_1.columns]
        print(cols)

        cic = [cic_1, cic_2, cic_3, cic_4, cic_5, cic_6, cic_7, cic_8]
        for df in cic:
            df.columns = cols

        training_dataset = pd.concat([cic_1, cic_2, cic_3, cic_4, cic_5, cic_6, cic_7, cic_8], ignore_index=True)
        print(training_dataset)
        return training_dataset


    def crear_features_adicional(self, df):
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

        return df


    def cargar_datos_analisis(self,ruta):
        """Funcion que carga una muestra de datos de trafico de red real"""
        test_data_real = pd.read_csv(ruta)
        return test_data_real

    def get_best_features (self):
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
        else: print("Scaler numerico no reconocido | standard, min_max, robust")
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
            scaler = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
        else:
            print("Scaler categorico no reconocido | label, robust")

        self.set_cat_scaler(scaler)


    def aplicar_transformador(self, X):
        """Aplicar transformador a datos de prediccion monitorizados con cicflowmeter"""

        X = X.copy()
        scaler = self.get_numeric_scaler()
        X = scaler.transform(X)
        return X


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

        # Limpiar siempre NaN e inf para evitar predicciones corruptas
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
        """
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
        """

        # codifica valores de proto, service y state para entrenamiento
        self.crear_mapeos_agrupados()

        for col in col_categoricas:
            valores_originales = datos_categoricos[col]
            if col == 'proto' and 'proto' in col_categoricas:
                X['proto'] = valores_originales.map(self.proto_map).fillna(0).astype(int)
            elif col == 'service':
                service_clean = X['service'].astype(str).str.lower()
                X['service'] = service_clean.map(self.service_map).fillna(0).astype(int)
            elif col == 'state' and 'state' in col_categoricas:
                state_clean = X['state'].astype(str).str.lower()
                X['state'] = state_clean.map(self.state_map).fillna(0).astype(int)

        # Debug
        #print(f"Service valores {X['service']}")
        #print(f"Proto valores {X['proto']}")
        #print(f"Status valores: {X['state']}")
        #X_procesado = pd.concat([X_escalado, X], axis=1)

        #print(X)
        return X

    def seleccionar_best_feature(self, X, y):
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
            selector = SelectKBest(f_classif, k=min(30, X.shape[1]))
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

        counts_antes = Counter(y)
        total_antes = sum(counts_antes.values())

        # Mostrar distribución ANTES del balanceo
        print("\n" + "="*60)
        print("DISTRIBUCIÓN ANTES DEL BALANCEO")
        print("="*60)
        for clase, cantidad in sorted(counts_antes.items(), key=lambda x: x[1], reverse=True):
            pct = cantidad / total_antes * 100
            print(f"  {clase:25s} {cantidad:8d}  ({pct:5.1f}%)")
        print(f"  {'TOTAL':25s} {total_antes:8d}")
        print("="*60)

        # Ajustar k_neighbors dinámicamente
        min_samples = min(counts_antes.values())
        k = min(5, min_samples - 1) if min_samples > 1 else 1

        # Estrategia proporcional: multiplicar muestras reales por un factor (max x5)
        # En vez de inflar todas a 15000, se respeta la cantidad original
        # Clases muy pequeñas (<200 muestras) se excluyen de SMOTE → class_weight las compensa
        smote_min_samples = 200
        smote_factor = 3  # multiplicar por 3 las muestras reales
        target_max = 40000

        target_oversample = {}
        for cls, count in counts_antes.items():
            if count >= smote_min_samples:
                target_nuevo = min(count * smote_factor, target_max)
                if target_nuevo > count:
                    target_oversample[cls] = target_nuevo
                    print(f"  [SMOTE] '{cls}': {count} → {target_nuevo} (x{target_nuevo/count:.1f})")
            else:
                print(f"  [SMOTE] Clase '{cls}' excluida ({count} muestras < {smote_min_samples}): se compensa con class_weight")

        target_undersample = {cls: target_max for cls, count in counts_antes.items() if count > target_max}

        pipeline = Pipeline([
            ('oversample', SMOTE(sampling_strategy=target_oversample, k_neighbors=k, random_state=42)),
            ('undersample', RandomUnderSampler(sampling_strategy=target_undersample, random_state=42))
        ])

        X_res, y_res = pipeline.fit_resample(X, y)

        counts_despues = Counter(y_res)
        total_despues = sum(counts_despues.values())

        # Mostrar distribución DESPUÉS del balanceo
        print("\n" + "="*60)
        print("DISTRIBUCIÓN DESPUÉS DEL BALANCEO")
        print("="*60)
        for clase, cantidad in sorted(counts_despues.items(), key=lambda x: x[1], reverse=True):
            pct = cantidad / total_despues * 100
            cambio = cantidad - counts_antes.get(clase, 0)
            signo = "+" if cambio >= 0 else ""
            print(f"  {clase:25s} {cantidad:8d}  ({pct:5.1f}%)  {signo}{cambio}")
        print(f"  {'TOTAL':25s} {total_despues:8d}  (antes: {total_antes})")
        print("="*60 + "\n")

        self.balanceado = True
        return X_res, y_res


    def entrenar_modelo(self, X_train, X_test, y_train, y_test,
                        experiment_name="RandomForest_experiment",
                        run_name=None,
                        tipo_modelo="unsw-nb15"):
        # Entrenar modelo
        logger.info("Entrenamiento del modelo RandomForestClassifier multiclase")
        print(f"Features utilizadas para entrenamiento: {len(self.features_seleccionadas)}")
        print(f"Conjunto entrenamiento: {X_train.shape}")
        print(f"Conjunto prueba: {X_test.shape}")

        # Mostrar distribución de clases en entrenamiento
        print("\nDistribucion de clases en entrenamiento:")
        distribucion_original = Counter(y_train)
        for clase, cantidad in sorted(distribucion_original.items()):
            print(f"  {clase}: {cantidad:6d}")

        mlflow.set_experiment(experiment_name)

        if tipo_modelo == "cic-ids2017":
            # Modelo CIC: más profundo, class_weight balanceado para TODAS las clases
            random_forest = RandomForestClassifier(
                criterion='entropy',
                max_depth=20,
                n_estimators=300,
                random_state=42,
                max_features='sqrt',
                class_weight='balanced',
                bootstrap=True,
                oob_score=True,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            )
            logger.info("Usando configuración CIC-IDS2017 con class_weight='balanced'")
        else:
            # Modelo UNSW-NB15
            random_forest = RandomForestClassifier(
                criterion='entropy',
                max_depth=10,
                n_estimators=150,
                random_state=42,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                min_samples_split=15,
                max_samples=0.8,
                n_jobs=-1
            )
            logger.info("Usando configuración UNSW-NB15")

        params = {
            'n_estimators': random_forest.n_estimators,
            'max_depth': random_forest.max_depth,
            'criterion': random_forest.criterion,
            'min_samples_split': random_forest.min_samples_split,
            'random_state': random_forest.random_state,
            'tipo_modelo': tipo_modelo
        }

        # Mostrar distribución de entrenamiento (balanceada) vs test (original)
        print("\n" + "="*60)
        print("DISTRIBUCIÓN ENTRENAMIENTO (balanceada)")
        print("="*60)
        dist_train = Counter(y_train)
        total_train = sum(dist_train.values())
        for clase, cantidad in sorted(dist_train.items(), key=lambda x: x[1], reverse=True):
            pct = cantidad / total_train * 100
            print(f"  {clase:25s} {cantidad:8d}  ({pct:5.1f}%)")
        print(f"  {'TOTAL':25s} {total_train:8d}")

        print("\nDISTRIBUCIÓN TEST (original sin balancear)")
        print("-"*60)
        dist_test = Counter(y_test)
        total_test = sum(dist_test.values())
        for clase, cantidad in sorted(dist_test.items(), key=lambda x: x[1], reverse=True):
            pct = cantidad / total_test * 100
            print(f"  {clase:25s} {cantidad:8d}  ({pct:5.1f}%)")
        print(f"  {'TOTAL':25s} {total_test:8d}")
        print("="*60 + "\n")

        with mlflow.start_run(run_name=run_name):

            logger.info("Entrenando modelo random forest..")
            mlflow.log_params(params)
            mlflow.log_param("scaler_type", type(self.scaler).__name__)
            mlflow.log_param("n_features", X_train.shape[1])

            t_ini = time.time()
            random_forest.fit(X_train, y_train)
            t_fin = time.time()
            print(f"tiempo entrenamiento: {t_fin - t_ini}")

            t_ini = time.time()
            y_pred = random_forest.predict(X_test)
            t_fin = time.time()
            print(f"tiempo test: {t_fin - t_ini}")
            self.obtener_resultados_clasificacion(y_test, y_pred)
            self.modelo = random_forest
                # Calcular métricas
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Registrar métricas
            mlflow.log_metrics({
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1
            })

            # Guardar matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            labels_cm = sorted(set(y_test) | set(y_pred))
            fig, ax = plt.subplots(figsize=(max(10, len(labels_cm)), max(8, len(labels_cm) * 0.7)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels_cm, yticklabels=labels_cm, ax=ax)
            ax.set_title('Matriz de Confusión')
            ax.set_ylabel('Real')
            ax.set_xlabel('Predicción')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            # Registrar modelo
            signature = infer_signature(X_train, random_forest.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=random_forest,
                artifact_path="model",
                signature=signature
            )

            print(f"\n Modelo entrenado - F1: {test_f1:.4f}")
            print(f"   MLflow Run ID: {mlflow.active_run().info.run_id}")


        return random_forest

    def prediccion_real_cic(self, X, modelo):
        logger.info("Realizando predicción multiclase sensible")

        # Obtenemos las probabilidades para cada clase
        y_pred_proba = modelo.predict_proba(X)
        clases = modelo.classes_.tolist()

        # Identificamos en qué posición está la clase 'BENIGN'
        idx_benign = clases.index('BENIGN')

        y_pred_final = []
        for proba in y_pred_proba:
            # ¿Es la probabilidad de que sea BENIGN menor al 60%?
            # (O dicho de otro modo: ¿Hay más de un 40% de sospecha de ataque?)
            if proba[idx_benign] < 0.60:
                # Si hay sospecha, quitamos la probabilidad de BENIGN
                # y buscamos cuál es el ataque más probable
                proba_solo_ataques = proba.copy()
                proba_solo_ataques[idx_benign] = 0
                y_pred_final.append(clases[np.argmax(proba_solo_ataques)])
            else:
                y_pred_final.append('BENIGN')

        print(y_pred_proba)
        return np.array(y_pred_final), y_pred_proba

    def prediccion_real2(self, X, modelo):
        logger.info("Realizando una prediccion de prueba")
        y_pred = modelo.predict(X)
        y_pred_proba = modelo.predict_proba(X)
        print(y_pred_proba)
        #self.obtener_resultados_clasificacion(y, y_pred)
        return y_pred, y_pred_proba


    def prediccion_real1(self, X, modelo):
        # 1. Obtener probabilidades
        y_proba = modelo.predict_proba(X)
        clases = modelo.classes_

        # 2. El ganador es el que tenga la probabilidad MÁXIMA
        idx_ganadores = np.argmax(y_proba, axis=1)
        y_pred = clases[idx_ganadores]

        # 3. La confianza es la probabilidad de ese ganador
        confianza = np.max(y_proba, axis=1)


        print(confianza)
        return y_pred, y_proba

    @classmethod
    def cargar_modelo(cls, ruta):
        data = joblib.load(str(ruta))
        if isinstance(data, dict):
            instancia = cls()
            instancia.modelo = data['modelo']
            instancia.scaler = data.get('scaler')
            instancia.features_seleccionadas = data.get('features_seleccionadas', [])
            instancia.best_features = data.get('best_features')
            return instancia
        return data

    def guardar_modelo(self, ruta):
        """Serializa una instancia de la clase"""
        data = {
            'modelo': self.modelo,
            'scaler': self.scaler,
            'features_seleccionadas': self.features_seleccionadas,
            'feature_selector': self.feature_selector,
            'best_features': self.best_features
        }
        joblib.dump(data, ruta)
        logger.info(f"Modelo guardado en {ruta} con toda la configuracion")

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
            'igmp': 4,
            'ipv6': 5,
            'ipv6-frag': 6,
            'ipv6-route': 7,
            'ipv6-opts': 8,
            'gre': 9,
            'sctp': 10,
            'rtp': 11,
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

    def codificar_val_cat(self, series, mapping, default_others=0):
        clean = series.astype(str).str.lower().fillna('unknown')
        encoded = clean.map(mapping)
        encoded = encoded.fillna(default_others).astype(int)
        return encoded

    def analizar_probabilidad_amenzas(y_proba, X):
        """Determina si la conexion tiene riesgo alto de ataque o no
           en base a probabilidad obtenida
        """
        threshold=0.5
        high_threat_mask = y_proba[:, 1] > threshold
        threats = X[high_threat_mask]

        for i, (_, row) in enumerate(threats.iterrows()):
            if row['sload'] > 50000 or row['dload'] > 50000:
                print('Alta velocidade de transferencia')
            if row['sbytes'] > row['dbytes'] * 3:
                print(" -Upload asimétricos mayor source que dest (posible exfiltración)")
            if row['dbytes'] > row['sbytes'] * 3:
                print(" -Download asimétrico (posible malware)")

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
        Calcula e imprime por-clase las métricas solicitadas con cálculo CORRECTO de TN.
        """
        # Asegurar arrays numpy
        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)
        n = len(y_test_arr)

        # Etiquetas presentes (orden determinista)
        labels = np.unique(np.concatenate([y_test_arr, y_pred_arr]))
        labels = list(labels)
        n_clases = len(labels)

        # Métricas globales
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

        # Matriz de confusión
        cm = confusion_matrix(y_test_arr, y_pred_arr, labels=labels)
        print("Matriz de confusion (filas=real, columnas=predicha)")
        print(cm)

        # --- CÁLCULO CORREGIDO DE MÉTRICAS POR CLASE ---
        total = int(cm.sum())

        # Inicializar arrays
        tp = np.diag(cm).astype(int)
        fp = np.zeros(n_clases, dtype=int)
        fn = np.zeros(n_clases, dtype=int)
        tn = np.zeros(n_clases, dtype=int)
        support = cm.sum(axis=1).astype(int)

        # Calcular FP, FN, TN para cada clase
        for i in range(n_clases):
            # FP = suma de la columna i menos TP[i]
            fp[i] = cm[:, i].sum() - tp[i]

            # FN = suma de la fila i menos TP[i]
            fn[i] = cm[i, :].sum() - tp[i]

            # TN = total - (TP + FP + FN) para esta clase
            # Esto es CORRECTO para multiclase: TN son todas las muestras que no son esta clase
            # y que fueron correctamente clasificadas como no-esta-clase
            tn[i] = total - (tp[i] + fp[i] + fn[i])

        # Comprobaciones
        assert tp.sum() + fn.sum() == total, "TP + FN sum != total"
        assert (tp + fp).sum() == total, "Predichas (TP+FP) sum != total"

        # Imprimir tabla
        header = f"{'Clase':25s} {'Support':>8s} {'TP':>8s} {'TN':>8s} {'FP':>8s} {'FN':>8s} {'TPR':>8s} {'FPR':>8s} {'TNR':>8s} {'FNR':>8s}"
        print(header)
        print('-' * len(header))

        for i, label in enumerate(labels):
            TP = int(tp[i])
            TN = int(tn[i])
            FP = int(fp[i])
            FN = int(fn[i])
            SUP = int(support[i])

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            TNR = TN / (FP + TN) if (FP + TN) > 0 else 0.0

            print(f"{str(label):25s} {SUP:8d} {TP:8d} {TN:8d} {FP:8d} {FN:8d} {TPR:8.2f} {FPR:8.2f} {TNR:8.2f} {FNR:8.2f}")

        print(f"Tasa de FPR = {FP/(FP+TN)}")
        print(f"Tasa de FNR = {FN/(FN+TP)}")
        detalle_rows = []
        for i, lab in enumerate(labels):
            TP = int(tp[i])
            TN = int(tn[i])
            FP = int(fp[i])
            FN = int(fn[i])
            Support = int(support[i])

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

        print("\nTABLA POR CLASE (DataFrame):")
        with pd.option_context('display.float_format', '{:0.2f}'.format):
            print(detalle_df[['Support','TP','TN','FP','FN','TPR','FPR','TNR','FNR']].to_string())

        # Reporte tradicional
        print("\nReporte de clasificación (scikit-learn):")
        print(classification_report(y_test_arr, y_pred_arr, labels=labels, zero_division=0, digits=4))

        self.plot_metricas_por_clase(y_test_arr, y_pred_arr)

        otros_df = pd.DataFrame()
        return detalle_df, otros_df, cm

def main():

    modelo_rf = ModeloRandomForest()

    modelo_rf.set_tipo_scaler_cat("label")
    modelo_rf.feature_selector = "kbest"

    #X_train = modelo_rf.cargar_datos()

    modelo_rf.select_scaler("robust")
    modelo_rf.select_cat_scaler()
    modelo = "cic"

    if modelo == "cic":
        X_train_cic = modelo_rf.cargar_datos_CIC()
        y_train_cic = X_train_cic['Label']
        X_train_cic = X_train_cic.drop('Label', axis=1)

        # Eliminar clases con muestras insuficientes para entrenamiento fiable
        clases_eliminar = ['Infiltration', 'Web Attack \u2013 SQL Injection', 'Heartbleed']
        mask = ~y_train_cic.isin(clases_eliminar)
        logger.info(f"Eliminando clases con pocas muestras: {clases_eliminar}")
        for cls in clases_eliminar:
            n = int((y_train_cic == cls).sum())
            if n > 0:
                logger.info(f"  Eliminada '{cls}': {n} muestras")
        X_train_cic = X_train_cic[mask].reset_index(drop=True)
        y_train_cic = y_train_cic[mask].reset_index(drop=True)

        # 1. Split antes de todo
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_cic, y_train_cic, test_size=0.3, random_state=42, stratify=y_train_cic
        )

        # 2. Preprocesar (limpiar NaN, inf)
        X_train = modelo_rf.preprocesar_datos_cic(X_train, train=True, limpiar_datos=True)
        X_test = modelo_rf.preprocesar_datos_cic(X_test, train=False, limpiar_datos=True)

        # 3. Seleccionar features (KBest + features obligatorias para portscan/DoS)
        best_features = modelo_rf.seleccionar_best_feature(X_train, y_train)

        # Features críticas que KBest puede no priorizar pero son esenciales
        features_obligatorias = [
            'DestinationPort',      # portscan: muchos puertos distintos
            'SYNFlagCount',         # portscan SYN scan
            'RSTFlagCount',         # portscan: puertos cerrados envían RST
            'TotalFwdPackets',      # portscan: pocos paquetes por flujo
            'TotalBackwardPackets', # portscan: 0 o 1 respuesta
            'FlowPackets/s',        # portscan/DDoS: alta tasa
            'FwdPackets/s',         # portscan/DDoS: alta tasa forward
            'Init_Win_bytes_forward', # fingerprinting de herramientas
        ]
        for f in features_obligatorias:
            if f in X_train.columns and f not in best_features:
                best_features.append(f)
                print(f"  + Añadida feature obligatoria: {f}")

        modelo_rf.features_seleccionadas = best_features
        print(f"\nFeatures seleccionadas ({len(best_features)}): {best_features}")

        # 4. Reducir a features seleccionadas ANTES de balancear
        X_train = X_train[best_features]
        X_test = X_test[best_features]

        # 5. Balancear datos (muestra distribución antes/después)
        X_train, y_train = modelo_rf.balancear_datos(X_train, y_train)

        # 6. Preprocesar final
        X_train = modelo_rf.preprocesar_datos_cic(X_train)
        X_test = modelo_rf.preprocesar_datos_cic(X_test, train=False)

        # 7. Entrenar con configuración CIC
        modelo = modelo_rf.entrenar_modelo(
            X_train, X_test, y_train, y_test,
            experiment_name="CIC_IDS2017",
            run_name="cic_balanced",
            tipo_modelo="cic-ids2017"
        )
        ruta = "../../models/modelo_rf_cic.joblib"
        modelo_rf.guardar_modelo(ruta)
        logger.info("Todo realizado con exito")
        return


    # Columnas a elimnar porque no es dificil monitorizarlas:
    X_train = modelo_rf.cargar_datos()
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
    #print(list(X_train_cic.columns))

    print(list(X_train.columns))

    features_train = X_train.columns.tolist()

    # proto, service
    middle_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts',
                       'sttl', 'dttl', 'tbytes', 'tpkts']
                       #'proto']
                    #service

    mini_features = ['sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'rate', 'proto', 'service', 'tbytes', 'tpkts', 'basymmetry', 'pasymmetry']
    modelo_rf.features_seleccionadas = mini_features

    X_train = X_train[mini_features]
    X_test = X_test[mini_features]

    X_train = modelo_rf.preprocesar_datos_unsw(X_train)
    X_test = modelo_rf.preprocesar_datos_unsw(X_test, train=False)

    # Antes del balanceo
    pd.Series(y_train).value_counts().plot(kind='bar', title='Distribución Original')
    X_train, y_train = modelo_rf.balancear_datos(X_train, y_train)
    pd.Series(y_train).value_counts().plot(kind='bar', title='Distribución Balanceada')

    modelo = modelo_rf.entrenar_modelo(X_train, X_test, y_train, y_test,
                                       experiment_name="UNSW_NB15",
                                       run_name="base_line")

    ruta = "../../models/modelo_rf_unsw.joblib"
    modelo_rf.guardar_modelo(ruta)
    print(modelo_rf.features_seleccionadas)
    #modelo.rf.prediccion_real(X_test, modelo)
    """
    datos_real = modelo_rf.cargar_datos_analisis("../../data/trafico_red_20250813_112614.csv")
    datos_real = datos_real[features_train]
    datos_real.columns = datos_real.columns.str.lower()

    datos_real = modelo_rf.limpiar_datos(datos_real, real_test=True)
    datos_real = modelo_rf.preprocesar_datos(datos_real, train=False)
    datos_real = datos_real[best_features]
    print(len(datos_real.columns))
    print(datos_real)
    print(X_test)
    modelo_rf.prediccion_real(datos_real, modelo)
    modelo_rf.guardar_modelo("../../models/random_forest.pkl")
    """

if __name__ == "__main__":
    main()
