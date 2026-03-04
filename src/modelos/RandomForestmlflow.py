import os
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize, RobustScaler, MinMaxScaler
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

# ==================== INTEGRACIÓN MLFLOW ====================
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
# ===========================================================

logger = logging.getLogger('logger_modelo_rc')
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
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
        
        training_set = pd.concat([data1, data2, data3, data4], ignore_index=True)
        training_set = self.crear_features_adicional(training_set)
        
        num_ataques = 0
        for _, col in training_data.iterrows():
            if col['label'] == 1:
               num_ataques += 1

        print(num_ataques)
        return training_set
         
    def cargar_datos_CIC(self):
        logger.info("Cargando datos CIC")
        cic_1 = pd.read_csv("../../datasets/CIC-1.csv")     
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

    def crear_features_adicional(self, df):
        """Features básicos derivados de métricas de flujo de Argus"""
        epsilon = 1e-8
        
        # Totales y ratios básicos
        df['tbytes'] = df['sbytes'] + df['dbytes']
        df['tpkts'] = df['spkts'] + df['dpkts']
        df['bratio'] = df['sbytes'] / (df['dbytes'] + 1)
        df['pratio'] = df['spkts'] / (df['dpkts'] + 1)
        df['rate'] = (df['sbytes'] + df['dbytes']) / (df['dur'] + epsilon) 
        
        # Tamaños promedio de paquetes
        df['avg_pkt_size_src'] = df['sbytes'] / (df['spkts'] + 1)
        df['avg_pkt_size_dst'] = df['dbytes'] / (df['dpkts'] + 1)
        df['avg_pkt_size_flow'] = df['tbytes'] / (df['tpkts'] + 1)
        
        # Asimetría
        df['basymmetry'] = abs(df['sbytes'] - df['dbytes']) / (df['tbytes'] + 1)
        df['pasymmetry'] = abs(df['spkts'] - df['dpkts']) / (df['tpkts'] + 1)
        
        # Características de eficiencia y densidad
        df['bytes_per_second'] = df['tbytes'] / (df['dur'] + epsilon)
        df['packets_per_second'] = df['tpkts'] / (df['dur'] + epsilon)
        
        return df

    def cargar_datos_analisis(self, ruta):
        """Funcion que carga una muestra de datos de trafico de red real"""
        test_data_real = pd.read_csv(ruta)
        return test_data_real
    
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
        """Selecciona el tipo de scaler"""
        if tipo == "standard":
            scaler = StandardScaler()
        elif tipo == "min_max":
            scaler = MinMaxScaler()
        elif tipo == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(f"Tipo de scaler '{tipo}' no reconocido. Usando StandardScaler.")
            scaler = StandardScaler()
        
        self.set_numeric_scaler(scaler)
        return scaler

    def select_cat_scaler(self):
        """Placeholder para scaler categórico"""
        pass

    def set_tipo_scaler_cat(self, tipo):
        """Placeholder para tipo de scaler categórico"""
        pass

    # ==================== MÉTODOS CON MLFLOW ====================
    
    def entrenar_modelo_con_mlflow(self, X_train, X_test, y_train, y_test, 
                                    experiment_name="RandomForest_Experiments",
                                    run_name=None,
                                    modelo_params=None):
        """
        Entrena el modelo Random Forest con tracking completo en MLflow
        
        Args:
            X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
            experiment_name: Nombre del experimento en MLflow
            run_name: Nombre específico para esta ejecución
            modelo_params: Diccionario con parámetros del modelo
        """
        
        # Configurar experimento MLflow
        mlflow.set_experiment(experiment_name)
        
        # Parámetros por defecto del modelo
        if modelo_params is None:
            modelo_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Iniciar run de MLflow
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # ============ REGISTRAR PARÁMETROS ============
            mlflow.log_params(modelo_params)
            mlflow.log_param("scaler_type", type(self.scaler).__name__ if self.scaler else "None")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", X_train.shape[0])
            mlflow.log_param("n_test_samples", X_test.shape[0])
            mlflow.log_param("feature_selector", self.feature_selector)
            mlflow.log_param("balanced", self.balanceado)
            
            # Registrar features seleccionadas
            if self.features_seleccionadas:
                mlflow.log_param("selected_features", str(self.features_seleccionadas[:10]))
            
            # ============ ENTRENAR MODELO ============
            logger.info("Entrenando Random Forest...")
            modelo = RandomForestClassifier(**modelo_params)
            modelo.fit(X_train, y_train)
            
            # ============ PREDICCIONES ============
            y_pred_train = modelo.predict(X_train)
            y_pred_test = modelo.predict(X_test)
            
            # ============ CALCULAR MÉTRICAS ============
            # Métricas de entrenamiento
            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            
            # Métricas de test
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            # ============ REGISTRAR MÉTRICAS ============
            mlflow.log_metrics({
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "overfitting_gap": train_accuracy - test_accuracy
            })
            
            # ============ MATRIZ DE CONFUSIÓN ============
            cm = confusion_matrix(y_test, y_pred_test)
            
            # Guardar matriz como imagen
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusión - Test Set')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predicha')
            
            confusion_matrix_path = "/tmp/confusion_matrix.png"
            plt.savefig(confusion_matrix_path)
            mlflow.log_artifact(confusion_matrix_path, "plots")
            plt.close()
            
            # ============ FEATURE IMPORTANCE ============
            if hasattr(modelo, 'feature_importances_'):
                importances = modelo.feature_importances_
                feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_train.shape[1])]
                
                # Crear DataFrame de importancias
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Guardar top 20 features
                plt.figure(figsize=(10, 8))
                top_n = min(20, len(importance_df))
                sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
                plt.title(f'Top {top_n} Feature Importances')
                plt.tight_layout()
                
                feature_importance_path = "/tmp/feature_importance.png"
                plt.savefig(feature_importance_path)
                mlflow.log_artifact(feature_importance_path, "plots")
                plt.close()
                
                # Guardar CSV de importancias
                importance_csv_path = "/tmp/feature_importances.csv"
                importance_df.to_csv(importance_csv_path, index=False)
                mlflow.log_artifact(importance_csv_path, "data")
            
            # ============ REGISTRAR MODELO ============
            # Inferir signature del modelo
            signature = infer_signature(X_train, modelo.predict(X_train))
            
            # Registrar modelo con mlflow
            mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"RandomForest_{experiment_name}"
            )
            
            # ============ GUARDAR ARTEFACTOS ADICIONALES ============
            # Guardar scalers si existen
            if self.scaler is not None:
                scaler_path = "/tmp/scaler.joblib"
                joblib.dump(self.scaler, scaler_path)
                mlflow.log_artifact(scaler_path, "preprocessors")
            
            # Guardar lista de features
            if self.features_seleccionadas:
                features_path = "/tmp/selected_features.txt"
                with open(features_path, 'w') as f:
                    f.write('\n'.join(self.features_seleccionadas))
                mlflow.log_artifact(features_path, "data")
            
            # ============ REPORTE DE CLASIFICACIÓN ============
            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            report_path = "/tmp/classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(classification_report(y_test, y_pred_test, zero_division=0))
            mlflow.log_artifact(report_path, "reports")
            
            # Registrar métricas por clase
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)
            
            # ============ IMPRIMIR RESUMEN ============
            print("\n" + "="*50)
            print("RESUMEN DEL ENTRENAMIENTO")
            print("="*50)
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"Experiment: {experiment_name}")
            print(f"\nMétricas de Test:")
            print(f"  Accuracy:  {test_accuracy:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall:    {test_recall:.4f}")
            print(f"  F1-Score:  {test_f1:.4f}")
            print(f"\nOverfitting Gap: {train_accuracy - test_accuracy:.4f}")
            print("="*50 + "\n")
            
            self.modelo = modelo
            return modelo

    def preprocesar_datos_unsw(self, X, train=True):
        """Método placeholder para preprocesamiento UNSW"""
        # Aquí deberías implementar tu lógica de preprocesamiento
        # Por ahora solo retorna los datos tal cual
        logger.info(f"Preprocesando datos UNSW (train={train})")
        
        if train and self.scaler is not None:
            # Identificar columnas numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        elif not train and self.scaler is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X

    def guardar_modelo(self, ruta):
        """Guarda el modelo usando joblib"""
        logger.info(f"Guardando modelo en {ruta}")
        joblib.dump({
            'modelo': self.modelo,
            'scaler': self.scaler,
            'features_seleccionadas': self.features_seleccionadas,
            'proto_map': self.proto_map,
            'state_map': self.state_map,
            'service_map': self.service_map
        }, ruta)

def main():
    """Función principal con integración MLflow"""
    
    modelo_rf = ModeloRandomForest()
    
    # Configuración
    modelo_rf.set_tipo_scaler_cat("label")
    modelo_rf.feature_selector = "kbest"
    modelo_rf.select_scaler("robust")
    modelo_rf.select_cat_scaler()
    
    # Seleccionar dataset
    modelo = "unsw"
    
    if modelo == "unsw":
        # Cargar datos UNSW
        X_train = modelo_rf.cargar_datos()
        X_train['attack_cat'].fillna('normal', inplace=True)
        X_train['attack_cat'] = X_train['attack_cat'].apply(lambda x: x.strip().lower())
        X_train['attack_cat'] = X_train['attack_cat'].replace('backdoors', 'backdoor')
        
        y_train = X_train['attack_cat']
        col_eliminar = ['attack_cat', 'label']
        X_train = X_train.drop(col_eliminar, axis=1)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        # Seleccionar features
        mini_features = ['sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'spkts', 
                        'dpkts', 'rate', 'proto', 'service', 'tbytes', 'tpkts', 
                        'basymmetry', 'pasymmetry']
        modelo_rf.features_seleccionadas = mini_features
        
        X_train = X_train[mini_features]
        X_test = X_test[mini_features]
        
        # Preprocesar
        X_train = modelo_rf.preprocesar_datos_unsw(X_train, train=True)
        X_test = modelo_rf.preprocesar_datos_unsw(X_test, train=False)
        
        # ============ ENTRENAR CON MLFLOW ============
        modelo_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        modelo = modelo_rf.entrenar_modelo_con_mlflow(
            X_train, X_test, y_train, y_test,
            experiment_name="UNSW_NB15_Attack_Detection",
            run_name="baseline_mini_features",
            modelo_params=modelo_params
        )
        
        # Guardar modelo tradicional también
        ruta = "../../models/modelo_rf_unsw_mlflow.joblib"
        modelo_rf.guardar_modelo(ruta)
        
        logger.info("✓ Entrenamiento completado con éxito")
        logger.info("✓ Revisa MLflow UI para ver los resultados")
        print("\nPara ver los resultados en MLflow UI, ejecuta:")
        print("  mlflow ui")
        print("Y abre http://localhost:5000 en tu navegador")

if __name__ == "__main__":
    main()