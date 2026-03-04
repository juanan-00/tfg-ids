#!/usr/bin/env python3
"""
comparar_modelos.py
Compara RandomForest, DecisionTree, KNN, NaiveBayes y SVM (LinearSVC)
sobre CIC-IDS2017 y UNSW-NB15.
Mismo preprocesado y mismo split para todos → comparación justa.
"""

import os
import sys
import time
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Definición de modelos ─────────────────────────────────────────────────────
# KNN y NaiveBayes necesitan escalado → SKPipeline con MinMaxScaler
# LinearSVC también se beneficia del escalado

MODELOS = {
    'RandomForest': RandomForestClassifier(
        criterion='entropy', max_depth=20, n_estimators=300,
        random_state=42, max_features='sqrt', class_weight='balanced',
        min_samples_split=5, min_samples_leaf=2, n_jobs=-1,
    ),
    'DecisionTree': DecisionTreeClassifier(
        criterion='gini', max_depth=None, min_samples_split=5,
        min_samples_leaf=2, max_features='sqrt', random_state=42,
        class_weight='balanced',
    ),
    'KNN': SKPipeline([
        ('scaler', MinMaxScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)),
    ]),
    'NaiveBayes': SKPipeline([
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB()),
    ]),
    'SVM': SKPipeline([
        ('scaler', MinMaxScaler()),
        ('svm', LinearSVC(max_iter=2000, random_state=42, class_weight='balanced')),
    ]),
}

# ─── Mapeos categóricos UNSW ───────────────────────────────────────────────────
PROTO_MAP = {
    'tcp': 1, 'udp': 2, 'icmp': 3, 'igmp': 4,
    'ipv6': 5, 'ipv6-frag': 6, 'ipv6-route': 7, 'ipv6-opts': 8,
    'gre': 9, 'sctp': 10, 'rtp': 11, 'unknown': 0, '-': 0,
}
SERVICE_MAP = {
    'http': 1, 'https': 2, 'ssl': 3, 'quic-ssl': 3,
    'ftp': 4, 'ftp-data': 5, 'dns': 6, 'dhcp': 7,
    'ssh': 8, 'telnet': 9, 'smtp': 10, 'pop3': 11, 'imap': 12,
    'snmp': 13, 'radius': 14, 'irc': 15, 'ntp': 16,
    'unknown': 0, '-': 0,
}
STATE_MAP = {
    'fin': 1, 'int': 2, 'con': 3, 'eco': 4,
    'req': 5, 'rst': 6, 'par': 7, 'urn': 8,
    'no': 9, 'unknown': 0, '-': 0,
}

# Features CIC: PortScan/DoS críticas que KBest puede omitir
FEATURES_OBLIGATORIAS_CIC = [
    'DestinationPort', 'SYNFlagCount', 'RSTFlagCount',
    'TotalFwdPackets', 'TotalBackwardPackets',
    'FlowPackets/s', 'FwdPackets/s', 'Init_Win_bytes_forward',
]

# ─── Utilidades comunes ────────────────────────────────────────────────────────

def limpiar_datos(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        # Convertir valores hexadecimales ('0x...') a entero
        if X[col].dtype == object:
            X[col] = X[col].apply(
                lambda v: int(v, 16) if isinstance(v, str) and v.startswith('0x') else v
            )
        X[col] = X[col].replace('-', np.nan)
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].replace([np.nan, np.inf, -np.inf], 0)
    return X


def filtrar_clases_minoritarias(X: pd.DataFrame, y: pd.Series, min_muestras: int = 200):
    counts = y.value_counts()
    eliminar = counts[counts < min_muestras].index.tolist()
    if eliminar:
        logger.info(f"Eliminando clases con < {min_muestras} muestras: {eliminar}")
        mask = ~y.isin(eliminar)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
    return X, y


def balancear_datos(X, y, factor=3, smote_min=200, target_max=40000, dataset_name=''):
    counts = Counter(y)
    target_over = {}
    for cls, cnt in counts.items():
        if cnt >= smote_min:
            nuevo = min(cnt * factor, target_max)
            if nuevo > cnt:
                target_over[cls] = nuevo
    target_under = {cls: target_max for cls, cnt in counts.items() if cnt > target_max}
    k = min(5, min(counts.values()) - 1) if min(counts.values()) > 1 else 1

    print("\nDistribución ANTES del balanceo:")
    total = sum(counts.values())
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:40s} {cnt:8d}  ({cnt/total*100:5.1f}%)")

    pipeline = ImbPipeline([
        ('over',  SMOTE(sampling_strategy=target_over, k_neighbors=k, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=target_under, random_state=42)),
    ])
    X_res, y_res = pipeline.fit_resample(X, y)

    counts_post = Counter(y_res)
    total_post = sum(counts_post.values())
    print("\nDistribución DESPUÉS del balanceo:")
    for cls, cnt in sorted(counts_post.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:40s} {cnt:8d}  ({cnt/total_post*100:5.1f}%)")

    plot_balanceo(counts, counts_post, dataset_name)
    return X_res, y_res


def entrenar_y_evaluar(nombre, clf, X_train_bal, y_train_bal, X_test, y_test):
    logger.info(f"Entrenando {nombre}...")
    t0 = time.time()
    clf.fit(X_train_bal, y_train_bal)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    t_pred = time.time() - t0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        'clf': clf, 'y_pred': y_pred, 'report': report,
        't_train': t_train, 't_pred': t_pred,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

# ─── Visualización ─────────────────────────────────────────────────────────────

def plot_matriz_confusion(nombre, y_test, y_pred, labels, dataset_name):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) * 0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Matriz de Confusión — {nombre} ({dataset_name})')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicción')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_metricas_por_clase(nombre, y_test, y_pred, dataset_name):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    clases = [k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')]
    x = np.arange(len(clases))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(clases) * 0.9), 6))
    ax.bar(x - width, [report[c]['precision'] for c in clases], width, label='Precision', color='steelblue')
    ax.bar(x,         [report[c]['recall']    for c in clases], width, label='Recall',    color='seagreen')
    ax.bar(x + width, [report[c]['f1-score']  for c in clases], width, label='F1-Score',  color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=40, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Puntuación')
    ax.set_title(f'Métricas por clase — {nombre} ({dataset_name})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_balanceo(counts_antes, counts_despues, dataset_name):
    clases = sorted(set(counts_antes) | set(counts_despues))
    x = np.arange(len(clases))
    width = 0.35

    antes   = [counts_antes.get(c, 0)   for c in clases]
    despues = [counts_despues.get(c, 0) for c in clases]

    fig, ax = plt.subplots(figsize=(max(10, len(clases) * 0.9), 6))
    ax.bar(x - width / 2, antes,   width, label='Antes',   color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, despues, width, label='Después', color='seagreen',  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Número de muestras')
    ax.set_title(f'Distribución de clases antes y después del balanceo ({dataset_name})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparacion_f1(resultados, dataset_name):
    nombres = list(resultados.keys())
    clases = sorted({
        cls for r in resultados.values()
        for cls in r['report']
        if cls not in ('accuracy', 'macro avg', 'weighted avg')
    })
    x = np.arange(len(clases))
    n = len(nombres)
    width = 0.8 / n
    colores = ['steelblue', 'seagreen', 'tomato', 'mediumpurple', 'darkorange']

    fig, ax = plt.subplots(figsize=(max(14, len(clases) * 1.1), 7))
    for i, (nombre, color) in enumerate(zip(nombres, colores)):
        f1s = [resultados[nombre]['report'].get(c, {}).get('f1-score', 0) for c in clases]
        ax.bar(x + (i - n / 2 + 0.5) * width, f1s, width, label=nombre, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(clases, rotation=40, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Comparación F1-Score por clase — {dataset_name}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def imprimir_tabla_resumen(resultados, dataset_name):
    print(f"\n{'='*78}")
    print(f"RESUMEN COMPARATIVO — {dataset_name}")
    print(f"{'='*78}")
    print(f"{'Modelo':15s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'T.Entreno':>10s}")
    print(f"{'-'*78}")
    for nombre, r in resultados.items():
        mins, secs = divmod(int(r['t_train']), 60)
        print(f"{nombre:15s} {r['accuracy']:10.4f} {r['precision']:10.4f} "
              f"{r['recall']:10.4f} {r['f1']:10.4f}   {mins}m {secs:02d}s")
    print(f"{'='*78}")


def plot_comparacion_global(resumen_cic, resumen_unsw):
    """Gráfica de barras comparando métricas globales entre datasets y modelos."""
    nombres = list(resumen_cic.keys())
    metricas = ['accuracy', 'precision', 'recall', 'f1']
    etiquetas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colores_cic  = ['steelblue', 'seagreen', 'tomato', 'mediumpurple', 'darkorange']
    colores_unsw = ['#5b8fc7', '#4caf7d', '#d94f3a', '#9b6cc4', '#cc7a00']

    fig, axes = plt.subplots(1, len(metricas), figsize=(18, 6), sharey=True)
    fig.suptitle('Comparación global de modelos: CIC-IDS2017 vs UNSW-NB15', fontsize=13)

    x = np.arange(len(nombres))
    width = 0.35

    for ax, metrica, etiqueta in zip(axes, metricas, etiquetas):
        vals_cic  = [resumen_cic[n][metrica]  for n in nombres]
        vals_unsw = [resumen_unsw[n][metrica] for n in nombres]
        ax.bar(x - width/2, vals_cic,  width, label='CIC-IDS2017',  color='steelblue', alpha=0.85)
        ax.bar(x + width/2, vals_unsw, width, label='UNSW-NB15',    color='tomato',    alpha=0.85)
        ax.set_title(etiqueta)
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ─── CIC-IDS2017 ───────────────────────────────────────────────────────────────

def preparar_cic():
    logger.info("=== CIC-IDS2017: cargando datos ===")
    dfs = []
    for i in range(1, 9):
        df = pd.read_csv(f"../../datasets/CIC-{i}.csv")
        df.columns = [c.replace(' ', '') for c in df.columns]
        dfs.append(df)
    dataset = pd.concat(dfs, ignore_index=True)
    logger.info(f"CIC cargado: {dataset.shape[0]:,} filas")

    y = dataset['Label']
    X = dataset.drop('Label', axis=1)

    # Eliminar clases explícitamente excluidas + las minoritarias
    excluidas = ['Infiltration', 'Web Attack \u2013 SQL Injection', 'Heartbleed']
    mask = ~y.isin(excluidas)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    X, y = filtrar_clases_minoritarias(X, y, min_muestras=200)

    logger.info(f"Clases CIC: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train = limpiar_datos(X_train.copy())
    X_test  = limpiar_datos(X_test.copy())

    # KBest k=30 + features obligatorias
    selector = SelectKBest(f_classif, k=min(30, X_train.shape[1]))
    selector.fit(X_train, y_train)
    best = X_train.columns[selector.get_support()].tolist()
    for f in FEATURES_OBLIGATORIAS_CIC:
        if f in X_train.columns and f not in best:
            best.append(f)
    logger.info(f"CIC features seleccionadas: {len(best)}")

    X_train, X_test = X_train[best], X_test[best]
    X_train_bal, y_train_bal = balancear_datos(X_train, y_train, dataset_name='CIC-IDS2017')

    return X_train_bal, y_train_bal, X_test, y_test, best


# ─── UNSW-NB15 ─────────────────────────────────────────────────────────────────

def preparar_unsw():
    logger.info("=== UNSW-NB15: cargando datos ===")

    # Nombres de columnas desde el archivo de features
    features_df = pd.read_csv("../../datasets/NUSW-NB15_features.csv", encoding='cp1252')
    cols = [c.replace(" ", "").strip().lower() for c in features_df["Name"].tolist()]

    dfs = []
    for i in range(1, 5):
        df = pd.read_csv(f"../../datasets/UNSW-NB15_{i}.csv", header=None)
        df.columns = cols
        dfs.append(df)
    dataset = pd.concat(dfs, ignore_index=True)
    logger.info(f"UNSW cargado: {dataset.shape[0]:,} filas")

    # Target
    dataset['attack_cat'] = dataset['attack_cat'].fillna('normal')
    dataset['attack_cat'] = dataset['attack_cat'].str.strip().str.lower()
    dataset['attack_cat'] = dataset['attack_cat'].replace('backdoors', 'backdoor')

    y = dataset['attack_cat']

    # Features derivadas
    epsilon = 1e-8
    dataset['tbytes']           = dataset['sbytes'] + dataset['dbytes']
    dataset['tpkts']            = dataset['spkts'] + dataset['dpkts']
    dataset['bratio']           = dataset['sbytes'] / (dataset['dbytes'] + 1)
    dataset['pratio']           = dataset['spkts'] / (dataset['dpkts'] + 1)
    dataset['rate_calc']        = dataset['tbytes'] / (dataset['dur'] + epsilon)
    dataset['avg_pkt_size_src'] = dataset['sbytes'] / (dataset['spkts'] + 1)
    dataset['avg_pkt_size_dst'] = dataset['dbytes'] / (dataset['dpkts'] + 1)
    dataset['avg_pkt_size']     = dataset['tbytes'] / (dataset['tpkts'] + 1)
    dataset['basymmetry']       = abs(dataset['sbytes'] - dataset['dbytes']) / (dataset['tbytes'] + 1)
    dataset['pasymmetry']       = abs(dataset['spkts'] - dataset['dpkts']) / (dataset['tpkts'] + 1)

    # Codificar categóricas
    for col, mapa in [('proto', PROTO_MAP), ('service', SERVICE_MAP), ('state', STATE_MAP)]:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(str).str.lower().map(mapa).fillna(0).astype(int)

    # Eliminar columnas no utilizables (IPs, timestamps, targets)
    drop_cols = ['srcip', 'dstip', 'stime', 'ltime', 'attack_cat', 'label']
    X = dataset.drop(columns=[c for c in drop_cols if c in dataset.columns])

    X, y = filtrar_clases_minoritarias(X, y, min_muestras=200)
    logger.info(f"Clases UNSW: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train = limpiar_datos(X_train.copy())
    X_test  = limpiar_datos(X_test.copy())

    # KBest k=20 (UNSW tiene menos features)
    selector = SelectKBest(f_classif, k=min(20, X_train.shape[1]))
    selector.fit(X_train, y_train)
    best = X_train.columns[selector.get_support()].tolist()
    logger.info(f"UNSW features seleccionadas: {len(best)} → {best}")

    X_train, X_test = X_train[best], X_test[best]
    X_train_bal, y_train_bal = balancear_datos(X_train, y_train, dataset_name='UNSW-NB15')

    return X_train_bal, y_train_bal, X_test, y_test, best


# ─── Pipeline de comparación ───────────────────────────────────────────────────

def comparar(dataset_name, X_train_bal, y_train_bal, X_test, y_test, features, rutas_base):
    resultados = {}
    labels_sorted = sorted(y_test.unique())

    for nombre, clf in MODELOS.items():
        logger.info(f"\n{'─'*55}\n{nombre} — {dataset_name}\n{'─'*55}")

        r = entrenar_y_evaluar(nombre, clf, X_train_bal, y_train_bal, X_test, y_test)
        resultados[nombre] = r

        print(f"\n{nombre} ({dataset_name}):")
        print(classification_report(y_test, r['y_pred'], zero_division=0, digits=4))

        sep = '─' * 44
        print(f"  {sep}")
        print(f"  Accuracy : {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall   : {r['recall']:.4f}")
        print(f"  F1-Score : {r['f1']:.4f}")
        mins, secs = divmod(int(r['t_train']), 60)
        print(f"  T.Entreno: {mins}m {secs:02d}s   T.Pred: {r['t_pred']:.3f}s")
        print(f"  {sep}\n")

        plot_matriz_confusion(nombre, y_test, r['y_pred'], labels_sorted, dataset_name)
        plot_metricas_por_clase(nombre, y_test, r['y_pred'], dataset_name)

        # Guardar modelo
        ruta = rutas_base.get(nombre)
        if ruta:
            os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
            joblib.dump({'modelo': clf, 'features_seleccionadas': features}, ruta)
            logger.info(f"Guardado en {ruta}")

    imprimir_tabla_resumen(resultados, dataset_name)
    plot_comparacion_f1(resultados, dataset_name)
    return resultados


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparación de modelos IDS",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=['cic', 'unsw', 'ambos'],
        default='ambos',
        help=(
            "Dataset a evaluar:\n"
            "  cic   → solo CIC-IDS2017\n"
            "  unsw  → solo UNSW-NB15\n"
            "  ambos → ambos datasets + comparación global (por defecto)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rutas_cic = {
        'RandomForest': '../../models/cmp_rf_cic.joblib',
        'DecisionTree': '../../models/cmp_dt_cic.joblib',
        'KNN':          '../../models/cmp_knn_cic.joblib',
        'NaiveBayes':   '../../models/cmp_nb_cic.joblib',
        'SVM':          '../../models/cmp_svm_cic.joblib',
    }
    rutas_unsw = {
        'RandomForest': '../../models/cmp_rf_unsw.joblib',
        'DecisionTree': '../../models/cmp_dt_unsw.joblib',
        'KNN':          '../../models/cmp_knn_unsw.joblib',
        'NaiveBayes':   '../../models/cmp_nb_unsw.joblib',
        'SVM':          '../../models/cmp_svm_unsw.joblib',
    }

    res_cic  = None
    res_unsw = None

    # ── CIC-IDS2017 ──
    if args.dataset in ('cic', 'ambos'):
        logger.info(">>> Ejecutando comparación CIC-IDS2017")
        X_tr_cic, y_tr_cic, X_te_cic, y_te_cic, feat_cic = preparar_cic()
        res_cic = comparar('CIC-IDS2017', X_tr_cic, y_tr_cic, X_te_cic, y_te_cic,
                           feat_cic, rutas_cic)

    # ── UNSW-NB15 ──
    if args.dataset in ('unsw', 'ambos'):
        logger.info(">>> Ejecutando comparación UNSW-NB15")
        X_tr_unsw, y_tr_unsw, X_te_unsw, y_te_unsw, feat_unsw = preparar_unsw()
        res_unsw = comparar('UNSW-NB15', X_tr_unsw, y_tr_unsw, X_te_unsw, y_te_unsw,
                            feat_unsw, rutas_unsw)

    # ── Comparación global (solo si se ejecutaron ambos) ──
    if res_cic is not None and res_unsw is not None:
        print("\n" + "=" * 78)
        print("COMPARACIÓN GLOBAL: CIC-IDS2017 vs UNSW-NB15")
        print("=" * 78)
        print(f"{'Modelo':15s} {'CIC-F1':>10s} {'UNSW-F1':>10s} {'CIC-Acc':>10s} {'UNSW-Acc':>10s}")
        print("-" * 55)
        for nombre in MODELOS:
            print(f"{nombre:15s} {res_cic[nombre]['f1']:10.4f} {res_unsw[nombre]['f1']:10.4f} "
                  f"{res_cic[nombre]['accuracy']:10.4f} {res_unsw[nombre]['accuracy']:10.4f}")
        print("=" * 78)
        plot_comparacion_global(res_cic, res_unsw)


if __name__ == "__main__":
    main()
