import re 
import os
import pandas as pd
import numpy as np
# 💡 IMPORTACIONES PARA PLOTEO
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
# --------------------
from sklearn.preprocessing import RobustScaler
from scipy.stats.mstats import winsorize 
from sklearn.compose import ColumnTransformer
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
from sklearn.model_selection import ParameterGrid
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, MeanShift, AffinityPropagation
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import skew
from scipy.spatial import ConvexHull

# Import hdbscan (si está disponible)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN no está disponible. Este algoritmo será ignorado.")

# 🔑 SEMILLAS A PROBAR
SEEDS_TO_TEST = [42]
RANDOM_SEED = SEEDS_TO_TEST[0] 
np.random.seed(RANDOM_SEED)

# ============================================================
# 🎯 CONFIGURACIÓN DE RUIDO (MODIFICABLE)
# ============================================================
# 🆕 CAMBIO: Aumentar el porcentaje máximo de ruido permitido
MAX_NOISE_PCT = 90.0  # Cambiar de 5.0% a 15.0%

# 🆕 NUEVO: Configuración adicional de ruido por algoritmo
NOISE_CONFIG = {
    'DBSCAN': {
        'max_noise': 98.0,      # DBSCAN puede tener más ruido naturalmente
        'min_cluster_size': 2   # Tamaño mínimo de cluster válido
    },
    'HDBSCAN': {
        'max_noise': 98.0,      # HDBSCAN también puede tener ruido elevado
        'min_cluster_size': 2
    },
    'default': {
        'max_noise': 98.0,      # Para otros algoritmos
        'min_cluster_size': 2
    }
}

print("=" * 70)
print("⚙️  CONFIGURACIÓN DE ACEPTACIÓN DE RUIDO")
print("=" * 70)
print(f"🔹 Ruido máximo general: {MAX_NOISE_PCT}%")
print(f"🔹 Ruido máximo para DBSCAN: {NOISE_CONFIG['DBSCAN']['max_noise']}%")
print(f"🔹 Ruido máximo para HDBSCAN: {NOISE_CONFIG['HDBSCAN']['max_noise']}%")
print("=" * 70 + "\n")

# ============================================================
# 1️⃣ Rutas y configuración (SOLO PARETO) 🔑
# ============================================================
PATH_PARETO = "/Users/eduardoruiz/Documents/GitHub/Precision-Oncology-for-Breast-Cancer-Diagnosis/Clinical_data_and_models_ids/ParetoSurface_CU_EA_extended_1226_Final_ALL_1000sol.csv"
PATH_CLINICAL = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/TCGA-BRCA.clinical.tsv"
PATH_SURVIVAL = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/TCGA-BRCA.survival.tsv.gz"
PATH_METADATA_SUP = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/MetaData.xlsx"
OUT_DIR = "resultados_de_pareto_UMAP_conruido2"
os.makedirs(OUT_DIR, exist_ok=True)
SUFFIX = "_Xomics_specificModel.mat"


def extract_model_id(model_name):
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2}[A-Z0-9]?)', model_name)
    if match:
        return match.group(0)[:16] 
    return model_name.split('_')[0].strip()[:16]

# ============================================================
# 2️⃣ CARGA, PREPARACIÓN Y AGREGACIÓN DE FEATURES DE PARETO 🔑
# ============================================================

def compute_pareto_extra_metrics(df):
    metrics = {}
    metrics['Pareto_n_solutions'] = len(df)
    vars_of_interest = ['CU_real', 'EA_real', 'ATP', 'Biomass']
    for var in vars_of_interest:
        if var in df.columns:
            metrics[f'Pareto_{var}_range'] = df[var].max() - df[var].min()
            metrics[f'Pareto_{var}_skew'] = skew(df[var], nan_policy='omit')

    hull_vars = [v for v in ['CU_real', 'EA_real', 'ATP'] if v in df.columns]
    if len(hull_vars) >= 2 and len(df) >= len(hull_vars) + 1:
        try:
            hull = ConvexHull(df[hull_vars].dropna().values)
            metrics['Pareto_front_volume'] = hull.volume
        except:
            metrics['Pareto_front_volume'] = np.nan
    else:
        metrics['Pareto_front_volume'] = np.nan

    metrics['Pareto_front_density'] = (
        metrics['Pareto_n_solutions'] / metrics['Pareto_front_volume']
        if metrics['Pareto_front_volume'] not in [0, np.nan] else np.nan
    )
    return pd.Series(metrics)

try:
    df_pareto = pd.read_csv(PATH_PARETO)
    PARETO_LOADED = True
except FileNotFoundError:
    print(f"⚠️ ADVERTENCIA: Archivo Pareto no encontrado.")
    PARETO_LOADED = False

if PARETO_LOADED:
    df_pareto["Model_ID_16"] = df_pareto["ModelName"].apply(extract_model_id)

    metrics_for_aggregation_pareto = [
        'CU_real', 'EA_real', 'Biomass', 'ATP', 'WarburgIndex', 'NitrogenAnaplerosis',
        'GrowthMetabolism', 'ATPConsumption', 'ATPProduction', 'RatioCU_EA'
    ]
    sa_cols_pareto = [col for col in df_pareto.columns if col.startswith('SA_')]

    agg_dict_pareto = {}
    for col in metrics_for_aggregation_pareto:
        if col in df_pareto.columns: agg_dict_pareto[col] = ['mean', 'std', 'median']
    for col in sa_cols_pareto:
        if col in df_pareto.columns: agg_dict_pareto[col] = ['mean', 'std']

    df_summary_pareto = df_pareto.groupby('Model_ID_16').agg(agg_dict_pareto)
    df_pareto_extra = (
        df_pareto
        .groupby('Model_ID_16')
        .apply(compute_pareto_extra_metrics)
        .reset_index()
        .rename(columns={'Model_ID_16': 'Model'})
    )

    df_summary_pareto.columns = [f"Pareto_{var}_{stat}" for var, stat in df_summary_pareto.columns]
    df_summary_pareto = df_summary_pareto.reset_index().rename(columns={'Model_ID_16': 'Model'})
    
    feature_cols_pareto = [col for col in df_summary_pareto.columns if col != 'Model']
    print(f"✅ Features de Pareto agregados: {len(feature_cols_pareto)} columnas.")

    df_features_merged = pd.merge(df_summary_pareto, df_pareto_extra, on='Model', how='left')

    feature_cols_pareto_extra = [c for c in df_pareto_extra.columns if c != 'Model']
    feature_cols = feature_cols_pareto + feature_cols_pareto_extra

# ============================================================
# 3️⃣ Preprocesamiento Unificado
# ============================================================
df_feat = df_features_merged[feature_cols].copy()

all_zero_cols = df_feat.columns[(df_feat == 0).all()].tolist()
zero_var_cols = df_feat.columns[df_feat.nunique(dropna=False) <= 1].tolist()
cols_to_drop = sorted(set(all_zero_cols + zero_var_cols))

print(f"🧹 Eliminando {len(cols_to_drop)} columnas sin información:")
for c in cols_to_drop:
    print(f"   - {c}")

df_features_merged = df_features_merged.drop(columns=cols_to_drop)
feature_cols = [c for c in feature_cols if c not in cols_to_drop]
print(f"✅ Features finales tras limpieza: {len(feature_cols)}")

X_combined = df_features_merged[feature_cols].values
imputer = SimpleImputer(missing_values=np.nan, strategy='median', fill_value=0) 
X_imputed = imputer.fit_transform(X_combined)

X_wins = np.array([winsorize(col, limits=[0.01, 0.01]) for col in X_imputed.T]).T
X_scaled = RobustScaler().fit_transform(X_wins)

print(f"✅ Preprocesamiento completo. X_scaled final generado con forma: {X_scaled.shape}")

# ============================================================
# 4️⃣ Reducción UMAP
# ============================================================
embedding_matrices = {} 
X_input = X_scaled

n_components_list = [2, 3]
n_neighbors_list = [10, 30, 50]
min_dist_list = [0.05, 0.3]
metric_list = ['euclidean', 'manhattan', 'cosine']

umap_param_grid = []
for n_comp in n_components_list:
    for n_neigh in n_neighbors_list:
        for m_dist in min_dist_list:
            for metric_type in metric_list:
                umap_param_grid.append({
                    'n_components': n_comp,
                    'n_neighbors': n_neigh,
                    'min_dist': m_dist,
                    'metric': metric_type
                })

print(f"\n🔸 Ejecutando UMAP en X_scaled ({X_scaled.shape[1]} features) con {len(umap_param_grid)} configuraciones y {len(SEEDS_TO_TEST)} semillas...")

for current_seed in SEEDS_TO_TEST:
    np.random.seed(current_seed) 
    for params in umap_param_grid:
        n_components = params['n_components']
        metric_name = params['metric']
        emb_name = f'UMAP_C{n_components}_NN{params["n_neighbors"]}_MD{params["min_dist"]}_M{metric_name}_S{current_seed}'
        try:
            umap_model = umap.UMAP(**params, random_state=current_seed)
            X_umap = umap_model.fit_transform(X_input)
            embedding_matrices[emb_name] = X_umap
        except Exception as e:
            print(f"Error al ejecutar UMAP con {emb_name}: {e}")
            continue

all_embeddings = {**embedding_matrices}
print(f"✅ Embeddings para clustering generados: {list(all_embeddings.keys())}")

# ============================================================
# 5️⃣ Funciones utilitarias (MODIFICADAS)
# ============================================================
def n_clusters_from_labels(labels):
    """Cuenta número de clusters excluyendo ruido (-1)"""
    return len(set(labels)) - (1 if -1 in labels else 0)

def compute_noise_pct(labels):
    """Calcula porcentaje de puntos clasificados como ruido"""
    return np.mean(labels == -1) * 100

# 🆕 NUEVA FUNCIÓN: Obtener límite de ruido por algoritmo
def get_max_noise_for_algorithm(alg_name):
    """
    Retorna el porcentaje máximo de ruido permitido para un algoritmo específico.
    """
    if alg_name in NOISE_CONFIG:
        return NOISE_CONFIG[alg_name]['max_noise']
    else:
        return NOISE_CONFIG['default']['max_noise']

# 🆕 NUEVA FUNCIÓN: Validar calidad del clustering considerando ruido
def validate_clustering_quality(labels, alg_name, X):
    """
    Valida si un clustering es aceptable considerando:
    - Porcentaje de ruido
    - Número de clusters válidos
    - Tamaño mínimo de clusters
    
    Returns:
        tuple: (is_valid, reason)
    """
    max_noise = get_max_noise_for_algorithm(alg_name)
    noise_pct = compute_noise_pct(labels)
    
    # Verificar ruido
    if noise_pct > max_noise:
        return False, f"Ruido excesivo: {noise_pct:.1f}% > {max_noise}%"
    
    # Contar clusters válidos (sin ruido)
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return False, f"Insuficientes clusters: {n_clusters} < 2"
    
    # Verificar tamaño mínimo de clusters
    min_size = NOISE_CONFIG.get(alg_name, NOISE_CONFIG['default'])['min_cluster_size']
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    
    if any(size < min_size for size in cluster_sizes):
        return False, f"Cluster muy pequeño: min={min(cluster_sizes)} < {min_size}"
    
    # Verificar que haya suficientes puntos no-ruido para calcular métricas
    if mask.sum() < 10:  # Al menos 10 puntos no-ruido
        return False, f"Muy pocos puntos válidos: {mask.sum()} < 10"
    
    return True, "OK"

# ============================================================
# 6️⃣ Rejillas de parámetros
# ============================================================
K_VALORES = range(2, 5) 

param_grids = {
    "KMeans": {'n_clusters': K_VALORES},
    "Agglomerative": {'n_clusters': K_VALORES, 'linkage': ['ward', 'average', 'complete']},
    "Birch": {'n_clusters': K_VALORES},
    "GMM": {'n_components': K_VALORES},
    "BayesianGaussianMixture": {'n_components': K_VALORES},
    "DBSCAN": {'eps': [0.5, 1.0, 1.5, 2.5, 5.0], 'min_samples': [3, 5, 8]},
    "HDBSCAN": {'min_cluster_size': [5, 10, 15]}
}
alg_classes = {
    'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering, 'Birch': Birch,
    'GMM': GaussianMixture, 'DBSCAN': DBSCAN,
    'MeanShift': MeanShift, 'AffinityPropagation': AffinityPropagation,
    'BayesianGaussianMixture': BayesianGaussianMixture,
}
if HDBSCAN_AVAILABLE:
    alg_classes['HDBSCAN'] = hdbscan.HDBSCAN
alg_classes = {k: v for k, v in alg_classes.items() if v is not None}

if 'MeanShift' in alg_classes and 'MeanShift' not in param_grids:
    from sklearn.cluster import estimate_bandwidth
    bw = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=min(len(X_scaled), 500)) if len(X_scaled) > 0 else 1.0
    param_grids['MeanShift'] = [{'bandwidth': [bw, bw*1.5, bw*0.5]}]
    
if 'AffinityPropagation' in alg_classes and 'AffinityPropagation' not in param_grids:
    param_grids['AffinityPropagation'] = [{'damping': [0.5, 0.9]}]
param_grids = {k: v for k, v in param_grids.items() if k in alg_classes}

# ============================================================
# 7️⃣ Búsqueda de clustering óptimo (MODIFICADA PARA ACEPTAR RUIDO)
# ============================================================
def optimize_clustering(alg_name, X):
    """
    🆕 MODIFICADA: Ahora acepta configuraciones con ruido según umbrales por algoritmo
    """
    best_score = -np.inf
    best_db = np.inf
    best_chi = -np.inf
    best_labels = None
    best_params = None
    best_noise = None
    best_seed = None 
    X_arr = np.asarray(X)
    alg_class = alg_classes[alg_name] 
    
    # 🆕 CAMBIO: Obtener límite de ruido específico del algoritmo
    max_noise_for_alg = get_max_noise_for_algorithm(alg_name)

    if alg_name in ["GMM", "BayesianGaussianMixture"]:
        max_comp = X_arr.shape[0] - 1
        current_gmm_comps = [k for k in param_grids.get(alg_name, {}).get('n_components', []) if k <= max_comp]
        if not current_gmm_comps:
             return {
                 "best_score": None, "best_db": None, "best_chi": None, 
                 "best_labels": None, "best_params": None, "noise_pct": None, 
                 "n_clusters_found": 0, "best_seed": None
             }
        current_grid = {'n_components': current_gmm_comps}
    else:
        current_grid = param_grids.get(alg_name, [{}])

    if isinstance(current_grid, dict):
        current_grid = ParameterGrid(current_grid)
    elif not isinstance(current_grid, ParameterGrid):
        current_grid = ParameterGrid([{}])

    if not list(current_grid):
        current_grid = ParameterGrid([{}])

    for current_seed in SEEDS_TO_TEST:
        if alg_name not in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch', 'AffinityPropagation'] and current_seed != SEEDS_TO_TEST[0]:
            continue

        for params in current_grid:
            try:
                if alg_name in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch']:
                    model = alg_class(**params, random_state=current_seed)
                else:
                    model = alg_class(**params)

                if alg_name == 'MeanShift':
                    model.fit(X_arr)
                    labels = model.labels_
                else:
                    labels = model.fit_predict(X_arr)

                # 🆕 CAMBIO: Usar validación flexible de ruido
                is_valid, reason = validate_clustering_quality(labels, alg_name, X_arr)
                
                if not is_valid:
                    # Opcional: descomentar para debug
                    # print(f"      Descartado {alg_name} con params {params}: {reason}")
                    continue
                
                noise_pct = compute_noise_pct(labels)
                mask = labels != -1
                n_clusters_valid = len(np.unique(labels[mask]))

                if n_clusters_valid > 1 and mask.sum() > 1:
                    score = silhouette_score(X_arr[mask], labels[mask])
                    db_score = davies_bouldin_score(X_arr[mask], labels[mask])
                    chi_score = calinski_harabasz_score(X_arr[mask], labels[mask])
                else:
                    score = -np.inf
                    db_score = np.inf
                    chi_score = -np.inf

                is_better = False
                if score > best_score:
                    is_better = True
                elif score == best_score:
                    if chi_score > best_chi:
                        is_better = True
                    elif chi_score == best_chi and db_score < best_db:
                        is_better = True

                if is_better:
                    best_score = score
                    best_db = db_score
                    best_chi = chi_score
                    best_params = params
                    best_labels = labels.copy()
                    best_noise = noise_pct
                    best_n_clusters = n_clusters_valid
                    best_seed = current_seed 

            except Exception as e:
                # Opcional: descomentar para debug
                # print(f"      Error en {alg_name}: {e}")
                continue

        if alg_name in ['Agglomerative', 'DBSCAN', 'HDBSCAN', 'MeanShift'] and current_seed == SEEDS_TO_TEST[0]:
            break

    if best_labels is None:
        return {
            "best_score": None, "best_db": None, "best_chi": None,
            "best_labels": None, "best_params": None, "noise_pct": None,
            "n_clusters_found": 0, "best_seed": None
        }

    return {
        "best_score": None if best_score == -np.inf else float(best_score),
        "best_db": None if best_db == np.inf else float(best_db),
        "best_chi": None if best_chi == -np.inf else float(best_chi),
        "best_labels": best_labels,
        "best_params": best_params,
        "noise_pct": float(best_noise),
        "n_clusters_found": best_n_clusters,
        "best_seed": best_seed
    }

# ============================================================
# 8️⃣ Ejecutar todos los algoritmos en todos los embeddings (MODIFICADO)
# ============================================================
rows = []
all_algorithm_names = list(alg_classes.keys())

print("\n" + "=" * 70)
print("🚀 INICIANDO CLUSTERING CON ACEPTACIÓN DE RUIDO")
print("=" * 70)

for reduction_name, X_emb in all_embeddings.items():
    print(f"\n{'='*70}")
    print(f"🔬 ANALIZANDO ESPACIO: {reduction_name} ({X_emb.shape[1]} dims)")
    print(f"{'='*70}")

    for alg in all_algorithm_names:
        if reduction_name == 'SCALED_RAW' and alg in ['DBSCAN', 'HDBSCAN'] and X_emb.shape[1] > 20:
            print(f"⚠️ Omitiendo {alg} en SCALED_RAW por alta dimensionalidad.")
            continue

        max_noise_alg = get_max_noise_for_algorithm(alg)
        print(f"\n🔹 Optimizando {alg} en {reduction_name} (ruido máx: {max_noise_alg}%)...")

        res = optimize_clustering(alg, X_emb)
        n_clusters = res.get('n_clusters_found', 0)
        best_seed = res.get('best_seed', 'N/A')
        noise_pct = res.get('noise_pct', None)

        # 🆕 CAMBIO: Condición más flexible para aceptar resultados
        if res['best_labels'] is not None and n_clusters > 1:
            # Validar una vez más antes de guardar
            is_valid, reason = validate_clustering_quality(res['best_labels'], alg, X_emb)
            
            if is_valid:
                rows.append({
                    'algorithm': alg,
                    'Optimal K/Comp': n_clusters,
                    'score': res['best_score'],
                    'Davies-Bouldin Score': res['best_db'],
                    'Calinski-Harabasz Score': res['best_chi'],
                    'labels': res['best_labels'],
                    'params': res['best_params'],
                    'noise': noise_pct,
                    'reduction': reduction_name,
                    'seed': best_seed 
                })
                print(f"✅ {alg} | K/Comp={n_clusters} | Seed={best_seed} | "
                      f"Silhouette: {res['best_score']:.3f} | Ruido={noise_pct:.2f}%")
            else:
                print(f"❌ {alg} descartado: {reason}")
        else:
            if noise_pct is not None:
                print(f"⚠️ {alg} no generó resultados válidos (ruido={noise_pct:.1f}%, clusters={n_clusters})")
            else:
                print(f"⚠️ {alg} no generó resultados válidos")

# ============================================================
# 9️⃣ FILTRADO Y EXPORTACIÓN (MODIFICADO)
# ============================================================
df_scores = pd.DataFrame(rows)
df_scores = df_scores[df_scores['labels'].notnull()].copy()

print(f"\n{'='*70}")
print(f"📊 RESUMEN DE RESULTADOS")
print(f"{'='*70}")
print(f"Total de configuraciones válidas: {len(df_scores)}")

if len(df_scores) > 0:
    print(f"\nEstadísticas de ruido:")
    print(f"  Mínimo: {df_scores['noise'].min():.2f}%")
    print(f"  Máximo: {df_scores['noise'].max():.2f}%")
    print(f"  Media: {df_scores['noise'].mean():.2f}%")
    print(f"  Mediana: {df_scores['noise'].median():.2f}%")
    
    print(f"\nEstadísticas de Silhouette Score:")
    print(f"  Mínimo: {df_scores['score'].min():.4f}")
    print(f"  Máximo: {df_scores['score'].max():.4f}")
    print(f"  Media: {df_scores['score'].mean():.4f}")

# 🆕 CAMBIO: Umbral de Silhouette más bajo para aceptar más modelos
SILHOUETTE_THRESHOLD = 0.05  # Bajado de 0.1 a 0.05

df_selected_models = df_scores[
    df_scores['score'] >= SILHOUETTE_THRESHOLD
].copy()

df_selected_models = df_selected_models.sort_values(
    by=['score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score', 'noise'],
    ascending=[False, False, True, True]
).reset_index(drop=True)

num_selected_models = len(df_selected_models)
print(f"\n{'='*70}")
print(f"🏆 MODELOS SELECCIONADOS")
print(f"{'='*70}")
print(f"Total de modelos con Silhouette >= {SILHOUETTE_THRESHOLD}: {num_selected_models}")

# Estadísticas por algoritmo
if num_selected_models > 0:
    print(f"\nModelos por algoritmo:")
    for alg in df_selected_models['algorithm'].unique():
        count = len(df_selected_models[df_selected_models['algorithm'] == alg])
        avg_noise = df_selected_models[df_selected_models['algorithm'] == alg]['noise'].mean()
        print(f"  {alg:25s}: {count:3d} modelos (ruido promedio: {avg_noise:.1f}%)")

# ─── FIX: usar df_features_merged['Model'] en lugar de patient_ids (no definido) ───
df_clusters_final = pd.DataFrame({'Model': df_features_merged['Model'].values})

for i, row in df_selected_models.iterrows():
    labels = np.asarray(row['labels'])

    score_str = f"S{row['score']:.2f}_DB{row['Davies-Bouldin Score']:.2f}_CH{row['Calinski-Harabasz Score']:.0f}"
    seed_str = f"_Seed{row['seed']}" if row['seed'] != 'N/A' else ""
    noise_str = f"_N{row['noise']:.1f}"  # 🆕 NUEVO: Incluir ruido en el nombre
    col_name = f"Cluster_{row['reduction']}_{row['algorithm']}_K{row['Optimal K/Comp']}_{score_str}{noise_str}{seed_str}"

    if len(labels) == df_features_merged.shape[0]:
        df_clusters_final[col_name] = labels

    if i < 10 or (num_selected_models <= 20 and i < num_selected_models):
        print(f"{i+1}. {row['reduction']} + {row['algorithm']} | "
              f"K/Comp: {row['Optimal K/Comp']} | "
              f"Seed: {row['seed']} | "
              f"Silhouette: {row['score']:.4f} | "
              f"Ruido: {row['noise']:.2f}%")
    elif i == 10 and num_selected_models > 20:
        print("... (Omitiendo modelos intermedios) ...")


CLUSTERS_PATH = os.path.join(OUT_DIR, "PatientClusters_Selected_PARETO_con_ruido.csv")
df_clusters_final.to_csv(CLUSTERS_PATH, index=False)
print(f"\n✅ Guardado {num_selected_models} clusterings seleccionados en: {CLUSTERS_PATH}")


# ============================================================
# 12️⃣ PLOTEO DEL MEJOR UMAP 2D (MODIFICADO PARA MOSTRAR RUIDO)
# ============================================================
df_top_to_plot = df_selected_models.head(10).copy()  # 🆕 Aumentado de 5 a 10

score_str_fn = df_top_to_plot.apply(
    lambda row: f"{row['reduction']}_{row['algorithm']}_K{row['Optimal K/Comp']}_S{row['score']:.2f}_DB{row['Davies-Bouldin Score']:.2f}_CH{row['Calinski-Harabasz Score']:.0f}_N{row['noise']:.1f}",
    axis=1
)
seed_str_fn = df_top_to_plot['seed'].apply(lambda x: f"_Seed{x}" if x != 'N/A' else "")
df_top_to_plot['Configuration'] = score_str_fn + seed_str_fn
df_top_to_plot = df_top_to_plot.rename(columns={'reduction': 'Embedding', 'labels': 'Labels'})
df_plot_ready = df_top_to_plot[['Embedding', 'Labels', 'Configuration', 'score', 'Optimal K/Comp', 'noise']].copy()


def generate_umap_plots(df_top_models, embedding_matrices, output_dir):
    """
    🆕 MODIFICADA: Ahora muestra puntos de ruido en gris
    """
    if df_top_models.empty:
        print("\n⚠️ No hay modelos TOP para graficar.")
        return

    print(f"\n📈 Generando gráficas UMAP para los {len(df_top_models)} modelos con mejor Score...")

    for i, row in df_top_models.reset_index(drop=True).iterrows():
        emb_key = row['Embedding']
        labels = row['Labels']
        config_name = row['Configuration']
        
        if 'SCALED_RAW' in emb_key:
            print(f"    > Advertencia: Omitiendo gráfica para {emb_key} (Alta dimensionalidad).")
            continue
            
        if emb_key not in embedding_matrices:
            print(f"Error: Matriz {emb_key} no encontrada para graficar.")
            continue

        X_umap = embedding_matrices[emb_key]
        n_dims = X_umap.shape[1]

        score = row['score']
        k_comp = row['Optimal K/Comp']
        noise = row['noise']

        # 🆕 CAMBIO: Incluir información de ruido en el título
        noise_count = np.sum(labels == -1)
        title = (f"TOP {i+1}: {config_name}\n"
                f"Silhouette: {score:.3f} | K={int(k_comp)} | "
                f"Ruido: {noise:.1f}% ({noise_count} pts)")
        
        filename = os.path.join(output_dir, f"TOP_{i+1}_{config_name}.png")

        df_plot = pd.DataFrame(X_umap, columns=[f'Dim{j+1}' for j in range(n_dims)])
        df_plot['Cluster'] = labels.astype(str)

        cluster_list = sorted(df_plot['Cluster'].unique())
        cluster_labels_for_palette = [c for c in cluster_list if c != '-1']
        
        # 🆕 CAMBIO: Paleta más grande y colores distintivos
        n_colors = max(len(cluster_labels_for_palette), 1)
        if n_colors <= 10:
            palette = sns.color_palette("tab10", n_colors=n_colors)
        else:
            palette = sns.color_palette("husl", n_colors=n_colors)
            
        color_map = {c: palette[j] for j, c in enumerate(cluster_labels_for_palette)}
        
        # 🆕 CAMBIO: Ruido en gris claro
        if '-1' in cluster_list: 
            color_map['-1'] = '#CCCCCC'  # Gris claro

        if n_dims == 2:
            plt.figure(figsize=(12, 10))
            
            # Primero plotear clusters válidos
            for cluster_label in cluster_labels_for_palette:
                df_subset = df_plot[df_plot['Cluster'] == cluster_label]
                plt.scatter(
                    df_subset['Dim1'], df_subset['Dim2'],
                    c=[color_map[cluster_label]], 
                    label=f'Cluster {cluster_label}',
                    alpha=0.7, s=60, edgecolors='black', linewidths=0.5
                )
            
            # Luego plotear ruido (para que quede al fondo visualmente)
            if '-1' in cluster_list:
                df_noise = df_plot[df_plot['Cluster'] == '-1']
                plt.scatter(
                    df_noise['Dim1'], df_noise['Dim2'],
                    c=[color_map['-1']], 
                    label=f'Ruido ({len(df_noise)} pts)',
                    alpha=0.4, s=40, marker='x'
                )
            
            plt.title(title, fontsize=11)
            plt.xlabel('UMAP Dim 1', fontsize=10)
            plt.ylabel('UMAP Dim 2', fontsize=10)
            plt.legend(title='Asignación', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"    > Gráfica 2D guardada: {filename}")

        elif n_dims == 3:
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Clusters válidos
            for cluster_label in cluster_labels_for_palette:
                df_subset = df_plot[df_plot['Cluster'] == cluster_label]
                ax.scatter(
                    df_subset['Dim1'], df_subset['Dim2'], df_subset['Dim3'], 
                    label=f'Cluster {cluster_label}', 
                    color=color_map.get(cluster_label, 'black'), 
                    alpha=0.7, s=60, edgecolors='black', linewidths=0.5
                )
            
            # Ruido
            if '-1' in cluster_list:
                df_noise = df_plot[df_plot['Cluster'] == '-1']
                ax.scatter(
                    df_noise['Dim1'], df_noise['Dim2'], df_noise['Dim3'],
                    label=f'Ruido ({len(df_noise)} pts)',
                    color=color_map['-1'],
                    alpha=0.3, s=40, marker='x'
                )
            
            ax.set_title(title, fontsize=11, pad=20)
            ax.set_xlabel('UMAP Dim 1', fontsize=10)
            ax.set_ylabel('UMAP Dim 2', fontsize=10)
            ax.set_zlabel('UMAP Dim 3', fontsize=10)
            ax.legend(title='Asignación', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"    > Gráfica 3D guardada: {filename}")

generate_umap_plots(df_plot_ready, all_embeddings, OUT_DIR)


# ============================================================
# 10️⃣ y 11️⃣ Merges Finales
# ============================================================

def load_and_standardize_clinical(path):
    try:
        df_source = pd.read_csv(path, sep="\t", compression='gzip') if path.endswith('.gz') else pd.read_csv(path, sep="\t")
        if 'sample' in df_source.columns:
            df_source['Model'] = df_source['sample'].apply(extract_model_id)
            return df_source
        else:
            return pd.DataFrame({'Model': []})
    except FileNotFoundError:
        return pd.DataFrame({'Model': []})

df_clinical_full = load_and_standardize_clinical(PATH_CLINICAL)
df_survival_full = load_and_standardize_clinical(PATH_SURVIVAL)

pareto_models = df_features_merged['Model'].unique().tolist() 
df_clinical_filtered = df_clinical_full[df_clinical_full["Model"].isin(pareto_models)]
df_survival_filtered = df_survival_full[df_survival_full["Model"].isin(pareto_models)]

base = df_features_merged.drop(columns=feature_cols).drop_duplicates(subset=["Model"]).copy()

def merge_and_report(left_df, right_df, right_name):
    cols_to_merge = [col for col in right_df.columns if col != 'Model']
    merged = left_df.merge(right_df.drop_duplicates(subset=["Model"]), on="Model", how="left", indicator=True)
    merged = merged.drop(columns=["_merge"])
    return merged

merged1 = merge_and_report(base, df_clinical_filtered, "Clinical")
merged2 = merge_and_report(merged1, df_survival_filtered, "Survival")
df_merged_final = merged2.drop_duplicates(subset=["Model"]).copy()

OUT_PATH = os.path.join(OUT_DIR, "Merged_TCGA_BRCA_AllData_safe_PARETO_con_ruido.csv")
df_merged_final.to_csv(OUT_PATH, index=False)
print(f"\n✅ Dataset final (sin features ni clusters) guardado en: {OUT_PATH}")

df_updated = df_merged_final.merge(df_features_merged, on='Model', how='left')

try:
    df_clusters = pd.read_csv(CLUSTERS_PATH).drop_duplicates(subset=["Model"])
    cluster_cols = [c for c in df_clusters.columns if c != "Model"]
    df_updated = df_updated.merge(df_clusters, on="Model", how="left")
    df_updated[cluster_cols] = df_updated[cluster_cols].fillna(-1).astype(int)

    OUT_UPDATED = os.path.join(OUT_DIR, "Merged_TCGA_BRCA_AllData_safe_withSelectedClusters_PARETO_con_ruido.csv")
    df_updated.to_csv(OUT_UPDATED, index=False)

    print(f"\n✅ Merge final con clusters guardado en: {OUT_UPDATED}")
    print("\nPreview de Clusters:")
    cluster_preview_cols = [c for c in df_updated.columns if c.startswith('Cluster_')][:5]  # Primeros 5
    if cluster_preview_cols:
        print(df_updated[['Model'] + cluster_preview_cols].head(8).to_string(index=False))

    # 🆕 NUEVO: Estadísticas de ruido por cluster
    print(f"\n📊 Estadísticas de Ruido por Clustering:")
    print("=" * 70)
    for col in cluster_preview_cols:
        noise_count = (df_updated[col] == -1).sum()
        total = len(df_updated)
        noise_pct = (noise_count / total) * 100
        n_clusters = df_updated[df_updated[col] != -1][col].nunique()
        print(f"{col[:60]:60s}: {noise_count:3d}/{total} ({noise_pct:5.1f}%) | Clusters: {n_clusters}")

except FileNotFoundError:
    print(f"⚠️ No se pudo cargar el archivo de clusters en {CLUSTERS_PATH}. El merge final no incluirá clusters.")

print("\n" + "=" * 70)
print("🎉 ANÁLISIS COMPLETADO CON ACEPTACIÓN DE RUIDO")
print("=" * 70)