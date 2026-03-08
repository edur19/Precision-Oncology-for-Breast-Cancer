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
# 1️⃣ Rutas y configuración (SOLO PARETO) 🔑
# ============================================================
# ❗ AJUSTA ESTA RUTA A TU UBICACIÓN LOCAL
# PATH_NORMAS12 ha sido eliminado.
PATH_PARETO = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/ParetoSurface_CU_EA_extended_1226_Final (1).csv" # Única fuente de features
PATH_CLINICAL = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/TCGA-BRCA.clinical.tsv"
PATH_SURVIVAL = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/TCGA-BRCA.survival.tsv.gz"
PATH_METADATA_SUP = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/MetaData.xlsx"
OUT_DIR = "resultados_de_pareto_sin_filtro" # Directorio de salida actualizado
os.makedirs(OUT_DIR, exist_ok=True)
SUFFIX = "_Xomics_specificModel.mat"


# 2.3 Estandarizar ID de Modelo Metabólico a 16 caracteres
def extract_model_id(model_name):
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2}[A-Z0-9]?)', model_name)
    if match:
        return match.group(0)[:16] 
    return model_name.split('_')[0].strip()[:16]

# ============================================================
# 2️⃣ CARGA, PREPARACIÓN Y AGREGACIÓN DE FEATURES DE PARETO 🔑
# ============================================================

# --- 2.1 Carga y Agregación de Pareto (Base Principal) ---
try:
    df_pareto = pd.read_csv(PATH_PARETO)
    PARETO_LOADED = True
except FileNotFoundError:
    print(f"❌ ERROR: Archivo Pareto no encontrado en {PATH_PARETO}. El pipeline se detiene.")
    PARETO_LOADED = False
    exit() # Detener si la única fuente de datos no se encuentra

if PARETO_LOADED:
    # Estandarización de ID de Pareto a 16 caracteres
    df_pareto["Model_ID_16"] = df_pareto["ModelName"].apply(extract_model_id)
    
    # 🚨 Definición de Métricas para Agregación de Pareto
    metrics_for_aggregation_pareto = [
        'CU_real', 'EA_real', 'Biomass', 'ATP', 'WarburgIndex', 'NitrogenAnaplerosis',
        'GrowthMetabolism', 'ATPConsumption', 'ATPProduction', 'RatioCU_EA'
    ]
    sa_cols_pareto = [col for col in df_pareto.columns if col.startswith('SA_')]

    # Diccionario de agregación
    agg_dict_pareto = {}
    for col in metrics_for_aggregation_pareto:
        if col in df_pareto.columns: agg_dict_pareto[col] = ['mean', 'std', 'median']
    for col in sa_cols_pareto:
        if col in df_pareto.columns: agg_dict_pareto[col] = ['mean', 'std']

    # Aplicar la agregación
    df_summary_pareto = df_pareto.groupby('Model_ID_16').agg(agg_dict_pareto)
    df_summary_pareto.columns = [f"Pareto_{var}_{stat}" for var, stat in df_summary_pareto.columns]
    df_summary_pareto = df_summary_pareto.reset_index().rename(columns={'Model_ID_16': 'Model'})
    
    df_features_merged = df_summary_pareto.copy()
    feature_cols = [col for col in df_features_merged.columns if col != 'Model']
    print(f"✅ Features de Pareto cargados y agregados: {len(feature_cols)} columnas.")

else:
    # Este bloque nunca debería ejecutarse debido al 'exit()' anterior
    exit()

patient_ids = df_features_merged["Model"].values
print(f"\nSe utiliza la base de Pareto con {len(patient_ids)} modelos.")


# ============================================================
# 3️⃣ Preprocesamiento Unificado (Imputación + Escalado) 🔑
# ============================================================

df_X = df_features_merged.copy()

# 3.1. Imputación
# No hay reemplazo de 0 por NaN aquí, ya que los datos de Pareto son agregaciones.
# Imputar TODAS las columnas con la mediana
X_combined = df_X[feature_cols].values
imputer = SimpleImputer(missing_values=np.nan, strategy='median', fill_value=0) 
X_imputed = imputer.fit_transform(X_combined)

# 3.2. Aplicar Winsorización (CONTROL DE OUTLIERS en las agregaciones)
# La Winsorización se mantiene ya que es buena práctica para las métricas de Pareto (mean/std/median).
X_wins = np.array([winsorize(col, limits=[0.01, 0.01]) for col in X_imputed.T]).T

# 3.3. Escalado robusto FINAL
X_scaled = RobustScaler().fit_transform(X_wins)

print(f"✅ Preprocesamiento completo. X_scaled final generado con forma: {X_scaled.shape}")

# El resto del código (Secciones 4-12) permanece igual ya que opera sobre X_scaled y feature_cols.

# ============================================================
# 4️⃣ Reducción UMAP con Múltiples Semillas y Grid Search 🔑
# ============================================================
# ... (El código de UMAP no cambia, utiliza X_scaled) ...

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

all_embeddings = {
    **embedding_matrices
}
print(f"✅ Embeddings para clustering generados: {list(all_embeddings.keys())}")


# ============================================================
# 5️⃣ Funciones utilitarias clustering
# ============================================================
def n_clusters_from_labels(labels):
    return len(set(labels)) - (1 if -1 in labels else 0)

def compute_noise_pct(labels):
    return np.mean(labels == -1) * 100

# ============================================================
# 6️⃣ Rejillas de parámetros
# ============================================================
K_VALORES = range(2, 15) 

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

# Estimación de ancho de banda para MeanShift
if 'MeanShift' in alg_classes and 'MeanShift' not in param_grids:
    from sklearn.cluster import estimate_bandwidth
    bw = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=min(len(X_scaled), 500)) if len(X_scaled) > 0 else 1.0
    param_grids['MeanShift'] = [{'bandwidth': [bw, bw*1.5, bw*0.5]}]
    
# Parámetros para AffinityPropagation
if 'AffinityPropagation' in alg_classes and 'AffinityPropagation' not in param_grids:
    param_grids['AffinityPropagation'] = [{'damping': [0.5, 0.9]}]
param_grids = {k: v for k, v in param_grids.items() if k in alg_classes}

# ============================================================
# 7️⃣ Búsqueda de clustering óptimo
# ============================================================
def optimize_clustering(alg_name, X):
    best_score = -np.inf; best_db = np.inf; best_chi = -np.inf;
    best_labels = None; best_params = None; best_noise = None
    best_seed = None 
    X_arr = np.asarray(X)
    alg_class = alg_classes[alg_name] 

    MAX_NOISE_PCT = 5.0 # Filtro estricto del 5%

    if alg_name in ["GMM", "BayesianGaussianMixture"]:
        max_comp = X_arr.shape[0] - 1
        current_gmm_comps = [k for k in param_grids.get(alg_name, {}).get('n_components', []) if k <= max_comp]
        if not current_gmm_comps:
             return {"best_score": None, "best_db": None, "best_chi": None, "best_labels": None, "best_params": None, "noise_pct": None, "n_clusters_found": 0, "best_seed": None}
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
                if alg_name in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch'] :
                    model = alg_class(**params, random_state=current_seed)
                else:
                    model = alg_class(**params)

                if alg_name == 'MeanShift':
                    model.fit(X_arr)
                    labels = model.labels_
                else:
                    labels = model.fit_predict(X_arr)

                noise_pct = compute_noise_pct(labels)

                if noise_pct <= MAX_NOISE_PCT:
                    mask = labels != -1
                    n_clusters_valid = len(np.unique(labels[mask]))

                    if n_clusters_valid > 1 and mask.sum() > 1:
                        score = silhouette_score(X_arr[mask], labels[mask])
                        db_score = davies_bouldin_score(X_arr[mask], labels[mask])
                        chi_score = calinski_harabasz_score(X_arr[mask], labels[mask])
                    else:
                        score = -np.inf; db_score = np.inf; chi_score = -np.inf;

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
                continue

        if alg_name in ['Agglomerative', 'DBSCAN', 'HDBSCAN', 'MeanShift'] and current_seed == SEEDS_TO_TEST[0]:
            break

    if best_labels is None:
        return {"best_score": None, "best_db": None, "best_chi": None, "best_labels": None, "best_params": None, "noise_pct": None, "n_clusters_found": 0, "best_seed": None}

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
# 8️⃣ Ejecutar todos los algoritmos en todos los embeddings
# ============================================================
rows = []
all_algorithm_names = list(alg_classes.keys())

for reduction_name, X_emb in all_embeddings.items():
    print(f"\n==============================================")
    print(f"🔬 ANALIZANDO ESPACIO: {reduction_name} ({X_emb.shape[1]} dims)")
    print(f"==============================================")

    for alg in all_algorithm_names:
        if reduction_name == 'SCALED_RAW' and alg in ['DBSCAN', 'HDBSCAN'] and X_emb.shape[1] > 20:
            print(f"⚠️ Omitiendo {alg} en SCALED_RAW por alta dimensionalidad.")
            continue

        print(f"\n🔹 Optimizando {alg} en {reduction_name}...")

        res = optimize_clustering(alg, X_emb)
        n_clusters = res.get('n_clusters_found', 0)
        best_seed = res.get('best_seed', 'N/A')

        if res['best_labels'] is not None and res['noise_pct'] <= 5.0 and n_clusters > 1:
            rows.append({
                'algorithm': alg,
                'Optimal K/Comp': n_clusters,
                'score': res['best_score'],
                'Davies-Bouldin Score': res['best_db'],
                'Calinski-Harabasz Score': res['best_chi'],
                'labels': res['best_labels'],
                'params': res['best_params'],
                'noise': res['noise_pct'],
                'reduction': reduction_name,
                'seed': best_seed 
            })
            print(f"✅ {alg} | K/Comp={n_clusters} | Seed={best_seed} | Silhouette Score: {res['best_score']:.3f} | Ruido={res['noise_pct']:.2f}%")
        else:
            if res['noise_pct'] is not None and res['noise_pct'] > 5.0:
                 print(f"❌ {alg} descartado (Ruido: {res['noise_pct']:.2f}% > 5%).")
            else:
                 print(f"⚠️ {alg} no obtuvo resultados válidos.")

# ============================================================
# 9️⃣ FILTRADO Y EXPORTACIÓN DE LOS ALGORITMOS SELECCIONADOS
# ============================================================
df_scores = pd.DataFrame(rows)
df_scores = df_scores[df_scores['labels'].notnull()].copy()

SILHOUETTE_THRESHOLD = 0.1 

df_selected_models = df_scores[
    df_scores['score'] >= SILHOUETTE_THRESHOLD
].copy()

df_selected_models = df_scores.sort_values(
    by=['score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score', 'noise'],
    ascending=[False, False, True, True]
).reset_index(drop=True)

num_selected_models = len(df_selected_models)
print(f"\n=== 🏆 Encontrados {num_selected_models} modelos con Silhouette Score >= {SILHOUETTE_THRESHOLD:.1f} (Ruido máximo 5%) ===")

df_clusters_final = pd.DataFrame({'Model': patient_ids})

for i, row in df_selected_models.iterrows():
    labels = np.asarray(row['labels'])

    score_str = f"S{row['score']:.2f}_DB{row['Davies-Bouldin Score']:.2f}_CH{row['Calinski-Harabasz Score']:.0f}"
    seed_str = f"_Seed{row['seed']}" if row['seed'] != 'N/A' else ""
    col_name = f"Cluster_{row['reduction']}_{row['algorithm']}_K{row['Optimal K/Comp']}_{score_str}{seed_str}"

    if len(labels) == df_features_merged.shape[0]: # Usar el tamaño final
        df_clusters_final[col_name] = labels

    if i < 5 or num_selected_models <= 10:
        print(f"{i+1}. {row['reduction']} + {row['algorithm']} | K/Comp: {row['Optimal K/Comp']} | Seed: {row['seed']} | Silhouette Score: {row['score']:.4f} | Ruido: {row['noise']:.2f}%")
    elif i == 5:
        print("... (Omitiendo modelos intermedios) ...")


CLUSTERS_PATH = os.path.join(OUT_DIR, "PatientClusters_Selected_PARETO_sinfiltro.csv")
df_clusters_final.to_csv(CLUSTERS_PATH, index=False)
print(f"\n✅ Guardado {num_selected_models} clusterings seleccionados en: {CLUSTERS_PATH}")


# ============================================================
# 12️⃣ PLOTEO DEL MEJOR UMAP 2D 🖼️
# ============================================================
df_top_to_plot = df_selected_models.head(5).copy() 

# 🚨 Pre-procesamiento para ploteo 🚨
score_str_fn = df_top_to_plot.apply(
    lambda row: f"{row['reduction']}_{row['algorithm']}_K{row['Optimal K/Comp']}_S{row['score']:.2f}_DB{row['Davies-Bouldin Score']:.2f}_CH{row['Calinski-Harabasz Score']:.0f}",
    axis=1
)
seed_str_fn = df_top_to_plot['seed'].apply(lambda x: f"_Seed{x}" if x != 'N/A' else "")
df_top_to_plot['Configuration'] = score_str_fn + seed_str_fn
df_top_to_plot = df_top_to_plot.rename(columns={'reduction': 'Embedding', 'labels': 'Labels'})
df_plot_ready = df_top_to_plot[['Embedding', 'Labels', 'Configuration', 'score', 'Optimal K/Comp', 'noise']].copy()


def generate_umap_plots(df_top_models, embedding_matrices, output_dir):
    """Genera y muestra las gráficas de UMAP para un subconjunto de los modelos."""

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

        title = f"TOP {i+1}: {config_name}\nSilhouette: {score:.3f} | K={int(k_comp)} | Ruido: {noise:.1f}%"
        filename = os.path.join(output_dir, f"TOP_{i+1}_{config_name}.png")

        df_plot = pd.DataFrame(X_umap, columns=[f'Dim{j+1}' for j in range(n_dims)])
        df_plot['Cluster'] = labels.astype(str)

        cluster_list = sorted(df_plot['Cluster'].unique())
        cluster_labels_for_palette = [c for c in cluster_list if c != '-1']
        palette = sns.color_palette("tab10", n_colors=max(len(cluster_labels_for_palette), 1))
        color_map = {c: palette[j] for j, c in enumerate(cluster_labels_for_palette)}
        if '-1' in cluster_list: color_map['-1'] = 'gray'

        if n_dims == 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='Dim1', y='Dim2', hue='Cluster', data=df_plot, palette=color_map, legend="full", alpha=0.7, s=50)
            plt.title(title, fontsize=12)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"    > Gráfica 2D guardada: {filename}")

        elif n_dims == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            for cluster_label in cluster_list:
                df_subset = df_plot[df_plot['Cluster'] == cluster_label]
                ax.scatter(df_subset['Dim1'], df_subset['Dim2'], df_subset['Dim3'], label=f'Cluster {cluster_label}', color=color_map.get(cluster_label, 'black'), alpha=0.7, s=50)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Dim1'); ax.set_ylabel('Dim2'); ax.set_zlabel('Dim3');
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"    > Gráfica 3D guardada: {filename}")

generate_umap_plots(df_plot_ready, all_embeddings, OUT_DIR)


# ============================================================
# 10️⃣ y 11️⃣ Merges Finales (Ajustados para usar solo la base Pareto)
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
# Nota: La base ya contiene los Model IDs de Pareto. No es necesario dropear features, solo usaremos df_features_merged.

def merge_and_report(left_df, right_df, right_name):
    # Asegura que el merge use solo los IDs de pacientes
    cols_to_merge = [col for col in right_df.columns if col != 'Model']
    merged = left_df.merge(right_df.drop_duplicates(subset=["Model"]), on="Model", how="left", indicator=True)
    merged = merged.drop(columns=["_merge"])
    return merged

# El merge se hace directamente sobre la base de datos que tiene los Model IDs de Pareto
merged1 = merge_and_report(base, df_clinical_filtered, "Clinical")
merged2 = merge_and_report(merged1, df_survival_filtered, "Survival")
df_merged_final = merged2.drop_duplicates(subset=["Model"]).copy()

OUT_PATH = os.path.join(OUT_DIR, "Merged_TCGA_BRCA_AllData_safe_PARETO_sinfiltro.csv")
df_merged_final.to_csv(OUT_PATH, index=False)
print(f"\n✅ Dataset final (sin features ni clusters) guardado en: {OUT_PATH}")

# Merge con features metabólicos y clusters
df_updated = df_merged_final.merge(
    df_features_merged,
    on='Model',
    how='left'
)

try:
    df_clusters = pd.read_csv(CLUSTERS_PATH).drop_duplicates(subset=["Model"])
    cluster_cols = [c for c in df_clusters.columns if c != "Model"]
    df_updated = df_updated.merge(df_clusters, on="Model", how="left")
    df_updated[cluster_cols] = df_updated[cluster_cols].fillna(-1).astype(int)

    OUT_UPDATED = os.path.join(OUT_DIR, "Merged_TCGA_BRCA_AllData_safe_withSelectedClusters_PARETO_sinfiltro.csv")
    df_updated.to_csv(OUT_UPDATED, index=False)

    print(f"\n✅ Merge final con clusters guardado en: {OUT_UPDATED}")
    print("\nPreview de Clusters:")
    cluster_preview_cols = [c for c in df_updated.columns if c.startswith('Cluster_')]
    print(df_updated[['Model'] + cluster_preview_cols].head(8).to_string(index=False))

except FileNotFoundError:
    print(f"⚠️ No se pudo cargar el archivo de clusters en {CLUSTERS_PATH}. El merge final no incluirá clusters.")