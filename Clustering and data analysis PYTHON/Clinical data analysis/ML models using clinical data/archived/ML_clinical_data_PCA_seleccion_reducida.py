# ============================================================
# 🚀 Pipeline completo: Análisis de clustering Clínico/Demográfico (CON GRAFICACIÓN PCA)
# VERSIÓN PCA - Reemplaza UMAP por PCA con grid de hiperparámetros
# ============================================================
# =========================
# 1. LIBRERÍAS Y CONFIGURACIÓN GLOBAL
# =========================
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# LIBRERÍAS DE VISUALIZACIÓN
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Importación condicional de HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️ Módulo 'hdbscan' no encontrado. Este algoritmo será omitido.")

# CONFIGURACIÓN DE RUTAS Y SEMILLAS
PATH_BASE = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/"
COL_SAMPLE_ID = 'sample'
COL_SAMPLE_TYPE = 'sample_type.samples'
OUTPUT_DIR = "Results_clustering_PCA_seleccion_reducida"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS_TO_TEST = [42, 123, 100]
RANDOM_SEED = SEEDS_TO_TEST[0]
np.random.seed(RANDOM_SEED)

# =======================================================
# 2. CARGA Y CREACIÓN DE LA BASE DE DATOS MAESTRA
# =======================================================

def load_data(filename):
    full_path = os.path.join(PATH_BASE, filename)
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(full_path)
        elif filename.endswith('.gz'):
            return pd.read_csv(full_path, sep="\t", compression='gzip')
        else:
            return pd.read_csv(full_path, sep="\t")
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo '{filename}' no encontrado en {PATH_BASE}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERROR al cargar '{filename}': {e}")
        return pd.DataFrame()

df_clinical    = load_data("TCGA-BRCA.clinical.tsv")
df_survival    = load_data("TCGA-BRCA.survival.tsv.gz")
df_metadata_raw = load_data("MetaData.xlsx")
df_model_names  = load_data("Model's_ids.txt")

for name, df in [("Clinical", df_clinical), ("Survival", df_survival), ("Model names", df_model_names)]:
    if df.empty:
        raise ValueError(f"❌ El archivo '{name}' es esencial y no pudo cargarse. Verifica la ruta.")

print(f"Bases de datos cargadas: Clínica ({len(df_clinical)}), Supervivencia ({len(df_survival)}), Metadata ({len(df_metadata_raw)})")


def extract_sample_id(filename):
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2}[A-Z0-9]?)', filename)
    if match:
        return match.group(0)[:16]
    return filename.split('_')[0].strip()[:16]


lista_modelos_unicos = df_model_names.iloc[:, 0].dropna().astype(str).tolist()
model_sample_ids = [extract_sample_id(m) for m in lista_modelos_unicos]
df_modelos_base = pd.DataFrame({
    COL_SAMPLE_ID: model_sample_ids,
    'modelo_path_completo': lista_modelos_unicos,
})

# =======================================================
# 3. MERGE PRINCIPAL
# =======================================================

metadata_cols_to_keep = [
    'Menopausal Status', 'Cancer Type', 'ER', 'PR', 'HER2', 'Subtype', 'Genetic Ancestry'
]

if not df_metadata_raw.empty and 'hidden' in df_metadata_raw.columns:
    df_metadata_clean = df_metadata_raw.copy()
    df_metadata_clean['temp_id'] = (df_metadata_clean['hidden'].astype(str)
                                    .str.replace(r'\.', '-', regex=True).str.slice(0, 16))
    df_metadata_clean.rename(columns={'temp_id': COL_SAMPLE_ID}, inplace=True)
    df_metadata_clean = df_metadata_clean.drop_duplicates(subset=[COL_SAMPLE_ID], keep='first')
    available_meta_cols = [c for c in metadata_cols_to_keep if c in df_metadata_clean.columns]
    df_metadata_clean = df_metadata_clean[[COL_SAMPLE_ID] + available_meta_cols]
    print("\n✅ Metadata molecular añadida con LEFT JOIN.")
else:
    print("❌ Metadata no cargada. Se continuará sin metadata molecular.")
    df_metadata_clean = pd.DataFrame()
    available_meta_cols = []

if 'sample' in df_clinical.columns:
    df_clinical['sample'] = df_clinical['sample'].apply(lambda x: extract_sample_id(str(x)))
if 'sample' in df_survival.columns:
    df_survival['sample'] = df_survival['sample'].apply(lambda x: extract_sample_id(str(x)))

df_survival_clean = (df_survival[['sample', 'OS.time', 'OS']]
                     .drop_duplicates(subset=['sample'], keep='first')
                     .rename(columns={'sample': COL_SAMPLE_ID}))

df_merged_clinical = pd.merge(
    df_modelos_base,
    df_clinical.drop(columns=['id', 'case_id'], errors='ignore'),
    on=COL_SAMPLE_ID, how='left'
)
df_final = pd.merge(df_merged_clinical, df_survival_clean, on=COL_SAMPLE_ID, how='left')

if not df_metadata_clean.empty:
    cols_overlap = [col for col in available_meta_cols if col in df_final.columns]
    df_final = pd.merge(
        df_final.drop(columns=cols_overlap, errors='ignore'),
        df_metadata_clean, on=COL_SAMPLE_ID, how='left'
    )

df = df_final.copy()
df_filtered = df.copy()

print(f"Filas totales finales para Clustering: {df.shape[0]}")
if COL_SAMPLE_TYPE in df_filtered.columns:
    print(df[COL_SAMPLE_TYPE].value_counts(dropna=False))
else:
    print(f"⚠️ Columna '{COL_SAMPLE_TYPE}' no encontrada.")

# =========================================================
# 2.2 Conversión de Tipos
# =========================================================
for c in ['is_ffpe.samples', 'oct_embedded.samples']:
    if c in df_filtered.columns:
        df_filtered[c] = df_filtered[c].replace({True: 1, False: 0})

for c in ['age_at_diagnosis.diagnoses', 'days_to_birth.demographic']:
    if c in df_filtered.columns and df_filtered[c].notna().any():
        if (df_filtered[c].dropna() > 1000).any():
            df_filtered[c] = df_filtered[c] / 365.25

# =========================
# 3. SELECCIÓN DE DESCRIPTORES CLÍNICOS
# =========================
descriptores_iniciales = [
    'ajcc_pathologic_stage.diagnoses', 'ajcc_pathologic_t.diagnoses',
    'ajcc_pathologic_n.diagnoses', 'ajcc_pathologic_m.diagnoses',
    'tumor_grade.diagnoses', 'morphology.diagnoses',
    'primary_diagnosis.diagnoses', 'treatment_type.treatments.diagnoses',
    'treatment_or_therapy.treatments.diagnoses', 'prior_treatment.diagnoses',
    'sample_type.samples', 'tissue_type.samples', 'tumor_descriptor.samples',
    'specimen_type.samples', 'is_ffpe.samples', 'oct_embedded.samples',
    'age_at_diagnosis.diagnoses', 'alcohol_history.exposures'
]
descriptores_alta_res = [
    'Menopausal Status', 'Cancer Type', 'ER', 'PR', 'HER2', 'Subtype', 'Genetic Ancestry'
]
descriptores_finales = descriptores_iniciales + descriptores_alta_res
final_cols = [c for c in descriptores_finales if c in df_filtered.columns]

if not final_cols:
    cols_to_exclude = ['submitter_id', 'sample', 'modelo_path_completo']
    final_cols = [col for col in df_filtered.columns if col not in cols_to_exclude]

if not final_cols:
    raise ValueError("No se encontraron columnas de descriptores válidas en el DataFrame.")

id_cols_to_keep = [c for c in ['submitter_id', 'sample', 'modelo_path_completo'] if c in df_filtered.columns]
df_aug = df_filtered[final_cols + id_cols_to_keep].copy()

# =========================
# 4. INGENIERÍA DE FEATURES
# =========================

def prior_treatment_flag(r):
    cols = ['prior_malignancy.diagnoses', 'prior_treatment.diagnoses', 'progression_or_recurrence.diagnoses']
    for c in cols:
        if c in r.index and pd.notna(r[c]):
            val = str(r[c]).lower().strip()
            if val in ['yes', 'true', 'had prior treatment', 'recurrence', 'progression']:
                return 1
    return 0

df_aug['Prior_Treatment_Flag'] = df_aug.apply(prior_treatment_flag, axis=1)

if all(c in df_aug.columns for c in ['ajcc_pathologic_m.diagnoses', 'ajcc_pathologic_n.diagnoses']):
    df_aug['Metastasis_Flag'] = df_aug.apply(
        lambda r: 1 if ('m1' in str(r['ajcc_pathologic_m.diagnoses']).lower() or
                         'n2' in str(r['ajcc_pathologic_n.diagnoses']).lower() or
                         'n3' in str(r['ajcc_pathologic_n.diagnoses']).lower()) else 0, axis=1)
else:
    df_aug['Metastasis_Flag'] = 0

receptor_cols = ['ER', 'PR', 'HER2']
if all(c in df_aug.columns for c in receptor_cols):
    def classify_molecular_subtype(row):
        er   = str(row['ER']).lower().strip()   if pd.notna(row['ER'])   else 'na'
        pr   = str(row['PR']).lower().strip()   if pd.notna(row['PR'])   else 'na'
        her2 = str(row['HER2']).lower().strip() if pd.notna(row['HER2']) else 'na'
        is_er_pos   = er   in ['positive', '+']
        is_pr_pos   = pr   in ['positive', '+']
        is_her2_pos = her2 in ['positive', '+', 'amplified', 'equivocal']
        if not is_er_pos and not is_pr_pos and not is_her2_pos:
            return 'Triple_Negative'
        elif is_her2_pos and (is_er_pos or is_pr_pos):
            return 'Luminal_HER2+'
        elif is_her2_pos and not (is_er_pos or is_pr_pos):
            return 'HER2_Enriched'
        elif is_er_pos or is_pr_pos:
            return 'Luminal_HR+'
        else:
            return 'Unknown'
    df_aug['ER_PR_HER2_Combo'] = df_aug.apply(classify_molecular_subtype, axis=1)
else:
    df_aug['ER_PR_HER2_Combo'] = 'Unknown'

# =========================
# 5. PREPROCESAMIENTO PARA ML
# =========================

label_cols_candidates = ['Molecular_Subtype', 'ER_PR_HER2_Combo', 'Subtype']
label_encoders = {}

for col in label_cols_candidates:
    if col in df_aug.columns and df_aug[col].dtype in ['object', 'category']:
        le = LabelEncoder()
        df_aug[f"{col}_encoded"] = le.fit_transform(df_aug[col].fillna('Unknown').astype(str))
        label_encoders[col] = le

id_and_meta_cols = set(id_cols_to_keep + ['submitter_id', 'sample', 'modelo_path_completo'])
numeric_cols = [c for c in df_aug.select_dtypes(include=['int64', 'float64', 'float32', 'int32', 'uint8']).columns
                if c not in id_and_meta_cols]
label_orig_cols = [col for col in label_cols_candidates
                   if col in df_aug.columns and f"{col}_encoded" in df_aug.columns]
categorical_cols = [c for c in df_aug.select_dtypes(include=['object', 'category']).columns
                    if c not in id_and_meta_cols and c not in label_orig_cols]

cols_to_impute_zero = [c for c in numeric_cols if df_aug[c].isna().any()]
if cols_to_impute_zero:
    imputer_num = SimpleImputer(strategy='constant', fill_value=0)
    df_aug.loc[:, cols_to_impute_zero] = imputer_num.fit_transform(df_aug.loc[:, cols_to_impute_zero])

for col in categorical_cols:
    df_aug[col] = df_aug[col].fillna('Missing').astype(str)

import sklearn
sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
ohe_kwargs = {'handle_unknown': 'ignore', 'sparse_output': False} if sklearn_version >= (1, 2) \
             else {'handle_unknown': 'ignore', 'sparse': False}

transformers = []
if numeric_cols:
    transformers.append(('num', StandardScaler(), numeric_cols))
if categorical_cols:
    transformers.append(('cat', OneHotEncoder(**ohe_kwargs), categorical_cols))

if not transformers:
    raise ValueError("No hay features numéricas o categóricas válidas para el preprocesamiento.")

preprocessor = ColumnTransformer(transformers, remainder='drop')
X_scaled = preprocessor.fit_transform(df_aug)

final_columns = []
if 'num' in preprocessor.named_transformers_:
    final_columns.extend(numeric_cols)
if 'cat' in preprocessor.named_transformers_ and preprocessor.named_transformers_['cat'] is not None:
    final_columns.extend(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))

final_columns = np.array(final_columns)
print(f"✅ ColumnTransformer completado: {len(final_columns)} features listos para ML.")

# =========================
# 6. EVALUACIÓN DE CLUSTERING
# =========================

K_VALORES = range(2, 10)

param_grids = {
    "KMeans": {'n_clusters': K_VALORES},
    "Agglomerative": {'n_clusters': K_VALORES, 'linkage': ['ward', 'average', 'complete']},
    "Birch": {'n_clusters': K_VALORES},
    "GMM": {'n_components': K_VALORES},
    "BayesianGaussianMixture": {'n_components': K_VALORES},
    "DBSCAN": {'eps': [0.5, 1.0, 1.5], 'min_samples': [5, 10, 20]},
    "MeanShift": {'bandwidth': [None]},
    "AffinityPropagation": {'damping': [0.5, 0.9]},
}

if HDBSCAN_AVAILABLE:
    param_grids["HDBSCAN"] = {'min_cluster_size': [5, 10, 20], 'min_samples': [None, 5, 10]}

alg_classes = {
    'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering,
    'Birch': Birch, 'GMM': GaussianMixture,
    'DBSCAN': DBSCAN, 'MeanShift': MeanShift,
    'AffinityPropagation': AffinityPropagation,
    'BayesianGaussianMixture': BayesianGaussianMixture,
}
if HDBSCAN_AVAILABLE:
    alg_classes['HDBSCAN'] = hdbscan.HDBSCAN

alg_classes  = {k: v for k, v in alg_classes.items()  if v is not None}
param_grids  = {k: v for k, v in param_grids.items()  if v}


def clustering_evaluation_table(X, algorithms, param_grids, seeds_list):
    results = {}
    summary_rows = []
    MAX_NOISE_PCT = 10.0

    for alg_name, alg_class in algorithms.items():
        best_score = -1
        best_params = best_labels = best_seed = None
        best_n_clusters = best_noise_pct = 0

        current_grid_dict = param_grids.get(alg_name, [{}])
        param_grid = ParameterGrid(current_grid_dict) if isinstance(current_grid_dict, dict) \
                     else ParameterGrid([current_grid_dict])

        for current_seed in seeds_list:
            if alg_name not in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch'] \
                    and current_seed != seeds_list[0]:
                continue
            for param_val in param_grid:
                try:
                    if alg_name in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch']:
                        model = alg_class(**param_val, random_state=current_seed)
                    else:
                        model = alg_class(**param_val)

                    labels = model.labels_ if alg_name == 'MeanShift' else model.fit_predict(X)
                    if alg_name == 'MeanShift':
                        model.fit(X)
                        labels = model.labels_

                    valid_mask = labels != -1
                    n_clusters  = len(set(labels[valid_mask]))
                    noise_pct   = (labels == -1).sum() / len(labels) * 100 if -1 in labels else 0.0

                    if noise_pct > MAX_NOISE_PCT:
                        continue
                    score = silhouette_score(X[valid_mask], labels[valid_mask]) \
                            if n_clusters > 1 and valid_mask.sum() > 1 else -1

                    if score > best_score:
                        best_score, best_params, best_labels = score, param_val, labels
                        best_n_clusters, best_noise_pct, best_seed = n_clusters, noise_pct, current_seed

                except Exception:
                    continue

        if best_labels is not None:
            try:
                mask = best_labels != -1
                db_score = davies_bouldin_score(X[mask], best_labels[mask])   if best_n_clusters > 1 else np.nan
                ch_score = calinski_harabasz_score(X[mask], best_labels[mask]) if best_n_clusters > 1 else np.nan
            except Exception:
                db_score = ch_score = np.nan

            results[alg_name] = {
                "best_score": best_score, "best_params": best_params, "best_labels": best_labels,
                "n_clusters": best_n_clusters, "noise_pct": best_noise_pct,
                "best_seed": best_seed, "db_score": db_score, "ch_score": ch_score
            }
            summary_rows.append({
                "Algorithm": alg_name, "Silhouette Score": best_score,
                "Davies-Bouldin Score": db_score, "Calinski-Harabasz Score": ch_score,
                "Best Params": best_params, "Clusters": best_n_clusters,
                "Noise %": best_noise_pct, "Best Seed": best_seed,
                "Unique Labels": np.unique(best_labels)
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="Silhouette Score", ascending=False)
    print("\n=== Resumen de clustering ===")
    print(summary_df.to_string(index=False))
    return results, summary_df


# =======================================================
# 7. GENERAR EMBEDDINGS PCA  ← reemplaza la sección UMAP
# =======================================================
X_input      = X_scaled
n_samples, n_features = X_input.shape
max_components = min(n_samples, n_features)

PCA_N_COMPONENTS_LIST = [2, 3, 5, 10, 15, 20, 50]
PCA_N_COMPONENTS_LIST = [c for c in PCA_N_COMPONENTS_LIST if c <= max_components]
PCA_WHITEN_LIST       = [False, True]
total_emb = len(PCA_N_COMPONENTS_LIST) * len(PCA_WHITEN_LIST)

print(f"\n🔸 Ejecutando PCA en X_scaled con grid de hiperparámetros:")
print(f"   Features de entrada : {n_features} métricas secundarias")
print(f"   Modelos             : {n_samples}")
print(f"   Componentes máx.    : {max_components}")
print(f"   Configuraciones     : {len(PCA_N_COMPONENTS_LIST)} n_components × "
      f"{len(PCA_WHITEN_LIST)} whiten = {total_emb} embeddings")

embedding_matrices = {}

for n_comp in PCA_N_COMPONENTS_LIST:
    for whiten in PCA_WHITEN_LIST:
        emb_name = f"PCA_C{n_comp}_W{int(whiten)}"
        try:
            pca_model = PCA(n_components=n_comp, whiten=whiten, random_state=RANDOM_SEED)
            embedding_matrices[emb_name] = pca_model.fit_transform(X_input)

            # Varianza explicada acumulada (informativo)
            var_exp = pca_model.explained_variance_ratio_.cumsum()[-1]
            print(f"   ✅ {emb_name} → varianza explicada acumulada: {var_exp:.3f}")
        except Exception as e:
            print(f"   ❌ Error en PCA {emb_name}: {e}")
            continue

print(f"\n✅ {len(embedding_matrices)} embeddings PCA generados.")

# ====================================
# 8. PIPELINE FINAL: CLUSTERING Y EXPORTACIÓN
# ====================================

df_ids = (df_aug[['sample']].copy().rename(columns={'sample': 'ModelName'})
          if 'sample' in df_aug.columns
          else df_aug[[id_cols_to_keep[0]]].copy().rename(columns={id_cols_to_keep[0]: 'ModelName'}))

df_clusters_master   = df_ids.copy()
all_clustering_results = []

for emb_name_base, X_emb in embedding_matrices.items():
    print(f"\n🔹 Clustering en Embedding: {emb_name_base}")
    results, summary_df = clustering_evaluation_table(X_emb, alg_classes, param_grids, seeds_list=SEEDS_TO_TEST)

    for index, row in summary_df.iterrows():
        if row['Algorithm'] not in results:
            continue

        seed_str     = f"_S{row['Best Seed']}" if row['Best Seed'] is not None else ""
        noise_pct_val = row['Noise %'] if pd.notna(row['Noise %']) else 0.0
        db_val        = row['Davies-Bouldin Score']    if pd.notna(row['Davies-Bouldin Score'])    else 0.0
        ch_val        = row['Calinski-Harabasz Score'] if pd.notna(row['Calinski-Harabasz Score']) else 0.0
        score_str     = f"S{row['Silhouette Score']:.2f}_DB{db_val:.2f}_CH{ch_val:.0f}"
        full_name     = f"{row['Algorithm']}_{emb_name_base}_K{row['Clusters']}_{score_str}{seed_str}"

        all_clustering_results.append({
            'Configuration': full_name, 'Algorithm': row['Algorithm'],
            'Embedding': emb_name_base, 'Silhouette Score': row['Silhouette Score'],
            'Davies-Bouldin Score': db_val, 'Calinski-Harabasz Score': ch_val,
            'Clusters': row['Clusters'], 'Noise_Pct': noise_pct_val,
            'Labels': results[row['Algorithm']]['best_labels'],
            'Embedding_Key': emb_name_base, 'Best Seed': row['Best Seed']
        })

        if row['Silhouette Score'] > 0:
            df_clusters_master[f"Cluster_{full_name}"] = results[row['Algorithm']]['best_labels']

output_filename_master = os.path.join(OUTPUT_DIR, "pacientes_clusterizados_todos_sinfiltro.csv")
df_clusters_master.to_csv(output_filename_master, index=False)
print(f"\n💾 CSV de clusters master guardado en: {output_filename_master}")

# ------------------------------------------------------------------------
# 9. FILTRADO Y EXPORTACIÓN DE LOS ALGORITMOS SELECCIONADOS
# ------------------------------------------------------------------------

df_all_results = pd.DataFrame(all_clustering_results)
SILHOUETTE_THRESHOLD = 0.1

df_selected_models = (df_all_results[df_all_results['Silhouette Score'] >= SILHOUETTE_THRESHOLD]
                      .sort_values(by=['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score'],
                                   ascending=[False, False, True])
                      .copy().reset_index(drop=True))

print(f"\n✨ Encontrados {len(df_selected_models)} modelos con Silhouette Score >= {SILHOUETTE_THRESHOLD:.1f}.")

if not df_selected_models.empty:
    df_selected_labels = df_ids.copy()
    for index, row in df_selected_models.iterrows():
        config_name, labels = row['Configuration'], row['Labels']
        if len(labels) == df_ids.shape[0]:
            df_selected_labels[config_name] = labels
        else:
            print(f"⚠️ Etiquetas de {config_name} ({len(labels)}) no coinciden con muestras ({df_ids.shape[0]}). Saltando.")

    output_filename_selected = os.path.join(OUTPUT_DIR, "pacientes_clusterizados_seleccion_sinfiltro2.csv")
    df_selected_labels.to_csv(output_filename_selected, index=False)
    print(f"\n💾 CSV de {len(df_selected_models)} modelos seleccionados guardado.")
else:
    print(f"\n⚠️ No se encontraron modelos con Silhouette Score >= {SILHOUETTE_THRESHOLD:.1f}.")

# ------------------------------------------------------------------------
# 10. GRAFICACIÓN DE LOS TOP N ALGORITMOS  ← adaptada para PCA
# ------------------------------------------------------------------------

df_top_to_plot = df_selected_models.head(5).copy()


def generate_pca_plots(df_top_models, embedding_matrices, output_dir):
    """
    Genera gráficas de scatter para los mejores modelos de clustering sobre embeddings PCA.
    - 2 componentes → scatter 2D
    - 3 componentes → scatter 3D
    - >3 componentes → matriz de pares (primeras 3 PCs)
    """
    if df_top_models.empty:
        return

    print(f"\n📈 Generando gráficas PCA para los {len(df_top_models)} modelos con mejor Score...")

    for i, row in df_top_models.reset_index(drop=True).iterrows():
        emb_key     = row['Embedding']
        labels      = row['Labels']
        config_name = row['Configuration']

        if emb_key not in embedding_matrices:
            print(f"   ❌ Matriz {emb_key} no encontrada. Saltando.")
            continue

        X_pca  = embedding_matrices[emb_key]
        n_dims = X_pca.shape[1]

        title = (f"TOP {i+1}: {config_name}\n"
                 f"Silhouette: {row['Silhouette Score']:.3f} | K={int(row['Clusters'])} | "
                 f"Ruido: {row['Noise_Pct']:.1f}%")

        safe_name = re.sub(r'[\\/*?:"<>|]', '_', config_name)
        filename  = os.path.join(output_dir, f"TOP_{i+1}_{safe_name}.png")

        df_plot = pd.DataFrame(X_pca[:, :min(n_dims, 3)],
                               columns=[f'PC{j+1}' for j in range(min(n_dims, 3))])
        df_plot['Cluster'] = labels.astype(str)

        cluster_list = sorted(df_plot['Cluster'].unique())
        non_noise    = [c for c in cluster_list if c != '-1']
        palette      = sns.color_palette("tab10", n_colors=max(len(non_noise), 1))
        color_map    = {c: palette[j] for j, c in enumerate(non_noise)}
        if '-1' in cluster_list:
            color_map['-1'] = 'gray'

        # ── 2D ──────────────────────────────────────────────────────────
        if n_dims == 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_plot,
                            palette=color_map, legend="full", alpha=0.7, s=50)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(title, fontsize=12)
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"   > Gráfica 2D guardada: {filename}")

        # ── 3D ──────────────────────────────────────────────────────────
        elif n_dims == 3:
            fig = plt.figure(figsize=(12, 10))
            ax  = fig.add_subplot(111, projection='3d')
            for cl in cluster_list:
                sub = df_plot[df_plot['Cluster'] == cl]
                ax.scatter(sub['PC1'], sub['PC2'], sub['PC3'],
                           label=f'Cluster {cl}',
                           color=color_map.get(cl, 'black'), alpha=0.7, s=50)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"   > Gráfica 3D guardada: {filename}")

        # ── >3D → matriz de pares con las primeras 3 PCs ─────────────────
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            pairs = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')]
            for ax, (px, py) in zip(axes, pairs):
                for cl in cluster_list:
                    sub = df_plot[df_plot['Cluster'] == cl]
                    ax.scatter(sub[px], sub[py],
                               label=f'Cluster {cl}',
                               color=color_map.get(cl, 'black'), alpha=0.7, s=30)
                ax.set_xlabel(px)
                ax.set_ylabel(py)
                ax.set_title(f'{px} vs {py}')
            axes[-1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.suptitle(title, fontsize=11)
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"   > Gráfica matriz de pares (PC1-3) guardada: {filename}")


generate_pca_plots(df_top_to_plot, embedding_matrices, OUTPUT_DIR)

# ========================================================================
# 11. GENERACIÓN DE BASE DE DATOS INTEGRADA
# ========================================================================

print("\n🔗 Generando archivo maestro integrado...")

sample_col = 'sample' if 'sample' in df_aug.columns else id_cols_to_keep[0]
df_clinica_final = df_aug.drop_duplicates(subset=[sample_col])

df_final_unificado = pd.merge(
    df_clusters_master, df_clinica_final,
    left_on='ModelName', right_on=sample_col, how='inner'
)

if sample_col in df_final_unificado.columns and sample_col != 'ModelName':
    df_final_unificado = df_final_unificado.drop(columns=[sample_col])

output_master_unificado = os.path.join(OUTPUT_DIR, "DATA_MASTER_CLUSTERS_Y_CLINICA.csv")
df_final_unificado.to_csv(output_master_unificado, index=False)

print(f"✅ ¡Éxito! Archivo maestro unificado guardado.")
print(f"📊 Dimensiones finales: {df_final_unificado.shape[0]} muestras × {df_final_unificado.shape[1]} columnas.")
print(f"📁 Ubicación: {output_master_unificado}")
print("\nPrimeras columnas del archivo unificado:")
print(df_final_unificado.columns[:10].tolist(), "... [etc]")