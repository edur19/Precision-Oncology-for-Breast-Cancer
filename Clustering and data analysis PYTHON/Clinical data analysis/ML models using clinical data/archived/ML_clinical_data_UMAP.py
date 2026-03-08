# ============================================================
# 🚀 Pipeline completo: Análisis de clustering Clínico/Demográfico (CON GRAFICACIÓN UMAP)
# ============================================================

# =========================
# 1. LIBRERÍAS Y CONFIGURACIÓN GLOBAL
# =========================
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import umap
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# LIBRERÍAS DE VISUALIZACIÓN AÑADIDAS
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
# ❗ AJUSTA ESTA RUTA A TU UBICACIÓN LOCAL
PATH_BASE = "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/Sistemas metabólicos/Proyecto_Tesis/Datos_actual/" 

# Definición de las claves
COL_SAMPLE_ID = 'sample'
COL_SAMPLE_TYPE = 'sample_type.samples'
OUTPUT_DIR = "Results_clustering_UMAP" # Cambiado el nombre del output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SEMILLAS A PROBAR
SEEDS_TO_TEST = [42, 123, 100, 2024]
RANDOM_SEED = SEEDS_TO_TEST[0]
np.random.seed(RANDOM_SEED)

# =======================================================
# 2. CARGA Y CREACIÓN DE LA BASE DE DATOS MAESTRA 🔑
# =======================================================

# 2.1 Carga robusta de archivos
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

df_clinical = load_data("TCGA-BRCA.clinical.tsv")
df_survival = load_data("TCGA-BRCA.survival.tsv.gz")
df_metadata_raw = load_data("MetaData.xlsx")
df_model_names = load_data("Model's_ids.txt")

print(f"Bases de datos cargadas: Clínica ({len(df_clinical)}), Supervivencia ({len(df_survival)}), Metadata ({len(df_metadata_raw)})")


# 🔑 FUNCIÓN DE EXTRACCIÓN DE ID AJUSTADA A 16 CARACTERES (TCGA-XX-XXXX-XXA)
def extract_sample_id(filename):
    match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2}[A-Z0-9]?)', filename)
    if match:
        return match.group(0)[:16]
    return filename.split('_')[0].strip()[:16]


lista_modelos_unicos = df_model_names.iloc[:, 0].dropna().astype(str).tolist()

model_sample_ids = [extract_sample_id(model) for model in lista_modelos_unicos]
df_modelos_base = pd.DataFrame({
    COL_SAMPLE_ID: model_sample_ids,
    'modelo_path_completo': lista_modelos_unicos,
})


# =======================================================
# 3. MERGE PRINCIPAL (FILTRO ESTRICTO DE METADATA ELIMINADO ❌)
# =======================================================

# --- 3.1 Procesar Metadata (YA NO SE USA PARA FILTRAR LA BASE) ---
if not df_metadata_raw.empty and 'hidden' in df_metadata_raw.columns:
    df_metadata_clean = df_metadata_raw.copy()
    
    # 🔑 ESTANDARIZAR CLAVE DE METADATA
    df_metadata_clean['temp_id'] = df_metadata_clean['hidden'].astype(str).str.replace(r'\.', '-', regex=True).str.slice(0, 16)
    df_metadata_clean.rename(columns={'temp_id': COL_SAMPLE_ID}, inplace=True)
    df_metadata_clean = df_metadata_clean.drop_duplicates(subset=[COL_SAMPLE_ID], keep='first')
    
    metadata_cols_to_keep = [
        'Menopausal Status', 'Cancer Type', 'ER', 'PR', 'HER2', 'Subtype', 'Genetic Ancestry'
    ]
    
    # ❌ ELIMINADO EL FILTRO: df_modelos_base ya no se reduce.

    print("\n✅ FILTRO DE METADATA ELIMINADO: Se usará la base de modelos COMPLETA. Metadata molecular se añadirá con LEFT JOIN.")
    
    df_metadata_clean = df_metadata_clean[[COL_SAMPLE_ID] + metadata_cols_to_keep]
else:
    print("❌ ERROR: Metadata no pudo ser cargada o no contiene la columna 'hidden'. Se usará la base completa.")
    df_metadata_clean = pd.DataFrame() 


# --- 3.2 Estandarización de IDs en bases externas (Mantenido) ---
if 'sample' in df_clinical.columns:
    df_clinical['sample'] = df_clinical['sample'].apply(lambda x: extract_sample_id(str(x)))
if 'sample' in df_survival.columns:
    df_survival['sample'] = df_survival['sample'].apply(lambda x: extract_sample_id(str(x)))


# --- 3.3 Merge con Datos Clínicos y Supervivencia (Left Merge sobre la base COMPLETA) ---
df_survival_clean = df_survival[['sample', 'OS.time', 'OS']].drop_duplicates(subset=['sample'], keep='first').rename(columns={'sample': COL_SAMPLE_ID})

df_merged_clinical = pd.merge(
    df_modelos_base,
    df_clinical.drop(columns=['id', 'case_id'], errors='ignore'),
    on=COL_SAMPLE_ID,
    how='left'
)

df_final = pd.merge(
    df_merged_clinical,
    df_survival_clean,
    on=COL_SAMPLE_ID,
    how='left'
)

# --- 3.4 Merge de Metadata (LEFT JOIN PURO) ---
if not df_metadata_clean.empty:
    cols_overlap = [col for col in metadata_cols_to_keep if col in df_final.columns]
    
    df_final = pd.merge(
        df_final.drop(columns=cols_overlap, errors='ignore'),
        df_metadata_clean,
        on=COL_SAMPLE_ID,
        how='left' 
    )

# --- 3.5 Inicialización para Clustering ---
df = df_final.copy()
df_filtered = df.copy()

print(f"Filas totales finales para Clustering (sin filtro de Metadata): {df.shape[0]}")
print(f"Conteo de Tipos de Muestra (Verificación):")
print(df[COL_SAMPLE_TYPE].value_counts(dropna=False))


# =========================================================
# 2.2 Conversión de Tipos (Reordenado aquí)
# =========================================================
for c in ['is_ffpe.samples', 'oct_embedded.samples']:
    if c in df_filtered.columns:
        df_filtered[c] = df_filtered[c].replace({True: 1, False: 0})

for c in ['age_at_diagnosis.diagnoses', 'days_to_birth.demographic']:
    if c in df_filtered.columns and (df_filtered[c] > 1000).any():
        df_filtered[c] = df_filtered[c] / 365.25

#=========================
#3. SELECCIÓN DE DESCRIPTORES CLÍNICOS (ACTUALIZADO CON METADATA)
#=========================
descriptores_iniciales = [
    'gender.demographic', 'age_at_index.demographic', 'days_to_birth.demographic',
    'race.demographic', 'ethnicity.demographic', 'vital_status.demographic',
    'disease_type', 'primary_site',
    'ajcc_pathologic_stage.diagnoses', 'ajcc_pathologic_t.diagnoses', 'ajcc_pathologic_n.diagnoses', 'ajcc_pathologic_m.diagnoses',
    'tumor_grade.diagnoses', 'morphology.diagnoses', 'primary_diagnosis.diagnoses',
    'prior_malignancy.diagnoses', 'progression_or_recurrence.diagnoses',
    'age_at_diagnosis.diagnoses', 'age_at_earliest_diagnosis_in_years.diagnoses.xena_derived',
    'year_of_diagnosis.diagnoses', 'icd_10_code.diagnoses', 'site_of_resection_or_biopsy.diagnoses',
    'last_known_disease_status.diagnoses', 'days_to_diagnosis.diagnoses', 'days_to_last_follow_up.diagnoses',
    'treatment_type.treatments.diagnoses', 'treatment_or_therapy.treatments.diagnoses', 'prior_treatment.diagnoses',
    'sample_type.samples', 'composition.samples', 'tissue_type.samples', 'tumor_descriptor.samples',
    'is_ffpe.samples', 'oct_embedded.samples', 'initial_weight.samples', 'specimen_type.samples',
    'alcohol_history.exposures', 'OS', 'OS.time'
]


# Nuevas variables de alta resolución
descriptores_alta_res = [
    'Menopausal Status', 'Cancer Type', 'ER', 'PR', 'HER2', 'Subtype', 'Genetic Ancestry'
]

descriptores_finales = descriptores_iniciales + descriptores_alta_res
final_cols = [c for c in descriptores_finales if c in df_filtered.columns]
if len(final_cols) == 0:
    cols_to_exclude = ['submitter_id', 'sample', 'modelo_path_completo']
    final_cols = [col for col in df_filtered.columns if col not in cols_to_exclude]

if not final_cols:
    raise ValueError("No se encontraron columnas de descriptores válidas en el DataFrame.")

df_aug = df_filtered[final_cols + ['submitter_id', 'sample', 'modelo_path_completo']].copy()

# =========================
# 4. INGENIERÍA DE FEATURES (ACTUALIZADO con variables moleculares)
# =========================

# 4.1) Long_Survivor (binaria)
surv_col = 'OS.time'
if surv_col in df_aug.columns:
    q75 = df_aug[surv_col].quantile(0.75)
    df_aug['Long_Survivor'] = (df_aug[surv_col] > q75).astype(int)
else:
    df_aug['Long_Survivor'] = 0

# 4.2) Age_Group
age_candidates = ['age_at_index.demographic', 'age_at_diagnosis.diagnoses', 'age_at_earliest_diagnosis_in_years.diagnoses.xena_derived']
age_src = next((c for c in age_candidates if c in df_aug.columns), None)

if age_src is not None:
    def age_group_row(a):
        if pd.isna(a): return 'Unknown'
        if a < 45: return 'Young'
        elif a <= 65: return 'Middle'
        else: return 'Old'
    df_aug['Age_Group'] = df_aug[age_src].apply(age_group_row)
else:
    df_aug['Age_Group'] = 'Unknown'

# 4.3) Survival_Label (0/1)
df_aug['Survival_Label'] = df_aug['OS'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x in ['dead','deceased','yes (dead)','1'] else 0 if x in ['alive', 'no (alive)', '0'] else np.nan)
df_aug['Survival_Label'] = df_aug['Survival_Label'].fillna(0).astype(int)

# 4.4) Prior_Treatment_Flag
def prior_treatment_flag(r):
    cols = ['prior_malignancy.diagnoses', 'prior_treatment.diagnoses', 'progression_or_recurrence.diagnoses']
    for c in cols:
        if c in df_aug.columns and pd.notna(r[c]):
            val = str(r[c]).lower().strip()
            if val in ['yes', 'true', 'had prior treatment', 'recurrence', 'progression']: return 1
    return 0
df_aug['Prior_Treatment_Flag'] = df_aug.apply(prior_treatment_flag, axis=1)

# 4.5) Metastasis_Flag
if all(c in df_aug.columns for c in ['ajcc_pathologic_m.diagnoses', 'ajcc_pathologic_n.diagnoses']):
    df_aug['Metastasis_Flag'] = df_aug.apply(
        lambda r: 1 if ('m1' in str(r['ajcc_pathologic_m.diagnoses']).lower() or
                         'n2' in str(r['ajcc_pathologic_n.diagnoses']).lower() or
                         'n3' in str(r['ajcc_pathologic_n.diagnoses']).lower()) else 0, axis=1)
else:
    df_aug['Metastasis_Flag'] = 0

# 4.6) Combinación de Receptores (Para alta correlación molecular)
receptor_cols = ['ER', 'PR', 'HER2']
if all(c in df_aug.columns for c in receptor_cols):
    
    def classify_molecular_subtype(row):
        er = str(row['ER']).lower().strip() if pd.notna(row['ER']) else 'na'
        pr = str(row['PR']).lower().strip() if pd.notna(row['PR']) else 'na'
        her2 = str(row['HER2']).lower().strip() if pd.notna(row['HER2']) else 'na'
        
        is_er_pos = er in ['positive', '+']
        is_pr_pos = pr in ['positive', '+']
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

# 5.1 Definición de columnas finales para ML
numeric_cols = df_aug.select_dtypes(include=['int64','float64','float32','int32', 'uint8']).columns.tolist()
categorical_cols = df_aug.select_dtypes(include=['object','category']).columns.tolist()

cols_to_remove_from_ml = ['submitter_id', 'sample', 'modelo_path_completo', 'OS', 'OS.time']
for c in cols_to_remove_from_ml:
    if c in numeric_cols: numeric_cols.remove(c)
    if c in categorical_cols: categorical_cols.remove(c)

label_cols_candidates = ['Molecular_Subtype', 'Age_Group', 'ER_PR_HER2_Combo', 'Subtype'] 

for col in label_cols_candidates:
    if col in categorical_cols:
        if col not in df_aug.select_dtypes(include=['object','category']).columns.tolist():
             continue
        df_aug[f"{col}_encoded"] = LabelEncoder().fit_transform(df_aug[col].fillna('Unknown').astype(str))
        categorical_cols.remove(col)


# 5.2 Imputación
cols_to_impute_zero = [c for c in numeric_cols if df_aug[c].isna().any()]
if len(cols_to_impute_zero) > 0:
    imputer_num = SimpleImputer(strategy='constant', fill_value=0)
    df_aug.loc[:, cols_to_impute_zero] = imputer_num.fit_transform(df_aug.loc[:, cols_to_impute_zero])

# Imputación de categóricas (Missing)
if len(categorical_cols) > 0:
    for col in categorical_cols:
        df_aug[col] = df_aug[col].fillna('Missing').astype(str)

# 5.3 ColumnTransformer (Escalado y OHE)
ohe_kwargs = {'handle_unknown': 'ignore', 'sparse_output': False}
transformers = []
if len(numeric_cols) > 0:
    transformers.append(('num', StandardScaler(), numeric_cols))
if len(categorical_cols) > 0:
    transformers.append(('cat', OneHotEncoder(**ohe_kwargs), categorical_cols))

if not transformers:
    raise ValueError("No hay features numéricas o categóricas válidas para el preprocesamiento.")

preprocessor = ColumnTransformer(transformers, remainder='drop')
X_scaled = preprocessor.fit_transform(df_aug)

final_columns = []
if 'num' in preprocessor.named_transformers_:
    final_columns.extend(numeric_cols)
if 'cat' in preprocessor.named_transformers_ and preprocessor.named_transformers_['cat'] is not None:
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    final_columns.extend(cat_feature_names)

final_columns = np.array(final_columns)
print(f"✅ ColumnTransformer completado: {len(final_columns)} features listos para ML.")

# =========================
# 6. EVALUACIÓN DE CLUSTERING (SETUP Y FUNCIÓN) 
# =========================

K_VALORES = range(2, 15)

param_grids = {
    "KMeans": {'n_clusters': K_VALORES},
    "Agglomerative": {'n_clusters': K_VALORES, 'linkage': ['ward', 'average', 'complete']},
    "Birch": {'n_clusters': K_VALORES},
    "GMM": {'n_components': K_VALORES},
    "BayesianGaussianMixture": {'n_components': K_VALORES},
    "DBSCAN": {'eps': [0.5, 1.0, 1.5], 'min_samples': [5, 10, 20]},
    "HDBSCAN": {'min_cluster_size': [5, 10, 20], 'min_samples': [None, 5, 10]} if HDBSCAN_AVAILABLE else {},
    "MeanShift": {'bandwidth': [None]},
    "AffinityPropagation": {'damping': [0.5, 0.9]},
}

alg_classes = {
    'KMeans': KMeans, 'Agglomerative': AgglomerativeClustering, 'Birch': Birch,
    'GMM': GaussianMixture, 'DBSCAN': DBSCAN, 'MeanShift': MeanShift,
    'AffinityPropagation': AffinityPropagation, 'BayesianGaussianMixture': BayesianGaussianMixture,
}

if HDBSCAN_AVAILABLE:
    alg_classes['HDBSCAN'] = hdbscan.HDBSCAN

alg_classes = {k: v for k, v in alg_classes.items() if v is not None}
param_grids = {k: v for k, v in param_grids.items() if v}


# FUNCIÓN DE EVALUACIÓN MODIFICADA PARA USAR MÚLTIPLES SEMILLAS Y FILTRO DE RUIDO
def clustering_evaluation_table(X, algorithms, param_grids, seeds_list):
    results = {}
    summary_rows = []
    
    MAX_NOISE_PCT = 10.0 # Usando 10% como umbral

    for alg_name, alg_class in algorithms.items():
        best_score = -1; best_params = None; best_labels = None
        best_n_clusters = 0; best_noise_pct = 0; best_seed = None

        current_grid_dict = param_grids.get(alg_name, [{}])

        if isinstance(current_grid_dict, dict) and any(isinstance(v, list) for v in current_grid_dict.values()):
            param_grid = ParameterGrid(current_grid_dict)
        else:
            param_grid = ParameterGrid([current_grid_dict])

        for current_seed in seeds_list:
            if alg_name not in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch'] and current_seed != seeds_list[0]:
                continue

            for param_val in param_grid:
                try:
                    if alg_name in ['GMM', 'KMeans', 'BayesianGaussianMixture', 'Birch']:
                        model = alg_class(**param_val, random_state=current_seed)
                    else:
                        model = alg_class(**param_val)
                    
                    if alg_name == 'MeanShift':
                        model.fit(X)
                        labels = model.labels_
                    else:
                        labels = model.fit_predict(X)

                    valid_mask = labels != -1
                    n_clusters = len(set(labels[valid_mask]))
                    noise_pct = (labels == -1).sum() / len(labels) * 100 if -1 in labels else 0.0

                    if noise_pct > MAX_NOISE_PCT:
                        continue 

                    if n_clusters > 1 and valid_mask.sum() > 1:
                        score = silhouette_score(X[valid_mask], labels[valid_mask])
                    else:
                        score = -1

                    if score > best_score:
                        best_score = score; best_params = param_val; best_labels = labels;
                        best_n_clusters = n_clusters; best_noise_pct = noise_pct; best_seed = current_seed

                except Exception as e:
                    continue
        
        if best_labels is not None:
            # Calcular Davies-Bouldin y Calinski-Harabasz solo para el mejor modelo encontrado
            try:
                if best_n_clusters > 1 and np.sum(best_labels != -1) > 1:
                    mask = best_labels != -1
                    db_score = davies_bouldin_score(X[mask], best_labels[mask])
                    ch_score = calinski_harabasz_score(X[mask], best_labels[mask])
                else:
                    db_score = np.nan
                    ch_score = np.nan
            except Exception:
                db_score = np.nan
                ch_score = np.nan

            results[alg_name] = {
                "best_score": best_score, "best_params": best_params, "best_labels": best_labels,
                "n_clusters": best_n_clusters, "noise_pct": best_noise_pct, "best_seed": best_seed,
                "db_score": db_score, "ch_score": ch_score
            }

            summary_rows.append({
                "Algorithm": alg_name, 
                "Silhouette Score": best_score, 
                "Davies-Bouldin Score": db_score,
                "Calinski-Harabasz Score": ch_score,
                "Best Params": best_params,
                "Clusters": best_n_clusters, 
                "Noise %": best_noise_pct, 
                "Best Seed": best_seed,
                "Unique Labels": np.unique(best_labels)
            })
        
    summary_df = pd.DataFrame(summary_rows).sort_values(by="Silhouette Score", ascending=False)
    print("\n=== Resumen de clustering ===")
    print(summary_df.to_string(index=False))

    return results, summary_df

# =======================================================
# 7. GENERAR EMBEDDINGS (SOLO UMAP DIRECTO, con múltiples semillas)
# =======================================================
embedding_results = {}
embedding_matrices = {}
X_input = X_scaled

# Definición de hiperparámetros de reducción
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

print(f"\n🔸 Ejecutando UMAP en X_scaled ({X_scaled.shape[1]} features) con {len(umap_param_grid)} configs y {len(SEEDS_TO_TEST)} semillas...")

# --- Bucle Externo de Semillas para UMAP ---
for current_seed in SEEDS_TO_TEST:
    np.random.seed(current_seed)
    for params in umap_param_grid:
        n_components = params['n_components']
        metric_name = params['metric']
        n_neighbors = params['n_neighbors']
        min_dist = params['min_dist']

        emb_name = f'UMAP_C{n_components}_NN{n_neighbors}_MD{min_dist}_M{metric_name}_S{current_seed}'

        try:
            umap_model = umap.UMAP(**params, random_state=current_seed)
            X_umap = umap_model.fit_transform(X_input)
            embedding_matrices[emb_name] = X_umap
            embedding_results[emb_name] = X_umap # Usamos solo el nombre del embedding como clave
        except Exception as e:
            print(f"Error al ejecutar UMAP con {emb_name}: {e}")
            continue

# ====================================
# 8. PIPELINE FINAL: CLUSTERING Y EXPORTACIÓN 
# ====================================

# Conservar IDs (clave del modelo)
df_ids = df_aug[['sample']].copy().rename(columns={'sample':'ModelName'})

df_clusters_master = df_ids.copy()
all_clustering_results = []

# Clustering sobre embeddings
for emb_name_base, X_emb in embedding_results.items():
    print(f"\n🔹 Clustering en Embedding: {emb_name_base}")

    results, summary_df = clustering_evaluation_table(X_emb, alg_classes, param_grids, seeds_list=SEEDS_TO_TEST)

    for index, row in summary_df.iterrows():
        if row['Algorithm'] not in results: continue

        seed_str = f"_S{row['Best Seed']}" if row['Best Seed'] is not None else ""
        noise_pct_val = row['Noise %'] if pd.notna(row['Noise %']) else 0.0
        score_str = f"S{row['Silhouette Score']:.2f}_DB{row['Davies-Bouldin Score']:.2f}_CH{row['Calinski-Harabasz Score']:.0f}"
        
        full_name = f"{row['Algorithm']}_{emb_name_base}_K{row['Clusters']}_{score_str}{seed_str}"

        all_clustering_results.append({
            'Configuration': full_name, 
            'Algorithm': row['Algorithm'], 
            'Embedding': emb_name_base,
            'Silhouette Score': row['Silhouette Score'], 
            'Davies-Bouldin Score': row['Davies-Bouldin Score'],
            'Calinski-Harabasz Score': row['Calinski-Harabasz Score'],
            'Clusters': row['Clusters'], 
            'Noise_Pct': noise_pct_val, 
            'Labels': results[row['Algorithm']]['best_labels'],
            'Embedding_Key': emb_name_base, 
            'Best Seed': row['Best Seed']
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

df_selected_models = df_all_results[
    (df_all_results['Silhouette Score'] >= SILHOUETTE_THRESHOLD)
].sort_values(
    by=['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score'],
    ascending=[False, False, True]
).copy().reset_index(drop=True)

num_selected_models = len(df_selected_models)
print(f"\n✨ Encontrados {num_selected_models} modelos con Silhouette Score >= {SILHOUETTE_THRESHOLD:.1f}.")

if not df_selected_models.empty:
    df_selected_labels = df_ids.copy()

    for index, row in df_selected_models.iterrows():
        config_name = row['Configuration']
        labels = row['Labels']

        if len(labels) == df_ids.shape[0]:
             df_selected_labels[config_name] = labels
        else:
             print(f"⚠️ Alerta: Etiquetas del cluster {config_name} ({len(labels)}) no coinciden con el número de muestras ({df_ids.shape[0]}). Saltando.")

    output_filename_selected = os.path.join(OUTPUT_DIR, "pacientes_clusterizados_seleccion_sinfiltro2.csv")
    df_selected_labels.to_csv(output_filename_selected, index=False)
    print(f"\n💾 CSV de los {num_selected_models} modelos seleccionados guardado en: {output_filename_selected}")
else:
    print(f"\n⚠️ No se encontraron modelos con Silhouette Score >= {SILHOUETTE_THRESHOLD:.1f} para generar el archivo de selección.")

# ------------------------------------------------------------------------
# 10. GRAFICACIÓN DE LOS TOP N ALGORITMOS (CORREGIDO)
# ------------------------------------------------------------------------

df_top_to_plot = df_selected_models.head(5).copy()

def generate_umap_plots(df_top_models, embedding_matrices, output_dir):
    if df_top_models.empty: return
    
    print(f"\n📈 Generando gráficas UMAP para los {len(df_top_models)} modelos con mejor Score...")

    for i, row in df_top_models.reset_index(drop=True).iterrows():
        emb_key = row['Embedding']
        labels = row['Labels']
        config_name = row['Configuration']

        if emb_key not in embedding_matrices: 
            print(f"Error: Matriz {emb_key} no encontrada para graficar.")
            continue

        X_umap = embedding_matrices[emb_key]
        n_dims = X_umap.shape[1]

        title = f"TOP {i+1}: {config_name}\nSilhouette: {row['Silhouette Score']:.3f} | K={int(row['Clusters'])} | Ruido: {row['Noise_Pct']:.1f}%"
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
            print(f"   > Gráfica 2D guardada: {filename}")


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
            print(f"   > Gráfica 3D guardada: {filename}")
        
        else:
            print(f"   > Advertencia: Omitiendo gráfica para {emb_key} (Dimensión {n_dims} > 3).")

generate_umap_plots(df_top_to_plot, embedding_matrices, OUTPUT_DIR)

# ========================================================================
# 11. GENERACIÓN DE BASE DE DATOS INTEGRADA (CLUSTERS + CLÍNICA + METADATA)
# ========================================================================

print("\n🔗 Generando archivo maestro integrado...")

# 1. Preparar la base clínica (usamos df_aug que ya tiene ingeniería de features)
# Quitamos duplicados por si acaso para asegurar un join 1:1 limpio
df_clinica_final = df_aug.drop_duplicates(subset=['sample'])

# 2. Realizar el merge con el dataframe de clusters master
# El dataframe 'df_clusters_master' tiene 'ModelName' como ID (que es el sample ID)
df_final_unificado = pd.merge(
    df_clusters_master, 
    df_clinica_final, 
    left_on='ModelName', 
    right_on='sample', 
    how='inner'
)

# 3. Limpieza de columnas redundantes post-merge
if 'sample' in df_final_unificado.columns:
    df_final_unificado = df_final_unificado.drop(columns=['sample'])

# 4. Guardar el archivo final
output_master_unificado = os.path.join(OUTPUT_DIR, "DATA_MASTER_CLUSTERS_Y_CLINICA.csv")
df_final_unificado.to_csv(output_master_unificado, index=False)

print(f"✅ ¡Éxito! Archivo maestro unificado guardado.")
print(f"📊 Dimensiones finales: {df_final_unificado.shape[0]} muestras x {df_final_unificado.shape[1]} columnas.")
print(f"📁 Ubicación: {output_master_unificado}")

# Opcional: Mostrar las primeras columnas para verificar
print("\nPrimeras columnas del archivo unificado:")
print(df_final_unificado.columns[:10].tolist(), "... [etc]")