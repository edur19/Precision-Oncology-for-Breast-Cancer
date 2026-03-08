
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 
from collections import defaultdict 
import sys 

# ============================================================
# 1️⃣ CONFIGURACIÓN Y CARGA DE DATOS 🔑
# ============================================================

ID_COLUMN = 'ModelName'
MIN_VALID_SAMPLES = 50 

# Rutas de tus archivos

path1 = "/Users/eduardoruiz/Documents/GitHub/Precision-Oncology-for-Breast-Cancer-Diagnosis/src/Clustering and data analysis PYTHON/Clinical data analysis/ML models using clinical data/Results_clustering_UMAP_seleccion_variables/pacientes_clusterizados_todos_sinfiltro.csv"
path2 = "/Users/eduardoruiz/Documents/GitHub/Precision-Oncology-for-Breast-Cancer-Diagnosis/src/Clustering and data analysis PYTHON/Metabolic data analysis/ML models using metabolic data/resultados_normas_pareto_UMAP_v2/PatientClusters_Selected_v2.csv"

def load_and_standardize_clusters(file_path, suffix, id_col='ModelName'):
    """Carga, limpia IDs de TCGA y renombra las columnas de cluster."""
    try:
        df = pd.read_csv(file_path, sep=None, engine='python') 
    except Exception as e:
        raise ValueError(f"Fallo al leer {file_path}. Error: {e}")

    df.columns = df.columns.str.strip()
    if id_col not in df.columns:
         raise ValueError(f"La columna ID '{id_col}' no se encontró en: {file_path}")

    # Limpieza de IDs de TCGA (mantener formato consistente)
    df[id_col] = df[id_col].astype(str).str.split('_').str[0].str.split('.').str[0].str.slice(0, 16)
    
    cluster_cols = [col for col in df.columns if col != id_col]
    rename_mapping = {col: f"{col}_{suffix}" for col in cluster_cols}
    df = df.rename(columns=rename_mapping)
    
    return df[[id_col] + list(rename_mapping.values())].copy()

# Cargar y fusionar
try:
    df1 = load_and_standardize_clusters(path1, suffix='C', id_col=ID_COLUMN)
    df2 = load_and_standardize_clusters(path2, suffix='M', id_col=ID_COLUMN)
    df_merged = df1.merge(df2, on=ID_COLUMN, how='inner')
    print(f"✅ Fusión exitosa. Pacientes comunes: {len(df_merged)}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit()

# Identificar columnas
cluster_cols_clinico = [col for col in df_merged.columns if col.endswith('_C')]
cluster_cols_metabolitos = [col for col in df_merged.columns if col.endswith('_M')]

# =================================================================
# 2️⃣ CÁLCULO DE MÉTRICAS (ARI/AMI) 📊
# =================================================================

def calculate_metrics_fixed(df, col1, col2):
    s1 = df[col1].replace(-1, np.nan)
    s2 = df[col2].replace(-1, np.nan)
    valid_idx = s1.dropna().index.intersection(s2.dropna().index)

    if len(valid_idx) < MIN_VALID_SAMPLES: return None

    labels1 = s1.loc[valid_idx].astype(int)
    labels2 = s2.loc[valid_idx].astype(int)

    if labels1.nunique() < 2 or labels2.nunique() < 2: return None

    return {
        'Clinico_Cluster': col1,
        'Metabolico_Cluster': col2,
        'ARI': adjusted_rand_score(labels1, labels2),
        'AMI': adjusted_mutual_info_score(labels1, labels2),
        'N_Muestras': len(valid_idx)
    }

results_list = []
for c_col in cluster_cols_clinico:
    for m_col in cluster_cols_metabolitos:
        res = calculate_metrics_fixed(df_merged, c_col, m_col)
        if res: results_list.append(res)

df_results = pd.DataFrame(results_list)
df_top10 = df_results.sort_values(by='ARI', ascending=False).head(10)

# =================================================================
# 3️⃣ VISUALIZACIÓN DEL TOP 10 📈
# =================================================================

pivot_top10 = df_top10.pivot_table(index='Clinico_Cluster', columns='Metabolico_Cluster', values='ARI')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_top10, annot=True, fmt=".4f", cmap='magma')
plt.title('Top 10 Correlaciones: Clínico vs Metabólico (ARI)', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("heatmap_top10_oncologia_paretoynormas.png")
plt.show()

# =================================================================
# 4️⃣ IDENTIFICACIÓN DEL SUBGRUPO DE PACIENTES 🕵️‍♂️
# =================================================================

# 1. Obtener el mejor par de modelos
best_pair = df_top10.iloc[0]
best_c_col = best_pair['Clinico_Cluster']
best_m_col = best_pair['Metabolico_Cluster']

# 2. Filtrar pacientes sin ruido en estos dos modelos
df_best = df_merged[[ID_COLUMN, best_c_col, best_m_col]].copy()
df_best = df_best[(df_best[best_c_col] != -1) & (df_best[best_m_col] != -1)]

# 3. Matriz de Contingencia (Confusion Matrix)
contingency_matrix = pd.crosstab(df_best[best_c_col], df_best[best_m_col])

#[Image of a confusion matrix heatmap]

plt.figure(figsize=(8, 6))
sns.heatmap(contingency_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title(f'Distribución de Pacientes:\n{best_c_col} vs {best_m_col}')
plt.xlabel('Clusters Metabólicos (M)')
plt.ylabel('Clusters Clínicos (C)')
plt.show()

# 4. Extraer el "Core" (La celda con más pacientes)
stacked = contingency_matrix.stack()
c_best, m_best = stacked.idxmax()
n_patients = stacked.max()

print(f"\n💡 RESULTADO DEL ANÁLISIS:")
print(f"El mayor acuerdo ocurre entre el Cluster Clínico '{c_best}' y el Metabólico '{m_best}'.")
print(f"Este grupo contiene {n_patients} pacientes de los {len(df_merged)} totales.")

# 5. Exportar IDs
pacientes_core = df_best[(df_best[best_c_col] == c_best) & (df_best[best_m_col] == m_best)]
pacientes_core[[ID_COLUMN]].to_csv("pacientes_core_correlacion_Paretoynormas.csv", index=False)

print(f"✅ Archivo 'pacientes_core_correlacion.csv' generado.")



# =================================================================
# 6️⃣ ANÁLISIS DE DIVERGENCIA (LOS 211 PACIENTES "REBELDES") 🚀
# =================================================================

# 1. Crear una columna de Categoría para todos los pacientes
def categorizar_paciente(row):
    if row[best_c_col] == c_best and row[best_m_col] == m_best:
        return 'Core (Concordantes)'
    elif row[best_c_col] == c_best and row[best_m_col] != m_best:
        return 'Divergencia Metabólica' # Misma clínica, distinto metabolismo
    elif row[best_c_col] != c_best and row[best_m_col] == m_best:
        return 'Divergencia Clínica'    # Mismo metabolismo, distinta clínica
    else:
        return 'Discrepancia Total'      # Distintos en ambos

df_best['Categoria_Analisis'] = df_best.apply(categorizar_paciente, axis=1)

# 2. Resumen estadístico de los grupos
resumen_grupos = df_best['Categoria_Analisis'].value_counts()
print("\n📊 DISTRIBUCIÓN DE LA COHORTE:")
print(resumen_grupos)

# 3. Visualización de la composición de la cohorte
plt.figure(figsize=(10, 6))
colors = sns.color_palette("pastel")[0:4]
resumen_grupos.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.1, 0.1, 0.1, 0.1))
plt.title('Composición de la Cohorte: Concordancia vs Divergencia')
plt.ylabel('')
plt.savefig("composicion_grupos_tesis.png")
plt.show()

# 4. Guardar los pacientes "Especiales" (Divergentes)
# Estos son los que podrían tener una respuesta al tratamiento distinta a la esperada
pacientes_divergentes = df_best[df_best['Categoria_Analisis'] != 'Core (Concordantes)']
pacientes_divergentes.to_csv("pacientes_divergentes_para_estudio_pyn.csv", index=False)

print(f"\n✅ Se han identificado {len(pacientes_divergentes)} pacientes divergentes.")
print("Archivo 'pacientes_divergentes_para_estudio.csv' listo para análisis clínico profundo.")

# =================================================================
# 7️⃣ VISUALIZACIÓN 3D DE CONVERGENCIA ENTRE LOS DOS MEJORES ALGORITMOS
# ========
# =================================================================
# 7️⃣ VISUALIZACIÓN 3D DE CONVERGENCIA ENTRE LOS DOS MEJORES ALGORITMOS
# =================================================================
# =================================================================
# 7️⃣ VISUALIZACIÓN 3D: RELACIÓN ENTRE LOS DOS ALGORITMOS GANADORES
# =================================================================

from mpl_toolkits.mplot3d import Axes3D

print("\n🚀 Visualización 3D de relación entre algoritmos ganadores...")

df_algo = df_best[[best_c_col, best_m_col]].copy()

# Convertir a enteros
df_algo[best_c_col] = df_algo[best_c_col].astype(int)
df_algo[best_m_col] = df_algo[best_m_col].astype(int)

# Contar pacientes por combinación
counts = df_algo.groupby([best_c_col, best_m_col]).size().reset_index(name='Count')

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    counts[best_c_col],
    counts[best_m_col],
    counts['Count'],
    s=counts['Count']*2,
    alpha=0.8
)

ax.set_xlabel(f'Cluster Clínico ({best_c_col})')
ax.set_ylabel(f'Cluster Metabólico ({best_m_col})')
ax.set_zlabel('Número de Pacientes')

ax.set_title('Distribución 3D de Pacientes entre Algoritmos')

plt.tight_layout()
plt.savefig("relacion_3D_algoritmos_ganadores.png")
plt.show()

print("✅ Figura guardada: relacion_3D_algoritmos_ganadores.png")
# =================================================================
# 7️⃣ VISUALIZACIÓN 3D – ALGORITMO CLÍNICO (COLORES POR CLUSTER)
# =================================================================

print("\n🚀 Visualizando algoritmo clínico...")

df_clin = df_best[[ID_COLUMN, best_c_col]].copy()
df_clin = df_clin.sort_values(by=best_c_col).reset_index(drop=True)

df_clin['X'] = np.arange(len(df_clin))
df_clin['Y'] = df_clin[best_c_col].astype(int)
df_clin['Z'] = np.random.normal(0, 0.2, len(df_clin))

clusters_clin = np.sort(df_clin['Y'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(clusters_clin)))

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

for c, color in zip(clusters_clin, colors):
    subset = df_clin[df_clin['Y'] == c]
    ax.scatter(
        subset['X'], subset['Y'], subset['Z'],
        label=f'Cluster {c}',
        color=color,
        alpha=0.85,
        s=45
    )

ax.set_title(
    f"Algoritmo Clínico\nDistribución 3D de Clusters\n({best_c_col})",
    fontsize=14,
    fontweight='bold'
)

ax.set_xlabel('Pacientes (ordenados)')
ax.set_ylabel('Cluster Clínico')
ax.set_zlabel('Jitter (visualización)')
ax.legend(title="Clusters", loc='best')

plt.tight_layout()
plt.savefig("clusters_3D_clinico.png", dpi=300)
plt.show()

# =================================================================
# 8️⃣ VISUALIZACIÓN 3D – ALGORITMO METABÓLICO (COLORES POR CLUSTER)
# =================================================================

print("\n🚀 Visualizando algoritmo metabólico...")

df_met = df_best[[ID_COLUMN, best_m_col]].copy()
df_met = df_met.sort_values(by=best_m_col).reset_index(drop=True)

df_met['X'] = np.arange(len(df_met))
df_met['Y'] = df_met[best_m_col].astype(int)
df_met['Z'] = np.random.normal(0, 0.2, len(df_met))

clusters_met = np.sort(df_met['Y'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(clusters_met)))

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

for c, color in zip(clusters_met, colors):
    subset = df_met[df_met['Y'] == c]
    ax.scatter(
        subset['X'], subset['Y'], subset['Z'],
        label=f'Cluster {c}',
        color=color,
        alpha=0.85,
        s=45
    )

ax.set_title(
    f"Algoritmo Metabólico\nDistribución 3D de Clusters\n({best_m_col})",
    fontsize=14,
    fontweight='bold'
)

ax.set_xlabel('Pacientes (ordenados)')
ax.set_ylabel('Cluster Metabólico')
ax.set_zlabel('Jitter (visualización)')
ax.legend(title="Clusters", loc='best')

plt.tight_layout()
plt.savefig("clusters_3D_metabolico.png", dpi=300)
plt.show()

print("✅ Figuras generadas con títulos y leyendas completas.")
