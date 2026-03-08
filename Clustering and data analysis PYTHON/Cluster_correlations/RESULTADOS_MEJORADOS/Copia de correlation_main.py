
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

path1 = "/Users/eduardoruiz/Documents/GitHub/Precision-Oncology-for-Breast-Cancer-Diagnosis/src/Clustering and data analysis PYTHON/Clinical data analysis/ML models using clinical data/Results_clustering_UMAP_seleccion_reducida_conmerge-DATOSNUEVOS/pacientes_clusterizados_todos_sinfiltro.csv"
path2 = "/Users/eduardoruiz/Documents/GitHub/Precision-Oncology-for-Breast-Cancer-Diagnosis/src/Clustering and data analysis PYTHON/Metabolic data analysis/ML models using metabolic data/resultados_TumorPhenotype_PCA_metrics_actualizado_sinl2/PatientClusters_TumorPhenotype_PCA.csv"

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
# 3️⃣ VISUALIZACIÓN DEL TOP 10 (VERSIÓN MEJORADA) 📈
# =================================================================

# Guardar el mejor par ANTES de reordenar
best_pair_row = df_results.sort_values(by='ARI', ascending=False).iloc[0]
best_c_col = best_pair_row['Clinico_Cluster']
best_m_col = best_pair_row['Metabolico_Cluster']

df_top10 = df_results.sort_values(by='ARI', ascending=False).head(10).copy()

import re

def short_clinico(name):
    """
    Input:  Cluster_KMeans_UMAP_C3_NN10_MD0.05_Mmanhattan_S123_K2_S0.80_DB0.15_CH948_S123_C
    Output: KMeans | UMAP C3 NN10 MD0.05 | K2 S0.80 DB0.15 CH948
    """
    m = re.search(
        r'Cluster_(\w+)_UMAP_(C\d+)_NN(\d+)_MD([\d.]+)_\w+_(S\d+)_K(\d+)_(S[\d.]+)_DB([\d.]+)_CH(\d+)',
        name
    )
    if m:
        algo, cx, nn, md, seed, k, s, db, ch = m.groups()
        return f"{algo}  |  UMAP {cx} NN{nn} MD{md} {seed}  |  K{k} {s} DB{db} CH{ch}"
    return name.replace('_C', '').replace('_', ' ')[:60]

def short_metabolico(name):
    """
    Input:  Cluster_PCA_C05_W1_DBSCAN_K2_S0.92_DB0.07_CH1139_Seed42_M
    Output: PCA C05 W1 | DBSCAN K2 S0.92 DB0.07 CH1139 Seed42
    """
    m = re.search(
        r'Cluster_(\w+)_(C\d+)_(W\d+)_(\w+)_K(\d+)_(S[\d.]+)_DB([\d.]+)_CH(\d+)_Seed(\d+)',
        name
    )
    if m:
        algo, comp, w, method, k, s, db, ch, seed = m.groups()
        return f"{algo} {comp} {w}  |  {method} K{k} {s} DB{db} CH{ch} Seed{seed}"
    return name.replace('_M', '').replace('_', ' ')[:60]

df_top10['Clinico_Short']    = df_top10['Clinico_Cluster'].apply(short_clinico)
df_top10['Metabolico_Short'] = df_top10['Metabolico_Cluster'].apply(short_metabolico)

# El metabólico es siempre el mismo → va en el xlabel
metabolico_label = df_top10['Metabolico_Short'].iloc[0]

# Ordenar ascendente para que el mejor quede arriba
df_top10 = df_top10.sort_values(by="ARI", ascending=True)

heatmap_data = df_top10[['ARI']].values

fig, ax = plt.subplots(figsize=(13, 9))

sns.heatmap(
    heatmap_data,
    annot=df_top10['ARI'].values.reshape(-1, 1),
    fmt=".3f",
    cmap="magma",
    cbar_kws={'label': 'Adjusted Rand Index', 'shrink': 0.6},
    yticklabels=df_top10['Clinico_Short'],
    xticklabels=[f"{metabolico_label}"],
    linewidths=0.6,
    linecolor="white",
    annot_kws={"size": 14, "weight": "bold"},
    ax=ax
)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=0)

ax.set_title(
    "Top 10 Concordancias entre Clustering Clínico y Metabólico",
    fontsize=15, fontweight='bold', pad=15
)
ax.set_ylabel("Algoritmo Clínico", fontsize=13, fontweight='bold')
ax.set_xlabel("Algoritmo Metabólico", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig("heatmap_top10_oncologia_paretoynormas.png", dpi=300, bbox_inches="tight")
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
plt.title(f'Distribución de Pacientes:')
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
plt.figure(figsize=(11,7))

labels = resumen_grupos.index
values = resumen_grupos.values

colors = sns.color_palette("pastel", len(values))

wedges, texts, autotexts = plt.pie(
    values,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=(0.05,0.15,0.15,0.10),
    pctdistance=0.7
)

# mover etiquetas fuera con líneas
for i, w in enumerate(wedges):
    ang = (w.theta2 + w.theta1) / 2
    x = np.cos(np.deg2rad(ang))
    y = np.sin(np.deg2rad(ang))

    # posiciones manuales para evitar superposición
    if i == 0:  # Core
        xytext = (1.45, -0.9)
    elif i == 1:  # Divergencia Clínica
        xytext = (-1.55, 0.9)
    elif i == 2:  # Divergencia Metabólica
        xytext = (-1.55, 0.75)
    else:  # Discrepancia Total
        xytext = (-1.55, 0.6)

    plt.annotate(
        labels[i],
        xy=(x, y),
        xytext=xytext,
        arrowprops=dict(arrowstyle="-", lw=1.2),
        ha="left" if xytext[0] > 0 else "right",
        va="center",
        fontsize=13
    )

plt.title("Composición de la Cohorte: Concordancia vs Divergencia", fontsize=16)
plt.tight_layout()
plt.savefig("composicion_grupos_tesis.png", dpi=300)
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

ax.set_xlabel('Pacientes ', fontweight='normal')

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

ax.set_xlabel('Pacientes ', fontweight='normal')
ax.legend(title="Clusters", loc='best')

plt.tight_layout()
plt.savefig("clusters_3D_metabolico.png", dpi=300)
plt.show()

print("✅ Figuras generadas con títulos y leyendas completas.")
