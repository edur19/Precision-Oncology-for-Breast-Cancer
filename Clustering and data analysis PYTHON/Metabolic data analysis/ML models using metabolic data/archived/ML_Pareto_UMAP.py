"""
Pareto Surface Clustering Pipeline
TCGA-BRCA Metabolic Flux Analysis
Refactored & Fixed Version
"""

import re
import os
import logging
import warnings
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, Birch,
    DBSCAN, MeanShift, AffinityPropagation,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import ParameterGrid
from scipy.stats.mstats import winsorize
from scipy.stats import skew
from scipy.spatial import ConvexHull
import umap

warnings.filterwarnings("ignore")

# ── HDBSCAN (optional) ───────────────────────────────────────────────────────
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    hdbscan = None
    HDBSCAN_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 0. CONFIGURATION
# ============================================================
@dataclass
class Config:
    path_pareto: str = (
        "/Users/eduardoruiz/Documents/GitHub/"
        "Precision-Oncology-for-Breast-Cancer-Diagnosis/src/"
        "Metabolic fluxes calculation MATLAB/ParetoSurfacenew.csv"
    )
    path_clinical: str = (
        "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/"
        "Sistemas metabólicos/Proyecto_Tesis/Datos_actual/TCGA-BRCA.clinical.tsv"
    )
    path_survival: str = (
        "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/"
        "Sistemas metabólicos/Proyecto_Tesis/Datos_actual/TCGA-BRCA.survival.tsv.gz"
    )
    path_metadata: str = (
        "/Users/eduardoruiz/Documents/MCBCI/MCBCI2/"
        "Sistemas metabólicos/Proyecto_Tesis/Datos_actual/MetaData.xlsx"
    )
    out_dir: str = "resultados_de_pareto_UMAP"
    seeds: tuple = (42)
    silhouette_threshold: float = 0.1
    max_noise_pct: float = 5.0
    k_range: range = field(default_factory=lambda: range(2, 15))
    suffix: str = "_Xomics_specificModel.mat"


# ============================================================
# 1. UTILITIES
# ============================================================
def get_units(col: str) -> str:
    """Return measurement units for a given Pareto column name."""
    patterns = {
        r"^SA_":                          "SA flux",
        r"WarburgIndex":                  "Warburg",
        r"NitrogenAna":                   "flux",
        r"GrowthMetab":                   "flux",
        r"ATPConsumption|ATPProduction":  "mmol/gDW·h",
        r"RatioCU_EA":                    "ratio",
        r"TotalTurnOver":                 "flux",
        r"MeanSaturation":                "[0-1]",
        r"GrowthEfficiency":              "[0-1]",
        r"MeanFlowCentrality":            "centrality",
        r"TotalReversibleFlux":           "flux",
        r"RedoxIndex":                    "flux",
        r"MFI":                           "count",
        r"AnabolismScore":                "ratio",
        r"Oncomet":                       "mmol/gDW·h",
        r"NADPHdemand":                   "flux/biomass",
        r"TCA_completeness":              "[0-1]",
        r"Lipid":                         "flux",
        r"GlnDependence":                 "[0-1]",
        r"CU_real|CU_range":              "flux units",
        r"EA_real|EA_range":              "flux units",
        r"Biomass":                       "gDW/h",
        r"^ATP_":                         "mmol/gDW·h",
        r"Hypervolume":                   "area",
        r"Pareto_length":                 "eu",
        r"slope":                         "ΔEA/ΔCU",
        r"Spacing_CV":                    "CV",
        r"convexity":                     "flux",
        r"N_soluciones":                  "n",
    }
    for pattern, unit in patterns.items():
        if re.search(pattern, col):
            return unit
    return "unknown"


def extract_model_id(model_name: str) -> str:
    """Extract 16-char TCGA model ID from a model name string."""
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2}[A-Z0-9]?)", model_name)
    if match:
        return match.group(0)[:16]
    return model_name.split("_")[0].strip()[:16]


def n_clusters_from_labels(labels: np.ndarray) -> int:
    return len(set(labels)) - (1 if -1 in labels else 0)


def compute_noise_pct(labels: np.ndarray) -> float:
    return float(np.mean(labels == -1) * 100)


# ============================================================
# 2. DATA LOADING
# ============================================================
def load_pareto(path: str) -> tuple[pd.DataFrame, bool]:
    """Load Pareto CSV. Returns (DataFrame, success_flag)."""
    try:
        df = pd.read_csv(path)
        df["Model_ID_16"] = df["ModelName"].apply(extract_model_id)
        logger.info(f"✅ Pareto cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df, True
    except FileNotFoundError:
        logger.error(f"❌ Archivo Pareto no encontrado: {path}")
        return pd.DataFrame(), False


def load_clinical(path: str) -> pd.DataFrame:
    """Load clinical or survival TSV (supports .gz). Returns DataFrame with Model column."""
    try:
        compression = "gzip" if path.endswith(".gz") else None
        df = pd.read_csv(path, sep="\t", compression=compression)
        if "sample" in df.columns:
            df["Model"] = df["sample"].apply(extract_model_id)
        else:
            logger.warning(f"⚠️ Columna 'sample' no encontrada en {path}. Retornando vacío.")
            return pd.DataFrame({"Model": []})
        logger.info(f"✅ Datos clínicos cargados desde {os.path.basename(path)}: {df.shape}")
        return df
    except FileNotFoundError:
        logger.warning(f"⚠️ Archivo no encontrado: {path}")
        return pd.DataFrame({"Model": []})


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def compute_pareto_extra_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute extra Pareto geometry metrics per model group."""
    metrics = {}
    if "CU_real" in group.columns and "EA_real" in group.columns:
        cu = group["CU_real"].dropna().values
        ea = group["EA_real"].dropna().values
        if len(cu) > 1:
            metrics["Pareto_CU_range"]    = float(cu.max() - cu.min())
            metrics["Pareto_EA_range"]    = float(ea.max() - ea.min())
            metrics["Pareto_n_solutions"] = len(group)
            metrics["Pareto_CU_skew"]     = float(skew(cu))
            metrics["Pareto_EA_skew"]     = float(skew(ea))
    return pd.Series(metrics)


def build_feature_matrix(df_pareto: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate Pareto metrics per Model_ID_16.
    Returns (merged_features_df, feature_column_names).
    """
    id_cols = {"ModelName", "Model_ID_16"}
    sa_cols = [c for c in df_pareto.columns if c.startswith("SA_")]
    metrics_cols = [
        c for c in df_pareto.columns
        if c not in id_cols and not c.startswith("SA_")
    ]

    # Build aggregation dictionary
    agg_dict: dict = {}
    for col in metrics_cols:
        if col in df_pareto.columns:
            agg_dict[col] = ["mean", "std", "median"]
    for col in sa_cols:
        if col in df_pareto.columns:
            agg_dict[col] = ["mean", "std"]

    df_summary = df_pareto.groupby("Model_ID_16").agg(agg_dict)
    df_summary.columns = [f"Pareto_{var}_{stat}" for var, stat in df_summary.columns]
    df_summary = df_summary.reset_index().rename(columns={"Model_ID_16": "Model"})

    # Extra geometric metrics
    df_extra = (
        df_pareto
        .groupby("Model_ID_16")
        .apply(compute_pareto_extra_metrics)
        .reset_index()
        .rename(columns={"Model_ID_16": "Model"})
    )

    df_merged = pd.merge(df_summary, df_extra, on="Model", how="left")
    feature_cols = [c for c in df_merged.columns if c != "Model"]
    logger.info(f"✅ Features de Pareto agregados: {len(feature_cols)} columnas.")
    return df_merged, feature_cols


# ============================================================
# 4. PREPROCESSING
# ============================================================
def preprocess_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Drop zero/constant columns, impute, winsorize, and scale.
    Returns (X_scaled, cleaned_df, cleaned_feature_cols).
    """
    df_feat = df[feature_cols].copy()

    # Drop zero / zero-variance columns
    zero_cols  = df_feat.columns[(df_feat == 0).all()].tolist()
    const_cols = df_feat.columns[df_feat.nunique(dropna=False) <= 1].tolist()
    cols_to_drop = sorted(set(zero_cols + const_cols))
    if cols_to_drop:
        logger.info(f"🧹 Eliminando {len(cols_to_drop)} columnas sin información.")
        df_feat = df_feat.drop(columns=cols_to_drop)

    clean_cols = df_feat.columns.tolist()
    df_clean   = df.drop(columns=cols_to_drop, errors="ignore")

    # Impute → winsorize → scale
    X = df_feat.values
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = np.array([winsorize(col, limits=[0.01, 0.01]) for col in X.T]).T
    X = RobustScaler().fit_transform(X)

    logger.info(f"✅ Preprocesamiento completo. X_scaled shape: {X.shape}")
    return X, df_clean, clean_cols


# ============================================================
# 5. UMAP REDUCTION
# ============================================================
def run_umap_grid(X: np.ndarray, seeds: tuple) -> dict[str, np.ndarray]:
    """
    Run UMAP over a full parameter grid.
    Returns dict mapping embedding name → 2D/3D array.
    """
    param_grid = [
        {"n_components": nc, "n_neighbors": nn, "min_dist": md, "metric": me}
        for nc in [2, 3]
        for nn in [10, 30, 50]
        for md in [0.05, 0.3]
        for me in ["euclidean", "manhattan", "cosine"]
    ]

    embeddings: dict[str, np.ndarray] = {}
    total = len(param_grid) * len(seeds)
    logger.info(
        f"🔸 Ejecutando UMAP ({X.shape[1]} features) — "
        f"{len(param_grid)} configuraciones × {len(seeds)} semillas = {total} runs."
    )

    for seed in seeds:
        np.random.seed(seed)
        for params in param_grid:
            name = (
                f"UMAP_C{params['n_components']}"
                f"_NN{params['n_neighbors']}"
                f"_MD{params['min_dist']}"
                f"_M{params['metric']}"
                f"_S{seed}"
            )
            try:
                X_emb = umap.UMAP(**params, random_state=seed).fit_transform(X)
                embeddings[name] = X_emb
            except Exception as exc:
                logger.warning(f"⚠️ UMAP falló para {name}: {exc}")

    logger.info(f"✅ Embeddings generados: {len(embeddings)}")
    return embeddings


# ============================================================
# 6. CLUSTERING
# ============================================================
def build_algorithm_registry(X_scaled: np.ndarray) -> tuple[dict, dict]:
    """
    Build algorithm class map and parameter grids.
    Returns (alg_classes, param_grids).
    """
    k_range = range(2, 15)

    alg_classes = {
        "KMeans":                  KMeans,
        "Agglomerative":           AgglomerativeClustering,
        "Birch":                   Birch,
        "GMM":                     GaussianMixture,
        "BayesianGaussianMixture": BayesianGaussianMixture,
        "DBSCAN":                  DBSCAN,
        "MeanShift":               MeanShift,
        "AffinityPropagation":     AffinityPropagation,
    }
    if HDBSCAN_AVAILABLE:
        alg_classes["HDBSCAN"] = hdbscan.HDBSCAN
    else:
        logger.warning("⚠️ HDBSCAN no disponible — ignorado.")

    bw = 1.0
    if len(X_scaled) > 0:
        bw = estimate_bandwidth(
            X_scaled, quantile=0.2, n_samples=min(len(X_scaled), 500)
        )

    param_grids: dict = {
        "KMeans":                  {"n_clusters": k_range},
        "Agglomerative":           {"n_clusters": k_range, "linkage": ["ward", "average", "complete"]},
        "Birch":                   {"n_clusters": k_range},
        "GMM":                     {"n_components": k_range},
        "BayesianGaussianMixture": {"n_components": k_range},
        "DBSCAN":                  {"eps": [0.5, 1.0, 1.5, 2.5, 5.0], "min_samples": [3, 5, 8]},
        "HDBSCAN":                 {"min_cluster_size": [5, 10, 15]},
        "MeanShift":               {"bandwidth": [bw * 0.5, bw, bw * 1.5]},
        "AffinityPropagation":     {"damping": [0.5, 0.9]},
    }

    # Keep only registered algorithms
    param_grids = {k: v for k, v in param_grids.items() if k in alg_classes}
    return alg_classes, param_grids


def optimize_clustering(
    alg_name: str,
    X: np.ndarray,
    alg_classes: dict,
    param_grids: dict,
    seeds: tuple,
    max_noise_pct: float = 5.0,
) -> dict:
    """
    Grid-search the best clustering for one algorithm on one embedding.
    Returns dict with best metrics, labels, params, and seed.
    """
    X_arr = np.asarray(X)
    alg_class = alg_classes[alg_name]

    best_score     = -np.inf
    best_db        = np.inf
    best_chi       = -np.inf
    best_labels    = None
    best_params    = None
    best_noise     = None
    best_n_clusters = 0          # ← FIX: initialize before loop
    best_seed      = None

    # Build grid
    raw_grid = param_grids.get(alg_name, [{}])
    if isinstance(raw_grid, dict):
        # Clamp GMM/BGM n_components to available samples
        if alg_name in ("GMM", "BayesianGaussianMixture"):
            max_comp = X_arr.shape[0] - 1
            comps = [k for k in raw_grid.get("n_components", []) if k <= max_comp]
            if not comps:
                return _empty_result()
            raw_grid = {"n_components": comps}
        grid = list(ParameterGrid(raw_grid))
    else:
        grid = list(ParameterGrid([{}]))

    if not grid:
        grid = [{}]

    # Algorithms that don't use a random_state — only run once
    deterministic = {"Agglomerative", "DBSCAN", "HDBSCAN", "MeanShift"}

    for seed in seeds:
        if alg_name in deterministic and seed != seeds[0]:
            continue

        for params in grid:
            try:
                uses_seed = alg_name in {
                    "KMeans", "GMM", "BayesianGaussianMixture",
                    "Birch", "AffinityPropagation",
                }
                model = (
                    alg_class(**params, random_state=seed)
                    if uses_seed
                    else alg_class(**params)
                )

                labels = (
                    model.fit(X_arr).labels_
                    if alg_name == "MeanShift"
                    else model.fit_predict(X_arr)
                )

                noise_pct = compute_noise_pct(labels)
                if noise_pct > max_noise_pct:
                    continue

                mask = labels != -1
                n_valid = len(np.unique(labels[mask]))
                if n_valid < 2 or mask.sum() < 2:
                    continue

                score = silhouette_score(X_arr[mask], labels[mask])
                db    = davies_bouldin_score(X_arr[mask], labels[mask])
                chi   = calinski_harabasz_score(X_arr[mask], labels[mask])

                if (
                    score > best_score
                    or (score == best_score and chi > best_chi)
                    or (score == best_score and chi == best_chi and db < best_db)
                ):
                    best_score      = score
                    best_db         = db
                    best_chi        = chi
                    best_params     = params
                    best_labels     = labels.copy()
                    best_noise      = noise_pct
                    best_n_clusters = n_valid
                    best_seed       = seed

            except Exception:
                continue

    if best_labels is None:
        return _empty_result()

    return {
        "best_score":     float(best_score),
        "best_db":        float(best_db),
        "best_chi":       float(best_chi),
        "best_labels":    best_labels,
        "best_params":    best_params,
        "noise_pct":      float(best_noise),
        "n_clusters_found": best_n_clusters,
        "best_seed":      best_seed,
    }


def _empty_result() -> dict:
    return {
        "best_score": None, "best_db": None, "best_chi": None,
        "best_labels": None, "best_params": None,
        "noise_pct": None, "n_clusters_found": 0, "best_seed": None,
    }


def run_clustering_search(
    embeddings: dict[str, np.ndarray],
    alg_classes: dict,
    param_grids: dict,
    seeds: tuple,
    max_noise_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Run all algorithms × all embeddings.
    Returns DataFrame with one row per valid clustering result.
    """
    rows = []
    for emb_name, X_emb in embeddings.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"🔬 ESPACIO: {emb_name} ({X_emb.shape[1]} dims)")

        for alg in alg_classes:
            # Skip density-based on raw high-dim space
            if emb_name == "SCALED_RAW" and alg in ("DBSCAN", "HDBSCAN") and X_emb.shape[1] > 20:
                logger.warning(f"⚠️ Omitiendo {alg} en alta dimensionalidad.")
                continue

            res = optimize_clustering(alg, X_emb, alg_classes, param_grids, seeds, max_noise_pct)
            k   = res["n_clusters_found"]

            if res["best_labels"] is not None and k > 1:
                rows.append({
                    "algorithm":              alg,
                    "Optimal K/Comp":         k,
                    "score":                  res["best_score"],
                    "Davies-Bouldin Score":   res["best_db"],
                    "Calinski-Harabasz Score":res["best_chi"],
                    "labels":                 res["best_labels"],
                    "params":                 res["best_params"],
                    "noise":                  res["noise_pct"],
                    "reduction":              emb_name,
                    "seed":                   res["best_seed"],
                })
                logger.info(
                    f"✅ {alg:30s} K={k:2d} | Seed={res['best_seed']} | "
                    f"Silhouette={res['best_score']:.3f} | Noise={res['noise_pct']:.2f}%"
                )
            else:
                logger.info(f"⚠️  {alg} — sin resultados válidos.")

    return pd.DataFrame(rows)


# ============================================================
# 7. SELECT & EXPORT CLUSTERS
# ============================================================
def select_and_export_clusters(
    df_scores: pd.DataFrame,
    df_features_merged: pd.DataFrame,
    out_dir: str,
    silhouette_threshold: float,
) -> tuple[pd.DataFrame, str]:
    """
    Filter, sort, and export best clustering results.
    Returns (df_clusters_final, output_path).
    """
    # FIX: filter AND sort in one step (previously the filter was overwritten)
    df_selected = (
        df_scores[df_scores["score"] >= silhouette_threshold]
        .sort_values(
            by=["score", "Calinski-Harabasz Score", "Davies-Bouldin Score", "noise"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"\n🏆 {len(df_selected)} modelos con Silhouette >= {silhouette_threshold:.2f}"
    )

    df_clusters = pd.DataFrame({"Model": df_features_merged["Model"].values})

    for i, row in df_selected.iterrows():
        labels = np.asarray(row["labels"])
        seed_s = f"_Seed{row['seed']}" if row["seed"] != "N/A" else ""
        col_name = (
            f"Cluster_{row['reduction']}_{row['algorithm']}"
            f"_K{row['Optimal K/Comp']}"
            f"_S{row['score']:.2f}"
            f"_DB{row['Davies-Bouldin Score']:.2f}"
            f"_CH{row['Calinski-Harabasz Score']:.0f}"
            f"{seed_s}"
        )
        if len(labels) == df_features_merged.shape[0]:
            df_clusters[col_name] = labels

        if i < 10:
            logger.info(
                f"  {i+1:3d}. {row['reduction']} + {row['algorithm']} "
                f"K={row['Optimal K/Comp']} Seed={row['seed']} "
                f"Sil={row['score']:.4f} Noise={row['noise']:.2f}%"
            )

    out_path = os.path.join(out_dir, "PatientClusters_Selected_PARETO.csv")
    df_clusters.to_csv(out_path, index=False)
    logger.info(f"✅ Clusters exportados: {out_path}")
    return df_clusters, out_path


# ============================================================
# 8. PLOTTING
# ============================================================
def generate_umap_plots(
    df_top: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    out_dir: str,
    top_n: int = 5,
) -> None:
    """Generate 2D/3D scatter plots for the top-N clustering results."""
    df_top = df_top.head(top_n).reset_index(drop=True)
    if df_top.empty:
        logger.warning("⚠️ No hay modelos para graficar.")
        return

    logger.info(f"📈 Generando {len(df_top)} gráficas UMAP...")

    for i, row in df_top.iterrows():
        emb_key = row["reduction"]
        labels  = np.asarray(row["labels"])
        if emb_key not in embeddings:
            logger.warning(f"⚠️ Embedding {emb_key} no encontrado.")
            continue

        X_emb  = embeddings[emb_key]
        n_dims = X_emb.shape[1]

        seed_s  = f"_Seed{row['seed']}" if row["seed"] != "N/A" else ""
        config  = (
            f"{emb_key}_{row['algorithm']}_K{row['Optimal K/Comp']}"
            f"_S{row['score']:.2f}_DB{row['Davies-Bouldin Score']:.2f}"
            f"_CH{row['Calinski-Harabasz Score']:.0f}{seed_s}"
        )
        title   = (
            f"TOP {i+1}: {config}\n"
            f"Silhouette: {row['score']:.3f} | K={int(row['Optimal K/Comp'])} | "
            f"Noise: {row['noise']:.1f}%"
        )
        outfile = os.path.join(out_dir, f"TOP_{i+1}_{config}.png")

        df_plot = pd.DataFrame(X_emb, columns=[f"Dim{j+1}" for j in range(n_dims)])
        df_plot["Cluster"] = labels.astype(str)

        cluster_list   = sorted(df_plot["Cluster"].unique())
        valid_clusters = [c for c in cluster_list if c != "-1"]
        palette        = sns.color_palette("tab10", n_colors=max(len(valid_clusters), 1))
        color_map      = {c: palette[j] for j, c in enumerate(valid_clusters)}
        if "-1" in cluster_list:
            color_map["-1"] = "gray"

        if n_dims == 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x="Dim1", y="Dim2", hue="Cluster",
                data=df_plot, palette=color_map,
                legend="full", alpha=0.7, s=50,
            )
            plt.title(title, fontsize=12)
            plt.savefig(outfile, bbox_inches="tight")
            plt.close()
            logger.info(f"    > 2D guardada: {outfile}")

        elif n_dims == 3:
            fig = plt.figure(figsize=(12, 10))
            ax  = fig.add_subplot(111, projection="3d")
            for c_label in cluster_list:
                sub = df_plot[df_plot["Cluster"] == c_label]
                ax.scatter(
                    sub["Dim1"], sub["Dim2"], sub["Dim3"],
                    label=f"Cluster {c_label}",
                    color=color_map.get(c_label, "black"),
                    alpha=0.7, s=50,
                )
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2"); ax.set_zlabel("Dim3")
            ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.savefig(outfile, bbox_inches="tight")
            plt.close()
            logger.info(f"    > 3D guardada: {outfile}")


# ============================================================
# 9. FINAL MERGE
# ============================================================
def build_final_dataset(
    df_features_merged: pd.DataFrame,
    feature_cols: list[str],
    df_clinical: pd.DataFrame,
    df_survival: pd.DataFrame,
    clusters_path: str,
    out_dir: str,
) -> pd.DataFrame:
    """
    Merge features + clinical + survival + clusters into one final CSV.
    """
    pareto_models = df_features_merged["Model"].unique().tolist()

    df_clin_filt = df_clinical[df_clinical["Model"].isin(pareto_models)]
    df_surv_filt = df_survival[df_survival["Model"].isin(pareto_models)]

    base = (
        df_features_merged
        .drop(columns=feature_cols, errors="ignore")
        .drop_duplicates(subset=["Model"])
        .copy()
    )

    def _merge(left, right):
        return left.merge(
            right.drop_duplicates(subset=["Model"]),
            on="Model", how="left",
        )

    merged = _merge(_merge(base, df_clin_filt), df_surv_filt)
    merged = merged.drop_duplicates(subset=["Model"])

    # Attach features
    merged = merged.merge(df_features_merged, on="Model", how="left")

    # Attach clusters
    try:
        df_clusters = pd.read_csv(clusters_path).drop_duplicates(subset=["Model"])
        cluster_cols = [c for c in df_clusters.columns if c != "Model"]
        merged = merged.merge(df_clusters, on="Model", how="left")
        merged[cluster_cols] = merged[cluster_cols].fillna(-1).astype(int)
        logger.info(f"✅ Clusters incorporados: {len(cluster_cols)} columnas.")
    except FileNotFoundError:
        logger.warning(f"⚠️ No se encontró el archivo de clusters: {clusters_path}")

    out_path = os.path.join(out_dir, "Merged_TCGA_BRCA_AllData_withClusters_PARETO.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"✅ Dataset final guardado: {out_path}")

    cluster_preview = [c for c in merged.columns if c.startswith("Cluster_")]
    if cluster_preview:
        print("\nPreview de Clusters:")
        print(merged[["Model"] + cluster_preview].head(8).to_string(index=False))

    return merged


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    cfg = Config()
    np.random.seed(cfg.seeds[0])
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ── 1. Load Pareto ────────────────────────────────────────
    df_pareto, pareto_loaded = load_pareto(cfg.path_pareto)
    if not pareto_loaded:
        raise RuntimeError("❌ No se pudo cargar el CSV Pareto. Abortando.")

    # ── 2. Feature engineering ────────────────────────────────
    df_features_merged, feature_cols = build_feature_matrix(df_pareto)

    # ── 3. Preprocessing ──────────────────────────────────────
    X_scaled, df_features_merged, feature_cols = preprocess_features(
        df_features_merged, feature_cols
    )

    # ── 4. UMAP ───────────────────────────────────────────────
    embeddings = run_umap_grid(X_scaled, cfg.seeds)

    # ── 5. Clustering ─────────────────────────────────────────
    alg_classes, param_grids = build_algorithm_registry(X_scaled)
    df_scores = run_clustering_search(
        embeddings, alg_classes, param_grids, cfg.seeds, cfg.max_noise_pct
    )

    if df_scores.empty:
        logger.error("❌ No se obtuvieron resultados de clustering válidos.")
        return

    # ── 6. Select & export clusters ───────────────────────────
    df_clusters_final, clusters_path = select_and_export_clusters(
        df_scores, df_features_merged, cfg.out_dir, cfg.silhouette_threshold
    )

    # ── 7. Plot top-5 ─────────────────────────────────────────
    df_top = (
        df_scores[df_scores["score"] >= cfg.silhouette_threshold]
        .sort_values("score", ascending=False)
        .head(5)
    )
    generate_umap_plots(df_top, embeddings, cfg.out_dir, top_n=5)

    # ── 8. Load clinical / survival ───────────────────────────
    df_clinical = load_clinical(cfg.path_clinical)
    df_survival = load_clinical(cfg.path_survival)

    # ── 9. Final merge ────────────────────────────────────────
    build_final_dataset(
        df_features_merged, feature_cols,
        df_clinical, df_survival,
        clusters_path, cfg.out_dir,
    )

    logger.info("\n🎉 Pipeline completado con éxito.")


if __name__ == "__main__":
    main()