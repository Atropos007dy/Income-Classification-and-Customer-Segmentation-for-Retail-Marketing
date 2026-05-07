from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_processed" / "cleaned_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "segmentation"


def main():
    df = pd.read_csv(DATA_PATH)

    target_col = "label"
    weight_col = "weight"

    numeric_features = [
        "age",
        "wage per hour",
        "weeks worked in year",
        "dividends from stocks",
        "capital gains",
        "capital losses",
    ]

    categorical_features = [
        "education",
        "marital stat",
        "sex",
        "race",
        "hispanic origin",
        "citizenship",
        "full or part time employment stat",
        "major occupation code",
        "major industry code",
        "class of worker",
        "tax filer stat",
        "family members under 18",
        "detailed household and family stat",
        "detailed household summary in household",
    ]

    features = numeric_features + categorical_features
    X = df[features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    results = []

    for k in [3, 4, 5, 6]:
        print(f"\nTraining KMeans with k={k}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clusterer", KMeans(n_clusters=k, random_state=42, n_init=10)),
            ]
        )

        cluster_labels = pipeline.fit_predict(X)
        X_transformed = pipeline.named_steps["preprocessor"].transform(X)

        # Silhouette can be slow on full data, so use a sample.
        sample_size = min(10000, X_transformed.shape[0])
        silhouette = silhouette_score(
            X_transformed[:sample_size],
            cluster_labels[:sample_size],
        )

        inertia = pipeline.named_steps["clusterer"].inertia_

        results.append({
            "k": k,
            "silhouette_score": silhouette,
            "inertia": inertia,
        })

        print(f"k={k}, silhouette={silhouette:.4f}, inertia={inertia:.2f}")

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "k_selection.csv"
    results_df.to_csv(results_path, index=False)

    print("\n--- K selection results ---")
    print(results_df)
    print(f"\nSaved k selection results to: {results_path}")



    # Fit final segmentation model with selected k.
    final_k = 5

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clusterer", KMeans(n_clusters=final_k, random_state=42, n_init=10)),
        ]
    )

    df["cluster"] = final_pipeline.fit_predict(X)

    segmented_path = OUTPUT_DIR / "segmented_dataset.csv"
    df.to_csv(segmented_path, index=False)
    print(f"\nSaved segmented dataset to: {segmented_path}")


        # Convert label to binary for profiling.
    df["label_binary"] = df[target_col].map({
        "- 50000.": 0,
        "50000+.": 1,
    })

    numeric_profile = df.groupby("cluster")[numeric_features + ["label_binary"]].mean()
    cluster_size = df["cluster"].value_counts().sort_index().rename("cluster_size")
    cluster_share = (df["cluster"].value_counts(normalize=True).sort_index()).rename("cluster_share")

    profile_df = pd.concat([cluster_size, cluster_share, numeric_profile], axis=1)

    profile_path = OUTPUT_DIR / "cluster_numeric_profile.csv"
    profile_df.to_csv(profile_path)
    print("\n--- Cluster numeric profile ---")
    print(profile_df)
    print(f"\nSaved cluster numeric profile to: {profile_path}")




    #df["cluster"] = final_pipeline.fit_predict(X)
        # ------------------------------------------------------------
    # PCA-style 2D visualization of the 5 clusters
    # ------------------------------------------------------------
    X_transformed_final = final_pipeline.named_steps["preprocessor"].transform(X)
    kmeans_model = final_pipeline.named_steps["clusterer"]

    # Sample a subset for plotting so the figure is not too crowded.
    sample_n = min(10000, X_transformed_final.shape[0])
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(X_transformed_final.shape[0], size=sample_n, replace=False)

    X_sample = X_transformed_final[sample_idx]
    cluster_sample = df["cluster"].iloc[sample_idx].to_numpy()

    # Use TruncatedSVD as a PCA-like 2D projection for sparse transformed features.
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_sample)

    # Project cluster centers into the same 2D space.
    centers_2d = svd.transform(kmeans_model.cluster_centers_)

    # Save 2D projection data if needed.
    projection_df = pd.DataFrame({
        "component_1": X_2d[:, 0],
        "component_2": X_2d[:, 1],
        "cluster": cluster_sample,
    })

    projection_path = OUTPUT_DIR / "cluster_2d_projection.csv"
    projection_df.to_csv(projection_path, index=False)
    print(f"Saved 2D projection data to: {projection_path}")

    # Plot
    plot_path = OUTPUT_DIR / "cluster_2d_plot.png"

    plt.figure(figsize=(9, 7))

    for cluster_id in sorted(projection_df["cluster"].unique()):
        subset = projection_df[projection_df["cluster"] == cluster_id]
        plt.scatter(
            subset["component_1"],
            subset["component_2"],
            s=8,
            alpha=0.45,
            label=f"Cluster {cluster_id}",
        )

    # Plot cluster centers as large black X markers
    plt.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        s=250,
        c="black",
        marker="X",
        label="Cluster Centers",
    )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("KMeans Customer Segments in 2D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved 2D cluster plot to: {plot_path}")
    

if __name__ == "__main__":
    main()