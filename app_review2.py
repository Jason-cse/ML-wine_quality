import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Wine Quality - Review 2",
    page_icon="🍷",
    layout="wide",
)

DATA_FILE = Path(__file__).resolve().parent / "winequality-red.csv"
RANDOM_STATE = 42


def add_features(frame):
    result = frame.copy()
    result["free_to_total_sulfur_ratio"] = result["free sulfur dioxide"] / (
        result["total sulfur dioxide"] + 1e-6
    )
    result["acidity_balance"] = (
        result["fixed acidity"]
        + result["volatile acidity"]
        + result["citric acid"]
    )
    result["sugar_to_alcohol_ratio"] = result["residual sugar"] / (
        result["alcohol"] + 1e-6
    )
    result["alcohol_squared"] = result["alcohol"] ** 2
    return result


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE).drop_duplicates().reset_index(drop=True)
    return add_features(data)


@st.cache_data
def calculate_clusters(data):
    cluster_features = data.drop(columns=["quality"])
    cluster_matrix = StandardScaler().fit_transform(cluster_features)
    inertias = {
        k: KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        .fit(cluster_matrix)
        .inertia_
        for k in range(2, 11)
    }
    kmeans_labels = KMeans(
        n_clusters=3, n_init=10, random_state=RANDOM_STATE
    ).fit_predict(cluster_matrix)
    hierarchical_labels = AgglomerativeClustering(
        n_clusters=3, linkage="ward"
    ).fit_predict(cluster_matrix)
    labels = {
        "K-Means": kmeans_labels,
        "Agglomerative Hierarchical": hierarchical_labels,
    }
    metrics = pd.DataFrame(
        [
            {
                "Algorithm": name,
                "Silhouette Score": silhouette_score(cluster_matrix, values),
                "Davies-Bouldin Index": davies_bouldin_score(
                    cluster_matrix, values
                ),
                "Calinski-Harabasz Index": calinski_harabasz_score(
                    cluster_matrix, values
                ),
            }
            for name, values in labels.items()
        ]
    )
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_values = pca.fit_transform(cluster_matrix)
    plot_data = pd.DataFrame(
        {
            "PC1": pca_values[:, 0],
            "PC2": pca_values[:, 1],
            "K-Means cluster": kmeans_labels.astype(str),
            "Hierarchical cluster": hierarchical_labels.astype(str),
        }
    )
    return inertias, metrics, plot_data, pca.explained_variance_ratio_.sum()


try:
    dataset = load_data()
except FileNotFoundError:
    st.error(
        f"Dataset file not found: `{DATA_FILE}`. Keep the app and CSV in the same folder."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the Review 2 app: {exc}")
    st.stop()


st.title("🍷 Wine Quality Clustering — Review 2")
st.write(
    "Explore unsupervised clusters in the red-wine quality dataset using "
    "K-Means and agglomerative hierarchical clustering."
)
st.caption("The quality label is excluded from all clustering features.")

st.subheader("Unsupervised clustering")
st.write("The selected k=3 matches the Review 2 notebook.")
inertias, cluster_metrics, plot_data, variance = calculate_clusters(dataset)
left, right = st.columns(2)
with left:
    st.write("K-Means elbow data")
    st.line_chart(pd.Series(inertias, name="Inertia"))
with right:
    st.write("Internal cluster metrics")
    st.dataframe(
        cluster_metrics.style.format(
            {
                "Silhouette Score": "{:.4f}",
                "Davies-Bouldin Index": "{:.4f}",
                "Calinski-Harabasz Index": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
st.write(f"PCA variance explained by two components: **{variance:.2%}**")
st.write("K-Means clusters in PCA space")
st.scatter_chart(
    plot_data,
    x="PC1",
    y="PC2",
    color="K-Means cluster",
)

st.divider()
st.caption(
    "These are estimates from the supplied dataset and are not a professional wine assessment."
)
