import json
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def main():
    with open("clustering_raggruppati/kitchen_top100_raggruppati.json", "r") as f:
        data = json.load(f)

    ids = [item["id"] for item in data]
    vectors = np.array([item["vector"] for item in data])

    emotions_list = [
       'Positive & Uplifting', 'Calm & Curious', 'Agitated/Reactive', 'Downcast & Vulnerable', 'Reflective & Anxious'
    ]
    toxicity_list = ["toxicity_high", "toxicity_medium", "toxicity_low"]
    readability_list = ["readability_excellent", "readability_good", "readability_average", "readability_poor", "readability_very_poor"]
    creativity_list = ["creativity_high", "creativity_medium", "creativity_low"]

    feature_names = emotions_list + toxicity_list + readability_list + creativity_list

    print("Conteggio delle feature nei vettori di input:")
    feature_counts = (vectors > 0).sum(axis=0)
    for feature, count in zip(feature_names, feature_counts):
        print(f"{feature}: {count}")

    wcss = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectors)
        wcss.append(kmeans.inertia_)

    k_opt = 4
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    silhouette_avg = silhouette_score(vectors, labels)
    print(f"\nSilhouette Score: {silhouette_avg:.2f}")

    centroids = kmeans.cluster_centers_

    medoids = []
    for i in range(k_opt):
        cluster_points = vectors[labels == i]
        distances = cdist(cluster_points, cluster_points, metric='hamming')
        medoid_idx = np.argmin(distances.sum(axis=1))
        medoids.append(cluster_points[medoid_idx])
    medoids = np.array(medoids)

    threshold = 0.3
    binary_centroids = np.where(centroids >= threshold, 1, 0)
    binary_medoids = medoids.astype(int)

    for i, (centroid, medoid) in enumerate(zip(centroids, medoids)):
        print(f"Cluster {i} - Valori originali delle feature (centroidi):")
        for j, value in enumerate(centroid):
            print(f"  {feature_names[j]}: {value:.2f}")
        print(f"Cluster {i} - Valori originali delle feature (medoidi):")
        for j, value in enumerate(medoid):
            print(f"  {feature_names[j]}: {value:.2f}")
        print(f"Cluster {i} - Numero di feature attive nel medoide: {np.sum(binary_medoids[i])}")

    clustered_data = [{"id": ids[i], "cluster": int(labels[i])} for i in range(len(ids))]
    output = {
        "clusters": clustered_data,
        "centroids": [{"cluster": i, "vector": binary_centroids[i].tolist()} for i in range(k_opt)],
        "medoids": [{"cluster": i, "vector": binary_medoids[i].tolist()} for i in range(k_opt)]
    }

    with open("clustering_results_raggruppati/kitchen_top100_comments_k_means_raggruppati_k=4.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Clustering completato")

if __name__ == "__main__":
    main()
