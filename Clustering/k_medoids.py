import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from kmedoids import KMedoids 

def main():
    with open("clustering_raggruppati/kitchen_top100_raggruppati.json", "r") as f:
        data = json.load(f)

    ids = [item["id"] for item in data]
    vectors = np.array([item["vector"] for item in data])

    emotions_list = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "trust", 
        "boredom", "envy", "guilt"
    ]
    toxicity_list = ["toxicity_high", "toxicity_medium", "toxicity_low"]
    readability_list = ["readability_excellent", "readability_good", "readability_average", "readability_poor", "readability_very_poor"]
    creativity_list = ["creativity_high", "creativity_medium", "creativity_low"]
    feature_names = emotions_list + toxicity_list + readability_list + creativity_list

    print("Conteggio delle feature nei vettori di input:")
    feature_counts = (vectors > 0).sum(axis=0)
    for feature, count in zip(feature_names, feature_counts):
        print(f"{feature}: {count}")

    k_opt = 4
    print(f"\nNumero di cluster selezionato manualmente: k = {k_opt}")

    model_final = KMedoids(n_clusters=k_opt, random_state=42, metric='hamming')
    model_final.fit(vectors)
    labels_final = model_final.labels_
    medoids_final = model_final.cluster_centers_

    silhouette_avg = silhouette_score(vectors, labels_final, metric='hamming')
    print(f"Silhouette Score (finale): {silhouette_avg:.2f}")

    binary_medoids = medoids_final.astype(int)

    for i, medoid in enumerate(medoids_final):
        print(f"\nCluster {i} - Valori originali delle feature (medoid):")
        for j, value in enumerate(medoid):
            print(f"  {feature_names[j]}: {value:.2f}")

    clustered_data = [{"id": ids[i], "cluster": int(labels_final[i])} for i in range(len(ids))]
    output = {
        "clusters": clustered_data,
        "medoids": [{"cluster": i, "vector": binary_medoids[i].tolist()} for i in range(k_opt)]
    }

    with open("clustering_results_raggruppati/kitchen_top100_comments_k_medoids_raggruppati.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nClustering completato con K-Medoids")

    medoid_df = pd.DataFrame(binary_medoids, columns=feature_names)
    medoid_df.index = [f"Cluster {i}" for i in range(k_opt)]
    print("\nMedoids (versione binaria):")
    print(medoid_df)

if __name__ == "__main__":
    main()
