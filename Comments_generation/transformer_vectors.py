import json

def main():
    emotions_list = [
       'Positive & Uplifting', 'Calm & Curious', 'Agitated/Reactive', 'Downcast & Vulnerable', 'Reflective & Anxious'
    ]
    toxicity_list = ["toxicity_high", "toxicity_medium", "toxicity_low"]
    readability_list = ["readability_excellent", "readability_good", "readability_average", "readability_poor", "readability_very_poor"]
    creativity_list = ["creativity_high", "creativity_medium", "creativity_low"]
    feature_names = emotions_list + toxicity_list + readability_list + creativity_list

    def extract_active_features(items):
        return [
            {
                "cluster": item["cluster"],
                "features": [
                    name for name, val in zip(feature_names, item["vector"]) if val == 1
                ]
            }
            for item in items
        ]

    with open("clustering_results_raggruppati/police_top100_comments_k_means_raggruppati.json", "r") as f:
        data = json.load(f)

    centroids = extract_active_features(data.get("centroids", []))
    medoids = extract_active_features(data.get("medoids", []))

    with open("classificazione/police_vettoriNONpuliti_centroidi.json", "w") as f:
        json.dump(centroids, f, indent=4)

    with open("classificazione/police_vettoriNONpuliti_medoidi.json", "w") as f:
        json.dump(medoids, f, indent=4)

if __name__ == "__main__":
    main()
