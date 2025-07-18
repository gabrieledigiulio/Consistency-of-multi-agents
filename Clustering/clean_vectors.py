import json
import numpy as np

def main():
    MIN_FEATURES_THRESHOLD = 5
    with open("clustering/kitchen_top100_comments_vettore.json", "r") as f:
        data = json.load(f)
    filtered_data = []
    total_vectors = len(data)
    for item in data:
        vector = np.array(item["vector"])
        non_zero_count = np.count_nonzero(vector)
        if non_zero_count >= MIN_FEATURES_THRESHOLD:
            filtered_data.append(item)
    filtered_vectors = len(filtered_data)
    output_path = "vettori_puliti/kitchen_top100_comments_vettori_puliti.json"
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Numero di vettori iniziali: {total_vectors}")
    print(f"Numero di vettori dopo la scrematura: {filtered_vectors}")
    print(f"Dati puliti salvati in {output_path}")

if __name__ == "__main__":
    main()
