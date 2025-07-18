import json

emotion_groups = {
    "Positive & Uplifting": ["joy", "amusement", "gratitude", "admiration", "love", "pride"],
    "Calm & Curious": ["curiosity", "optimism", "approval", "trust", "relief"],
    "Agitated/Reactive": ["anger", "annoyance", "disapproval", "desire", "disgust", "surprise"],
    "Downcast & Vulnerable": ["sadness", "disappointment", "grief", "guilt", "remorse", "nervousness"],
    "Reflective & Anxious": ["fear", "confusion", "embarrassment", "caring", "realization", "boredom", "envy"]
}

emotions_list = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "trust", 
    "boredom", "envy", "guilt"
]

emotion_index_map = {emotion: i for i, emotion in enumerate(emotions_list)}

def transform_entry(entry):
    vec = entry["vector"]
    grouped_emotions = []
    for group in emotion_groups.values():
        active = int(any(vec[emotion_index_map[e]] for e in group))
        grouped_emotions.append(active)
    rest = vec[len(emotions_list):]
    return {
        "id": entry["id"],
        "vector": grouped_emotions + rest
    }

def transform_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transformed = [transform_entry(entry) for entry in data]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2)

def main():
    transform_json(
        "vettori_puliti/kitchen_top100_comments_vettori_puliti.json",
        "clustering_raggruppati/kitchen_top100_raggruppati_vettori_puliti.json"
    )

if __name__ == "__main__":
    main()
