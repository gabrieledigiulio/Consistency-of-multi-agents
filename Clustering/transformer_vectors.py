import json

def process_json_data(input_file, output_file):
    emotions_list = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "trust", 
        "boredom", "envy", "guilt"
    ]
    toxicity_list = ["high", "medium", "low"]
    readability_list = ["excellent", "good", "average", "poor", "very poor"]
    creativity_list = ["high", "medium", "low"]
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        try:
            json_data = json.load(infile)
            if not isinstance(json_data, list):
                raise ValueError("Il file JSON non è una lista.")
        except json.JSONDecodeError:
            infile.seek(0)
            json_data = [json.loads(line) for line in infile]
    
    for item in json_data:
        post_id = item.get("id", "")
        vector = []
        emotions = item.get("emotions", [])
        vector.extend([1 if emotion in emotions else 0 for emotion in emotions_list])
        toxicity = item.get("toxicity", "")
        vector.extend([1 if toxicity == level else 0 for level in toxicity_list])
        readability = item.get("readability", "")
        vector.extend([1 if readability == level else 0 for level in readability_list])
        creativity = item.get("creativity", "")
        vector.extend([1 if creativity == level else 0 for level in creativity_list])
        results.append({"id": post_id, "vector": vector})
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

def main():
    input_file = "./results/kitchen_top100_comments_annotated.jsonl"
    output_file = "./clustering/kitchen_top100_comments_vettore.jsonl"
    process_json_data(input_file, output_file)

if __name__ == "__main__":
    main()
