import autogen
import json
import os

def build_prompt(cluster_id, features):
    prompt = f"""
The following is a user profile from a Reddit discussion about police brutality and law enforcement.

Profile C{cluster_id}
Traits: {', '.join(features)}

Based on this profile, generate a sample Reddit post. Use a tone and content that reflect the listed traits.
The post should feel authentic and thoughtful, without exaggeration. It can focus on personal experiences, reflections, opinions, or questions related to police brutality, justice, or law enforcement policies.
Write only the Reddit post itself. Do not include any explanations, prefaces, emoji or metadata.
"""
    return prompt.strip()

def main():
    config_list_gemma = [
        {
            'base_url': "http://localhost:11434/v1",
            'api_key': "NULL",
            'model': 'gemma2:2b',
            'price': [0.001, 0.002]  
        }       
    ]

    llm_config_gemma = {
        "config_list": config_list_gemma
    }

    llm_agent = autogen.AssistantAgent(
        name="Administrator",
        llm_config=llm_config_gemma,
    )

    input_files = [
        "dataset_iniziale_police/police_vettoriNONpuliti_centroidi.json",
        "dataset_iniziale_police/police_vettoriNONpuliti_medoidi.json",
        "dataset_iniziale_police/police_vettoripuliti_centroidi.json",
        "dataset_iniziale_police/police_vettoripuliti_medoidi.json"
    ]

    n = 10

    for input_path in input_files:
        print(f"Caricamento del file: {input_path}...")
        with open(input_path, "r") as f:
            clusters = json.load(f)

        print(f"{len(clusters)} cluster caricati da {input_path}\n")
        generated_posts = []

        for i, cluster in enumerate(clusters):
            cluster_id = cluster['cluster']
            features = cluster['features']

            for j in range(n):
                print(f"Cluster {cluster_id} ({i + 1}/{len(clusters)}) — Features: {features} — Esempio {j + 1}/{n}")
                prompt = build_prompt(cluster_id, features)
                print("Invio prompt al modello...")

                conversable = autogen.ConversableAgent(
                    name="CommentGenerators",
                    llm_config=llm_config_gemma,
                    human_input_mode="NEVER"
                )

                response = conversable.initiate_chat(
                    recipient=llm_agent,
                    message=prompt,
                    max_turns=1
                )

                generated_text = response.chat_history[-1]["content"].strip()
                print("Commento generato.")

                generated_posts.append({
                    "cluster": cluster_id,
                    "traits": features,
                    "prompt": prompt,
                    "generated_text": generated_text
                })

        output_filename = os.path.basename(input_path).replace(".json", "_generati.json")
        output_path = os.path.join("risultati_police", output_filename)
        with open(output_path, "w") as f:
            json.dump(generated_posts, f, indent=4)

        print(f"File generato salvato in: {output_path}\n")

    print("Generazione completata per tutti i file.")

if __name__ == "__main__":
    main()
