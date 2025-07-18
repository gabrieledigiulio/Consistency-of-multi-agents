import autogen
import json
import os

def mapp_emotions(emotions_list):    
    emotion_groups = {
        "Positive & Uplifting": ["joy", "amusement", "gratitude", "admiration", "love", "pride"],
        "Calm & Curious": ["curiosity", "optimism", "approval", "trust", "relief"],
        "Agitated/Reactive": ["anger", "annoyance", "disapproval", "desire", "disgust", "surprise"],
        "Downcast & Vulnerable": ["sadness", "disappointment", "grief", "remorse", "nervousness"],
        "Reflective & Anxious": ["fear", "confusion", "embarrassment", "caring", "realization"]
    }
    groups_found = set()
    for emotion in emotions_list:
        emotion_clean = emotion.lower().strip()
        for group_name, group_emotions in emotion_groups.items():
            if emotion_clean in group_emotions:
                groups_found.add(group_name)
                break
    return list(groups_found)

def extract_trait(response, trait_type): 
    response_lower = response.lower()
    possible_values = {
        'emotions': ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment","excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism","pride", "realization", "relief", "remorse", "sadness", "surprise", "trust", "boredom", "envy", "guilt"],
        'readability': ['excellent', 'good', 'average', 'poor', 'very poor'],
        'toxicity': ['high', 'medium', 'low'],
        'creativity': ['high', 'medium', 'low']
    }
    if trait_type == 'emotions':
        emotions_found = []
        for emotion in possible_values['emotions']:
            if emotion in response_lower:
                emotions_found.append(emotion)
        return emotions_found
    else:
        for value in possible_values[trait_type]:
            if value in response_lower:
                trait_value = f"{trait_type}_{value.replace(' ', '_')}"
                return trait_value
        return 'unknown'

def process_item(item, user_agent, emotion_annotator, readability_annotator, toxicity_annotator, creativity_annotator):
    generated_text = item.get("generated_text", "")
    traits_original = item.get("traits", [])
    emotion_annotator.initiate_chat(
        user_agent, 
        message=f"Analyze the emotions in this text: {generated_text}",
        silent=True,
        max_round=1
    )
    emotion_response = emotion_annotator.chat_messages[user_agent][-1]["content"]
    readability_annotator.initiate_chat(
        user_agent,
        message=f"Evaluate the readability of this text: {generated_text}",
        silent=True,
        max_round=1
    )
    readability_response = readability_annotator.chat_messages[user_agent][-1]["content"]
    toxicity_annotator.initiate_chat(
        user_agent,
        message=f"Evaluate the toxicity of this text: {generated_text}",
        silent=True,
        max_round=1
    )
    toxicity_response = toxicity_annotator.chat_messages[user_agent][-1]["content"]
    creativity_annotator.initiate_chat(
        user_agent,
        message=f"Evaluate the creativity of this text: {generated_text}",
        silent=True,
        max_round=1
    )
    creativity_response = creativity_annotator.chat_messages[user_agent][-1]["content"]

    emotions_extracted = extract_trait(emotion_response, 'emotions')
    emotion_groups_detected = mapp_emotions(emotions_extracted)
    readability_value = extract_trait(readability_response, 'readability')
    toxicity_value = extract_trait(toxicity_response, 'toxicity')
    creativity_value = extract_trait(creativity_response, 'creativity')
    traits_detected = emotion_groups_detected + [readability_value, toxicity_value, creativity_value]
    set_original = set(traits_original)
    set_detected = set(traits_detected)
    intersection = set_original & set_detected
    union = set_original | set_detected
    jaccard_similarity = len(intersection) / len(union) if union else 0
    result = {
        "traits_original": traits_original,
        "traits_detected": traits_detected,
        "traits_in_common": list(intersection),
        "jaccard": jaccard_similarity,
        "generated_text": generated_text,        
    }
    return result

def load_input(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    data_config = {
        'base_url': "http://localhost:11434/v1",
        'api_key': "NULL",
        'model': 'gemma2:2b',
        'price': [0.001, 0.002]
    }
    llm_config = {"config_list": [data_config]}
    input_files = [
        "risultati_kitchen/kitchen_vettoriNONpuliti_centroidi_generati.json",
        "risultati_kitchen/kitchen_vettoriNONpuliti_medoidi_generati.json", 
        "risultati_kitchen/kitchen_vettoripuliti_centroidi_generati.json",
        "risultati_kitchen/kitchen_vettoripuliti_medoidi_generati.json",
        "risultati_police/police_vettoriNONpuliti_centroidi_generati.json",
        "risultati_police/police_vettoriNONpuliti_medoidi_generati.json",
        "risultati_police/police_vettoripuliti_centroidi_generati.json", 
        "risultati_police/police_vettoripuliti_medoidi_generati.json"
    ]
    output_folder = "risultati_finali"
    max_items = None
    user_agent = autogen.AssistantAgent(
        name="User",
        llm_config=llm_config,
        system_message="You are an online user. You will receive text to analyze. Simply repeat the text exactly as given when asked.",
        max_consecutive_auto_reply=1,
    )
    emotion_annotator = autogen.AssistantAgent(
        name="EmotionAnnotator",
        llm_config=llm_config,
        system_message=(
            "You are an emotion detection expert. After having read the user's response to your request, act as follows: 1. Consider a model capable of detecting a broad range of emotions as encoded in the GoEmotions taxonomy, which includes: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, and trust. 2. Detect the emotions in the user's texts. 3. Format your response strictly as a Python list of emotions, with each emotion enclosed in double asterisks: e.g., ['**admiration**', '**joy**'].  - The emotions can be more than one. - **You must not use any word that is not on the provided GoEmotions list.**  - **The response must only contain the list of emotions, without explanations, descriptions, or any additional text.** - **Do not use synonyms, alternative spellings, or add any additional commentary.** Use only the exact words provided in the GoEmotions list. Failure to follow these instructions will result in your response being invalid. You are expected to produce only the list of emotions as described. "
        ),
        max_consecutive_auto_reply=1,
    )
    readability_annotator = autogen.AssistantAgent(
        name="ReadabilityAnnotator",
        llm_config=llm_config,
        system_message=(
            "You are a readability expert. Evaluate how easy the given text is to read. "
            "Respond with ONLY one of these exact words: **Excellent**, **Good**, **Average**, **Poor**, **Very poor**"
        ),
        max_consecutive_auto_reply=1,
    )
    toxicity_annotator = autogen.AssistantAgent(  
        name="ToxicityAnnotator",
        llm_config=llm_config,
        system_message=(
            "You are a toxicity detection expert. Evaluate the toxicity level of the given text. "
            "Respond with ONLY one of these exact words: **High**, **Medium**, **Low**"
        ),
        max_consecutive_auto_reply=1,
    )
    creativity_annotator = autogen.AssistantAgent(
        name="CreativityAnnotator",
        llm_config=llm_config,
        system_message=(
            "You are a creativity assessment expert. Evaluate the creativity level of the given text. "
            "Respond with ONLY one of these exact words: **High**, **Medium**, **Low**"
        ),
        max_consecutive_auto_reply=1,
    )
    for input_file in input_files:
        data = load_input(input_file)
        results = []
        processed_count = 0
        for idx, item in enumerate(data):
            if max_items is not None and processed_count >= max_items:
                break
            result = process_item(item, user_agent, emotion_annotator, readability_annotator, toxicity_annotator, creativity_annotator)
            if result is not None:
                results.append(result)
                processed_count += 1
        input_filename = os.path.basename(input_file).replace('.json', '')
        output_filename = f"{input_filename}_annotated.json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
