import autogen
import json
import os
import re

def main():
    output_folder = "results"

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

    u1 = autogen.AssistantAgent(
        name="Agent",
        llm_config=llm_config_gemma, 
        system_message="You are role-playing as an online user. Act as requested by the Handler.",
        max_consecutive_auto_reply=1,
    )

    u2 = autogen.AssistantAgent(
        name="Handler",
        llm_config=llm_config_gemma, 
        system_message="You are the Handler that specifies the actions the user has to perform. After having read the user's response to your request, act as follows: 1. Consider a model capable of detecting a broad range of emotions as encoded in the GoEmotions taxonomy, which includes: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, and trust. 2. Detect the emotions in the user's texts. 3. Format your response strictly as a Python list of emotions, with each emotion enclosed in double asterisks: e.g., ['**admiration**', '**joy**'].  - The emotions can be more than one. - **You must not use any word that is not on the provided GoEmotions list.**  - **The response must only contain the list of emotions, without explanations, descriptions, or any additional text.** - **Do not use synonyms, alternative spellings, or add any additional commentary.** Use only the exact words provided in the GoEmotions list. Failure to follow these instructions will result in your response being invalid. You are expected to produce only the list of emotions as described. ",
        max_consecutive_auto_reply=1,
    )

    u3 = autogen.AssistantAgent(
        name="ReadabilityAnnotator",
        llm_config=llm_config_gemma,
        system_message="You are the Readability Annotator that evaluates how easy it is to read the user's text.\nAfter reading the user's response, act as follows:\n - Assess the readability of the text, considering clarity, structure, and ease of understanding.\n - Format your response as one of the following words enclosed in **: '**Excellent**', '**Good**', '**Average**', '**Poor**', '**Very poor**'. You must not use any word outside of this list.",
        max_consecutive_auto_reply=1,
    )

    u4 = autogen.AssistantAgent(
        name="ToxicityAnnotator",
        llm_config=llm_config_gemma,
        system_message="You are the Toxicity Annotator that detects toxic language in the user's text.\nAfter reading the user's response, act as follows:\n - Analyze the text for offensive, aggressive, or harmful language.\n - Format your response as '**High**', '**Medium**', or '**Low**'. You must not use any word outside of this list.",
        max_consecutive_auto_reply=1,
    )

    u5 = autogen.AssistantAgent(
        name="CreativityAnnotator",
        llm_config=llm_config_gemma,
        system_message="You are the Creativity Annotator that evaluates the creativity of the user's text.\nAfter reading the user's response, act as follows:\n - Analyze the text for originality, innovation, and expressiveness.\n - Format your response as '**High**', '**Medium**', or '**Low**'. You must not use any word outside of this list.",
        max_consecutive_auto_reply=1,
    )

    u2.initiate_chat(
        u1,
        message="I am the Handler. Write a tweet discussing a topic of kitchen.",
        silent=True,
        max_round=1,
    )

    emotion_eval = u2.chat_messages[u1][-1]["content"].lower()
    post_text = u2.chat_messages[u1][-2]["content"]

    u3.initiate_chat(
        u1,
        message=f'I am the Handler. repeat this text word by word, you cannot add any word just repeat this text: {post_text}',
        silent=True,
        max_round=1,
    )

    u4.initiate_chat(
        u1,
        message=f'I am the Handler. repeat this text word by word, you cannot add any word just repeat this text: {post_text}',
        silent=True,
        max_round=1,
    )

    u5.initiate_chat(
        u1,
        message=f'I am the Handler. repeat this text word by word, you cannot add any word just repeat this text: {post_text}',
        silent=True,
        max_round=1,
    )

    readability_eval = u3.chat_messages[u1][-1]["content"].strip().lower()
    post_text2 = u3.chat_messages[u1][-2]["content"]
    toxicity_eval = u4.chat_messages[u1][-1]["content"].strip().lower()
    post_text3 = u4.chat_messages[u1][-2]["content"]
    creativity_eval = u5.chat_messages[u1][-1]["content"].strip().lower()
    post_text4 = u5.chat_messages[u1][-2]["content"]

    emotions_pattern = r"\*\*['\"]?(.*?)['\"]?\*\*"
    detected_emotions_raw = re.findall(emotions_pattern, emotion_eval)
    emotions_list = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "trust", 
        "boredom", "envy", "guilt"
    ]
    detected_emotions = [emotion.strip() for emotion in detected_emotions_raw if emotion.strip() in emotions_list]
    readability_list = [readability for readability in ["excellent", "good", "average", "poor", "very poor"] if f"**{readability}**" in readability_eval]
    toxicity_list = [toxicity for toxicity in ["high", "medium", "low"] if f"**{toxicity}**" in toxicity_eval]
    creativity_list = [creativity for creativity in ["high", "medium", "low"] if f"**{creativity}**" in creativity_eval]

    print("Emozioni rilevate:", detected_emotions)

    output_path = os.path.join(output_folder, "valutazioni.json")

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as json_file:
            try:
                data = json.load(json_file)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    results = { 
        "testo": post_text,
        "emozioni": detected_emotions,  
        "tossicità": toxicity_list,
        "leggibilità": readability_list,
        "creatività": creativity_list
    }
    data.append(results)

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print("Risultati salvati nel file JSON.")

if __name__ == "__main__":
    main()
