import json
import re
from collections import Counter
from datetime import datetime
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

def main():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    with open('./Datasets/police_top100_comments.json', 'r') as f:
        data = json.load(f)

    total_comments = len(data)

    authors_counter = Counter([comment['author'] for comment in data])
    sorted_authors = authors_counter.most_common()

    total_words = sum(len(comment['body'].split()) for comment in data)
    average_words_per_comment = total_words / total_comments

    comments3words = sum(1 for comment in data if len(comment['body'].split()) > 3)
    comments4words = sum(1 for comment in data if len(comment['body'].split()) > 4)

    comments_per_day = Counter()
    for comment in data:
        date_str = comment['created_utc']
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y").date()
            comments_per_day[date] += 1
        except ValueError:
            pass
    sorted_comments_per_day = sorted(comments_per_day.items())

    comments_per_post = Counter([comment['link_id'] for comment in data])
    sorted_comments_per_post = comments_per_post.most_common()

    total_posts = len(comments_per_post)
    average_comments_per_post = total_comments / total_posts

    word_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    positive_comments = 0
    negative_comments = 0
    neutral_comments = 0
    subjective_comments = 0
    objective_comments = 0
    total_subjectivity = 0.0

    users_with_more_than_10_comments_and_4_words = Counter()
    comments_matching_criteria = 0

    for comment in data:
        words = re.findall(r'\w+', comment['body'].lower())
        filtered_words = [word for word in words if word not in stop_words]
        word_counter.update(filtered_words)
        bigram_counter.update(zip(filtered_words, filtered_words[1:]))
        trigram_counter.update(zip(filtered_words, filtered_words[1:], filtered_words[2:]))

        blob = TextBlob(comment['body'])
        sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if sentiment > 0:
            positive_comments += 1
        elif sentiment < 0:
            negative_comments += 1
        else:
            neutral_comments += 1

        total_subjectivity += subjectivity
        if subjectivity > 0.5:
            subjective_comments += 1
        else:
            objective_comments += 1

        if len(comment['body'].split()) > 4:
            users_with_more_than_10_comments_and_4_words[comment['author']] += 1
            if authors_counter[comment['author']] > 10:
                comments_matching_criteria += 1

    average_subjectivity = total_subjectivity / total_comments if total_comments > 0 else 0

    author10comments = sum(1 for author, count in authors_counter.items() if count > 10)
    author10comments4words = sum(1 for author, count in users_with_more_than_10_comments_and_4_words.items() if count > 10)

    top_10_words = word_counter.most_common(10)
    top_10_bigrams = bigram_counter.most_common(10)
    top_10_trigrams = trigram_counter.most_common(10)

    with open('kitchen_top100_comments.txt', 'w') as f:
        f.write(f"Numero totale di commenti: {total_comments}\n")
        f.write(f"Numero medio di parole per commento: {average_words_per_comment:.2f}\n")
        f.write(f"Numero di commenti con più di 3 parole: {comments3words}\n")
        f.write(f"Numero di commenti con più di 4 parole: {comments4words}\n")
        f.write(f"\nNumero di post unici: {total_posts}\n")
        f.write(f"Numero medio di commenti per post: {average_comments_per_post:.2f}\n")
        f.write(f"\nNumero di utenti che hanno scritto più di 10 commenti: {author10comments}\n")
        f.write(f"Numero di utenti che hanno scritto più di 10 commenti di una lunghezza di più di 4 parole: {author10comments4words}\n")
        f.write(f"Numero di commenti che rispecchiano il parametro (autori e lunghezza parole): {comments_matching_criteria}\n")
        f.write("\nRisultati della sentiment analysis:\n")
        f.write(f"Numero di commenti positivi: {positive_comments}\n")
        f.write(f"Numero di commenti negativi: {negative_comments}\n")
        f.write(f"Numero di commenti neutri: {neutral_comments}\n")
        f.write("\nRisultati dell'analisi di soggettività:\n")
        f.write(f"Numero di commenti soggettivi: {subjective_comments}\n")
        f.write(f"Numero di commenti oggettivi: {objective_comments}\n")
        f.write(f"Soggettività media dei commenti: {average_subjectivity:.2f}\n")
        f.write("\nLe 10 parole più frequenti nei commenti:\n")
        for word, count in top_10_words:
            f.write(f"{word}: {count} occorrenze\n")
        f.write("\nI 10 bigrammi più frequenti nei commenti:\n")
        for bigram, count in top_10_bigrams:
            f.write(f"{' '.join(bigram)}: {count} occorrenze\n")
        f.write("\nI 10 trigrammi più frequenti nei commenti:\n")
        for trigram, count in top_10_trigrams:
            f.write(f"{' '.join(trigram)}: {count} occorrenze\n")
        f.write("\nLista di utenti in ordine di numero di commenti:\n")
        for author, count in sorted_authors:
            f.write(f"{author}: {count} commenti\n")
        f.write("\nNumero di commenti per giorno:\n")
        for date, count in sorted_comments_per_day:
            f.write(f"{date}: {count} commenti\n")
        f.write("\nNumero di commenti per ciascun post (in ordine dal più grande al più piccolo):\n")
        for link_id, count in sorted_comments_per_post:
            f.write(f"{link_id}: {count} commenti\n")

    print("Analisi completata.")

if __name__ == "__main__":
    main()
