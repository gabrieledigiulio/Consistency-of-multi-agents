import json
import datetime
import re
from collections import defaultdict, Counter

def main():
    input_file = "./Datasets/2020PoliceBrutality_comments.json"
    output_file = "./NewDatasets/police_top100_comments.json"
    months_to_filter = [1,2,3,4,5,6,7,8,9,10,11,12]
    url_pattern = re.compile(r'(http[s]?://|www\.)\S+')

    try:
        author_comment_count = Counter()
        comments_by_author = defaultdict(list)

        with open(input_file, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    comment = json.loads(line)
                    timestamp = comment.get("created_utc")
                    if timestamp is not None:
                        date = datetime.datetime.utcfromtimestamp(int(timestamp))
                        body = comment.get("body", "")
                        author = comment.get("author")
                        if (date.month in months_to_filter and author != "[deleted]" 
                                and body != "[deleted]" and not url_pattern.search(body)):
                            word_count = len(body.split())
                            if word_count > 4:
                                author_comment_count[author] += 1
                                comments_by_author[author].append(comment)
                except json.JSONDecodeError:
                    print("Errore nel parsing di una riga. Riga saltata.")

        top_authors = [author for author, count in author_comment_count.most_common(100) if count >= 10]

        filtered_data = []
        for author in top_authors:
            for comment in comments_by_author[author]:
                if len(comment['body'].split()) > 4:
                    timestamp = comment.get("created_utc")
                    date = datetime.datetime.utcfromtimestamp(int(timestamp))
                    filtered_data.append({
                        "author": comment.get("author"),
                        "body": comment.get("body"),
                        "id": comment.get("id"),
                        "subreddit": comment.get("subreddit"),
                        "link_id": comment.get("link_id"),
                        "created_utc": date.strftime("%d/%m/%Y")
                    })

        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(filtered_data, outfile, indent=4)

        print(f"File filtrato salvato in: {output_file}")

    except FileNotFoundError:
        print(f"Il file {input_file} non è stato trovato.")
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()
