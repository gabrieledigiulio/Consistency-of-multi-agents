import json
import os
from collections import defaultdict

def calculate_confusion_matrix(original_traits, detected_traits, feature):
    true_positive_condition = feature in original_traits
    predicted_positive_condition = feature in detected_traits
    if true_positive_condition and predicted_positive_condition:
        return int(1), int(0), int(0), int(0)
    elif not true_positive_condition and predicted_positive_condition:
        return int(0), int(1), int(0), int(0)
    elif true_positive_condition and not predicted_positive_condition:
        return int(0), int(0), int(1), int(0)
    else:
        return int(0), int(0), int(0), int(1)

def calculate_metrics(tp, fp, fn, tn):
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def analyze_clusters(cluster_feature_results, clusters_found, total_features, out_f):
    out_f.write("RISULTATI SPECIFICI PER CLUSTER:\n")
    out_f.write("-"*40 + "\n")
    for cluster_id in sorted(clusters_found):
        cluster_samples = sum([cluster_feature_results[cluster_id][feature]['tp'] + 
                             cluster_feature_results[cluster_id][feature]['fp'] + 
                             cluster_feature_results[cluster_id][feature]['fn'] + 
                             cluster_feature_results[cluster_id][feature]['tn'] 
                             for feature in total_features]) // len(total_features)
        out_f.write(f"\nCluster {cluster_id} (campioni: {cluster_samples}):\n")
        for feature in total_features:
            tp = int(cluster_feature_results[cluster_id][feature]['tp'])
            fp = int(cluster_feature_results[cluster_id][feature]['fp'])
            fn = int(cluster_feature_results[cluster_id][feature]['fn'])
            tn = int(cluster_feature_results[cluster_id][feature]['tn'])
            precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)
            out_f.write(f"  {feature}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (TP:{tp}, FP:{fp}, FN:{fn}, TN:{tn})\n")

def analyze_single_file(file_feature_results, total_features, out_f):
    out_f.write("\nRISULTATI SPECIFICI DEL FILE:\n")
    out_f.write("-"*30 + "\n")
    for feature in total_features:
        tp = int(file_feature_results[feature]['tp'])
        fp = int(file_feature_results[feature]['fp'])
        fn = int(file_feature_results[feature]['fn'])
        tn = int(file_feature_results[feature]['tn'])
        precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)
        out_f.write(f"\n{feature}:\n")
        out_f.write(f"  Matrice di Confusione - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
        out_f.write(f"  Precisione: {precision:.4f}\n")
        out_f.write(f"  Recall: {recall:.4f}\n") 
        out_f.write(f"  F1-Score: {f1:.4f}\n")

def analyze_global_results(feature_results, total_features, out_f):
    out_f.write("\n\n" + "="*60 + "\n")
    out_f.write("RISULTATI GLOBALI ATTRAVERSO TUTTI I FILE\n")
    out_f.write("="*60 + "\n")
    precision_list = []
    recall_list = []
    f1_list = []
    for feature in total_features:
        tp = int(feature_results[feature]['tp'])
        fp = int(feature_results[feature]['fp'])
        fn = int(feature_results[feature]['fn'])
        tn = int(feature_results[feature]['tn'])
        precision, recall, f1 = calculate_metrics(tp, fp, fn, tn)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        out_f.write(f"\n{feature}:\n")
        out_f.write(f"  Matrice di Confusione - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
        out_f.write(f"  Precisione: {precision:.4f}\n")
        out_f.write(f"  Recall: {recall:.4f}\n")
        out_f.write(f"  F1-Score: {f1:.4f}\n")
    precision_mean = sum(precision_list) / len(precision_list) if precision_list else 0
    recall_mean = sum(recall_list) / len(recall_list) if recall_list else 0
    f1_mean = sum(f1_list) / len(f1_list) if f1_list else 0
    out_f.write(f"\n\nSTATISTICHE RIASSUNTIVE:\n")
    out_f.write(f"Precisione Media: {precision_mean:.4f}\n")
    out_f.write(f"Recall Medio: {recall_mean:.4f}\n") 
    out_f.write(f"F1-Score Medio: {f1_mean:.4f}\n")
    if f1_list:
        best_f1_idx = f1_list.index(max(f1_list))
        worst_f1_idx = f1_list.index(min(f1_list))
        out_f.write(f"Miglior F1-Score: {max(f1_list):.4f} ({total_features[best_f1_idx]})\n")
        out_f.write(f"Peggior F1-Score: {min(f1_list):.4f} ({total_features[worst_f1_idx]})\n")

def process_files(input_files, total_features, out_f):
    feature_results = {}
    for feature in total_features:
        feature_results[feature] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Attenzione: File {file_path} non trovato, saltando...")
            out_f.write(f"Attenzione: File {file_path} non trovato, saltando...\n")
            continue
        print(f"Processando file: {file_path}")
        out_f.write(f"\nProcessando file: {file_path}\n")
        out_f.write("="*50 + "\n")
        file_feature_results = {}
        cluster_feature_results = defaultdict(lambda: {})
        for feature in total_features:
            file_feature_results[feature] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            clusters_found = set()
            for line in f:
                data = json.loads(line.strip())
                line_count += 1
                original_traits = data.get('traits_original', [])
                detected_traits = data.get('traits_detected', [])
                cluster_id = data.get('cluster', 'unknown')
                clusters_found.add(cluster_id)
                if cluster_id not in cluster_feature_results:
                    cluster_feature_results[cluster_id] = {}
                    for feature in total_features:
                        cluster_feature_results[cluster_id][feature] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                for feature in total_features:
                    tp, fp, fn, tn = calculate_confusion_matrix(original_traits, detected_traits, feature)
                    file_feature_results[feature]['tp'] += tp
                    file_feature_results[feature]['fp'] += fp  
                    file_feature_results[feature]['fn'] += fn
                    file_feature_results[feature]['tn'] += tn
                    cluster_feature_results[cluster_id][feature]['tp'] += tp
                    cluster_feature_results[cluster_id][feature]['fp'] += fp
                    cluster_feature_results[cluster_id][feature]['fn'] += fn
                    cluster_feature_results[cluster_id][feature]['tn'] += tn
                    feature_results[feature]['tp'] += tp
                    feature_results[feature]['fp'] += fp
                    feature_results[feature]['fn'] += fn
                    feature_results[feature]['tn'] += tn
        out_f.write(f"Campioni totali processati: {line_count}\n")
        out_f.write(f"Cluster trovati: {sorted(list(clusters_found))}\n\n")
        analyze_clusters(cluster_feature_results, clusters_found, total_features, out_f)
        analyze_single_file(file_feature_results, total_features, out_f)
    analyze_global_results(feature_results, total_features, out_f)

def main():
    output_txt = "analysis_results.txt"
    total_features = [
        "Positive & Uplifting", "Calm & Curious", "Agitated/Reactive", "Downcast & Vulnerable", "Reflective & Anxious",
        "toxicity_low", "toxicity_medium", "toxicity_high",
        "creativity_low", "creativity_medium", "creativity_high",
        "readability_excellent", "readability_good", "readability_average", "readability_poor", "readability_very_poor"
    ]
    input_files = [
        "risultati_finali_cluster/kitchen_vettoriNONpuliti_centroidi_generati_annotated_c.jsonl",
        "risultati_finali_cluster/kitchen_vettoriNONpuliti_medoidi_generati_annotated_c.jsonl", 
        "risultati_finali_cluster/kitchen_vettoripuliti_centroidi_generati_annotated_c.jsonl",
        "risultati_finali_cluster/kitchen_vettoripuliti_medoidi_generati_annotated_c.jsonl",
        "risultati_finali_cluster/police_vettoriNONpuliti_centroidi_generati_annotated_c.jsonl",
        "risultati_finali_cluster/police_vettoriNONpuliti_medoidi_generati_annotated_c.jsonl",
        "risultati_finali_cluster/police_vettoripuliti_centroidi_generati_annotated_c.jsonl", 
        "risultati_finali_cluster/police_vettoripuliti_medoidi_generati_annotated_c.jsonl"
    ]
    out_f = open(output_txt, "w", encoding="utf-8")
    out_f.write("RISULTATI DELL'ANALISI DELLE CARATTERISTICHE\n")
    out_f.write("="*60 + "\n")
    out_f.write(f"Caratteristiche totali analizzate: {len(total_features)}\n")
    out_f.write(f"File totali da processare: {len(input_files)}\n\n")
    process_files(input_files, total_features, out_f)
    out_f.close()
    print(f"Analisi completata! Risultati salvati in {output_txt}")

if __name__ == "__main__":
    main()
