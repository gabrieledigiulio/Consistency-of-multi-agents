import os
import json

def make_output_path(input_path, output_dir):
    base = os.path.basename(input_path)
    name, ext = os.path.splitext(base)
    new_name = f"{name}_c{ext}"
    return os.path.join(output_dir, new_name)

def main():
    input_files = [
        "risultati_finali/kitchen_vettoriNONpuliti_centroidi_generati_annotated.jsonl",
        "risultati_finali/kitchen_vettoriNONpuliti_medoidi_generati_annotated.jsonl",
        "risultati_finali/kitchen_vettoripuliti_centroidi_generati_annotated.jsonl",
        "risultati_finali/kitchen_vettoripuliti_medoidi_generati_annotated.jsonl",
        "risultati_finali/police_vettoriNONpuliti_centroidi_generati_annotated.jsonl",
        "risultati_finali/police_vettoriNONpuliti_medoidi_generati_annotated.jsonl",
        "risultati_finali/police_vettoripuliti_centroidi_generati_annotated.jsonl",
        "risultati_finali/police_vettoripuliti_medoidi_generati_annotated.jsonl"
    ]

    output_dir = "risultati_finali_cluster"
    os.makedirs(output_dir, exist_ok=True)

    for infile in input_files:
        if not os.path.isfile(infile):
            print(f"Attenzione: file non trovato: {infile}")
            continue

        outfile = make_output_path(infile, output_dir)

        with open(infile, "r", encoding="utf-8") as fin, \
             open(outfile, "w", encoding="utf-8") as fout:

            prev_traits = None
            cluster_idx = 0

            for line_number, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Errore JSON ({infile}, riga {line_number}): {e}")
                    continue

                traits_orig = record.get("traits_original", [])
                if not isinstance(traits_orig, list):
                    traits_orig = list(traits_orig)

                if prev_traits is None:
                    record["cluster"] = cluster_idx
                    prev_traits = traits_orig
                else:
                    if traits_orig == prev_traits:
                        record["cluster"] = cluster_idx
                    else:
                        cluster_idx += 1
                        record["cluster"] = cluster_idx
                        prev_traits = traits_orig

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Creato: {outfile}")

if __name__ == "__main__":
    main()
