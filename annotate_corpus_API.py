import requests
import json
import pandas as pd
from tqdm import tqdm


def find_ratio(d):
    for key, value in d.items():

        # Check keys containing "ratio"
        if "ratio_subordination" in key.lower():
            if isinstance(value, (int, float)) and value > 1:
                return key, value

        # Recurse inside nested dictionaries
        if isinstance(value, dict):
            result = find_ratio(value)
            if result:
                return result

    return None

input_file_path = "Qualtrics_Annotations_B.csv"
df = pd.read_csv(input_file_path, sep='\t')



for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    text = row["text"]
    text_indice = row["text_indice"]
    r = requests.post(url="https://cental.uclouvain.be/iread4skills/annotator/", json={"raw_text": text})
    final_json = json.loads(r.text)
    #result = find_ratio(final_json)
    #if result:
    #    key, value = result
    #    print(f"This id has a ratio > 1: {text_indice} ({key} = {value})")
    with open('./outputs/%s.json' %text_indice, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)


#ids_ratio_bt_one = [1121, 1035, 128, 1455, 1093, 1879, 272]
