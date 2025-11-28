import requests
import json
import pandas as pd
from tqdm import tqdm



input_file_path = "Qualtrics_Annotations_B.csv"
df = pd.read_csv(input_file_path, sep='\t')


for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    text = row["text"]
    text_indice = row["text_indice"]
    r = requests.post(url="https://cental.uclouvain.be/iread4skills/annotator/", json={"raw_text": text})
    final_json = json.loads(r.text)
    with open('./outputs/%s.json' %text_indice, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)



