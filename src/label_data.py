import os
import time
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-2.5-flash')

read_file_path = "../sample_data/processed/genius-clean-with-title-artist-5000.csv"
output_path = "../sample_data/processed/Labeled_genius-clean-with-title-artist-5000.csv"
df = pd.read_csv(read_file_path)

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)


QUERIES = [
    "love and heartbreak",
    "party and dance",
    "lonely rain night",
    "quiet fog morning",
    "zero-gravity space exploration",
    "dancing in the club with friends until the sun comes up and forgetting all problems",
    "driving down the highway with windows down feeling free and wild",
    "hanging out with best friends making memories that will last forever and laughing together",
    "standing up against the world and fighting for what is right despite the odds",
    "looking into your eyes and realizing you are the only one I want to spend my life with",
]
query_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(QUERIES)])

if not os.path.exists(output_path):
    cols = list(df.columns) + [f'q{i+1}' for i in range(10)]
    pd.DataFrame(columns=cols).to_csv(output_path, index=False)

processed_count = len(pd.read_csv(output_path))
print(f"Resuming from row {processed_count}...")

for i, row in tqdm(df.iloc[processed_count:].iterrows(), total=len(df) - processed_count):
    prompt = f"""
    Analyze lyrics. Return ONLY a JSON list of 10 integers (1=relevant, 0=not) for these themes:
    {query_str}

    Song: {row['title']} - {row['artist']}
    Lyrics: {str(row['lyrics'])[:5000]}
    """

    try:
        response = model.generate_content(prompt)
        scores = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        save_row = row.to_dict()
        for idx, score in enumerate(scores):
            save_row[f'q{idx + 1}'] = score

        pd.DataFrame([save_row]).to_csv(output_path, mode='a', header=False, index=False)

    except Exception as e:
        print(f"Error row {i}: {e}")







