import time
import json
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Helpers ------------------
def seconds_to_mmss(s):
    s = int(round(s))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"

def create_embedding(text_list, retry=1):
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=20)
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:
        if retry > 0:
            time.sleep(0.5)
            return create_embedding(text_list, retry - 1)
        raise

def inference(prompt, retry=1):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if retry > 0:
            time.sleep(0.5)
            return inference(prompt, retry - 1)
        raise

# ------------------ Load embeddings DB ------------------
df = joblib.load("embeddings.joblib")  # expect columns: embedding (np.array), title, number, start, end, text
# ensure embedding column is array-like
emb_matrix = np.vstack(df['embedding'].values)

# ------------------ Loop state ------------------
embedding_cache = {}   # query_text -> embedding vector
query_count = 0

print("Interactive RAG CLI. Ask a question about the course videos.")
print("Type 'q' or 'quit' to exit.\n")

while True:
    incoming_query = input("Ask a Question (or 'q' to quit): ").strip()
    if not incoming_query:
        continue
    if incoming_query.lower() in ("q", "quit"):
        print("Exiting. Goodbye.")
        break

    query_count += 1
    try:
        # --- get or create embedding (caching per session) ---
        if incoming_query in embedding_cache:
            question_embedding = embedding_cache[incoming_query]
        else:
            embeddings = create_embedding([incoming_query])
            question_embedding = embeddings[0]
            embedding_cache[incoming_query] = question_embedding

        # --- cosine similarity search ---
        similarities = cosine_similarity(emb_matrix, [question_embedding]).flatten()
        top_results = 3
        max_indx = similarities.argsort()[::-1][:top_results]
        new_df = df.iloc[max_indx].copy()
        new_df = new_df.reset_index(drop=True)

        # --- Build a small human-friendly preview for CLI output ---
        preview_lines = []
        for i, row in new_df.iterrows():
            title = row.get("title", "unknown title")
            number = row.get("number", "")
            start = row.get("start", 0)
            end = row.get("end", 0)
            text = row.get("text", "")[:200].replace("\n", " ")
            preview_lines.append({
                "rank": i + 1,
                "title": title,
                "number": number,
                "start_s": int(start),
                "end_s": int(end),
                "start": seconds_to_mmss(start),
                "end": seconds_to_mmss(end),
                "text": text,
                "score": float(similarities[max_indx[i]])
            })

        # print quick preview in CLI so user sees where it looked
        print("\n--- top chunks used for answer ---")
        for p in preview_lines:
            print(f"[{p['rank']}] Video: {p['title']} (#{p['number']})  {p['start']}–{p['end']}  score={p['score']:.4f}")
            print(f"    {p['text']}")
        print("----------------------------------\n")

        # --- Create prompt for the LLM (keep same schema as yours) ---
        chunks_json = new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")
        prompt = f'''I am teaching python in my python programming course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{chunks_json}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
Be concise and include timestamps in mm:ss format as well as the raw seconds.
'''
        # Save the prompt for logging (append)
        with open("prompt.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps({"query_num": query_count, "query": incoming_query, "prompt": prompt}) + "\n")

        # --- Call the model & handle response ---
        model_resp = inference(prompt)
        # your inference returns a dict; earlier you used ["response"]
        response_text = model_resp.get("response") if isinstance(model_resp, dict) else str(model_resp)

        # Print model response
        print("=== Model answer ===")
        print(response_text)
        print("====================\n")

        # Save the response (append)
        with open("response.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps({"query_num": query_count, "query": incoming_query, "response": response_text}) + "\n")

        # small pause to avoid hammering local server if user types quickly
        time.sleep(0.08)

    except Exception as e:
        print(f"Error while answering: {e}")
        print("You can retry or type 'q' to quit.\n")
