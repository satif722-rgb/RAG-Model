import os
import json

# How many small chunks to merge into one big chunk
N = 10

INPUT_DIR = "jsons"
OUTPUT_DIR = "jsons_merged"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(INPUT_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get chunks safely
    chunks = data.get("chunks", [])
    if not chunks:
        print(f"Skipping {filename}: no 'chunks' found or it's empty.")
        continue

    new_chunks = []

    # Go in steps of N instead of math.ceil nonsense
    for i in range(0, len(chunks), N):
        chunk_group = chunks[i : i + N]

        merged_chunk = {
            # use the number of the first chunk in this group
            "number": chunk_group[0].get("number", i // N + 1),
            "title": chunk_group[0].get("title", ""),
            "start": chunk_group[0].get("start"),
            "end": chunk_group[-1].get("end"),
            "text": " ".join(c.get("text", "") for c in chunk_group),
        }

        new_chunks.append(merged_chunk)

    # Keep other top-level keys, just replace chunks
    new_data = data.copy()
    new_data["chunks"] = new_chunks

    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Processed: {filename} -> {out_path}")

print("All done.")
