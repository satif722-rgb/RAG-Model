🔍 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about YouTube video content.
https://youtu.be/vLqTf2b6GZw?si=s8g28ArYdKdKH7ts this is the link of the video it is apna college python tutorial.

The system:

Converts YouTube videos into audio

Transcribes audio into text

Splits the text into meaningful chunks

Generates vector embeddings for each chunk

Stores embeddings in a vector database

Retrieves the most relevant chunks using cosine similarity

Uses a Large Language Model (LLM) to generate accurate answers

This enables context-aware question answering directly from video content.

🧠 Why RAG?

Large Language Models alone can hallucinate or miss video-specific details.
RAG solves this by:

Retrieving relevant video content first

Then generating answers grounded in that content

🛠️ Technologies Used
🔹 Language & Libraries

Python

Pandas

NumPy

scikit-learn

joblib

requests

🔹 Embedding Model

bge-m3
Used to convert text chunks into dense vector embeddings.

🔹 Similarity Search

Cosine Similarity (from sklearn.metrics.pairwise)

🔹 LLM

LLaMA 3.2 (served locally using Ollama API)

🔹 Vector Storage

Embeddings stored using joblib for fast loading and retrieval.

🔄 Project Workflow
1️⃣ YouTube Video → Audio

The YouTube video is converted into an audio file.

2️⃣ Audio → Text

Audio is transcribed into text using a speech-to-text process.

3️⃣ Text Chunking

The transcription is split into smaller chunks to preserve semantic meaning.

4️⃣ Embedding Creation

Each chunk is converted into a vector embedding using the bge-m3 model.

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

5️⃣ Vector Database

All embeddings are stored along with metadata (title, timestamps, text).

df = joblib.load("embeddings.joblib")

6️⃣ Retrieval Using Cosine Similarity

User questions are embedded and matched against stored vectors.

similarities = cosine_similarity(
    np.vstack(df['embedding'].values),
    [question_embedding]
).flatten()


Top relevant chunks are selected.

7️⃣ Answer Generation (RAG)

Retrieved chunks are passed to the LLM to generate a grounded answer.

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return r.json()

🧪 Example Use Case

Ask: “Where is gradient descent explained in the video?”

System:

Retrieves relevant timestamps

Responds with video number + exact time

Prevents unrelated questions

🚀 Key Features

✔️ Local LLM (No paid API)
✔️ Semantic search on video content
✔️ Timestamp-aware answers
✔️ Scalable chunk-based retrieval
✔️ Practical RAG implementation

📂 Repository Structure (High Level)
RAG-Model/
│
├── audio_processing/           
├── text_chunking/
├── embedding_generation/
├── embeddings.joblib
├── inference.py
├── requirements.txt
└── README.md

🎯 Future Improvements

Replace joblib with FAISS / Chroma

Add UI (Streamlit / FastAPI)

Support multiple videos

Add citation highlighting in answers

👨‍💻 Author

Siddiqui Atif Iqbal
Data Science & Machine Learning Enthusiast
