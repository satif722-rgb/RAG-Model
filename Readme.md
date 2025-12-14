# 🎥 RAG Model for YouTube Video Question Answering

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions about YouTube video content.

The system converts a video into audio, transcribes it into text, splits the text into chunks, generates vector embeddings, retrieves the most relevant chunks using semantic similarity, and finally generates answers using a **local LLM**.

---

## 🔍 What This Project Does

✔ Converts YouTube videos into audio  
✔ Transcribes audio into text  
✔ Chunks long text into smaller semantic pieces  
✔ Generates embeddings for each chunk  
✔ Stores embeddings in a vector database  
✔ Retrieves relevant chunks using cosine similarity  
✔ Uses an LLM to answer questions **grounded in video content**

This avoids hallucination and ensures answers are based on the actual video.

---

## 🧠 Why RAG?

Large Language Models alone do not “know” your video content.

**RAG = Retrieval + Generation**
- Retrieval finds the most relevant video segments
- Generation produces answers using only that retrieved context

This improves accuracy, relevance, and trust.

---

## 🛠 Tech Stack

### Language & Core Libraries
- Python
- pandas
- numpy
- scikit-learn
- joblib
- requests

### Models
- **Embedding Model:** `bge-m3`
- **LLM:** `LLaMA 3.x` (served locally via Ollama)

### Similarity Search
- Cosine Similarity (`sklearn.metrics.pairwise`)

### Storage
- Vector embeddings stored using `joblib`

---

## 🔄 Project Pipeline
YouTube Video
↓
Audio (MP3)
↓
Transcription (JSON)
↓
Text Chunking
↓
Embedding Generation (bge-m3)
↓
Vector Database (joblib)
↓
Cosine Similarity Search
↓
LLM Answer Generation (LLaMA)


---

## 📁 Project Structure



RAG-Model/
│
├── video_to_mp3.py # Convert YouTube video to MP3
├── mp3_to_json.py # Transcribe audio to JSON
├── preprocess_json.py # Chunk text & create embeddings
├── embeddings.joblib # Stored vector database
├── inference.py # Query + retrieval + LLM response
├── prompt.txt # Prompt template for LLM
├── requirements.txt
└── README.md


---

## 🧩 How It Works (Technical)

### 1️⃣ Text Embeddings
Each text chunk is converted into a dense vector using `bge-m3`.

```python
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

2️⃣ Semantic Retrieval

The user question is embedded and compared with stored vectors using cosine similarity.

similarities = cosine_similarity(
    np.vstack(df['embedding'].values),
    [question_embedding]
).flatten()


Top-K relevant chunks are selected.

3️⃣ Answer Generation

The retrieved chunks are passed to the LLM along with the question.

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return r.json()


The LLM answers only using the retrieved context.

🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start Ollama
ollama serve


Make sure these models are installed:

ollama pull bge-m3
ollama pull llama3.2

3️⃣ Run the Pipeline
python video_to_mp3.py
python mp3_to_json.py
python preprocess_json.py
python inference.py

🧪 Example Use Case

Question:

Where is Gradient Descent explained in the video?

System Output:

Retrieves relevant video segments

Uses timestamps + content

Generates a precise answer from the video itself

📌 Key Features

Fully local RAG system (no paid APIs)

Semantic search on video content

Timestamp-aware answers

Scalable chunk-based retrieval

Practical real-world RAG implementation

🔮 Future Improvements

Replace joblib with FAISS or ChromaDB

Add Streamlit / FastAPI UI

Support multiple videos

Highlight exact timestamps in answers

Add source citations in output

👤 Author

Siddiqui Atif Iqbal
Machine Learning & Data Science Learner

⭐ If you find this useful

Star the repository and feel free to fork or contribute.


---

### Final honest advice
This README **alone** upgrades your project quality a lot.  
Next level would be:
- Clean folder naming
- Add sample output screenshot
- Add `requirements.txt` properly

If you want, I can:
- Review your repo structure
- Make it resume-ready
- Write a LinkedIn post for this project

Say the word.
