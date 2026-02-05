import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/cleaned_jobs.csv")

embeddings = model.encode(df['text'].tolist())
dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

faiss.write_index(index, "models/job_index.faiss")
np.save("models/job_embeddings.npy", embeddings)

def recommend_jobs(query, top_k=5):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec), top_k)
    return df.iloc[I[0]][['Job Title', 'Job Description']]
