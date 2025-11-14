# scripts/build_index.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dim
CHUNKS_FILE = "data/chunks.jsonl"
INDEX_OUT = "embeddings/faiss.index"
META_OUT = "embeddings/metadata.npy"

model = SentenceTransformer(MODEL_NAME)
texts = []
metas = []
ids = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ids.append(obj["id"])
        texts.append(obj["text"])
        metas.append(obj["meta"])

# compute embeddings in batches
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")  # faiss needs float32

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_OUT)
np.save(META_OUT, np.array(ids))
# optionally save texts or a mapping id->text
import pickle
with open("embeddings/id2text.pkl", "wb") as f:
    pickle.dump(texts, f)
