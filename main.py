# main.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
import torch
from sklearn.cluster import KMeans

# --- FastAPI アプリ設定 ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 環境変数からOpenAI APIキーを取得 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

# --- 埋め込み取得関数 ---
def get_embeddings(texts):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": texts,
        "model": "text-embedding-3-small"
    }
    res = requests.post(url, headers=headers, json=data)
    res.raise_for_status()  # エラーがあれば例外を発生させる
    return [item["embedding"] for item in res.json()["data"]]

# --- コサイン類似度計算 ---
def cosine_similarity(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

# --- POSTエンドポイント: 質問を受け取り、代表質問を返す ---
@app.post("/cluster")
async def cluster(request: Request):
    body = await request.json()
    questions = body["questions"]
    if len(questions) < 1:
        return {"representative_questions": []}

    embeddings = get_embeddings(questions)
    X = np.array(embeddings)
    n_clusters = min(3, len(questions))  # 少ない質問数でも動作
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    reps = []
    for cid in range(n_clusters):
        idx = [i for i, lbl in enumerate(labels) if lbl == cid]
        cluster_vecs = [torch.tensor(embeddings[i]) for i in idx]
        centroid = torch.mean(torch.stack(cluster_vecs), dim=0)
        sims = [cosine_similarity(centroid, v).item() for v in cluster_vecs]
        max_i = int(np.argmax(sims))
        reps.append({
            "cluster": cid,
            "question": questions[idx[max_i]],
            "similarity": round(sims[max_i], 4)
        })

    return {"representative_questions": reps}
