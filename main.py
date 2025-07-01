from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, Filter, MatchValue, FieldCondition
from qdrant_client.http.models import PointStruct, VectorParams
import json
from typing import Any, Generator, Optional, Dict, List
import uuid
from tqdm import tqdm
from qdrant_client.http.models.models import ScoredPoint
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = QdrantClient(host="localhost", port=6333, timeout=60.0)
openai_client = OpenAI(api_key=openai_api_key)
COLLECTION_NAME='arxiv_papers'
app = FastAPI()

r'''
vectors_config={"size": 1536, "distance": "Cosine"}
file_path = r'C:\Users\Dominika\PycharmProjects\archive\ml-arxiv-embeddings.json'

def stream_json(file_path) -> Generator[Any, Any, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def count_lines(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def load_vectors_to_qdrant(file_path, batch_size=100) -> None:
    total_lines = count_lines(file_path)
    batch = []
    total_records = 0
    skipped = 0

    print(f"📂 Ładowanie danych z pliku: {file_path}")
    print(f"📊 Liczba rekordów: {total_lines}")

    for record in tqdm(stream_json(file_path), total=total_lines, desc="🔄 Przetwarzanie", unit="rekord"):
        embedding = record.get("embedding")
        if embedding is None:
            skipped += 1
            continue

        point_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=record["id"]))
        payload = {k: v for k, v in record.items() if k != "embedding"}

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        batch.append(point)

        if len(batch) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_records += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        total_records += len(batch)

    print("\n✅ Ładowanie zakończone.")
    print(f"✔️ Wczytano: {total_records} punktów.")
    print(f"⛔ Pominięto: {skipped} rekordów bez embeddingu.")

if not client.collection_exists(collection_name=f"{COLLECTION_NAME}"):
    client.create_collection(
        collection_name="arxiv_papers",
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
            )
    )
    load_vectors_to_qdrant(file_path, batch_size=500)
'''

class SearchRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5

class SearchResult(BaseModel):
    id: str
    payload: Dict
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

def get_embedding(query: str) -> list[float]:
    cleaned_query = query.replace("\n", " ")
    response = openai_client.embeddings.create(
        input=cleaned_query,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def extract_author_name(query: str) -> str | None:
    match = re.search(r"by\s+([A-Za-z\s\-]+)", query)
    if match:
        author = match.group(1).strip()
        return author
    return None

def search_similar(embedding: list[float], query: str, top_k: int = 3) -> list[str]:
    author = extract_author_name(query)
    must_filter = None
    if author:
        must_filter = Filter(must=[FieldCondition(key="authors", match=models.MatchText(text=author))])

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=must_filter
    ).points

    #return [p.payload["id"] for p in results if "id" in p.payload]
    return results

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    embedding = get_embedding(request.query)
    results = search_similar(embedding, request.query, top_k=request.top_n)
    search_results = [
        SearchResult(
            id=point.id,
            payload=point.payload,
            score=point.score
        )
        for point in results
    ]
    return SearchResponse(results=search_results)