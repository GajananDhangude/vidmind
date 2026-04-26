from langchain_community.vectorstores import Chroma
import os
from pathlib import Path
from app.services.chunker import split_text
from app.services.transcript import fetch_transcripts, get_video_id

from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[3]
VECTOR_STORE_PATH = str(BASE_DIR / "storage" / "vector_store")
COLLECTION_NAME = "rag_system"

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)


vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_STORE_PATH
)


def get_collection(url:str):
    video_id = get_video_id(url)

    texts = fetch_transcripts(url)
    chunks = split_text(texts, video_id=video_id, with_metadata=True)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    chunk_texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "video_id": chunk["video_id"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    existing = vector_store.get(where={"video_id": video_id})
    existing_ids = existing.get("ids", [])
    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_texts(
        ids=chunk_ids,
        texts=chunk_texts,
        metadatas=metadatas,
    )

    return vector_store


# if __name__ == "__main__":

#     url = "https://www.youtube.com/watch?v=JTVx6i6MzVw&t=50s"

#     collection = create_collection(url)

#     print("Collection created: ", collection.name)
