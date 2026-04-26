from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=0
)


def generate_id(video_id: str, chunk_index: int, text: str) -> str:
    payload = f"{video_id}:{chunk_index}:{text}"
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def split_text(texts: str, video_id: str | None = None, with_metadata: bool = False):
    chunks = text_splitter.split_text(texts)

    if not with_metadata:
        print("Number of chunks: ", len(chunks))
        return chunks

    if not video_id:
        raise ValueError("video_id is required when with_metadata=True")

    enriched_chunks = []
    for index, chunk in enumerate(chunks):
        enriched_chunks.append(
            {
                "text": chunk,
                "chunk_id": generate_id(video_id, index, chunk),
                "video_id": video_id,
                "chunk_index": index,
            }
        )

    print("Number of chunks: ", len(chunks))
    return enriched_chunks



if __name__ == "__main__":
    sample_text = "This is a sample text to demonstrate the chunking process. " * 100

    chunks = split_text(sample_text)

    print(chunks)

