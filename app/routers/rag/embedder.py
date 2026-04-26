from sentence_transformers import SentenceTransformer



model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_chunks(chunks):

    embeddings = model.encode(chunks , normalize_embeddings=True)

    print(embeddings.shape)
    return embeddings



# if __name__ == "__main__":
#     chunks = ["This is a test sentence.", "Another test sentence."]

#     embeddings = embed_chunks(chunks)

#     print(embeddings)

