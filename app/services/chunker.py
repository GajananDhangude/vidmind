from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=0
)

def split_text(texts):

    chunks = text_splitter.split_text(texts)
    print("Number of chunks: ", len(chunks))
    return chunks


