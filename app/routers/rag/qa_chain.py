from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from app.routers.rag.vector_store import vector_store
from app.services.transcript import get_video_id
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def query_rag(query_text: str, url: str):
    video_id = get_video_id(url)

    retriever = vector_store.as_retriever(
        search_kwargs = {
            "k":3,
            "filter": {"video_id": video_id},
        }
    )

    # retrieved_docs = retriever.invoke(quert_text)


    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering based ONLY on the provided context.
Context:
{context}
Question:
{question}
Answer clearly and concisely:
""")
    

    rag_chain = (
        {
            "context":retriever | format_docs,
            "question":RunnablePassthrough()

        }
        |prompt
        |llm
        |StrOutputParser()
    )

    response = rag_chain.invoke(query_text)

    return {
        "response": response,
        # "contexts":[doc.page_content for doc in retrieved_docs],
    }


if __name__ == "__main__":

    query_text = "What is the main topic of the video?"

    responce = query_rag(query_text)

    print("Responce: ", responce)