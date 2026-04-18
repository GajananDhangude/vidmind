from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


summarizer_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature = 0.1
)


summarizer_llm2 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature = 0.1
)