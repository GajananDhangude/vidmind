from app.services.llm import summarizer_llm2
from app.routers.map_reduce.map_chunks import map_chunks

def final_summary(url:str) -> dict:

    chunks = map_chunks(url)

    combined = "\n\n".join(chunks)

    prompt = f"""
Summarize the following content into a "Key Takeaways" format. 
Follow this exact structure strictly:

1. A header line: "Learning [Topic Name]: Here are the key takeaways from the video:"
2. Three bullet points that cover:
    - The traditional/legacy approach mentioned.
    - The modern/AI-driven solution.
    - The specific technical benefits (e.g., efficiency, generalization).
3. A concluding sentence starting with "This foundational concept is crucial for..."

Style: Use professional, technical language. Avoid introductory filler.

Content to summarize: {combined}
"""

    response = summarizer_llm2.invoke([
        {'role': 'user', 'content': prompt}
    ])

    res = response.content

    return {
        "summary": res
    }


# if __name__ == "__main__":

#     result = final_summary("https://www.youtube.com/watch?v=JTVx6i6MzVw&t=50s")

#     print(result["summary"])