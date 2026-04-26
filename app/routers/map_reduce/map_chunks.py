from tqdm import tqdm
from app.services.chunker import split_text
from app.services.llm import summarizer_llm
from app.services.transcript import fetch_transcripts

def map_chunks(url:str) -> str:

    transcript = fetch_transcripts(url)
    chunks = split_text(transcript)

    summary = []

    for i , chunk in enumerate(tqdm(chunks)):

        prompt = f"""Summarize this transcript segment concisely. Capture the main points, key ideas, and any important quotes in one lines.
    Segment {i+1}/{len(chunks)}:
    {chunk}
    Summary:"""

        response = summarizer_llm.invoke([
            {'role':'user' , 'content':prompt}
        ])
    
        summary.append(response.content)

    return summary        
    