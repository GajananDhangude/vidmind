from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


yt_api = YouTubeTranscriptApi()

def get_video_id(url:str)-> str:

    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL")
    
    query_params = parse_qs(parsed.query)

    video_id = query_params['v'][0]

    return video_id

def fetch_transcripts(url:str) -> str:

    video_id = get_video_id(url)

    transcript = yt_api.fetch(video_id)

    return " ".join(item.text for item in transcript)


# if __name__ == "__main__":

#     text = fetch_transcripts("https://www.youtube.com/watch?v=JTVx6i6MzVw&t=50s")

#     print(text)