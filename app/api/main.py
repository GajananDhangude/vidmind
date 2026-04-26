from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.routers.map_reduce.get_summaries import final_summary
from app.routers.rag.qa_chain import query_rag
from app.routers.rag.vector_store import get_collection



class VideoRequest(BaseModel):
	url: str = Field(..., description="YouTube video URL")
	mode: str = Field(default="summary", description="summary or qa")
	question: str | None = Field(default=None, description="Question for QA mode")


app = FastAPI(title="AI Video Summarizer API", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
	return {"status": "ok"}


@app.post("/api/video")
def process_video(payload: VideoRequest) -> dict:
	mode = payload.mode.lower().strip()

	try:
		if mode == "summary":
			return final_summary(payload.url)

		if mode == "qa":
			if not payload.question:
				raise HTTPException(status_code=400, detail="question is required for qa mode")

			# Build/update collection for the requested video before querying.
			get_collection(payload.url)
			return query_rag(payload.question, payload.url)

		raise HTTPException(status_code=400, detail="mode must be 'summary' or 'qa'")
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
