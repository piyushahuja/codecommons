# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.utils.logging import setup_logging
from app.services.llm import LLMService
import logging

setup_logging(log_file='app.log')

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    message: str

# Response model
class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        llm_service = LLMService()
        response = await llm_service.prompt(request.message, system_prompt=f"You are a math teacher conducting a tutorial on {request.message}. Come up with a list of concepts that you have to check that the student understands. Then come up with a few practice problems to check their understanding." )
        logger.info(f"Response: {response}")

        return ChatResponse(
            response=response,
            error=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Basic health check
@app.get("/health")
async def health_check():
    logger.info("Health check")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)