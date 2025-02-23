# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlmodel import SQLModel, Field, Session, create_engine, select
from app.utils.logging import setup_logging
from app.services.llm import LLMService
from dotenv import load_dotenv
from uvicorn import Config, Server
import logging
import os
import uvicorn
import socketio
import asyncio

load_dotenv()

setup_logging(log_file='app.log')

logger = logging.getLogger(__name__)

# Database initialization
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database.db")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=5,         # Maximum number of connections in pool
    max_overflow=10      # Maximum number of connections that can be created beyond pool_size
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

class Prompt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_name: str
    prompt: str

# Create Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# Initialize FastAPI app
app = FastAPI()

# Create ASGI app by wrapping Socket.IO
socket_app = socketio.ASGIApp(sio, app)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def chat_message(sid, data):
    logger.info(f"Received data from {sid}: {data}")
    llm_service = LLMService()

    # Process the received data
    response = await llm_service.prompt(data, system_prompt="You are a comedian.")

    # Emit response back to client
    await sio.emit('response_event', {'response': response})

    logger.info(f"Sent response to {sid}: {response}")

# Create tables on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Request model
class ChatRequest(BaseModel):
    message: str
    prompt_name: Optional[str]

# Response model
class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

# Create prompt
@app.post("/prompt")
async def create_prompt(prompt: Prompt, session: Session = Depends(get_session)):
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt

# Update prompt
@app.put("/prompt/{prompt_id}")
async def update_prompt(prompt_id: int, prompt: Prompt, session: Session = Depends(get_session)):
    db_prompt = session.get(Prompt, prompt_id)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    prompt.id = prompt_id
    session.add(prompt)
    session.commit()
    session.refresh(prompt)
    return prompt

# Get prompts
@app.get("/prompts")
async def get_prompts(session: Session = Depends(get_session)):
    prompts = session.exec(select(Prompt)).all()
    return prompts

# Delete prompt
@app.delete("/prompt/{prompt_id}")
async def delete_prompt(prompt_id: int, session: Session = Depends(get_session)):
    prompt = session.get(Prompt, prompt_id)
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    session.delete(prompt)
    session.commit()
    return {"message": "Prompt deleted"}

@app.post("/chat", response_model = ChatResponse)
async def chat_endpoint(request: ChatRequest, session: Session = Depends(get_session)):
    try:
        llm_service = LLMService()

        if request.prompt_name:
            # Get prompt from database
            system_prompt = session.execute(select(Prompt).where(Prompt.prompt_name == request.prompt_name))
            system_prompt = system_prompt.scalars().first()
            if not system_prompt:
                raise HTTPException(status_code=404, detail="Prompt not found")
            system_prompt = system_prompt.prompt

        response = await llm_service.prompt(request.message, system_prompt=system_prompt)
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

# Custom uvicorn server setup to handle both HTTP and WebSocket
def run_server():
    config = Config(
        app=socket_app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio"
    )
    server = Server(config)
    asyncio.run(server.serve())

if __name__ == "__main__":
    run_server()