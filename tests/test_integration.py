import pytest
from app.services.llm import LLMService
from main import app
from fastapi.testclient import TestClient
import os

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_openai_integration():
    """Test actual OpenAI API integration."""
    # Make sure we have an API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    
    llm_service = LLMService(api_key=api_key)
    
    # Test with a simple, deterministic prompt
    prompt = "What is 2+2? Answer with just the number."
    response = await llm_service.prompt(prompt)
    
    # Check if we got a reasonable response
    assert response.strip() in ["4", "Four", "four"], f"Unexpected response: {response}"

def test_chat_endpoint_integration(test_client):
    """Test the chat endpoint with real OpenAI API calls."""
    # Make sure we have an API key
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment")
    
    # Test with a simple prompt
    response = test_client.post(
        "/chat",
        json={"message": "What is 2+2? Answer with just the number."}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["error"] is None
    assert data["response"].strip() in ["4", "Four", "four"], f"Unexpected response: {data['response']}"

def test_full_conversation_flow(test_client):
    """Test a full conversation flow with real API calls."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment")
    
    # First, check health
    health_response = test_client.get("/health")
    assert health_response.status_code == 200
    
    # Then have a conversation
    conversation = [
        "Pretend you are a math teacher. What is 2+2?",
        "Now explain why that's correct.",
        "Give me a slightly harder math problem."
    ]
    
    previous_responses = []
    for message in conversation:
        response = test_client.post("/chat", json={"message": message})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["error"] is None
        # Make sure each response is different from previous ones
        assert data["response"] not in previous_responses
        previous_responses.append(data["response"])