import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv
import os

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

load_dotenv()

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role
    content: str

    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)

class LLMService:
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize the LLM service with OpenAI client.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, will try to get from environment.
            default_system_prompt: Default system message to use when not explicitly provided
        """
        try:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not provided and not found in environment")
            
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.default_system_prompt = default_system_prompt
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise

    async def get_chat_completion(
        self,
        request: ChatCompletionRequest,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Get a chat completion from OpenAI API with retry logic.
        
        Args:
            request: Validated chat completion request
            retry_attempts: Number of retry attempts for recoverable errors
            retry_delay: Delay between retries in seconds
            
        Returns:
            OpenAI ChatCompletion response object
            
        Raises:
            ValueError: For validation errors
            APIError: For OpenAI API errors
        """
        attempt = 0
        last_error = None

        while attempt < retry_attempts:
            try:
                logger.info(f"Sending chat completion request with {len(request.messages)} messages")
                
                response = await self.client.chat.completions.create(
                    model=request.model,
                    messages=[msg.dict() for msg in request.messages],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                logger.info("Successfully received chat completion response")
                return response

            except RateLimitError as e:
                logger.warning(f"Rate limit reached (attempt {attempt + 1}/{retry_attempts})")
                last_error = e
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                
            except APIConnectionError as e:
                logger.warning(f"API connection error (attempt {attempt + 1}/{retry_attempts}): {str(e)}")
                last_error = e
                await asyncio.sleep(retry_delay)
                
            except APIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error during chat completion: {str(e)}")
                raise

            attempt += 1

        logger.error("Max retry attempts reached")
        raise last_error

    async def prompt(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simple method to send a prompt and get a response string.
        
        Args:
            prompt: The user's prompt
            system_prompt: Optional system prompt to override default
            **kwargs: Additional arguments to pass to chat completion (e.g., temperature, model)
            
        Returns:
            The assistant's response as a string
        """
        messages = [
            Message(
                role=Role.SYSTEM,
                content=system_prompt or self.default_system_prompt
            ),
            Message(role=Role.USER, content=prompt)
        ]
        
        request = ChatCompletionRequest(
            messages=messages,
            **kwargs
        )
        
        response = await self.get_chat_completion(request)
        return response.choices[0].message.content

# Example usage:
"""
async def example():
    llm = LLMService()
    
    # Simple usage
    response = await llm.prompt("Tell me a joke")
    print(response)
    
    # Advanced usage with custom parameters
    response = await llm.prompt(
        "Tell me a joke",
        system_prompt="You are a comedian.",
        temperature=0.9,
        model="gpt-4"
    )
    print(response)
"""