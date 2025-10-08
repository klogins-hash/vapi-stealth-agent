"""Groq client implementation for the Microsoft Agent Framework."""

import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from groq import Groq, AsyncGroq
from pydantic import BaseModel
import json


class GroqConfig(BaseModel):
    """Configuration for Groq client."""
    api_key: str
    model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False


class GroqMessage(BaseModel):
    """Message format for Groq API."""
    role: str
    content: str
    name: Optional[str] = None


class GroqResponse(BaseModel):
    """Response from Groq API."""
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: str


class GroqClient:
    """Client for interacting with Groq API."""
    
    def __init__(self, config: Optional[GroqConfig] = None):
        """Initialize Groq client with configuration."""
        if config is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            
            config = GroqConfig(
                api_key=api_key,
                model=os.getenv("DEFAULT_GROQ_MODEL", "llama3-70b-8192"),
                temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
            )
        
        self.config = config
        self.client = Groq(api_key=config.api_key)
        self.async_client = AsyncGroq(api_key=config.api_key)
    
    def chat_completion(
        self,
        messages: List[GroqMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> GroqResponse:
        """Send chat completion request to Groq."""
        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Convert messages to Groq format
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return response  # Return generator for streaming
        
        return GroqResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump(),
            finish_reason=response.choices[0].finish_reason
        )
    
    async def async_chat_completion(
        self,
        messages: List[GroqMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> GroqResponse:
        """Send async chat completion request to Groq."""
        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Convert messages to Groq format
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            return response  # Return async generator for streaming
        
        return GroqResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump(),
            finish_reason=response.choices[0].finish_reason
        )
    
    async def stream_chat_completion(
        self,
        messages: List[GroqMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response from Groq."""
        model = model or self.config.model
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Convert messages to Groq format
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
