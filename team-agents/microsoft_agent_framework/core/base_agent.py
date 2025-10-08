"""Base agent implementation for the Microsoft Agent Framework."""

from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from abc import ABC, abstractmethod
from pydantic import BaseModel
import asyncio
import json
from datetime import datetime

from .groq_client import GroqClient, GroqMessage, GroqResponse
from .agent_thread import AgentThread, ThreadMessage
from .context_provider import ContextProvider, InMemoryContextProvider


class AgentConfig(BaseModel):
    """Configuration for an AI agent."""
    name: str
    instructions: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: List[str] = []
    metadata: Dict[str, Any] = {}


class AgentRunResponse(BaseModel):
    """Response from agent execution."""
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: str
    metadata: Dict[str, Any] = {}
    
    def __str__(self) -> str:
        """Return the content as string representation."""
        return self.content
    
    @property
    def text(self) -> str:
        """Get the text content."""
        return self.content


class AgentRunResponseUpdate(BaseModel):
    """Streaming update from agent execution."""
    content: str
    is_complete: bool = False
    metadata: Dict[str, Any] = {}
    
    def __str__(self) -> str:
        """Return the content as string representation.""" 
        return self.content
    
    @property
    def text(self) -> str:
        """Get the text content."""
        return self.content


class BaseAgent(ABC):
    """Base class for AI agents in the Microsoft Agent Framework."""
    
    def __init__(
        self,
        config: AgentConfig,
        groq_client: GroqClient,
        context_provider: Optional[ContextProvider] = None,
        thread: Optional[AgentThread] = None
    ):
        """Initialize the base agent."""
        self.config = config
        self.groq_client = groq_client
        self.context_provider = context_provider or InMemoryContextProvider()
        self.thread = thread or AgentThread()
        self.tools: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Add system message with instructions
        if config.instructions:
            self.thread.add_system_message(config.instructions)
    
    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Add a tool function to the agent."""
        self.tools[name] = func
        # Store tool metadata
        if not hasattr(func, '_tool_metadata'):
            func._tool_metadata = {"name": name, "description": description}
    
    def add_middleware(self, middleware_func: Callable) -> None:
        """Add middleware function to intercept agent actions."""
        self.middleware.append(middleware_func)
    
    async def _apply_middleware(self, action: str, data: Any) -> Any:
        """Apply middleware functions to agent actions."""
        for middleware in self.middleware:
            data = await middleware(action, data)
        return data
    
    async def _get_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant context for the query."""
        return await self.context_provider.get_context(query, limit=5)
    
    async def _prepare_messages(self, user_input: str) -> List[GroqMessage]:
        """Prepare messages for the LLM including context and history."""
        messages = []
        
        # Get conversation history
        history = self.thread.get_conversation_history()
        
        # Convert to GroqMessage format
        for msg in history:
            messages.append(GroqMessage(role=msg["role"], content=msg["content"]))
        
        # Add user message
        user_message = GroqMessage(role="user", content=user_input)
        messages.append(user_message)
        
        # Get relevant context
        context_items = await self._get_relevant_context(user_input)
        if context_items:
            context_content = "Relevant context:\n"
            for item in context_items:
                context_content += f"- {item['content']}\n"
            
            # Insert context before the user message
            context_message = GroqMessage(role="system", content=context_content)
            messages.insert(-1, context_message)
        
        return messages
    
    async def run_async(self, user_input: str) -> AgentRunResponse:
        """Run the agent with user input and return response."""
        # Add user message to thread
        self.thread.add_user_message(user_input)
        
        # Apply middleware
        user_input = await self._apply_middleware("user_input", user_input)
        
        # Prepare messages
        messages = await self._prepare_messages(user_input)
        
        # Get response from Groq
        response = await self.groq_client.async_chat_completion(
            messages=messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Apply middleware to response
        response = await self._apply_middleware("response", response)
        
        # Add assistant message to thread
        self.thread.add_assistant_message(response.content)
        
        # Store interaction in context
        await self.context_provider.add_context(
            f"User: {user_input}\nAssistant: {response.content}",
            metadata={"timestamp": datetime.now().isoformat(), "agent": self.config.name}
        )
        
        return AgentRunResponse(
            content=response.content,
            model=response.model,
            usage=response.usage,
            finish_reason=response.finish_reason,
            metadata={"agent_name": self.config.name}
        )
    
    async def run_streaming_async(self, user_input: str) -> AsyncGenerator[AgentRunResponseUpdate, None]:
        """Run the agent with streaming response."""
        # Add user message to thread
        self.thread.add_user_message(user_input)
        
        # Apply middleware
        user_input = await self._apply_middleware("user_input", user_input)
        
        # Prepare messages
        messages = await self._prepare_messages(user_input)
        
        # Stream response from Groq
        full_content = ""
        async for chunk in self.groq_client.stream_chat_completion(
            messages=messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        ):
            full_content += chunk
            
            # Apply middleware to chunk
            chunk = await self._apply_middleware("stream_chunk", chunk)
            
            yield AgentRunResponseUpdate(
                content=chunk,
                is_complete=False,
                metadata={"agent_name": self.config.name}
            )
        
        # Final update
        yield AgentRunResponseUpdate(
            content="",
            is_complete=True,
            metadata={"agent_name": self.config.name, "full_content": full_content}
        )
        
        # Add assistant message to thread
        self.thread.add_assistant_message(full_content)
        
        # Store interaction in context
        await self.context_provider.add_context(
            f"User: {user_input}\nAssistant: {full_content}",
            metadata={"timestamp": datetime.now().isoformat(), "agent": self.config.name}
        )
    
    def get_thread(self) -> AgentThread:
        """Get the agent's thread."""
        return self.thread
    
    def clear_thread(self) -> None:
        """Clear the agent's conversation thread."""
        # Keep system messages
        system_messages = self.thread.get_messages(role="system")
        self.thread.clear_messages()
        for msg in system_messages:
            self.thread.add_message(msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "config": self.config.model_dump(),
            "thread": self.thread.to_dict(),
            "tools": list(self.tools.keys())
        }


class ChatCompletionAgent(BaseAgent):
    """Chat completion agent implementation."""
    
    def __init__(
        self,
        instructions: str,
        name: str,
        groq_client: GroqClient,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context_provider: Optional[ContextProvider] = None,
        thread: Optional[AgentThread] = None
    ):
        """Initialize chat completion agent."""
        config = AgentConfig(
            name=name,
            instructions=instructions,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        super().__init__(
            config=config,
            groq_client=groq_client,
            context_provider=context_provider,
            thread=thread
        )
