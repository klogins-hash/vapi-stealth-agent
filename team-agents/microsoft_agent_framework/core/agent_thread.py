"""Agent thread for state management in the Microsoft Agent Framework."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid


class ThreadMessage(BaseModel):
    """Message in an agent thread."""
    id: str
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    
    @classmethod
    def create_user_message(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "ThreadMessage":
        """Create a user message."""
        return cls(
            id=str(uuid.uuid4()),
            role="user",
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
    
    @classmethod
    def create_assistant_message(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "ThreadMessage":
        """Create an assistant message."""
        return cls(
            id=str(uuid.uuid4()),
            role="assistant",
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
    
    @classmethod
    def create_system_message(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "ThreadMessage":
        """Create a system message."""
        return cls(
            id=str(uuid.uuid4()),
            role="system",
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )


class AgentThread:
    """Thread for managing conversation state and history."""
    
    def __init__(self, thread_id: Optional[str] = None):
        """Initialize agent thread."""
        self.id = thread_id or str(uuid.uuid4())
        self.messages: List[ThreadMessage] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, message: ThreadMessage) -> None:
        """Add a message to the thread."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ThreadMessage:
        """Add a user message to the thread."""
        message = ThreadMessage.create_user_message(content, metadata)
        self.add_message(message)
        return message
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ThreadMessage:
        """Add an assistant message to the thread."""
        message = ThreadMessage.create_assistant_message(content, metadata)
        self.add_message(message)
        return message
    
    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ThreadMessage:
        """Add a system message to the thread."""
        message = ThreadMessage.create_system_message(content, metadata)
        self.add_message(message)
        return message
    
    def get_messages(self, role: Optional[str] = None, limit: Optional[int] = None) -> List[ThreadMessage]:
        """Get messages from the thread, optionally filtered by role and limited."""
        messages = self.messages
        
        if role:
            messages = [msg for msg in messages if msg.role == role]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_history(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get conversation history in a format suitable for LLM APIs."""
        history = []
        
        for message in self.messages:
            if not include_system and message.role == "system":
                continue
            
            history.append({
                "role": message.role,
                "content": message.content
            })
        
        return history
    
    def clear_messages(self) -> None:
        """Clear all messages from the thread."""
        self.messages.clear()
        self.updated_at = datetime.now()
    
    def get_last_message(self, role: Optional[str] = None) -> Optional[ThreadMessage]:
        """Get the last message, optionally filtered by role."""
        messages = self.get_messages(role=role)
        return messages[-1] if messages else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thread to dictionary."""
        return {
            "id": self.id,
            "messages": [msg.model_dump() for msg in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentThread":
        """Create thread from dictionary."""
        thread = cls(thread_id=data["id"])
        thread.metadata = data.get("metadata", {})
        thread.created_at = datetime.fromisoformat(data["created_at"])
        thread.updated_at = datetime.fromisoformat(data["updated_at"])
        
        for msg_data in data.get("messages", []):
            message = ThreadMessage(**msg_data)
            thread.messages.append(message)
        
        return thread
