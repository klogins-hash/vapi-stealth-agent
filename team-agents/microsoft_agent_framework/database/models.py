"""Database models for the Microsoft Agent Framework."""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class Agent(Base):
    """Agent model for storing agent configurations."""
    
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    template_name = Column(String(100), nullable=True)
    instructions = Column(Text, nullable=False)
    model = Column(String(100), default="llama-3.1-70b-versatile")
    temperature = Column(String(10), default="0.7")
    max_tokens = Column(Integer, default=4096)
    tools = Column(JSON, default=list)
    agent_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "template_name": self.template_name,
            "instructions": self.instructions,
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "metadata": self.agent_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active
        }


class Conversation(Base):
    """Conversation model for storing conversation threads."""
    
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    title = Column(String(255), nullable=True)
    agent_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert conversation to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "title": self.title,
            "metadata": self.agent_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "message_count": len(self.messages) if self.messages else 0
        }


class Message(Base):
    """Message model for storing conversation messages."""
    
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    agent_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def to_dict(self):
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.agent_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AgentTemplate(Base):
    """Template model for storing agent templates."""
    
    __tablename__ = "agent_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=False)
    model = Column(String(100), default="llama-3.1-70b-versatile")
    temperature = Column(String(10), default="0.7")
    max_tokens = Column(Integer, default=4096)
    tools = Column(JSON, default=list)
    agent_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        """Convert template to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "metadata": self.agent_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active
        }
