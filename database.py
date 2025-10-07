"""
Database integration for VAPI Stealth Agent
Handles conversation history, user preferences, and agent orchestration data
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Boolean, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import text
import uuid
import numpy as np

# Database configuration from Northflank environment variables (private network)
DATABASE_URL = os.environ.get("NF_POSTGRESQL_POSTGRES_URI")
if not DATABASE_URL:
    print("⚠️ No PostgreSQL connection string found")
    DATABASE_URL = "sqlite:///fallback.db"  # Fallback for development
else:
    print(f"✅ Using Northflank private network PostgreSQL: {DATABASE_URL.split('@')[1].split('/')[0] if '@' in DATABASE_URL else 'unknown'}")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ConversationHistory(Base):
    """Store conversation history for context and learning"""
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)  # VAPI call ID or user session
    user_id = Column(String(255), index=True, nullable=True)  # If available from VAPI
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Message data
    role = Column(String(50))  # 'user', 'assistant', 'system'
    content = Column(Text)
    model_used = Column(String(100), nullable=True)  # Which Groq model was used
    
    # Metadata
    response_time_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    intent_analysis = Column(JSON, nullable=True)  # Store intent analysis results
    agents_called = Column(JSON, nullable=True)  # Which specialized agents were used

class UserPreferences(Base):
    """Store user preferences and patterns"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Preferences
    preferred_response_style = Column(String(100), default="conversational")  # formal, casual, technical
    preferred_response_length = Column(String(50), default="medium")  # short, medium, long
    topics_of_interest = Column(JSON, default=list)  # List of topics user frequently asks about
    
    # Usage patterns
    total_conversations = Column(Integer, default=0)
    favorite_agents = Column(JSON, default=list)  # Which specialized agents user uses most
    timezone = Column(String(50), nullable=True)
    language_preference = Column(String(10), default="en")

class AgentMetrics(Base):
    """Track agent performance and usage"""
    __tablename__ = "agent_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Agent info
    agent_name = Column(String(100))  # 'github_analysis', 'stealth_orchestrator', etc.
    operation = Column(String(100))  # 'intent_analysis', 'response_synthesis', etc.
    
    # Performance metrics
    response_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Usage data
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)

class DatabaseManager:
    """Manage database operations for the stealth agent"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("✅ Database tables initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def log_conversation(self, session_id: str, role: str, content: str, 
                        model_used: str = None, response_time_ms: int = None,
                        intent_analysis: Dict = None, agents_called: List = None,
                        user_id: str = None):
        """Log conversation message"""
        try:
            with self.get_session() as db:
                conversation = ConversationHistory(
                    session_id=session_id,
                    user_id=user_id,
                    role=role,
                    content=content,
                    model_used=model_used,
                    response_time_ms=response_time_ms,
                    intent_analysis=intent_analysis,
                    agents_called=agents_called
                )
                db.add(conversation)
                db.commit()
        except Exception as e:
            print(f"❌ Failed to log conversation: {e}")
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for context"""
        try:
            with self.get_session() as db:
                conversations = db.query(ConversationHistory)\
                    .filter(ConversationHistory.session_id == session_id)\
                    .order_by(ConversationHistory.timestamp.desc())\
                    .limit(limit)\
                    .all()
                
                return [{
                    "role": conv.role,
                    "content": conv.content,
                    "timestamp": conv.timestamp.isoformat(),
                    "model_used": conv.model_used
                } for conv in reversed(conversations)]
        except Exception as e:
            print(f"❌ Failed to get conversation history: {e}")
            return []
    
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        try:
            with self.get_session() as db:
                user_pref = db.query(UserPreferences)\
                    .filter(UserPreferences.user_id == user_id)\
                    .first()
                
                if not user_pref:
                    user_pref = UserPreferences(user_id=user_id)
                    db.add(user_pref)
                
                # Update preferences
                for key, value in preferences.items():
                    if hasattr(user_pref, key):
                        setattr(user_pref, key, value)
                
                user_pref.updated_at = datetime.now(timezone.utc)
                db.commit()
        except Exception as e:
            print(f"❌ Failed to update user preferences: {e}")
    
    def log_agent_metrics(self, agent_name: str, operation: str, 
                         response_time_ms: int, success: bool = True,
                         error_message: str = None, model_used: str = None,
                         input_tokens: int = None, output_tokens: int = None):
        """Log agent performance metrics"""
        try:
            with self.get_session() as db:
                metrics = AgentMetrics(
                    agent_name=agent_name,
                    operation=operation,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=error_message,
                    model_used=model_used,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
                db.add(metrics)
                db.commit()
        except Exception as e:
            print(f"❌ Failed to log agent metrics: {e}")
    
    def get_user_context(self, user_id: str) -> Dict:
        """Get user context for personalized responses"""
        try:
            with self.get_session() as db:
                user_pref = db.query(UserPreferences)\
                    .filter(UserPreferences.user_id == user_id)\
                    .first()
                
                if user_pref:
                    return {
                        "preferred_style": user_pref.preferred_response_style,
                        "preferred_length": user_pref.preferred_response_length,
                        "topics_of_interest": user_pref.topics_of_interest,
                        "total_conversations": user_pref.total_conversations,
                        "favorite_agents": user_pref.favorite_agents
                    }
                return {}
        except Exception as e:
            print(f"❌ Failed to get user context: {e}")
            return {}
    
    def get_agent_performance(self, agent_name: str = None, hours: int = 24) -> Dict:
        """Get agent performance statistics"""
        try:
            with self.get_session() as db:
                query = db.query(AgentMetrics)
                
                if agent_name:
                    query = query.filter(AgentMetrics.agent_name == agent_name)
                
                # Filter by time
                since = datetime.now(timezone.utc) - timedelta(hours=hours)
                query = query.filter(AgentMetrics.timestamp >= since)
                
                metrics = query.all()
                
                if not metrics:
                    return {"total_calls": 0}
                
                total_calls = len(metrics)
                successful_calls = sum(1 for m in metrics if m.success)
                avg_response_time = sum(m.response_time_ms for m in metrics) / total_calls
                
                return {
                    "total_calls": total_calls,
                    "success_rate": successful_calls / total_calls,
                    "avg_response_time_ms": avg_response_time,
                    "total_tokens": sum(m.input_tokens or 0 for m in metrics) + sum(m.output_tokens or 0 for m in metrics)
                }
        except Exception as e:
            print(f"❌ Failed to get agent performance: {e}")
            return {"error": str(e)}

# Global database manager instance
db_manager = DatabaseManager()
