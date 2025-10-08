"""
Microsoft Agent Framework - A framework for building Microsoft-integrated agents using Groq models.
"""

__version__ = "0.1.0"
__author__ = "Microsoft Agent Framework Team"

from .core.agent_builder import AgentBuilder, AgentTemplate
from .core.base_agent import BaseAgent, ChatCompletionAgent, AgentConfig
from .core.groq_client import GroqClient, GroqConfig
from .core.context_provider import ContextProvider, InMemoryContextProvider, FileContextProvider
from .core.agent_thread import AgentThread
from .core.team_orchestrator import TeamOrchestrator, TeamMember, Task

__all__ = [
    "BaseAgent", "ChatCompletionAgent", "AgentConfig", 
    "AgentBuilder", "AgentTemplate", 
    "GroqClient", "GroqConfig", 
    "ContextProvider", "InMemoryContextProvider", "FileContextProvider", 
    "AgentThread", 
    "TeamOrchestrator", "TeamMember", "Task"
]
