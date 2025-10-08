"""Context provider for agent memory and context management."""

from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime


class ContextProvider(ABC):
    """Abstract base class for context providers."""
    
    @abstractmethod
    async def get_context(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant context based on query."""
        pass
    
    @abstractmethod
    async def add_context(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add content to the context store."""
        pass
    
    @abstractmethod
    async def update_context(self, context_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing context."""
        pass
    
    @abstractmethod
    async def delete_context(self, context_id: str) -> bool:
        """Delete context by ID."""
        pass


class InMemoryContextProvider(ContextProvider):
    """In-memory implementation of context provider."""
    
    def __init__(self):
        """Initialize in-memory context provider."""
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
    
    async def get_context(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant context based on query (simple text matching)."""
        results = []
        query_lower = query.lower()
        
        for context_id, context_data in self.contexts.items():
            content = context_data.get("content", "").lower()
            if query_lower in content:
                results.append({
                    "id": context_id,
                    "content": context_data.get("content", ""),
                    "metadata": context_data.get("metadata", {}),
                    "timestamp": context_data.get("timestamp")
                })
        
        # Sort by timestamp (most recent first) and limit results
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]
    
    async def add_context(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add content to the context store."""
        self._counter += 1
        context_id = f"ctx_{self._counter}"
        
        self.contexts[context_id] = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return context_id
    
    async def update_context(self, context_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing context."""
        if context_id not in self.contexts:
            return False
        
        self.contexts[context_id]["content"] = content
        if metadata is not None:
            self.contexts[context_id]["metadata"] = metadata
        self.contexts[context_id]["timestamp"] = datetime.now().isoformat()
        
        return True
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context by ID."""
        if context_id in self.contexts:
            del self.contexts[context_id]
            return True
        return False


class FileContextProvider(ContextProvider):
    """File-based context provider."""
    
    def __init__(self, file_path: str):
        """Initialize file-based context provider."""
        self.file_path = file_path
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._load_contexts()
    
    def _load_contexts(self):
        """Load contexts from file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.contexts = data.get("contexts", {})
                self._counter = data.get("counter", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            self.contexts = {}
            self._counter = 0
    
    def _save_contexts(self):
        """Save contexts to file."""
        data = {
            "contexts": self.contexts,
            "counter": self._counter
        }
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def get_context(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant context based on query."""
        results = []
        query_lower = query.lower()
        
        for context_id, context_data in self.contexts.items():
            content = context_data.get("content", "").lower()
            if query_lower in content:
                results.append({
                    "id": context_id,
                    "content": context_data.get("content", ""),
                    "metadata": context_data.get("metadata", {}),
                    "timestamp": context_data.get("timestamp")
                })
        
        # Sort by timestamp (most recent first) and limit results
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]
    
    async def add_context(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add content to the context store."""
        self._counter += 1
        context_id = f"ctx_{self._counter}"
        
        self.contexts[context_id] = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_contexts()
        return context_id
    
    async def update_context(self, context_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing context."""
        if context_id not in self.contexts:
            return False
        
        self.contexts[context_id]["content"] = content
        if metadata is not None:
            self.contexts[context_id]["metadata"] = metadata
        self.contexts[context_id]["timestamp"] = datetime.now().isoformat()
        
        self._save_contexts()
        return True
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete context by ID."""
        if context_id in self.contexts:
            del self.contexts[context_id]
            self._save_contexts()
            return True
        return False
