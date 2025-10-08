"""File-related tools for agents."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiofiles


class FileTools:
    """File-related tools for agents."""
    
    def __init__(self, base_directory: Optional[str] = None):
        """Initialize file tools with optional base directory restriction."""
        self.base_directory = Path(base_directory) if base_directory else None
    
    def _validate_path(self, file_path: str) -> Path:
        """Validate and resolve file path."""
        path = Path(file_path).resolve()
        
        if self.base_directory:
            try:
                path.relative_to(self.base_directory.resolve())
            except ValueError:
                raise PermissionError(f"Access denied: {file_path} is outside allowed directory")
        
        return path
    
    async def read_file(self, file_path: str) -> str:
        """Tool: Read content from a file."""
        try:
            path = self._validate_path(file_path)
            
            if not path.exists():
                return f"File not found: {file_path}"
            
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return f"Content of {file_path}:\n{content}"
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"
    
    async def write_file(self, file_path: str, content: str) -> str:
        """Tool: Write content to a file."""
        try:
            path = self._validate_path(file_path)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            return f"Successfully wrote content to {file_path}"
        except Exception as e:
            return f"Error writing to {file_path}: {str(e)}"
    
    async def append_file(self, file_path: str, content: str) -> str:
        """Tool: Append content to a file."""
        try:
            path = self._validate_path(file_path)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(path, 'a', encoding='utf-8') as f:
                await f.write(content)
            
            return f"Successfully appended content to {file_path}"
        except Exception as e:
            return f"Error appending to {file_path}: {str(e)}"
    
    async def list_directory(self, directory_path: str) -> str:
        """Tool: List contents of a directory."""
        try:
            path = self._validate_path(directory_path)
            
            if not path.exists():
                return f"Directory not found: {directory_path}"
            
            if not path.is_dir():
                return f"Not a directory: {directory_path}"
            
            items = []
            for item in path.iterdir():
                item_type = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else "-"
                items.append(f"{item_type:4} {size:>10} {item.name}")
            
            return f"Contents of {directory_path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing {directory_path}: {str(e)}"
    
    async def file_exists(self, file_path: str) -> str:
        """Tool: Check if a file exists."""
        try:
            path = self._validate_path(file_path)
            exists = path.exists()
            return f"File {file_path} {'exists' if exists else 'does not exist'}"
        except Exception as e:
            return f"Error checking {file_path}: {str(e)}"
    
    async def delete_file(self, file_path: str) -> str:
        """Tool: Delete a file."""
        try:
            path = self._validate_path(file_path)
            
            if not path.exists():
                return f"File not found: {file_path}"
            
            if path.is_file():
                path.unlink()
                return f"Successfully deleted file: {file_path}"
            else:
                return f"Not a file: {file_path}"
        except Exception as e:
            return f"Error deleting {file_path}: {str(e)}"
    
    async def read_json(self, file_path: str) -> str:
        """Tool: Read and parse JSON file."""
        try:
            path = self._validate_path(file_path)
            
            if not path.exists():
                return f"File not found: {file_path}"
            
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                return f"JSON content of {file_path}:\n{json.dumps(data, indent=2)}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON in {file_path}: {str(e)}"
        except Exception as e:
            return f"Error reading JSON from {file_path}: {str(e)}"
    
    async def write_json(self, file_path: str, data: Dict[str, Any]) -> str:
        """Tool: Write data as JSON to file."""
        try:
            path = self._validate_path(file_path)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(json_content)
            
            return f"Successfully wrote JSON data to {file_path}"
        except Exception as e:
            return f"Error writing JSON to {file_path}: {str(e)}"
