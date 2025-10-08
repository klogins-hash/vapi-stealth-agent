"""Code-related tools for agents."""

import ast
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio


class CodeTools:
    """Code-related tools for agents."""
    
    def __init__(self, allowed_languages: Optional[List[str]] = None):
        """Initialize code tools with optional language restrictions."""
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
    
    async def execute_python(self, code: str, timeout: int = 30) -> str:
        """Tool: Execute Python code safely."""
        if "python" not in self.allowed_languages:
            return "Python execution not allowed"
        
        try:
            # Basic safety check - prevent dangerous imports
            dangerous_modules = ["os", "subprocess", "sys", "shutil", "pathlib"]
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            return f"Dangerous import detected: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_modules:
                        return f"Dangerous import detected: {node.module}"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        'python3', temp_file,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    ),
                    timeout=timeout
                )
                
                stdout, stderr = await result.communicate()
                
                output = []
                if stdout:
                    output.append(f"Output:\n{stdout.decode()}")
                if stderr:
                    output.append(f"Errors:\n{stderr.decode()}")
                
                return "\n".join(output) if output else "Code executed successfully (no output)"
                
            finally:
                os.unlink(temp_file)
                
        except asyncio.TimeoutError:
            return f"Code execution timed out after {timeout} seconds"
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"
        except Exception as e:
            return f"Error executing Python code: {str(e)}"
    
    async def validate_python_syntax(self, code: str) -> str:
        """Tool: Validate Python code syntax."""
        try:
            ast.parse(code)
            return "Python syntax is valid"
        except SyntaxError as e:
            return f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return f"Error validating syntax: {str(e)}"
    
    async def analyze_python_code(self, code: str) -> str:
        """Tool: Analyze Python code structure."""
        try:
            tree = ast.parse(code)
            
            analysis = {
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    analysis["imports"].append(f"from {node.module}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis["variables"].append(target.id)
            
            result = ["Code Analysis:"]
            for category, items in analysis.items():
                if items:
                    result.append(f"{category.title()}: {', '.join(set(items))}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error analyzing code: {str(e)}"
    
    async def format_code(self, code: str, language: str = "python") -> str:
        """Tool: Format code (basic implementation)."""
        if language == "python":
            try:
                # Basic Python formatting - in production, use black or autopep8
                tree = ast.parse(code)
                # This is a simplified formatter - real implementation would use proper tools
                return f"Formatted {language} code:\n{code}"
            except Exception as e:
                return f"Error formatting Python code: {str(e)}"
        else:
            return f"Code formatting for {language} not implemented yet"
    
    async def run_shell_command(self, command: str, timeout: int = 30) -> str:
        """Tool: Run shell command (restricted)."""
        if "bash" not in self.allowed_languages:
            return "Shell command execution not allowed"
        
        # Basic safety check - prevent dangerous commands
        dangerous_commands = ["rm", "del", "format", "sudo", "su", "chmod", "chown"]
        command_parts = command.split()
        
        if any(dangerous in command_parts[0] for dangerous in dangerous_commands):
            return f"Dangerous command detected: {command_parts[0]}"
        
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=timeout
            )
            
            stdout, stderr = await result.communicate()
            
            output = []
            if stdout:
                output.append(f"Output:\n{stdout.decode()}")
            if stderr:
                output.append(f"Errors:\n{stderr.decode()}")
            
            return "\n".join(output) if output else "Command executed successfully (no output)"
            
        except asyncio.TimeoutError:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    async def create_project_structure(self, project_name: str, project_type: str = "python") -> str:
        """Tool: Create basic project structure."""
        try:
            base_path = Path(project_name)
            
            if base_path.exists():
                return f"Project directory {project_name} already exists"
            
            if project_type == "python":
                # Create Python project structure
                base_path.mkdir()
                (base_path / "src").mkdir()
                (base_path / "tests").mkdir()
                (base_path / "docs").mkdir()
                
                # Create basic files
                (base_path / "README.md").write_text(f"# {project_name}\n\nDescription of your project.")
                (base_path / "requirements.txt").write_text("# Add your dependencies here\n")
                (base_path / ".gitignore").write_text("__pycache__/\n*.pyc\n.env\n")
                (base_path / "src" / "__init__.py").write_text("")
                
                return f"Created Python project structure for {project_name}"
            
            else:
                return f"Project type {project_type} not supported yet"
                
        except Exception as e:
            return f"Error creating project structure: {str(e)}"
