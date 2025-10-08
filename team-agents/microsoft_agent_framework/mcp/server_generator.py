"""MCP server generator for converting APIs to MCP servers."""

import json
import os
import tempfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

from .api_parser import APIDefinition, APIEndpoint, APIParameter

logger = logging.getLogger(__name__)


@dataclass
class MCPServerCode:
    """Generated MCP server code and metadata."""
    server_name: str
    code: str
    requirements: List[str]
    config: Dict[str, Any]
    transport_type: str  # 'stdio', 'http', 'websocket'
    entry_point: str


@dataclass
class MCPServerInfo:
    """Information about a deployed MCP server."""
    name: str
    transport_type: str
    connection_info: Dict[str, Any]
    capabilities: List[str]
    status: str = "active"
    health_endpoint: Optional[str] = None


class MCPServerGenerator:
    """Generate MCP servers from API specifications."""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        
    def generate_stdio_server(self, api_def: APIDefinition, server_name: Optional[str] = None) -> MCPServerCode:
        """Generate stdio-based MCP server."""
        server_name = server_name or self._sanitize_name(api_def.title)
        
        # Generate server code
        code = self._generate_stdio_server_code(api_def, server_name)
        
        # Define requirements
        requirements = [
            "mcp>=1.0.0",
            "httpx>=0.25.0",
            "pydantic>=2.0.0",
            "asyncio-mqtt>=0.11.0" if self._needs_mqtt(api_def) else None
        ]
        requirements = [req for req in requirements if req is not None]
        
        # Server configuration
        config = {
            "name": server_name,
            "description": api_def.description,
            "version": api_def.version,
            "api_base_url": api_def.base_url,
            "endpoints": len(api_def.endpoints)
        }
        
        return MCPServerCode(
            server_name=server_name,
            code=code,
            requirements=requirements,
            config=config,
            transport_type="stdio",
            entry_point=f"{server_name}_mcp_server.py"
        )
    
    def generate_http_server(self, api_def: APIDefinition, server_name: Optional[str] = None) -> MCPServerCode:
        """Generate HTTP-based MCP server."""
        server_name = server_name or self._sanitize_name(api_def.title)
        
        # Generate server code
        code = self._generate_http_server_code(api_def, server_name)
        
        # Define requirements
        requirements = [
            "mcp>=1.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "httpx>=0.25.0",
            "pydantic>=2.0.0"
        ]
        
        # Server configuration
        config = {
            "name": server_name,
            "description": api_def.description,
            "version": api_def.version,
            "api_base_url": api_def.base_url,
            "endpoints": len(api_def.endpoints),
            "port": 8000,
            "host": "0.0.0.0"
        }
        
        return MCPServerCode(
            server_name=server_name,
            code=code,
            requirements=requirements,
            config=config,
            transport_type="http",
            entry_point=f"{server_name}_mcp_server.py"
        )
    
    def generate_websocket_server(self, api_def: APIDefinition, server_name: Optional[str] = None) -> MCPServerCode:
        """Generate WebSocket-based MCP server."""
        server_name = server_name or self._sanitize_name(api_def.title)
        
        # Generate server code
        code = self._generate_websocket_server_code(api_def, server_name)
        
        # Define requirements
        requirements = [
            "mcp>=1.0.0",
            "websockets>=11.0.0",
            "httpx>=0.25.0",
            "pydantic>=2.0.0",
            "asyncio>=3.4.3"
        ]
        
        # Server configuration
        config = {
            "name": server_name,
            "description": api_def.description,
            "version": api_def.version,
            "api_base_url": api_def.base_url,
            "endpoints": len(api_def.endpoints),
            "port": 8001,
            "host": "0.0.0.0"
        }
        
        return MCPServerCode(
            server_name=server_name,
            code=code,
            requirements=requirements,
            config=config,
            transport_type="websocket",
            entry_point=f"{server_name}_mcp_server.py"
        )
    
    def _generate_stdio_server_code(self, api_def: APIDefinition, server_name: str) -> str:
        """Generate stdio MCP server code."""
        tools_code = self._generate_tools_code(api_def)
        
        return f'''#!/usr/bin/env python3
"""
MCP Server for {api_def.title}
Generated automatically from API specification
Transport: stdio
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


class {server_name.title()}MCPServer:
    """MCP Server for {api_def.title}."""
    
    def __init__(self):
        self.server = Server("{server_name}")
        self.http_client = httpx.AsyncClient()
        self.base_url = "{api_def.base_url}"
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all API endpoints as MCP tools."""
{self._indent_code(tools_code, 8)}
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.http_client.aclose()

{self._generate_tool_implementations(api_def)}

async def main():
    """Main entry point for stdio MCP server."""
    server_instance = {server_name.title()}MCPServer()
    
    try:
        async with stdio_server() as streams:
            await server_instance.server.run(
                streams[0], streams[1], server_instance.server.create_initialization_options()
            )
    finally:
        await server_instance.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _generate_http_server_code(self, api_def: APIDefinition, server_name: str) -> str:
        """Generate HTTP MCP server code."""
        tools_code = self._generate_tools_code(api_def)
        
        return f'''#!/usr/bin/env python3
"""
MCP Server for {api_def.title}
Generated automatically from API specification
Transport: HTTP
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
import httpx
from fastapi import FastAPI, HTTPException
from mcp.server import Server
from mcp.server.fastapi import create_fastapi_app
from mcp.types import Tool, TextContent
import uvicorn


class {server_name.title()}MCPServer:
    """MCP Server for {api_def.title}."""
    
    def __init__(self):
        self.server = Server("{server_name}")
        self.http_client = httpx.AsyncClient()
        self.base_url = "{api_def.base_url}"
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all API endpoints as MCP tools."""
{self._indent_code(tools_code, 8)}
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.http_client.aclose()

{self._generate_tool_implementations(api_def)}

# Create FastAPI app
server_instance = {server_name.title()}MCPServer()
app = create_fastapi_app(server_instance.server)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {{"status": "healthy", "server": "{server_name}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _generate_websocket_server_code(self, api_def: APIDefinition, server_name: str) -> str:
        """Generate WebSocket MCP server code."""
        tools_code = self._generate_tools_code(api_def)
        
        return f'''#!/usr/bin/env python3
"""
MCP Server for {api_def.title}
Generated automatically from API specification
Transport: WebSocket
"""

import asyncio
import json
import websockets
from typing import Any, Dict, List, Optional
import httpx
from mcp.server import Server
from mcp.server.websockets import websocket_server
from mcp.types import Tool, TextContent


class {server_name.title()}MCPServer:
    """MCP Server for {api_def.title}."""
    
    def __init__(self):
        self.server = Server("{server_name}")
        self.http_client = httpx.AsyncClient()
        self.base_url = "{api_def.base_url}"
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all API endpoints as MCP tools."""
{self._indent_code(tools_code, 8)}
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.http_client.aclose()

{self._generate_tool_implementations(api_def)}

async def main():
    """Main entry point for WebSocket MCP server."""
    server_instance = {server_name.title()}MCPServer()
    
    try:
        async with websocket_server() as server_context:
            await server_instance.server.run_websocket(
                "0.0.0.0", 8001, server_context
            )
    finally:
        await server_instance.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _generate_tools_code(self, api_def: APIDefinition) -> str:
        """Generate tool registration code for all endpoints."""
        tools_code = []
        
        for endpoint in api_def.endpoints:
            tool_name = self._generate_tool_name(endpoint)
            description = endpoint.description or endpoint.summary or f"{endpoint.method} {endpoint.path}"
            
            # Generate parameters schema
            params_schema = self._generate_parameters_schema(endpoint)
            
            tools_code.append(f'''
@self.server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="{tool_name}",
            description="{description}",
            inputSchema={json.dumps(params_schema, indent=8)}
        )
    ]

@self.server.call_tool()
async def call_{tool_name}(arguments: Dict[str, Any]) -> List[TextContent]:
    return await self._{tool_name}(arguments)
''')
        
        return '\n'.join(tools_code)
    
    def _generate_tool_implementations(self, api_def: APIDefinition) -> str:
        """Generate tool implementation methods."""
        implementations = []
        
        for endpoint in api_def.endpoints:
            tool_name = self._generate_tool_name(endpoint)
            impl_code = self._generate_endpoint_implementation(endpoint, tool_name)
            implementations.append(impl_code)
        
        return '\n\n'.join(implementations)
    
    def _generate_endpoint_implementation(self, endpoint: APIEndpoint, tool_name: str) -> str:
        """Generate implementation for a specific endpoint."""
        method = endpoint.method.lower()
        path = endpoint.path
        
        # Build URL construction code
        url_code = f'url = f"{self.base_url}{path}"'
        
        # Handle path parameters
        if '{' in path:
            path_params = []
            for param in endpoint.parameters:
                if param.name in path:
                    path_params.append(f'"{param.name}": arguments.get("{param.name}")')
            
            if path_params:
                url_code = f'''
        path_params = {{{', '.join(path_params)}}}
        url = f"{self.base_url}{path}".format(**path_params)'''
        
        # Handle query parameters
        query_params = [p for p in endpoint.parameters if p.name not in path]
        query_code = ""
        if query_params:
            query_code = '''
        # Build query parameters
        params = {}
        for param_name in arguments:
            if param_name not in path_params and arguments[param_name] is not None:
                params[param_name] = arguments[param_name]'''
        
        # Handle request body
        body_code = ""
        if endpoint.request_body and method in ['post', 'put', 'patch']:
            body_code = '''
        # Handle request body
        json_data = arguments.get("body") or {}'''
        
        return f'''    async def _{tool_name}(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Implementation for {endpoint.method} {endpoint.path}."""
        try:{url_code}{query_code}{body_code}
            
            # Make HTTP request
            response = await self.http_client.{method}(
                url{"," if query_params else ""}
                {f"params=params," if query_params else ""}
                {f"json=json_data," if body_code else ""}
                timeout=30.0
            )
            response.raise_for_status()
            
            # Return response
            result = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text", 
                text=f"Error calling {endpoint.method} {endpoint.path}: {{str(e)}}"
            )]'''
    
    def _generate_parameters_schema(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate JSON schema for endpoint parameters."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in endpoint.parameters:
            param_schema = {
                "type": self._map_parameter_type(param.type),
                "description": param.description
            }
            
            if param.enum_values:
                param_schema["enum"] = param.enum_values
            
            if param.default is not None:
                param_schema["default"] = param.default
            
            schema["properties"][param.name] = param_schema
            
            if param.required:
                schema["required"].append(param.name)
        
        # Add request body parameter if needed
        if endpoint.request_body:
            schema["properties"]["body"] = {
                "type": "object",
                "description": "Request body data"
            }
        
        return schema
    
    def _generate_tool_name(self, endpoint: APIEndpoint) -> str:
        """Generate a valid tool name from endpoint."""
        if endpoint.operation_id:
            return self._sanitize_name(endpoint.operation_id)
        
        # Generate from method and path
        path_parts = [part for part in endpoint.path.split('/') if part and not part.startswith('{')]
        method = endpoint.method.lower()
        
        if path_parts:
            name = f"{method}_{'_'.join(path_parts)}"
        else:
            name = f"{method}_root"
        
        return self._sanitize_name(name)
    
    def _map_parameter_type(self, param_type: str) -> str:
        """Map API parameter type to JSON schema type."""
        type_mapping = {
            'integer': 'number',
            'int': 'number',
            'float': 'number',
            'double': 'number',
            'boolean': 'boolean',
            'bool': 'boolean',
            'array': 'array',
            'object': 'object'
        }
        return type_mapping.get(param_type.lower(), 'string')
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as identifier."""
        import re
        # Remove special characters and replace with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isdigit():
            sanitized = f"api_{sanitized}"
        
        return sanitized.lower() or "api_tool"
    
    def _needs_mqtt(self, api_def: APIDefinition) -> bool:
        """Check if API needs MQTT support."""
        # Simple heuristic - check for real-time or streaming endpoints
        for endpoint in api_def.endpoints:
            if any(keyword in endpoint.description.lower() for keyword in ['stream', 'websocket', 'realtime', 'mqtt']):
                return True
        return False
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line for line in code.split('\n'))
    
    def deploy_server(self, server_code: MCPServerCode, deployment_target: str = "local") -> MCPServerInfo:
        """Deploy generated MCP server."""
        try:
            if deployment_target == "local":
                return self._deploy_local(server_code)
            elif deployment_target == "docker":
                return self._deploy_docker(server_code)
            elif deployment_target == "kubernetes":
                return self._deploy_kubernetes(server_code)
            else:
                raise ValueError(f"Unsupported deployment target: {deployment_target}")
                
        except Exception as e:
            logger.error(f"Error deploying MCP server: {e}")
            raise
    
    def _deploy_local(self, server_code: MCPServerCode) -> MCPServerInfo:
        """Deploy MCP server locally."""
        # Create temporary directory for server files
        temp_dir = tempfile.mkdtemp(prefix=f"mcp_{server_code.server_name}_")
        
        # Write server code
        server_file = Path(temp_dir) / server_code.entry_point
        with open(server_file, 'w') as f:
            f.write(server_code.code)
        
        # Write requirements
        requirements_file = Path(temp_dir) / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(server_code.requirements))
        
        # Write config
        config_file = Path(temp_dir) / "config.json"
        with open(config_file, 'w') as f:
            json.dump(server_code.config, f, indent=2)
        
        # Determine connection info based on transport type
        if server_code.transport_type == "stdio":
            connection_info = {
                "type": "stdio",
                "command": f"python {server_file}",
                "working_directory": temp_dir
            }
        elif server_code.transport_type == "http":
            connection_info = {
                "type": "http",
                "url": f"http://localhost:{server_code.config.get('port', 8000)}",
                "command": f"python {server_file}",
                "working_directory": temp_dir
            }
        else:  # websocket
            connection_info = {
                "type": "websocket", 
                "url": f"ws://localhost:{server_code.config.get('port', 8001)}",
                "command": f"python {server_file}",
                "working_directory": temp_dir
            }
        
        # Extract capabilities from API definition
        capabilities = [f"{endpoint.method} {endpoint.path}" for endpoint in server_code.config.get('endpoints', [])]
        
        return MCPServerInfo(
            name=server_code.server_name,
            transport_type=server_code.transport_type,
            connection_info=connection_info,
            capabilities=capabilities,
            status="deployed",
            health_endpoint=connection_info.get("url", "") + "/health" if server_code.transport_type == "http" else None
        )
    
    def _deploy_docker(self, server_code: MCPServerCode) -> MCPServerInfo:
        """Deploy MCP server using Docker."""
        # TODO: Implement Docker deployment
        raise NotImplementedError("Docker deployment not yet implemented")
    
    def _deploy_kubernetes(self, server_code: MCPServerCode) -> MCPServerInfo:
        """Deploy MCP server to Kubernetes.""" 
        # TODO: Implement Kubernetes deployment
        raise NotImplementedError("Kubernetes deployment not yet implemented")
