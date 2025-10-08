"""MCP client for connecting to MCP servers."""

import asyncio
import json
import subprocess
import websockets
import httpx
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

from .registry import MCPServer

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Represents an MCP tool from a server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_id: str
    server_name: str


class MCPClient:
    """Client for connecting to and using MCP servers."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def connect_to_server(self, server: MCPServer) -> bool:
        """Connect to an MCP server."""
        try:
            if server.transport_type == "stdio":
                return await self._connect_stdio(server)
            elif server.transport_type == "http":
                return await self._connect_http(server)
            elif server.transport_type == "websocket":
                return await self._connect_websocket(server)
            else:
                logger.error(f"Unsupported transport type: {server.transport_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to server {server.name}: {e}")
            return False
    
    async def _connect_stdio(self, server: MCPServer) -> bool:
        """Connect to stdio-based MCP server."""
        command = server.connection_info.get("command")
        working_dir = server.connection_info.get("working_directory")
        
        if not command:
            logger.error(f"No command specified for stdio server {server.name}")
            return False
        
        try:
            # Start the process
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            self.connections[server.id] = {
                "type": "stdio",
                "process": process,
                "server": server
            }
            
            logger.info(f"Connected to stdio MCP server: {server.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting stdio server {server.name}: {e}")
            return False
    
    async def _connect_http(self, server: MCPServer) -> bool:
        """Connect to HTTP-based MCP server."""
        url = server.connection_info.get("url")
        
        if not url:
            logger.error(f"No URL specified for HTTP server {server.name}")
            return False
        
        try:
            # Test connection with health check
            health_url = f"{url}/health"
            response = await self.http_client.get(health_url)
            
            if response.status_code == 200:
                self.connections[server.id] = {
                    "type": "http",
                    "url": url,
                    "server": server
                }
                logger.info(f"Connected to HTTP MCP server: {server.name}")
                return True
            else:
                logger.error(f"HTTP server {server.name} health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to HTTP server {server.name}: {e}")
            return False
    
    async def _connect_websocket(self, server: MCPServer) -> bool:
        """Connect to WebSocket-based MCP server."""
        url = server.connection_info.get("url")
        
        if not url:
            logger.error(f"No URL specified for WebSocket server {server.name}")
            return False
        
        try:
            # Test WebSocket connection
            websocket = await websockets.connect(url)
            
            self.connections[server.id] = {
                "type": "websocket",
                "websocket": websocket,
                "url": url,
                "server": server
            }
            
            logger.info(f"Connected to WebSocket MCP server: {server.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket server {server.name}: {e}")
            return False
    
    async def list_tools(self, server_id: str) -> List[MCPTool]:
        """List available tools from an MCP server."""
        if server_id not in self.connections:
            logger.error(f"Not connected to server {server_id}")
            return []
        
        connection = self.connections[server_id]
        
        try:
            if connection["type"] == "stdio":
                return await self._list_tools_stdio(connection)
            elif connection["type"] == "http":
                return await self._list_tools_http(connection)
            elif connection["type"] == "websocket":
                return await self._list_tools_websocket(connection)
        except Exception as e:
            logger.error(f"Error listing tools from server {server_id}: {e}")
            return []
    
    async def _list_tools_stdio(self, connection: Dict[str, Any]) -> List[MCPTool]:
        """List tools from stdio MCP server."""
        # TODO: Implement MCP protocol communication for stdio
        # For now, return mock tools based on server capabilities
        server = connection["server"]
        tools = []
        
        for i, capability in enumerate(server.capabilities):
            tool = MCPTool(
                name=f"tool_{i}",
                description=capability,
                input_schema={"type": "object", "properties": {}},
                server_id=server.id,
                server_name=server.name
            )
            tools.append(tool)
        
        return tools
    
    async def _list_tools_http(self, connection: Dict[str, Any]) -> List[MCPTool]:
        """List tools from HTTP MCP server."""
        url = connection["url"]
        server = connection["server"]
        
        try:
            response = await self.http_client.get(f"{url}/tools")
            if response.status_code == 200:
                tools_data = response.json()
                tools = []
                
                for tool_data in tools_data.get("tools", []):
                    tool = MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        server_id=server.id,
                        server_name=server.name
                    )
                    tools.append(tool)
                
                return tools
            else:
                logger.error(f"Failed to list tools from HTTP server: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing tools from HTTP server: {e}")
            return []
    
    async def _list_tools_websocket(self, connection: Dict[str, Any]) -> List[MCPTool]:
        """List tools from WebSocket MCP server."""
        websocket = connection["websocket"]
        server = connection["server"]
        
        try:
            # Send list tools request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
            
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                tools = []
                for tool_data in response_data["result"].get("tools", []):
                    tool = MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        server_id=server.id,
                        server_name=server.name
                    )
                    tools.append(tool)
                
                return tools
            else:
                logger.error(f"Error in WebSocket response: {response_data.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing tools from WebSocket server: {e}")
            return []
    
    async def call_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call a tool on an MCP server."""
        if server_id not in self.connections:
            logger.error(f"Not connected to server {server_id}")
            return None
        
        connection = self.connections[server_id]
        
        try:
            if connection["type"] == "stdio":
                return await self._call_tool_stdio(connection, tool_name, arguments)
            elif connection["type"] == "http":
                return await self._call_tool_http(connection, tool_name, arguments)
            elif connection["type"] == "websocket":
                return await self._call_tool_websocket(connection, tool_name, arguments)
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on server {server_id}: {e}")
            return None
    
    async def _call_tool_stdio(self, connection: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call tool on stdio MCP server."""
        # TODO: Implement MCP protocol communication for stdio
        return f"Mock response from stdio tool {tool_name} with args {arguments}"
    
    async def _call_tool_http(self, connection: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call tool on HTTP MCP server."""
        url = connection["url"]
        
        try:
            request_data = {
                "name": tool_name,
                "arguments": arguments
            }
            
            response = await self.http_client.post(f"{url}/tools/call", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("content", [{}])[0].get("text", "No response")
            else:
                logger.error(f"Tool call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling HTTP tool: {e}")
            return None
    
    async def _call_tool_websocket(self, connection: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call tool on WebSocket MCP server."""
        websocket = connection["websocket"]
        
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                content = response_data["result"].get("content", [])
                if content and len(content) > 0:
                    return content[0].get("text", "No response")
                return "Empty response"
            else:
                error = response_data.get("error", {})
                logger.error(f"Tool call error: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling WebSocket tool: {e}")
            return None
    
    async def disconnect_from_server(self, server_id: str) -> bool:
        """Disconnect from an MCP server."""
        if server_id not in self.connections:
            return False
        
        connection = self.connections[server_id]
        
        try:
            if connection["type"] == "stdio":
                process = connection["process"]
                process.terminate()
                await process.wait()
            elif connection["type"] == "websocket":
                websocket = connection["websocket"]
                await websocket.close()
            
            del self.connections[server_id]
            logger.info(f"Disconnected from MCP server: {server_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from server {server_id}: {e}")
            return False
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        server_ids = list(self.connections.keys())
        for server_id in server_ids:
            await self.disconnect_from_server(server_id)
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server IDs."""
        return list(self.connections.keys())
    
    def is_connected(self, server_id: str) -> bool:
        """Check if connected to a specific server."""
        return server_id in self.connections
    
    async def cleanup(self):
        """Cleanup client resources."""
        await self.disconnect_all()
        await self.http_client.aclose()
