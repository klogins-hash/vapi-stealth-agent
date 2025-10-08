"""MCP server registry for managing and discovering MCP servers."""

import json
import asyncio
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

from .server_generator import MCPServerInfo

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Represents a registered MCP server."""
    id: str
    name: str
    description: str
    transport_type: str  # 'stdio', 'http', 'websocket'
    connection_info: Dict[str, Any]
    capabilities: List[str]
    tags: List[str] = field(default_factory=list)
    status: str = "active"
    health_endpoint: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # 'healthy', 'unhealthy', 'unknown'
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPRegistry:
    """Central registry for MCP servers and APIs."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
    async def register_mcp_server(self, server_info: MCPServerInfo, tags: Optional[List[str]] = None) -> str:
        """Register an MCP server in the registry."""
        server_id = f"mcp_{server_info.name}_{int(datetime.utcnow().timestamp())}"
        
        server = MCPServer(
            id=server_id,
            name=server_info.name,
            description=f"MCP server for {server_info.name}",
            transport_type=server_info.transport_type,
            connection_info=server_info.connection_info,
            capabilities=server_info.capabilities,
            tags=tags or [],
            status=server_info.status,
            health_endpoint=server_info.health_endpoint
        )
        
        self.servers[server_id] = server
        
        # Perform initial health check
        if server.health_endpoint:
            await self._check_server_health(server_id)
        
        logger.info(f"Registered MCP server: {server.name} ({server_id})")
        return server_id
    
    def get_server(self, server_id: str) -> Optional[MCPServer]:
        """Get server by ID."""
        return self.servers.get(server_id)
    
    def list_servers(self, status_filter: Optional[str] = None, transport_filter: Optional[str] = None) -> List[MCPServer]:
        """List all registered servers with optional filters."""
        servers = list(self.servers.values())
        
        if status_filter:
            servers = [s for s in servers if s.status == status_filter]
        
        if transport_filter:
            servers = [s for s in servers if s.transport_type == transport_filter]
        
        return servers
    
    def find_servers_by_capability(self, capability: str) -> List[MCPServer]:
        """Find servers that provide a specific capability."""
        matching_servers = []
        
        for server in self.servers.values():
            if any(capability.lower() in cap.lower() for cap in server.capabilities):
                matching_servers.append(server)
        
        return matching_servers
    
    def find_servers_by_tag(self, tag: str) -> List[MCPServer]:
        """Find servers with a specific tag."""
        return [server for server in self.servers.values() if tag in server.tags]
    
    def search_servers(self, query: str) -> List[MCPServer]:
        """Search servers by name, description, or capabilities."""
        query = query.lower()
        matching_servers = []
        
        for server in self.servers.values():
            if (query in server.name.lower() or 
                query in server.description.lower() or
                any(query in cap.lower() for cap in server.capabilities) or
                any(query in tag.lower() for tag in server.tags)):
                matching_servers.append(server)
        
        return matching_servers
    
    async def health_check_servers(self) -> Dict[str, bool]:
        """Perform health checks on all servers with health endpoints."""
        health_results = {}
        
        for server_id, server in self.servers.items():
            if server.health_endpoint:
                is_healthy = await self._check_server_health(server_id)
                health_results[server_id] = is_healthy
            else:
                health_results[server_id] = None  # No health check available
        
        return health_results
    
    async def _check_server_health(self, server_id: str) -> bool:
        """Check health of a specific server."""
        server = self.servers.get(server_id)
        if not server or not server.health_endpoint:
            return False
        
        try:
            response = await self.http_client.get(server.health_endpoint)
            is_healthy = response.status_code == 200
            
            # Update server health status
            server.last_health_check = datetime.utcnow()
            server.health_status = "healthy" if is_healthy else "unhealthy"
            
            logger.debug(f"Health check for {server.name}: {'healthy' if is_healthy else 'unhealthy'}")
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Health check failed for {server.name}: {e}")
            server.last_health_check = datetime.utcnow()
            server.health_status = "unhealthy"
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """Remove a server from the registry."""
        if server_id in self.servers:
            server = self.servers.pop(server_id)
            logger.info(f"Removed MCP server: {server.name} ({server_id})")
            return True
        return False
    
    def update_server_status(self, server_id: str, status: str) -> bool:
        """Update server status."""
        if server_id in self.servers:
            self.servers[server_id].status = status
            logger.info(f"Updated server {server_id} status to: {status}")
            return True
        return False
    
    def add_server_tags(self, server_id: str, tags: List[str]) -> bool:
        """Add tags to a server."""
        if server_id in self.servers:
            server = self.servers[server_id]
            server.tags.extend(tag for tag in tags if tag not in server.tags)
            logger.info(f"Added tags {tags} to server {server_id}")
            return True
        return False
    
    def remove_server_tags(self, server_id: str, tags: List[str]) -> bool:
        """Remove tags from a server."""
        if server_id in self.servers:
            server = self.servers[server_id]
            server.tags = [tag for tag in server.tags if tag not in tags]
            logger.info(f"Removed tags {tags} from server {server_id}")
            return True
        return False
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        servers = list(self.servers.values())
        
        stats = {
            "total_servers": len(servers),
            "active_servers": len([s for s in servers if s.status == "active"]),
            "transport_types": {},
            "health_status": {},
            "capabilities_count": 0,
            "most_common_tags": {}
        }
        
        # Count by transport type
        for server in servers:
            transport = server.transport_type
            stats["transport_types"][transport] = stats["transport_types"].get(transport, 0) + 1
        
        # Count by health status
        for server in servers:
            health = server.health_status
            stats["health_status"][health] = stats["health_status"].get(health, 0) + 1
        
        # Count total capabilities
        stats["capabilities_count"] = sum(len(server.capabilities) for server in servers)
        
        # Count tags
        all_tags = []
        for server in servers:
            all_tags.extend(server.tags)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Get top 10 most common tags
        stats["most_common_tags"] = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry data for backup or transfer."""
        return {
            "servers": {server_id: asdict(server) for server_id, server in self.servers.items()},
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
    
    def import_registry(self, registry_data: Dict[str, Any]) -> int:
        """Import registry data from backup."""
        imported_count = 0
        
        servers_data = registry_data.get("servers", {})
        for server_id, server_dict in servers_data.items():
            try:
                # Convert datetime strings back to datetime objects
                if "created_at" in server_dict:
                    server_dict["created_at"] = datetime.fromisoformat(server_dict["created_at"])
                if "last_health_check" in server_dict and server_dict["last_health_check"]:
                    server_dict["last_health_check"] = datetime.fromisoformat(server_dict["last_health_check"])
                
                server = MCPServer(**server_dict)
                self.servers[server_id] = server
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Error importing server {server_id}: {e}")
        
        logger.info(f"Imported {imported_count} servers to registry")
        return imported_count
    
    async def discover_apis(self, domain: str) -> List[Dict[str, Any]]:
        """Discover APIs in a domain (basic implementation)."""
        discovered_apis = []
        
        # Common API discovery patterns
        discovery_patterns = [
            f"https://{domain}/api",
            f"https://{domain}/api/v1", 
            f"https://{domain}/api/v2",
            f"https://api.{domain}",
            f"https://{domain}/swagger.json",
            f"https://{domain}/openapi.json",
            f"https://{domain}/.well-known/api"
        ]
        
        for url in discovery_patterns:
            try:
                response = await self.http_client.get(url)
                if response.status_code == 200:
                    api_info = {
                        "url": url,
                        "status": "discovered",
                        "content_type": response.headers.get("content-type", ""),
                        "size": len(response.content)
                    }
                    
                    # Try to extract API information
                    if "application/json" in response.headers.get("content-type", ""):
                        try:
                            data = response.json()
                            if "openapi" in data or "swagger" in data:
                                api_info["type"] = "openapi"
                                api_info["title"] = data.get("info", {}).get("title", "Unknown API")
                                api_info["version"] = data.get("info", {}).get("version", "unknown")
                            elif "data" in data and "__schema" in data.get("data", {}):
                                api_info["type"] = "graphql"
                        except:
                            pass
                    
                    discovered_apis.append(api_info)
                    
            except Exception as e:
                logger.debug(f"Discovery failed for {url}: {e}")
        
        logger.info(f"Discovered {len(discovered_apis)} APIs for domain {domain}")
        return discovered_apis
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.http_client.aclose()


# Global registry instance
_global_registry: Optional[MCPRegistry] = None


def get_registry() -> MCPRegistry:
    """Get the global MCP registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MCPRegistry()
    return _global_registry


async def cleanup_registry():
    """Cleanup the global registry."""
    global _global_registry
    if _global_registry:
        await _global_registry.cleanup()
        _global_registry = None
