"""Web-related tools for agents."""

import aiohttp
import json
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import asyncio


class WebTools:
    """Web-related tools for agents."""
    
    def __init__(self):
        """Initialize web tools."""
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def fetch_url(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Tool: Fetch content from a URL."""
        try:
            session = await self._get_session()
            async with session.get(url, headers=headers or {}) as response:
                if response.status == 200:
                    content = await response.text()
                    return f"Successfully fetched content from {url}:\n{content[:1000]}..."
                else:
                    return f"Error fetching {url}: HTTP {response.status}"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    async def post_json(self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
        """Tool: Post JSON data to a URL."""
        try:
            session = await self._get_session()
            default_headers = {"Content-Type": "application/json"}
            if headers:
                default_headers.update(headers)
            
            async with session.post(url, json=data, headers=default_headers) as response:
                response_text = await response.text()
                return f"POST to {url} - Status: {response.status}\nResponse: {response_text[:500]}..."
        except Exception as e:
            return f"Error posting to {url}: {str(e)}"
    
    async def check_url_status(self, url: str) -> str:
        """Tool: Check if a URL is accessible."""
        try:
            session = await self._get_session()
            async with session.head(url) as response:
                return f"URL {url} is accessible - Status: {response.status}"
        except Exception as e:
            return f"URL {url} is not accessible: {str(e)}"
    
    async def search_web(self, query: str) -> str:
        """Tool: Simple web search (placeholder - would need actual search API)."""
        # This is a placeholder - in a real implementation, you'd integrate with
        # a search API like Google Custom Search, Bing, or DuckDuckGo
        return f"Web search for '{query}' - This would return search results in a real implementation. Consider integrating with a search API."
    
    async def close(self):
        """Close the web session."""
        if self.session and not self.session.closed:
            await self.session.close()
