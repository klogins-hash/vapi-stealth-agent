"""Microsoft Graph API integration for the Agent Framework."""

import os
from typing import Dict, List, Any, Optional
import aiohttp
import json
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from pydantic import BaseModel


class GraphAPIConfig(BaseModel):
    """Configuration for Microsoft Graph API."""
    client_id: str
    client_secret: Optional[str] = None
    tenant_id: str
    scopes: List[str] = ["https://graph.microsoft.com/.default"]


class GraphAPIClient:
    """Client for Microsoft Graph API operations."""
    
    def __init__(self, config: Optional[GraphAPIConfig] = None):
        """Initialize Graph API client."""
        if config is None:
            config = GraphAPIConfig(
                client_id=os.getenv("MICROSOFT_CLIENT_ID", ""),
                client_secret=os.getenv("MICROSOFT_CLIENT_SECRET"),
                tenant_id=os.getenv("MICROSOFT_TENANT_ID", "")
            )
        
        self.config = config
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.token = None
        self.token_expires = None
        
        # Initialize credential
        if config.client_secret:
            self.credential = ClientSecretCredential(
                tenant_id=config.tenant_id,
                client_id=config.client_id,
                client_secret=config.client_secret
            )
        else:
            self.credential = DefaultAzureCredential()
    
    async def _get_access_token(self) -> str:
        """Get access token for Graph API."""
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            return self.token
        
        token_result = self.credential.get_token(*self.config.scopes)
        self.token = token_result.token
        self.token_expires = datetime.fromtimestamp(token_result.expires_on)
        
        return self.token
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Graph API."""
        token = await self._get_access_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"Graph API error {response.status}: {error_text}")
                
                return await response.json()
    
    async def get_user_profile(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user profile information."""
        endpoint = f"/users/{user_id}" if user_id else "/me"
        return await self._make_request("GET", endpoint)
    
    async def get_calendar_events(
        self,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get calendar events for a user."""
        endpoint = f"/users/{user_id}/events" if user_id else "/me/events"
        
        params = {}
        if start_time and end_time:
            params["$filter"] = f"start/dateTime ge '{start_time.isoformat()}' and end/dateTime le '{end_time.isoformat()}'"
        
        response = await self._make_request("GET", endpoint, params=params)
        return response.get("value", [])
    
    async def create_calendar_event(
        self,
        subject: str,
        start_time: datetime,
        end_time: datetime,
        attendees: Optional[List[str]] = None,
        body: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a calendar event."""
        endpoint = f"/users/{user_id}/events" if user_id else "/me/events"
        
        event_data = {
            "subject": subject,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "UTC"
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "UTC"
            }
        }
        
        if body:
            event_data["body"] = {
                "contentType": "text",
                "content": body
            }
        
        if attendees:
            event_data["attendees"] = [
                {
                    "emailAddress": {"address": email, "name": email},
                    "type": "required"
                }
                for email in attendees
            ]
        
        return await self._make_request("POST", endpoint, data=event_data)
    
    async def get_teams(self) -> List[Dict[str, Any]]:
        """Get user's teams."""
        response = await self._make_request("GET", "/me/joinedTeams")
        return response.get("value", [])
    
    async def get_team_channels(self, team_id: str) -> List[Dict[str, Any]]:
        """Get channels for a team."""
        response = await self._make_request("GET", f"/teams/{team_id}/channels")
        return response.get("value", [])
    
    async def send_teams_message(
        self,
        team_id: str,
        channel_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Send a message to a Teams channel."""
        endpoint = f"/teams/{team_id}/channels/{channel_id}/messages"
        
        message_data = {
            "body": {
                "contentType": "text",
                "content": message
            }
        }
        
        return await self._make_request("POST", endpoint, data=message_data)
    
    async def search_files(
        self,
        query: str,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for files in OneDrive."""
        endpoint = f"/users/{user_id}/drive/root/search(q='{query}')" if user_id else f"/me/drive/root/search(q='{query}')"
        response = await self._make_request("GET", endpoint)
        return response.get("value", [])
    
    async def get_emails(
        self,
        user_id: Optional[str] = None,
        folder: str = "inbox",
        top: int = 10
    ) -> List[Dict[str, Any]]:
        """Get emails from a folder."""
        endpoint = f"/users/{user_id}/mailFolders/{folder}/messages" if user_id else f"/me/mailFolders/{folder}/messages"
        params = {"$top": top, "$orderby": "receivedDateTime desc"}
        
        response = await self._make_request("GET", endpoint, params=params)
        return response.get("value", [])


class GraphAPITools:
    """Tools for Graph API integration with agents."""
    
    def __init__(self, graph_client: GraphAPIClient):
        """Initialize Graph API tools."""
        self.graph_client = graph_client
    
    async def get_my_profile(self) -> str:
        """Tool: Get current user's profile."""
        try:
            profile = await self.graph_client.get_user_profile()
            return f"User Profile: {profile.get('displayName', 'Unknown')} ({profile.get('mail', 'No email')})"
        except Exception as e:
            return f"Error getting profile: {str(e)}"
    
    async def get_today_events(self) -> str:
        """Tool: Get today's calendar events."""
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            events = await self.graph_client.get_calendar_events(
                start_time=today,
                end_time=tomorrow
            )
            
            if not events:
                return "No events scheduled for today."
            
            event_list = []
            for event in events:
                start_time = event.get('start', {}).get('dateTime', 'Unknown time')
                subject = event.get('subject', 'No subject')
                event_list.append(f"- {start_time}: {subject}")
            
            return f"Today's events:\n" + "\n".join(event_list)
        except Exception as e:
            return f"Error getting events: {str(e)}"
    
    async def schedule_meeting(
        self,
        subject: str,
        start_time_str: str,
        duration_minutes: int = 60,
        attendees: Optional[str] = None
    ) -> str:
        """Tool: Schedule a meeting."""
        try:
            # Parse start time (simplified - in production, use proper date parsing)
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            attendee_list = []
            if attendees:
                attendee_list = [email.strip() for email in attendees.split(',')]
            
            event = await self.graph_client.create_calendar_event(
                subject=subject,
                start_time=start_time,
                end_time=end_time,
                attendees=attendee_list
            )
            
            return f"Meeting '{subject}' scheduled successfully. Event ID: {event.get('id', 'Unknown')}"
        except Exception as e:
            return f"Error scheduling meeting: {str(e)}"
    
    async def search_my_files(self, query: str) -> str:
        """Tool: Search files in OneDrive."""
        try:
            files = await self.graph_client.search_files(query)
            
            if not files:
                return f"No files found matching '{query}'."
            
            file_list = []
            for file in files[:5]:  # Limit to 5 results
                name = file.get('name', 'Unknown')
                web_url = file.get('webUrl', 'No URL')
                file_list.append(f"- {name}: {web_url}")
            
            return f"Files matching '{query}':\n" + "\n".join(file_list)
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    async def get_recent_emails(self, count: int = 5) -> str:
        """Tool: Get recent emails."""
        try:
            emails = await self.graph_client.get_emails(top=count)
            
            if not emails:
                return "No recent emails found."
            
            email_list = []
            for email in emails:
                subject = email.get('subject', 'No subject')
                sender = email.get('from', {}).get('emailAddress', {}).get('name', 'Unknown sender')
                received = email.get('receivedDateTime', 'Unknown time')
                email_list.append(f"- From {sender}: {subject} ({received})")
            
            return f"Recent emails:\n" + "\n".join(email_list)
        except Exception as e:
            return f"Error getting emails: {str(e)}"
