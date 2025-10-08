"""
Simple Microsoft Agent Framework API for testing the frontend.
This version focuses on the Team Orchestrator functionality without complex dependencies.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager
from groq import Groq

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str

class TeamMember:
    def __init__(self, role: str, specialties: List[str], instructions: str):
        self.role = role
        self.specialties = specialties
        self.instructions = instructions
        self.available = True
        self.current_task = None
        self.tasks_completed = 0

class SimpleTeamOrchestrator:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.team_members = self._initialize_team()
    
    def _initialize_team(self):
        return {
            "code_assistant": TeamMember(
                role="code_assistant",
                specialties=["programming", "debugging", "code_review", "software_development"],
                instructions="You are a Code Assistant. You excel at programming, debugging, code review, and software development. Provide clear, working code solutions."
            ),
            "data_analyst": TeamMember(
                role="data_analyst", 
                specialties=["data_analysis", "statistics", "visualization", "reporting"],
                instructions="You are a Data Analyst. You excel at data analysis, statistics, visualization, and reporting. Help users understand their data."
            ),
            "customer_support": TeamMember(
                role="customer_support",
                specialties=["help_desk", "issue_resolution", "user_assistance", "troubleshooting"],
                instructions="You are a Customer Support specialist. You excel at help desk support, issue resolution, user assistance, and troubleshooting."
            ),
            "api_integrator": TeamMember(
                role="api_integrator",
                specialties=["api_integration", "external_services", "webhooks", "mcp_servers"],
                instructions="You are an API Integration specialist. You excel at connecting to external APIs, setting up webhooks, and integrating external services."
            ),
            "content_creator": TeamMember(
                role="content_creator",
                specialties=["writing", "documentation", "creative_content", "communication"],
                instructions="You are a Content Creator. You excel at writing, documentation, creative content, and communication. Create clear, engaging content."
            )
        }
    
    async def chat(self, message: str) -> str:
        # Analyze the message and determine which specialist to use
        analysis_prompt = f"""
        Analyze this user request and determine which specialist should handle it:
        
        Request: "{message}"
        
        Available specialists:
        - code_assistant: programming, debugging, code_review, software_development
        - data_analyst: data_analysis, statistics, visualization, reporting  
        - customer_support: help_desk, issue_resolution, user_assistance, troubleshooting
        - api_integrator: api_integration, external_services, webhooks, mcp_servers
        - content_creator: writing, documentation, creative_content, communication
        
        Respond with just the specialist name (e.g., "code_assistant") that best matches this request.
        """
        
        try:
            # Get specialist assignment
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=50
            )
            
            assigned_specialist = response.choices[0].message.content.strip().lower()
            
            # Default to code_assistant if no clear match
            if assigned_specialist not in self.team_members:
                assigned_specialist = "code_assistant"
            
            # Get the specialist
            specialist = self.team_members[assigned_specialist]
            specialist.available = False
            specialist.current_task = message[:50] + "..."
            
            # Have the specialist handle the request
            specialist_prompt = f"""
            {specialist.instructions}
            
            User Request: {message}
            
            Please provide a comprehensive, helpful response to this request.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": specialist_prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=4096
            )
            
            # Update specialist status
            specialist.available = True
            specialist.current_task = None
            specialist.tasks_completed += 1
            
            # Return response with team lead coordination
            team_response = f"**Team Lead**: I've assigned this to our {specialist.role.replace('_', ' ').title()} specialist.\n\n"
            team_response += f"**{specialist.role.replace('_', ' ').title()}**: {response.choices[0].message.content}"
            
            return team_response
            
        except Exception as e:
            return f"I apologize, but I encountered an error coordinating with the team: {str(e)}"
    
    def get_team_status(self):
        team_status = {}
        for role, member in self.team_members.items():
            team_status[role] = {
                "role": role,
                "specialties": member.specialties,
                "available": member.available,
                "current_task": member.current_task,
                "tasks_completed": member.tasks_completed
            }
        
        return {
            "team_members": team_status,
            "total_members": len(self.team_members),
            "available_members": sum(1 for m in self.team_members.values() if m.available),
            "active_tasks": sum(1 for m in self.team_members.values() if m.current_task)
        }

# Global variables
team_orchestrator: Optional[SimpleTeamOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global team_orchestrator
    
    # Startup
    print("🚀 Starting Microsoft Agent Framework...")
    
    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable is required")
        raise ValueError("GROQ_API_KEY is required")
    
    # Initialize team orchestrator
    team_orchestrator = SimpleTeamOrchestrator(groq_api_key)
    print("✅ Team orchestrator initialized")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Microsoft Agent Framework...")
    print("✅ Cleanup completed")

# Create FastAPI app
app = FastAPI(
    title="Microsoft Agent Framework API",
    description="API for building and managing AI agents with Groq models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "https://web--microsoft-agent-framework--4h7vh8ddvxpx.code.run",  # Production
        "*"  # Allow all origins for now
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Microsoft Agent Framework API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

# Team Orchestrator Endpoints
@app.post("/team/chat")
async def chat_with_team(request: ChatRequest):
    """Chat with the team orchestrator - single entry point for all communication."""
    if not team_orchestrator:
        raise HTTPException(status_code=500, detail="Team orchestrator not initialized")
    
    try:
        response = await team_orchestrator.chat(request.message)
        return {
            "response": response,
            "orchestrator": "Team Lead",
            "message": request.message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with team: {str(e)}")

@app.get("/team/status")
async def get_team_status():
    """Get current team status and member availability."""
    if not team_orchestrator:
        raise HTTPException(status_code=500, detail="Team orchestrator not initialized")
    
    try:
        status = team_orchestrator.get_team_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting team status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
