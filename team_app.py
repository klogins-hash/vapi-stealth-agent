"""FastAPI web application for the Microsoft Agent Framework."""

import os
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager

from src.microsoft_agent_framework import AgentBuilder, TeamOrchestrator
from src.microsoft_agent_framework.database import DatabaseManager, get_database, init_database
from src.microsoft_agent_framework.database.models import Agent as AgentModel, Conversation, Message
from src.microsoft_agent_framework.tools import WebTools, FileTools, CodeTools
from src.microsoft_agent_framework.mcp import APISpecificationParser, MCPServerGenerator, get_registry


# Pydantic models for API
class CreateAgentRequest(BaseModel):
    name: str
    template_name: Optional[str] = None
    instructions: Optional[str] = None
    model: Optional[str] = "llama-3.1-70b-versatile"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    tools: Optional[List[str]] = []


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_id: str


class GenerateMCPServerRequest(BaseModel):
    api_source: str
    api_type: str = "openapi"  # openapi, graphql, rest_discovery, postman
    server_type: str = "http"  # stdio, http, websocket
    deployment_target: str = "local"  # local, docker, kubernetes
    server_name: Optional[str] = None


class CreateAgentWithAPIRequest(BaseModel):
    name: str
    api_specs: List[str]
    instructions: str
    template_name: Optional[str] = None
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.7


class DiscoverAPIsRequest(BaseModel):
    domain: str


# Global variables
agent_builder: Optional[AgentBuilder] = None
team_orchestrator: Optional[TeamOrchestrator] = None
web_tools: Optional[WebTools] = None
file_tools: Optional[FileTools] = None
code_tools: Optional[CodeTools] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent_builder, team_orchestrator, web_tools, file_tools, code_tools
    
    # Startup
    print("🚀 Starting Microsoft Agent Framework...")
    
    try:
        # Initialize database
        await init_database()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")
        print("🔄 Continuing without database (will retry on first request)")
    
    # Initialize agent builder and tools
    agent_builder = AgentBuilder()
    team_orchestrator = TeamOrchestrator()
    web_tools = WebTools()
    file_tools = FileTools()
    code_tools = CodeTools()
    
    # Register tools
    agent_builder.register_tool("fetch_url", web_tools.fetch_url, "Fetch content from a URL")
    agent_builder.register_tool("read_file", file_tools.read_file, "Read content from a file")
    agent_builder.register_tool("write_file", file_tools.write_file, "Write content to a file")
    agent_builder.register_tool("execute_python", code_tools.execute_python, "Execute Python code")
    agent_builder.register_tool("validate_syntax", code_tools.validate_python_syntax, "Validate Python syntax")
    
    print("✅ Agent builder, team orchestrator, and tools initialized")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Microsoft Agent Framework...")
    if web_tools:
        await web_tools.close()
    
    if agent_builder:
        await agent_builder.cleanup()
    
    db = get_database()
    await db.close()
    print("✅ Cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Microsoft Agent Framework API",
    description="API for building and managing AI agents with Groq models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "https://web--microsoft-agent-framework--4h7vh8ddvxpx.code.run",  # Production
        "*"  # Allow all origins for now - configure appropriately for production
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
async def health_check():
    """Health check endpoint for Kubernetes probes."""
    from datetime import datetime
    
    # Check if core services are initialized
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "agent_builder": agent_builder is not None,
            "team_orchestrator": team_orchestrator is not None,
            "web_tools": web_tools is not None,
            "file_tools": file_tools is not None,
            "code_tools": code_tools is not None
        }
    }
    
    # If any core service is not initialized, return unhealthy
    if not all(health_status["services"].values()):
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@app.get("/templates")
async def list_templates():
    """List available agent templates."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    templates = agent_builder.list_templates()
    return {"templates": templates}


@app.post("/agents")
async def create_agent(request: CreateAgentRequest):
    """Create a new agent."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        # Create agent using builder
        if request.template_name:
            agent = agent_builder.create_agent_from_template(
                template_name=request.template_name,
                name=request.name,
                custom_instructions=request.instructions,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        else:
            if not request.instructions:
                raise HTTPException(status_code=400, detail="Instructions required for custom agent")
            
            agent = agent_builder.create_custom_agent(
                name=request.name,
                instructions=request.instructions,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools
            )
        
        # Save agent to database
        db = get_database()
        async with db.get_session() as session:
            db_agent = AgentModel(
                name=agent.config.name,
                template_name=request.template_name,
                instructions=agent.config.instructions,
                model=agent.config.model or "llama3-70b-8192",
                temperature=str(agent.config.temperature or 0.7),
                max_tokens=agent.config.max_tokens or 4096,
                tools=request.tools or [],
                metadata=agent.config.metadata
            )
            session.add(db_agent)
            await session.flush()
            
            return {
                "agent_id": db_agent.id,
                "name": db_agent.name,
                "template_name": db_agent.template_name,
                "message": "Agent created successfully"
            }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List all agents."""
    db = get_database()
    async with db.get_session() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(AgentModel).where(AgentModel.is_active == True)
        )
        agents = result.scalars().all()
        
        return {
            "agents": [agent.to_dict() for agent in agents]
        }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent details."""
    db = get_database()
    async with db.get_session() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(AgentModel).where(AgentModel.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent.to_dict()


@app.post("/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: str, request: ChatRequest):
    """Chat with an agent."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    db = get_database()
    
    try:
        async with db.get_session() as session:
            from sqlalchemy import select
            
            # Get agent from database
            result = await session.execute(
                select(AgentModel).where(AgentModel.id == agent_id)
            )
            db_agent = result.scalar_one_or_none()
            
            if not db_agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Create agent instance
            agent = agent_builder.create_custom_agent(
                name=db_agent.name,
                instructions=db_agent.instructions,
                model=db_agent.model,
                temperature=float(db_agent.temperature),
                max_tokens=db_agent.max_tokens,
                tools=db_agent.tools
            )
            
            # Get or create conversation
            conversation_id = request.conversation_id
            if conversation_id:
                result = await session.execute(
                    select(Conversation).where(Conversation.id == conversation_id)
                )
                conversation = result.scalar_one_or_none()
                if not conversation:
                    raise HTTPException(status_code=404, detail="Conversation not found")
            else:
                conversation = Conversation(
                    agent_id=agent_id,
                    title=f"Chat with {db_agent.name}"
                )
                session.add(conversation)
                await session.flush()
                conversation_id = conversation.id
            
            # Add user message to database
            user_message = Message(
                conversation_id=conversation_id,
                role="user",
                content=request.message
            )
            session.add(user_message)
            
            # Get agent response
            response = await agent.run_async(request.message)
            
            # Add assistant message to database
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=response.content
            )
            session.add(assistant_message)
            
            return ChatResponse(
                response=response.content,
                conversation_id=conversation_id,
                agent_id=agent_id
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_id}/chat/stream")
async def stream_chat_with_agent(agent_id: str, request: ChatRequest):
    """Stream chat with an agent."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    async def generate_response():
        db = get_database()
        
        try:
            async with db.get_session() as session:
                from sqlalchemy import select
                
                # Get agent from database
                result = await session.execute(
                    select(AgentModel).where(AgentModel.id == agent_id)
                )
                db_agent = result.scalar_one_or_none()
                
                if not db_agent:
                    yield f"data: {json.dumps({'error': 'Agent not found'})}\n\n"
                    return
                
                # Create agent instance
                agent = agent_builder.create_custom_agent(
                    name=db_agent.name,
                    instructions=db_agent.instructions,
                    model=db_agent.model,
                    temperature=float(db_agent.temperature),
                    max_tokens=db_agent.max_tokens,
                    tools=db_agent.tools
                )
                
                # Stream response
                full_response = ""
                async for update in agent.run_streaming_async(request.message):
                    if not update.is_complete:
                        full_response += update.content
                        yield f"data: {json.dumps({'content': update.content, 'done': False})}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    db = get_database()
    async with db.get_session() as session:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "conversation": conversation.to_dict(),
            "messages": [msg.to_dict() for msg in conversation.messages]
        }


@app.post("/build-agent")
async def build_agent_from_description(description: str):
    """Build an agent from natural language description."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        recommendation = await agent_builder.get_agent_recommendation(description)
        return {
            "recommendation": recommendation,
            "description": description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MCP Integration Endpoints

@app.post("/mcp/generate-server")
async def generate_mcp_server(request: GenerateMCPServerRequest):
    """Generate MCP server from API specification."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        server_info = await agent_builder.create_mcp_server_from_api(
            api_source=request.api_source,
            api_type=request.api_type,
            server_type=request.server_type,
            deployment_target=request.deployment_target,
            server_name=request.server_name
        )
        
        return {
            "server_info": {
                "name": server_info.name,
                "transport_type": server_info.transport_type,
                "connection_info": server_info.connection_info,
                "capabilities": server_info.capabilities,
                "status": server_info.status
            },
            "message": f"MCP server '{server_info.name}' generated and deployed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating MCP server: {str(e)}")


@app.get("/mcp/servers")
async def list_mcp_servers():
    """List all registered MCP servers."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        servers = agent_builder.list_mcp_servers()
        return {
            "servers": servers,
            "count": len(servers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing MCP servers: {str(e)}")


@app.get("/mcp/servers/search")
async def search_mcp_servers(query: str):
    """Search MCP servers by capability or name."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        results = agent_builder.search_mcp_servers(query)
        return {
            "results": results,
            "query": query,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching MCP servers: {str(e)}")


@app.get("/mcp/servers/health")
async def health_check_mcp_servers():
    """Perform health checks on all MCP servers."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        health_results = await agent_builder.health_check_mcp_servers()
        healthy_count = sum(1 for status in health_results.values() if status is True)
        total_count = len(health_results)
        
        return {
            "health_results": health_results,
            "summary": {
                "healthy": healthy_count,
                "total": total_count,
                "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking MCP server health: {str(e)}")


@app.post("/mcp/discover-apis")
async def discover_apis(request: DiscoverAPIsRequest):
    """Discover APIs in a domain."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        discovered_apis = await agent_builder.discover_apis(request.domain)
        return {
            "domain": request.domain,
            "discovered_apis": discovered_apis,
            "count": len(discovered_apis)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error discovering APIs: {str(e)}")


@app.post("/agents/create-with-api")
async def create_agent_with_api_integration(request: CreateAgentWithAPIRequest):
    """Create agent with automatic API integration via MCP."""
    if not agent_builder:
        raise HTTPException(status_code=500, detail="Agent builder not initialized")
    
    try:
        # Create agent with API integration
        agent = await agent_builder.create_agent_with_api_integration(
            name=request.name,
            api_specs=request.api_specs,
            instructions=request.instructions,
            template_name=request.template_name,
            model=request.model,
            temperature=request.temperature
        )
        
        # Store agent in database
        db = get_database()
        async with db.get_session() as session:
            agent_model = AgentModel(
                name=request.name,
                instructions=request.instructions,
                model=request.model,
                temperature=str(request.temperature),
                tools=request.api_specs,
                agent_metadata={"api_integrated": True, "api_specs": request.api_specs}
            )
            session.add(agent_model)
            await session.commit()
            await session.refresh(agent_model)
            
            return {
                "agent": agent_model.to_dict(),
                "message": f"Agent '{request.name}' created with API integration",
                "api_integrations": len(request.api_specs)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent with API integration: {str(e)}")


@app.get("/mcp/templates")
async def list_mcp_templates():
    """List available MCP server templates."""
    return {
        "templates": [
            {
                "name": "rest_api",
                "description": "Standard REST API with CRUD operations",
                "supported_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "features": ["Path parameters", "Query parameters", "Request body handling"]
            },
            {
                "name": "graphql_api", 
                "description": "GraphQL API with queries and mutations",
                "supported_operations": ["Query", "Mutation", "Subscription"],
                "features": ["Query operations", "Variable handling", "Introspection"]
            },
            {
                "name": "webhook_api",
                "description": "Webhook-based event-driven API",
                "supported_events": ["create", "update", "delete", "custom"],
                "features": ["Event subscription", "Signature verification", "Retry mechanisms"]
            },
            {
                "name": "streaming_api",
                "description": "Real-time streaming data API", 
                "supported_protocols": ["WebSocket", "Server-Sent Events", "HTTP Streaming"],
                "features": ["Real-time data", "Connection management", "Backpressure handling"]
            }
        ]
    }


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
        status = await team_orchestrator.get_team_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting team status: {str(e)}")


@app.post("/team/add-member")
async def add_team_member(
    role: str,
    specialties: List[str],
    template_name: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    api_specs: Optional[List[str]] = None
):
    """Add a new team member with specific role and specialties."""
    if not team_orchestrator:
        raise HTTPException(status_code=500, detail="Team orchestrator not initialized")
    
    try:
        member_id = await team_orchestrator.add_team_member(
            role=role,
            specialties=specialties,
            template_name=template_name,
            custom_instructions=custom_instructions,
            api_specs=api_specs
        )
        
        return {
            "member_id": member_id,
            "role": role,
            "specialties": specialties,
            "message": f"Team member '{role}' added successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding team member: {str(e)}")


@app.post("/team/enhance-member")
async def enhance_team_member_with_api(role: str, api_specs: List[str]):
    """Enhance an existing team member with API integration capabilities."""
    if not team_orchestrator:
        raise HTTPException(status_code=500, detail="Team orchestrator not initialized")
    
    try:
        success = await team_orchestrator.add_api_integration_to_member(role, api_specs)
        
        if success:
            return {
                "role": role,
                "api_specs": api_specs,
                "message": f"Enhanced '{role}' with API integration capabilities"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Team member '{role}' not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing team member: {str(e)}")


@app.get("/team/members")
async def list_team_members():
    """List all team members and their capabilities."""
    if not team_orchestrator:
        raise HTTPException(status_code=500, detail="Team orchestrator not initialized")
    
    try:
        status = await team_orchestrator.get_team_status()
        return {
            "team_members": status["team_members"],
            "summary": {
                "total_members": status["total_members"],
                "available_members": status["available_members"],
                "active_tasks": status["active_tasks"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing team members: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
