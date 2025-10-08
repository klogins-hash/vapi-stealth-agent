"""Team Orchestrator for managing and coordinating multiple agents."""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base_agent import BaseAgent, ChatCompletionAgent, AgentConfig
from .groq_client import GroqClient
from .context_provider import InMemoryContextProvider
from .agent_builder import AgentBuilder

logger = logging.getLogger(__name__)


@dataclass
class TeamMember:
    """Represents a team member agent."""
    agent: BaseAgent
    role: str
    specialties: List[str]
    availability: bool = True
    current_task: Optional[str] = None
    performance_score: float = 1.0
    task_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Task:
    """Represents a task to be executed."""
    id: str
    description: str
    priority: str = "medium"  # low, medium, high, urgent
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class TeamOrchestrator:
    """Central orchestrator that manages and coordinates a team of specialized agents."""
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        """Initialize the team orchestrator."""
        self.groq_client = groq_client or GroqClient()
        self.agent_builder = AgentBuilder(groq_client)
        
        # Team management
        self.team_members: Dict[str, TeamMember] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
        
        # Create the orchestrator agent
        self.orchestrator_agent = self._create_orchestrator_agent()
        
        # Initialize with default team members
        asyncio.create_task(self._initialize_default_team())
    
    def _create_orchestrator_agent(self) -> BaseAgent:
        """Create the main orchestrator agent."""
        config = AgentConfig(
            name="Team Lead",
            instructions="""You are the Team Lead and central orchestrator for a team of specialized AI agents. 

Your responsibilities:
1. **Task Analysis**: Break down complex requests into specific tasks
2. **Agent Assignment**: Assign tasks to the most suitable team members based on their specialties
3. **Coordination**: Manage task dependencies and workflow
4. **Communication**: Provide clear, consolidated responses to users
5. **Quality Control**: Review and synthesize results from team members

Your team includes:
- Code Assistant: Software development, debugging, code review
- Data Analyst: Data analysis, visualization, statistics
- Customer Support: Help desk, issue resolution, user assistance  
- API Integrator: API connections, MCP server creation, external integrations
- Content Creator: Writing, documentation, creative content

When you receive a request:
1. Analyze what needs to be done
2. Determine which team members should handle it
3. Break it into specific tasks if needed
4. Coordinate the work
5. Provide a unified response

Be professional, efficient, and always consider the best team member for each task.""",
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=4096
        )
        
        return ChatCompletionAgent(
            config=config,
            groq_client=self.groq_client,
            context_provider=InMemoryContextProvider()
        )
    
    async def _initialize_default_team(self):
        """Initialize the team with default specialized agents."""
        try:
            # Code Assistant
            await self.add_team_member(
                role="code_assistant",
                specialties=["programming", "debugging", "code_review", "software_development"],
                template_name="code_assistant"
            )
            
            # Data Analyst
            await self.add_team_member(
                role="data_analyst", 
                specialties=["data_analysis", "statistics", "visualization", "reporting"],
                template_name="data_analyst"
            )
            
            # Customer Support
            await self.add_team_member(
                role="customer_support",
                specialties=["help_desk", "issue_resolution", "user_assistance", "troubleshooting"],
                template_name="customer_support"
            )
            
            # API Integrator (uses MCP capabilities)
            await self.add_team_member(
                role="api_integrator",
                specialties=["api_integration", "mcp_servers", "external_services", "webhooks"],
                custom_instructions="""You are an API Integration Specialist. You excel at:
                - Connecting to external APIs and services
                - Creating MCP servers from API specifications
                - Setting up webhooks and real-time integrations
                - Troubleshooting API connectivity issues
                - Building agents with API tool integration
                
                Use your MCP capabilities to integrate with any external service."""
            )
            
            # Content Creator
            await self.add_team_member(
                role="content_creator",
                specialties=["writing", "documentation", "creative_content", "communication"],
                custom_instructions="""You are a Content Creation Specialist. You excel at:
                - Writing clear, engaging content
                - Creating technical documentation
                - Developing marketing materials
                - Crafting user guides and tutorials
                - Editing and proofreading content
                
                Always maintain a professional yet approachable tone."""
            )
            
            logger.info(f"Team initialized with {len(self.team_members)} members")
            
        except Exception as e:
            logger.error(f"Error initializing team: {e}")
    
    async def add_team_member(
        self, 
        role: str, 
        specialties: List[str],
        template_name: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        api_specs: Optional[List[str]] = None
    ) -> str:
        """Add a new team member."""
        try:
            if api_specs:
                # Create agent with API integration
                agent = await self.agent_builder.create_agent_with_api_integration(
                    name=f"{role.replace('_', ' ').title()} Agent",
                    api_specs=api_specs,
                    instructions=custom_instructions or f"You are a {role.replace('_', ' ')} specialist.",
                    template_name=template_name
                )
            elif template_name:
                # Create from template
                agent = self.agent_builder.create_agent_from_template(
                    template_name,
                    name=f"{role.replace('_', ' ').title()} Agent"
                )
                if custom_instructions:
                    agent.config.instructions = custom_instructions
            else:
                # Create custom agent
                config = AgentConfig(
                    name=f"{role.replace('_', ' ').title()} Agent",
                    instructions=custom_instructions or f"You are a {role.replace('_', ' ')} specialist.",
                    model="llama-3.1-70b-versatile",
                    temperature=0.7
                )
                agent = ChatCompletionAgent(
                    config=config,
                    groq_client=self.groq_client,
                    context_provider=InMemoryContextProvider()
                )
            
            # Add to team
            team_member = TeamMember(
                agent=agent,
                role=role,
                specialties=specialties
            )
            
            self.team_members[role] = team_member
            logger.info(f"Added team member: {role}")
            return role
            
        except Exception as e:
            logger.error(f"Error adding team member {role}: {e}")
            raise
    
    async def process_request(self, user_request: str) -> str:
        """Process a user request through the team orchestrator."""
        try:
            # First, let the orchestrator analyze the request
            analysis_prompt = f"""
            Analyze this user request and determine the best approach:
            
            Request: "{user_request}"
            
            Available team members and their specialties:
            {self._get_team_summary()}
            
            Please provide:
            1. Task breakdown (if complex)
            2. Which team member(s) should handle this
            3. Any specific instructions for the assigned agent(s)
            4. Priority level (low/medium/high/urgent)
            
            Format your response as JSON:
            {{
                "analysis": "your analysis",
                "tasks": [
                    {{
                        "description": "task description",
                        "assigned_to": "team_member_role",
                        "priority": "medium",
                        "instructions": "specific instructions"
                    }}
                ],
                "coordination_needed": true/false
            }}
            """
            
            orchestrator_response = await self.orchestrator_agent.run_async(analysis_prompt)
            
            try:
                # Parse the orchestrator's analysis
                analysis = json.loads(orchestrator_response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return await self._handle_simple_request(user_request)
            
            # Execute the tasks
            if len(analysis.get("tasks", [])) == 1:
                # Single task - direct execution
                task = analysis["tasks"][0]
                result = await self._execute_task(task, user_request)
                return result
            elif len(analysis.get("tasks", [])) > 1:
                # Multiple tasks - coordinate execution
                return await self._coordinate_multiple_tasks(analysis["tasks"], user_request)
            else:
                # No specific tasks - handle directly
                return await self._handle_simple_request(user_request)
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    async def _execute_task(self, task_info: Dict[str, Any], original_request: str) -> str:
        """Execute a single task with the assigned team member."""
        assigned_role = task_info.get("assigned_to")
        
        if assigned_role not in self.team_members:
            return f"Sorry, I don't have a {assigned_role} available on the team."
        
        team_member = self.team_members[assigned_role]
        
        if not team_member.availability:
            return f"The {assigned_role} is currently busy. Please try again later."
        
        try:
            # Mark as busy
            team_member.availability = False
            team_member.current_task = task_info.get("description", original_request)
            
            # Prepare the request for the specialist
            specialist_prompt = f"""
            {task_info.get('instructions', '')}
            
            User Request: {original_request}
            Specific Task: {task_info.get('description', 'Handle the user request')}
            
            Please provide a comprehensive response for this task.
            """
            
            # Execute with the specialist
            response = await team_member.agent.run_async(specialist_prompt)
            
            # Update task history
            task_record = {
                "request": original_request,
                "task": task_info.get("description"),
                "completed_at": datetime.utcnow().isoformat(),
                "success": True
            }
            team_member.task_history.append(task_record)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error executing task with {assigned_role}: {e}")
            return f"I encountered an error while working on this task: {str(e)}"
        
        finally:
            # Mark as available again
            team_member.availability = True
            team_member.current_task = None
    
    async def _coordinate_multiple_tasks(self, tasks: List[Dict[str, Any]], original_request: str) -> str:
        """Coordinate execution of multiple tasks."""
        results = []
        
        # Execute tasks (could be done in parallel for independent tasks)
        for task in tasks:
            result = await self._execute_task(task, original_request)
            results.append({
                "task": task.get("description"),
                "result": result
            })
        
        # Let the orchestrator synthesize the results
        synthesis_prompt = f"""
        I coordinated multiple team members to handle this request: "{original_request}"
        
        Here are the results from each specialist:
        {json.dumps(results, indent=2)}
        
        Please provide a unified, comprehensive response that synthesizes these results into a coherent answer for the user.
        """
        
        final_response = await self.orchestrator_agent.run_async(synthesis_prompt)
        return final_response.content
    
    async def _handle_simple_request(self, user_request: str) -> str:
        """Handle simple requests directly with the orchestrator."""
        response = await self.orchestrator_agent.run_async(user_request)
        return response.content
    
    def _get_team_summary(self) -> str:
        """Get a summary of available team members."""
        summary = []
        for role, member in self.team_members.items():
            status = "Available" if member.availability else f"Busy with: {member.current_task}"
            summary.append(f"- {role}: {', '.join(member.specialties)} ({status})")
        return "\n".join(summary)
    
    async def get_team_status(self) -> Dict[str, Any]:
        """Get current team status."""
        team_status = {}
        
        for role, member in self.team_members.items():
            team_status[role] = {
                "role": role,
                "specialties": member.specialties,
                "available": member.availability,
                "current_task": member.current_task,
                "performance_score": member.performance_score,
                "tasks_completed": len(member.task_history)
            }
        
        return {
            "team_members": team_status,
            "total_members": len(self.team_members),
            "available_members": sum(1 for m in self.team_members.values() if m.availability),
            "active_tasks": sum(1 for m in self.team_members.values() if m.current_task)
        }
    
    async def add_api_integration_to_member(self, role: str, api_specs: List[str]) -> bool:
        """Add API integration capabilities to an existing team member."""
        if role not in self.team_members:
            return False
        
        try:
            # Create new agent with API integration
            member = self.team_members[role]
            enhanced_agent = await self.agent_builder.create_agent_with_api_integration(
                name=member.agent.config.name,
                api_specs=api_specs,
                instructions=member.agent.config.instructions,
                model=member.agent.config.model,
                temperature=member.agent.config.temperature
            )
            
            # Replace the agent
            member.agent = enhanced_agent
            member.specialties.extend(["api_integration", "external_services"])
            
            logger.info(f"Enhanced {role} with API integration: {api_specs}")
            return True
            
        except Exception as e:
            logger.error(f"Error enhancing {role} with API integration: {e}")
            return False
    
    async def chat(self, message: str) -> str:
        """Main chat interface - single entry point for all communication."""
        return await self.process_request(message)
