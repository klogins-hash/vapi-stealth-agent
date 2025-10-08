"""
Business Integrator + 5-Person Team Integration
Connects the EOS business framework with the Microsoft Agent Framework team
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional
from pathlib import Path

# Add team-agents to Python path
sys.path.append(str(Path(__file__).parent / "team-agents"))

try:
    from microsoft_agent_framework import AgentBuilder
    from microsoft_agent_framework.tools import WebTools, FileTools, CodeTools
    TEAM_FRAMEWORK_AVAILABLE = True
    print("✅ Microsoft Agent Framework loaded successfully")
except ImportError as e:
    print(f"⚠️ Microsoft Agent Framework not available: {e}")
    TEAM_FRAMEWORK_AVAILABLE = False

class BusinessTeamIntegrator:
    """
    Integrates the Business Integrator with the 5-Person Team Framework
    """
    
    def __init__(self):
        self.team_available = TEAM_FRAMEWORK_AVAILABLE
        self.agents = {}
        
        if self.team_available:
            self.builder = AgentBuilder()
            self._initialize_team_agents()
    
    def _initialize_team_agents(self):
        """Initialize the 5-person team agents"""
        try:
            # Define the 5-person team roles
            team_roles = {
                "lead_guy": {
                    "template": "coordinator",
                    "description": "Team coordinator and project manager",
                    "tools": ["web", "file", "communication"]
                },
                "technical_lead": {
                    "template": "developer", 
                    "description": "Technical architecture and development lead",
                    "tools": ["code", "file", "web"]
                },
                "business_analyst": {
                    "template": "analyst",
                    "description": "Business requirements and process analyst", 
                    "tools": ["web", "file", "data"]
                },
                "qa_specialist": {
                    "template": "tester",
                    "description": "Quality assurance and testing specialist",
                    "tools": ["code", "file", "web"]
                },
                "deployment_engineer": {
                    "template": "devops",
                    "description": "Deployment and infrastructure specialist",
                    "tools": ["code", "file", "web"]
                }
            }
            
            print(f"🎯 Initializing {len(team_roles)} team agents...")
            for role, config in team_roles.items():
                self.agents[role] = {
                    "config": config,
                    "status": "initialized",
                    "last_task": None
                }
                
            print(f"✅ Team agents initialized: {list(self.agents.keys())}")
            
        except Exception as e:
            print(f"❌ Failed to initialize team agents: {e}")
    
    async def delegate_task(self, task: str, role: str = "lead_guy", context: Dict = None) -> str:
        """
        Delegate a task to a specific team member
        """
        if not self.team_available:
            return f"Team framework not available. Task '{task}' noted for manual coordination."
        
        if role not in self.agents:
            return f"Role '{role}' not found. Available roles: {list(self.agents.keys())}"
        
        try:
            # Update agent status
            self.agents[role]["last_task"] = task
            self.agents[role]["status"] = "working"
            
            # For now, return a structured response
            # In full implementation, this would actually create and run the agent
            response = f"""Task delegated to {role}:
            
Task: {task}
Status: Assigned and in progress
Agent: {self.agents[role]['config']['description']}
Tools: {', '.join(self.agents[role]['config']['tools'])}

I'll coordinate with the {role} and provide updates on progress."""
            
            return response
            
        except Exception as e:
            return f"Failed to delegate task to {role}: {str(e)}"
    
    def get_team_status(self) -> Dict:
        """
        Get current status of all team members
        """
        if not self.team_available:
            return {"status": "Team framework not available", "agents": {}}
        
        team_status = {
            "framework_status": "operational",
            "total_agents": len(self.agents),
            "agents": {}
        }
        
        for role, agent in self.agents.items():
            team_status["agents"][role] = {
                "description": agent["config"]["description"],
                "status": agent["status"],
                "last_task": agent["last_task"],
                "tools": agent["config"]["tools"]
            }
        
        return team_status
    
    async def coordinate_project(self, project_name: str, requirements: List[str]) -> str:
        """
        Coordinate a multi-agent project across the team
        """
        if not self.team_available:
            return f"Project '{project_name}' noted. Team framework not available for coordination."
        
        # Assign tasks based on requirements
        task_assignments = []
        
        for req in requirements:
            req_lower = req.lower()
            
            if any(word in req_lower for word in ['code', 'develop', 'implement', 'build']):
                assigned_role = "technical_lead"
            elif any(word in req_lower for word in ['test', 'qa', 'quality', 'verify']):
                assigned_role = "qa_specialist"
            elif any(word in req_lower for word in ['deploy', 'infrastructure', 'server']):
                assigned_role = "deployment_engineer"
            elif any(word in req_lower for word in ['analyze', 'business', 'requirements']):
                assigned_role = "business_analyst"
            else:
                assigned_role = "lead_guy"
            
            task_assignments.append({
                "requirement": req,
                "assigned_to": assigned_role
            })
        
        # Generate coordination response
        response = f"""Project '{project_name}' coordination initiated:

Total requirements: {len(requirements)}
Team assignments:
"""
        
        for assignment in task_assignments:
            response += f"• {assignment['requirement']} → {assignment['assigned_to']}\n"
        
        response += f"\nLead guy will coordinate overall project execution. All agents are ready to begin work."
        
        return response

# Global instance for use in stealth_agent.py
business_team_integrator = BusinessTeamIntegrator()

if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        print("🧪 Testing Business Team Integration...")
        
        # Test team status
        status = business_team_integrator.get_team_status()
        print(f"Team Status: {status}")
        
        # Test task delegation
        result = await business_team_integrator.delegate_task(
            "Review and optimize the business integrator deployment",
            "technical_lead"
        )
        print(f"Delegation Result: {result}")
        
        # Test project coordination
        project_result = await business_team_integrator.coordinate_project(
            "EOS Framework Enhancement",
            [
                "Analyze current EOS implementation",
                "Develop enhanced reporting features", 
                "Test integration with lead-guy service",
                "Deploy to Northflank production"
            ]
        )
        print(f"Project Coordination: {project_result}")
    
    asyncio.run(test_integration())
