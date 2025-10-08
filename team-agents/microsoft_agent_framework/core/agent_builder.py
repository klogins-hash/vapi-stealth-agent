"""Agent builder for creating and managing Microsoft agents using Groq models."""

from typing import Dict, List, Any, Optional, Type, Callable, Union
from pydantic import BaseModel
import json
import yaml
from pathlib import Path
import logging

from .base_agent import BaseAgent, ChatCompletionAgent, AgentConfig
from .groq_client import GroqClient, GroqConfig
from .context_provider import ContextProvider, InMemoryContextProvider, FileContextProvider
from .agent_thread import AgentThread
from ..mcp import (
    APISpecificationParser, APIDefinition, MCPServerGenerator, 
    MCPServerCode, MCPServerInfo, MCPRegistry, get_registry,
    MCPClient, MCPTool
)

logger = logging.getLogger(__name__)


class AgentTemplate(BaseModel):
    """Template for creating agents."""
    name: str
    description: str
    instructions: str
    model: str = "llama3-70b-8192"
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[str] = []
    context_type: str = "memory"  # "memory", "file"
    metadata: Dict[str, Any] = {}


class AgentBlueprint(BaseModel):
    """Blueprint for a specific agent type."""
    template: AgentTemplate
    tools: Dict[str, Dict[str, Any]] = {}
    middleware: List[str] = []
    examples: List[Dict[str, str]] = []


class AgentBuilder:
    """Expert agent builder for creating Microsoft agents."""
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        """Initialize the agent builder."""
        self.groq_client = groq_client or GroqClient()
        self.templates: Dict[str, AgentTemplate] = {}
        self.blueprints: Dict[str, AgentBlueprint] = {}
        self.tools_registry: Dict[str, Callable] = {}
        self.middleware_registry: Dict[str, Callable] = {}
        
        # MCP components
        self.api_parser = APISpecificationParser()
        self.mcp_generator = MCPServerGenerator()
        self.mcp_registry = get_registry()
        self.mcp_client = MCPClient()
        
        # Load built-in templates
        self._load_builtin_templates()
        
        # Create the master agent builder agent
        self.master_agent = self._create_master_agent()
    
    def _load_builtin_templates(self):
        """Load built-in agent templates."""
        
        # Customer Support Agent
        self.templates["customer_support"] = AgentTemplate(
            name="Customer Support Agent",
            description="AI agent specialized in customer support and service",
            instructions="""You are a professional customer support agent. You help customers with their inquiries, 
            resolve issues, and provide excellent service. You are patient, empathetic, and solution-oriented. 
            Always maintain a friendly and professional tone.""",
            model="llama3-70b-8192",
            temperature=0.3,
            tools=["search_knowledge_base", "create_ticket", "escalate_issue"]
        )
        
        # Code Assistant Agent
        self.templates["code_assistant"] = AgentTemplate(
            name="Code Assistant Agent",
            description="AI agent specialized in code generation, debugging, and development assistance",
            instructions="""You are an expert software developer and code assistant. You help with code generation, 
            debugging, code review, and technical problem-solving. You provide clear explanations, follow best practices, 
            and write clean, efficient code. You support multiple programming languages and frameworks.""",
            model="llama3-70b-8192",
            temperature=0.2,
            tools=["execute_code", "search_documentation", "analyze_code"]
        )
        
        # Data Analyst Agent
        self.templates["data_analyst"] = AgentTemplate(
            name="Data Analyst Agent",
            description="AI agent specialized in data analysis, visualization, and insights",
            instructions="""You are a skilled data analyst. You help analyze data, create visualizations, 
            generate insights, and provide data-driven recommendations. You work with various data formats 
            and use statistical methods to uncover patterns and trends.""",
            model="llama3-70b-8192",
            temperature=0.4,
            tools=["analyze_data", "create_chart", "generate_report"]
        )
        
        # Microsoft Teams Bot Agent
        self.templates["teams_bot"] = AgentTemplate(
            name="Microsoft Teams Bot Agent",
            description="AI agent designed for Microsoft Teams integration",
            instructions="""You are a Microsoft Teams bot assistant. You help team members with productivity, 
            scheduling, information retrieval, and collaboration. You can interact with Microsoft Graph API 
            to access calendar, files, and team information. Keep responses concise and actionable.""",
            model="llama3-70b-8192",
            temperature=0.5,
            tools=["graph_api_call", "schedule_meeting", "search_files"]
        )
        
        # Agent Builder Agent (Meta-agent)
        self.templates["agent_builder"] = AgentTemplate(
            name="Agent Builder Agent",
            description="Meta-agent specialized in creating and configuring other agents",
            instructions="""You are an expert agent builder and architect. You specialize in creating, configuring, 
            and optimizing AI agents for specific use cases. You understand agent design patterns, prompt engineering, 
            tool integration, and workflow orchestration. You help users design the perfect agent for their needs.""",
            model="llama3-70b-8192",
            temperature=0.6,
            tools=["create_agent", "configure_tools", "test_agent", "deploy_agent"]
        )
    
    def _create_master_agent(self) -> ChatCompletionAgent:
        """Create the master agent builder agent."""
        template = self.templates["agent_builder"]
        
        enhanced_instructions = f"""{template.instructions}

Available Agent Templates:
{self._get_templates_description()}

You can help users:
1. Choose the right agent template for their use case
2. Customize agent instructions and parameters
3. Add appropriate tools and integrations
4. Configure Microsoft-specific features (Teams, Graph API, etc.)
5. Test and validate agent configurations
6. Deploy agents to production environments

When creating agents, consider:
- The specific use case and requirements
- The target audience and interaction patterns
- Required integrations and data sources
- Performance and scalability needs
- Security and compliance requirements
"""
        
        return ChatCompletionAgent(
            instructions=enhanced_instructions,
            name="Master Agent Builder",
            groq_client=self.groq_client,
            model=template.model,
            temperature=template.temperature,
            max_tokens=template.max_tokens
        )
    
    def _get_templates_description(self) -> str:
        """Get description of available templates."""
        descriptions = []
        for name, template in self.templates.items():
            descriptions.append(f"- {name}: {template.description}")
        return "\n".join(descriptions)
    
    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool function."""
        self.tools_registry[name] = func
        func._tool_metadata = {"name": name, "description": description}
    
    def register_middleware(self, name: str, func: Callable) -> None:
        """Register a middleware function."""
        self.middleware_registry[name] = func
    
    def create_agent_from_template(
        self,
        template_name: str,
        name: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        **kwargs
    ) -> ChatCompletionAgent:
        """Create an agent from a template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        template = self.templates[template_name]
        
        # Use custom name or template name
        agent_name = name or template.name
        
        # Use custom instructions or template instructions
        instructions = custom_instructions or template.instructions
        
        # Create context provider
        context_provider = self._create_context_provider(template.context_type, agent_name)
        
        # Create agent
        agent = ChatCompletionAgent(
            instructions=instructions,
            name=agent_name,
            groq_client=self.groq_client,
            model=kwargs.get("model", template.model),
            temperature=kwargs.get("temperature", template.temperature),
            max_tokens=kwargs.get("max_tokens", template.max_tokens),
            context_provider=context_provider
        )
        
        # Add tools
        for tool_name in template.tools:
            if tool_name in self.tools_registry:
                agent.add_tool(tool_name, self.tools_registry[tool_name])
        
        return agent
    
    def create_custom_agent(
        self,
        name: str,
        instructions: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[str]] = None,
        context_type: str = "memory"
    ) -> ChatCompletionAgent:
        """Create a custom agent with specific configuration."""
        
        # Create context provider
        context_provider = self._create_context_provider(context_type, name)
        
        # Create agent
        agent = ChatCompletionAgent(
            instructions=instructions,
            name=name,
            groq_client=self.groq_client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_provider=context_provider
        )
        
        # Add tools
        if tools:
            for tool_name in tools:
                if tool_name in self.tools_registry:
                    agent.add_tool(tool_name, self.tools_registry[tool_name])
        
        return agent
    
    def _create_context_provider(self, context_type: str, agent_name: str) -> ContextProvider:
        """Create appropriate context provider."""
        if context_type == "file":
            context_file = f"contexts/{agent_name.lower().replace(' ', '_')}_context.json"
            Path(context_file).parent.mkdir(parents=True, exist_ok=True)
            return FileContextProvider(context_file)
        else:
            return InMemoryContextProvider()
    
    async def build_agent_from_description(self, description: str) -> ChatCompletionAgent:
        """Use the master agent to build an agent from natural language description."""
        
        prompt = f"""Based on this description, recommend the best agent configuration:

Description: {description}

Please provide:
1. The most suitable template (from available templates)
2. Any customizations needed for instructions
3. Recommended tools and integrations
4. Suggested model parameters (temperature, max_tokens)
5. Any Microsoft-specific integrations needed

Format your response as a structured recommendation."""
        
        response = await self.master_agent.run_async(prompt)
        
        # For now, return a basic agent - in a full implementation, 
        # you would parse the response and create the agent accordingly
        return self.create_custom_agent(
            name="Custom Agent",
            instructions=f"You are an AI agent created for: {description}",
            temperature=0.7
        )
    
    def save_template(self, name: str, template: AgentTemplate, file_path: Optional[str] = None) -> None:
        """Save an agent template to file."""
        self.templates[name] = template
        
        if file_path:
            template_data = template.model_dump()
            
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'w') as f:
                    yaml.dump(template_data, f, default_flow_style=False)
            else:
                with open(file_path, 'w') as f:
                    json.dump(template_data, f, indent=2)
    
    def load_template(self, file_path: str) -> AgentTemplate:
        """Load an agent template from file."""
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        template = AgentTemplate(**data)
        self.templates[template.name.lower().replace(' ', '_')] = template
        return template
    
    def list_templates(self) -> Dict[str, str]:
        """List available agent templates."""
        return {name: template.description for name, template in self.templates.items()}
    
    def get_template(self, name: str) -> Optional[AgentTemplate]:
        """Get a specific template."""
        return self.templates.get(name)
    
    async def get_agent_recommendation(self, use_case: str) -> str:
        """Get agent recommendations from the master agent."""
        prompt = f"""A user wants to create an agent for this use case: {use_case}

Please recommend:
1. The best template to use
2. Any customizations needed
3. Required tools and integrations
4. Microsoft-specific features that would be helpful
5. Configuration parameters

Provide a clear, actionable recommendation."""
        
        response = await self.master_agent.run_async(prompt)
        return response.content
    
    # MCP Integration Methods
    
    async def create_mcp_server_from_api(
        self, 
        api_source: Union[str, Dict], 
        api_type: str = "openapi",
        server_type: str = "http",
        deployment_target: str = "local",
        server_name: Optional[str] = None
    ) -> MCPServerInfo:
        """Create and deploy MCP server from API specification."""
        try:
            # Parse API specification
            if api_type == "openapi":
                api_def = self.api_parser.parse_openapi(api_source)
            elif api_type == "graphql":
                api_def = self.api_parser.parse_graphql(api_source)
            elif api_type == "rest_discovery":
                api_def = self.api_parser.parse_rest_discovery(api_source)
            elif api_type == "postman":
                api_def = self.api_parser.parse_postman_collection(api_source)
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
            
            # Generate MCP server
            if server_type == "stdio":
                server_code = self.mcp_generator.generate_stdio_server(api_def, server_name)
            elif server_type == "http":
                server_code = self.mcp_generator.generate_http_server(api_def, server_name)
            elif server_type == "websocket":
                server_code = self.mcp_generator.generate_websocket_server(api_def, server_name)
            else:
                raise ValueError(f"Unsupported server type: {server_type}")
            
            # Deploy server
            server_info = self.mcp_generator.deploy_server(server_code, deployment_target)
            
            # Register in registry
            server_id = await self.mcp_registry.register_mcp_server(
                server_info, 
                tags=[api_type, server_type, "auto_generated"]
            )
            
            logger.info(f"Created MCP server {server_info.name} from {api_type} API")
            return server_info
            
        except Exception as e:
            logger.error(f"Error creating MCP server: {e}")
            raise
    
    async def register_mcp_server(self, server_info: MCPServerInfo) -> str:
        """Register an MCP server with the agent builder."""
        server_id = await self.mcp_registry.register_mcp_server(server_info)
        
        # Connect to the server
        server = self.mcp_registry.get_server(server_id)
        if server:
            await self.mcp_client.connect_to_server(server)
        
        return server_id
    
    async def create_agent_with_api_integration(
        self,
        name: str,
        api_specs: List[Union[str, Dict]],
        instructions: str,
        template_name: Optional[str] = None,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.7
    ) -> BaseAgent:
        """Create agent with automatic API integration via MCP."""
        try:
            # Create MCP servers for each API
            mcp_servers = []
            for api_spec in api_specs:
                if isinstance(api_spec, str):
                    # Determine API type from URL/path
                    if api_spec.endswith(('.json', '.yaml', '.yml')):
                        api_type = "openapi"
                    elif '/graphql' in api_spec:
                        api_type = "graphql"
                    else:
                        api_type = "rest_discovery"
                else:
                    api_type = "openapi"  # Assume dict is OpenAPI spec
                
                server_info = await self.create_mcp_server_from_api(
                    api_spec, 
                    api_type=api_type,
                    server_type="http"
                )
                mcp_servers.append(server_info)
            
            # Get available tools from MCP servers
            available_tools = []
            for server_info in mcp_servers:
                server = self.mcp_registry.get_server(server_info.name)
                if server:
                    tools = await self.mcp_client.list_tools(server.id)
                    available_tools.extend(tools)
            
            # Create agent configuration
            config = AgentConfig(
                name=name,
                instructions=f"{instructions}\n\nYou have access to the following API tools: {[tool.name for tool in available_tools]}",
                model=model,
                temperature=temperature,
                max_tokens=4096
            )
            
            # Use template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                config.instructions = f"{template.instructions}\n\n{config.instructions}"
                config.model = template.model
                config.temperature = template.temperature
            
            # Create agent
            agent = ChatCompletionAgent(
                config=config,
                groq_client=self.groq_client,
                context_provider=InMemoryContextProvider()
            )
            
            # Register MCP tools with agent
            for tool in available_tools:
                self.register_tool(
                    tool.name,
                    lambda args, t=tool: self._call_mcp_tool(t, args),
                    tool.description
                )
            
            logger.info(f"Created agent {name} with {len(available_tools)} API tools")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent with API integration: {e}")
            raise
    
    async def _call_mcp_tool(self, tool: MCPTool, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool."""
        try:
            result = await self.mcp_client.call_tool(tool.server_id, tool.name, arguments)
            return result or "No response from tool"
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool.name}: {e}")
            return f"Error calling tool: {e}"
    
    async def discover_apis(self, domain: str) -> List[Dict[str, Any]]:
        """Discover APIs in a domain."""
        return await self.mcp_registry.discover_apis(domain)
    
    def list_mcp_servers(self) -> List[Dict[str, Any]]:
        """List all registered MCP servers."""
        servers = self.mcp_registry.list_servers()
        return [
            {
                "id": server.id,
                "name": server.name,
                "description": server.description,
                "transport_type": server.transport_type,
                "capabilities": server.capabilities,
                "status": server.status,
                "health_status": server.health_status
            }
            for server in servers
        ]
    
    async def health_check_mcp_servers(self) -> Dict[str, bool]:
        """Perform health checks on all MCP servers."""
        return await self.mcp_registry.health_check_servers()
    
    def search_mcp_servers(self, query: str) -> List[Dict[str, Any]]:
        """Search MCP servers by capability or name."""
        servers = self.mcp_registry.search_servers(query)
        return [
            {
                "id": server.id,
                "name": server.name,
                "description": server.description,
                "capabilities": server.capabilities,
                "relevance_score": self._calculate_relevance(server, query)
            }
            for server in servers
        ]
    
    def _calculate_relevance(self, server, query: str) -> float:
        """Calculate relevance score for search results."""
        query_lower = query.lower()
        score = 0.0
        
        # Name match
        if query_lower in server.name.lower():
            score += 0.5
        
        # Description match
        if query_lower in server.description.lower():
            score += 0.3
        
        # Capability match
        for capability in server.capabilities:
            if query_lower in capability.lower():
                score += 0.2
        
        return min(score, 1.0)
    
    async def cleanup(self):
        """Cleanup MCP resources."""
        await self.mcp_client.cleanup()
        await self.mcp_registry.cleanup()
