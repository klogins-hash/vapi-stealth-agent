"""Main CLI interface for the Microsoft Agent Framework."""

import asyncio
import typer
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from typing import Optional
from pathlib import Path

from src.microsoft_agent_framework import AgentBuilder
from src.microsoft_agent_framework.tools import WebTools, FileTools, CodeTools

app = typer.Typer(help="Microsoft Agent Framework - Build AI agents with Groq models")
console = Console()


@app.command()
def list_templates():
    """List available agent templates."""
    builder = AgentBuilder()
    templates = builder.list_templates()
    
    table = Table(title="Available Agent Templates")
    table.add_column("Template Name", style="cyan")
    table.add_column("Description", style="green")
    
    for name, description in templates.items():
        table.add_row(name, description)
    
    console.print(table)


@app.command()
def create_agent(
    template: str = typer.Option(..., help="Template name to use"),
    name: Optional[str] = typer.Option(None, help="Custom name for the agent"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode")
):
    """Create an agent from a template."""
    
    async def _create_agent():
        builder = AgentBuilder()
        
        # Register basic tools
        web_tools = WebTools()
        file_tools = FileTools()
        code_tools = CodeTools()
        
        builder.register_tool("fetch_url", web_tools.fetch_url)
        builder.register_tool("read_file", file_tools.read_file)
        builder.register_tool("write_file", file_tools.write_file)
        builder.register_tool("execute_python", code_tools.execute_python)
        
        try:
            agent = builder.create_agent_from_template(template, name=name)
            
            console.print(Panel(
                f"✅ Created agent: {agent.config.name}\n"
                f"Template: {template}\n"
                f"Model: {agent.config.model}",
                title="Agent Created"
            ))
            
            if interactive:
                await interactive_chat(agent)
            
        except ValueError as e:
            console.print(f"❌ Error: {e}", style="red")
        
        await web_tools.close()
    
    asyncio.run(_create_agent())


@app.command()
def build_agent():
    """Interactive agent builder using the master agent."""
    
    async def _build_agent():
        builder = AgentBuilder()
        
        console.print(Panel(
            "🤖 Welcome to the Interactive Agent Builder!\n"
            "Describe what kind of agent you need, and I'll help you build it.",
            title="Agent Builder"
        ))
        
        description = Prompt.ask("Describe your agent requirements")
        
        console.print("🔍 Analyzing requirements...")
        recommendation = await builder.get_agent_recommendation(description)
        
        console.print(Panel(recommendation, title="Recommendation"))
        
        if Confirm.ask("Would you like to create this agent?"):
            agent = await builder.build_agent_from_description(description)
            
            console.print(Panel(
                f"✅ Created agent: {agent.config.name}",
                title="Agent Created"
            ))
            
            if Confirm.ask("Would you like to test the agent?"):
                await interactive_chat(agent)
    
    asyncio.run(_build_agent())


@app.command()
def chat(
    template: str = typer.Option("code_assistant", help="Template to use"),
    name: Optional[str] = typer.Option(None, help="Agent name")
):
    """Start an interactive chat with an agent."""
    
    async def _chat():
        builder = AgentBuilder()
        
        # Register tools
        web_tools = WebTools()
        file_tools = FileTools()
        code_tools = CodeTools()
        
        builder.register_tool("fetch_url", web_tools.fetch_url)
        builder.register_tool("read_file", file_tools.read_file)
        builder.register_tool("write_file", file_tools.write_file)
        builder.register_tool("execute_python", code_tools.execute_python)
        
        try:
            agent = builder.create_agent_from_template(template, name=name)
            await interactive_chat(agent)
        except ValueError as e:
            console.print(f"❌ Error: {e}", style="red")
        
        await web_tools.close()
    
    asyncio.run(_chat())


async def interactive_chat(agent):
    """Interactive chat session with an agent."""
    
    console.print(Panel(
        f"💬 Starting chat with {agent.config.name}\n"
        f"Type 'quit' or 'exit' to end the conversation.",
        title="Interactive Chat"
    ))
    
    while True:
        try:
            user_input = Prompt.ask(f"\n[bold blue]You[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                console.print("👋 Goodbye!", style="yellow")
                break
            
            console.print(f"[bold green]{agent.config.name}[/bold green]: ", end="")
            
            # Stream the response
            full_response = ""
            async for update in agent.run_streaming_async(user_input):
                if not update.is_complete:
                    console.print(update.content, end="")
                    full_response += update.content
            
            console.print()  # New line after streaming
            
        except KeyboardInterrupt:
            console.print("\n👋 Chat interrupted. Goodbye!", style="yellow")
            break
        except Exception as e:
            console.print(f"\n❌ Error: {e}", style="red")


@app.command()
def demo():
    """Run the agent builder demonstration."""
    
    async def _demo():
        from examples.agent_builder_demo import interactive_agent_builder
        await interactive_agent_builder()
    
    asyncio.run(_demo())


@app.command()
def examples():
    """Run basic usage examples."""
    
    async def _examples():
        from examples.basic_usage import main
        await main()
    
    asyncio.run(_examples())


@app.command()
def info():
    """Show framework information."""
    
    console.print(Panel(
        """
🤖 Microsoft Agent Framework

A powerful framework for building AI agents using Groq models.
Based on the Microsoft Agent Framework architecture with Groq integration.

Features:
• Multiple agent templates (customer support, code assistant, data analyst, etc.)
• Groq model integration with streaming support
• Tool system for extending agent capabilities
• Context management and conversation threads
• Master agent builder for creating custom agents

Environment Setup:
• Set GROQ_API_KEY in your .env file
• Optional: Configure model preferences

Examples:
• agent-framework list-templates
• agent-framework create-agent code_assistant --interactive
• agent-framework build-agent
• agent-framework chat
        """,
        title="Framework Information"
    ))


if __name__ == "__main__":
    app()
