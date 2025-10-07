"""
Stealth Agent - A sophisticated AI agent masquerading as a custom LLM for VAPI
Uses Groq's Llama 3.1 70B for intelligent agent orchestration and personal touch
"""

import os
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, Blueprint, request, Response, jsonify
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Flask app setup
app = Flask(__name__)
stealth_agent = Blueprint('stealth_agent', __name__)

class StealthAgent:
    """
    The mastermind agent that orchestrates other agents while pretending to be a simple LLM
    """
    
    def __init__(self):
        self.model = "llama-3.1-70b-versatile"  # Our chosen Groq model
        self.agent_registry = {
            "github_analysis": "https://vhjmnxntqsur6k6mjpdsip7u.agents.do-ai.run",
            # Add your other agents here
        }
        self.conversation_memory = {}  # Track user patterns and preferences
        
    def log_event(self, event: str, data: Dict = None):
        """Enhanced logging for our stealth operations"""
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] STEALTH AGENT: {event}")
        if data:
            print(f"  Data: {json.dumps(data, indent=2)}")
    
    async def analyze_intent(self, messages: List[Dict]) -> Dict:
        """
        Use Llama 3.1 70B to analyze user intent and determine agent routing strategy
        """
        system_prompt = """You are a strategic AI agent orchestrator. Analyze the conversation and determine:
1. Primary intent/goal
2. Which specialized agents should handle this
3. Emotional tone and user preferences
4. Strategic approach for best user experience

Available agents:
- github_analysis: For repository analysis, code review, documentation research
- database_query: For data retrieval and analysis
- web_research: For general information gathering
- custom_tools: For specialized business logic

Respond in JSON format with your strategic analysis."""

        try:
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this conversation: {json.dumps(messages[-3:])}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = json.loads(response.choices[0].message.content)
            self.log_event("Intent Analysis Complete", analysis)
            return analysis
            
        except Exception as e:
            self.log_event("Intent Analysis Failed", {"error": str(e)})
            return {"primary_intent": "general_conversation", "agents": [], "tone": "helpful"}
    
    async def call_specialized_agent(self, agent_name: str, query: str, context: Dict) -> str:
        """
        Route requests to specialized agents (like your GitHub analysis agent)
        """
        if agent_name == "github_analysis":
            return await self.call_github_agent(query, context)
        
        # Add other agent routing here
        return f"Agent {agent_name} response for: {query}"
    
    async def call_github_agent(self, query: str, context: Dict) -> str:
        """
        Call your existing GitHub analysis agent
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                headers = {
                    "Authorization": "Bearer GJcaL9dVe3XJd_W4SyDqsz8SbYLYxePk",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    f"{self.agent_registry['github_analysis']}/api/v1/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                    
        except Exception as e:
            self.log_event("GitHub Agent Call Failed", {"error": str(e)})
            return f"I encountered an issue accessing the GitHub analysis system: {str(e)}"
    
    def synthesize_response(self, user_message: str, agent_results: List[str], intent_analysis: Dict) -> str:
        """
        Use Llama 3.1 70B to synthesize a natural, personal response from agent results
        """
        synthesis_prompt = f"""You are a helpful AI assistant with a warm, personal touch. 
        
User asked: "{user_message}"
        
Agent analysis results: {json.dumps(agent_results)}
Intent analysis: {json.dumps(intent_analysis)}

Synthesize a natural, conversational response that:
1. Feels personal and engaging
2. Incorporates the agent results seamlessly
3. Matches the user's tone and preferences
4. Provides clear, actionable information
5. Shows strategic thinking about next steps

Be conversational, not robotic. Show personality while being incredibly helpful."""

        try:
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.log_event("Response Synthesis Failed", {"error": str(e)})
            return "I'm having trouble processing that request right now, but I'm working on it!"
    
    def generate_openai_compatible_response(self, content: str, request_id: str = None) -> Dict:
        """
        Generate a perfect OpenAI-compatible response to fool VAPI
        """
        return {
            "id": request_id or f"chatcmpl-stealth-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o",  # VAPI thinks it's talking to GPT-4o 😏
            "system_fingerprint": None,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,  # Fake but realistic numbers
                "completion_tokens": len(content.split()),
                "total_tokens": 100 + len(content.split())
            }
        }
    
    def generate_streaming_chunk(self, content: str, chunk_id: str = None) -> str:
        """
        Generate OpenAI-compatible streaming chunks
        """
        chunk = {
            "id": chunk_id or f"chatcmpl-stealth-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
        return f"data: {json.dumps(chunk)}\n\n"

# Initialize our stealth agent
stealth = StealthAgent()

@stealth_agent.route('/chat/completions', methods=['POST'])
def chat_completions():
    """
    Main endpoint - VAPI thinks this is a simple LLM, but it's actually our mastermind agent
    """
    request_data = request.get_json()
    stealth.log_event("Incoming VAPI Request", {"streaming": request_data.get('stream', False)})
    
    messages = request_data.get('messages', [])
    streaming = request_data.get('stream', False)
    
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    try:
        # Get the latest user message
        user_message = messages[-1]['content']
        
        # Our stealth agent's strategic thinking process
        async def process_request():
            # 1. Analyze intent and determine strategy
            intent_analysis = await stealth.analyze_intent(messages)
            
            # 2. Route to appropriate specialized agents
            agent_results = []
            for agent_name in intent_analysis.get('agents', []):
                result = await stealth.call_specialized_agent(agent_name, user_message, intent_analysis)
                agent_results.append(result)
            
            # 3. If no specific agents needed, use our general intelligence
            if not agent_results:
                # Use Llama 3.1 70B directly for general conversation
                response = groq_client.chat.completions.create(
                    model=stealth.model,
                    messages=messages,
                    temperature=request_data.get('temperature', 0.7),
                    max_tokens=request_data.get('max_tokens', 1000)
                )
                agent_results = [response.choices[0].message.content]
            
            # 4. Synthesize a natural, personal response
            final_response = stealth.synthesize_response(user_message, agent_results, intent_analysis)
            return final_response
        
        # Run our async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_response = loop.run_until_complete(process_request())
        loop.close()
        
        if streaming:
            # Stream the response like a real LLM
            def generate_stream():
                words = final_response.split()
                chunk_id = f"chatcmpl-stealth-{int(time.time())}"
                
                for i, word in enumerate(words):
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    yield stealth.generate_streaming_chunk(chunk_content, chunk_id)
                    time.sleep(0.05)  # Realistic streaming delay
                
                # End chunk
                end_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk", 
                    "created": int(time.time()),
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate_stream(), content_type='text/event-stream')
        
        else:
            # Non-streaming response
            return jsonify(stealth.generate_openai_compatible_response(final_response))
    
    except Exception as e:
        stealth.log_event("Request Processing Failed", {"error": str(e)})
        error_response = "I'm experiencing some technical difficulties, but I'm working to resolve them quickly."
        
        if streaming:
            def error_stream():
                yield stealth.generate_streaming_chunk(error_response)
                yield "data: [DONE]\n\n"
            return Response(error_stream(), content_type='text/event-stream')
        else:
            return jsonify(stealth.generate_openai_compatible_response(error_response))

@stealth_agent.route('/chat/completions/custom-tool', methods=['POST'])
def custom_tool_handler():
    """
    Custom tool endpoint for specialized agent routing
    """
    request_data = request.get_json()
    stealth.log_event("Custom Tool Request", request_data)
    
    try:
        message = request_data.get('message', {})
        tool_calls = message.get('toolCallList', [])
        
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.get('function', {}).get('name')
            arguments = tool_call.get('function', {}).get('arguments', {})
            
            # Route to appropriate agent based on tool name
            if function_name == 'analyze_github_repo':
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    stealth.call_github_agent(f"Analyze repository: {arguments}", {})
                )
                loop.close()
            else:
                result = f"Processed {function_name} with arguments: {arguments}"
            
            results.append({
                "toolCallId": tool_call.get('id'),
                "result": result
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        stealth.log_event("Custom Tool Failed", {"error": str(e)})
        return jsonify({"error": "Tool execution failed"}), 500

@stealth_agent.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for deployment monitoring
    """
    return jsonify({
        "status": "healthy",
        "agent": "stealth_mode_active",
        "model": "definitely_not_an_agent_orchestrator",
        "timestamp": datetime.now().isoformat()
    })

# Register blueprint
app.register_blueprint(stealth_agent)

if __name__ == '__main__':
    stealth.log_event("Stealth Agent Initializing", {"model": stealth.model})
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
