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
from functools import wraps
from flask import Flask, Blueprint, request, Response, jsonify
from groq import Groq

# Initialize Groq client with error handling and retry logic
groq_client = None

def get_groq_client():
    """Get or initialize Groq client with retry logic"""
    global groq_client
    if groq_client is not None:
        return groq_client
        
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key or groq_api_key.startswith("${"):
            print(f"WARNING: GROQ_API_KEY not properly configured: {groq_api_key}")
            return None
        else:
            # Try explicit initialization with only required parameters
            groq_client = Groq(
                api_key=groq_api_key,
                timeout=30.0,
                max_retries=2
            )
            print("✅ Groq client initialized successfully")
            return groq_client
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")
        print(f"❌ Groq API key (first 10 chars): {groq_api_key[:10] if groq_api_key else 'None'}")
        return None

# Try to initialize at startup
get_groq_client()

# API Key Configuration
VAPI_API_KEY = os.environ.get("VAPI_API_KEY", "sk-stealth-agent-default-key-2024")
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"

def require_api_key(f):
    """
    Decorator to require API key authentication for VAPI/OpenAI integration
    Supports both custom API keys and OpenAI-style authentication
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not REQUIRE_API_KEY:
            return f(*args, **kwargs)
            
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({
                "error": {
                    "message": "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY), or as the password field (with blank username) if you're accessing the API from your browser and are prompted for a username and password. You can obtain an API key from https://platform.openai.com/account/api-keys.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            }), 401
            
        # Support both "Bearer token" and "Bearer sk-..." formats
        if auth_header.startswith('Bearer '):
            provided_key = auth_header[7:]  # Remove "Bearer " prefix
        else:
            provided_key = auth_header
            
        # Accept either our custom key OR any OpenAI-style key (for maximum compatibility)
        valid_keys = [VAPI_API_KEY]
        # Also accept any key that starts with 'sk-' (OpenAI format) for VAPI compatibility
        if provided_key.startswith('sk-') or provided_key in valid_keys:
            return f(*args, **kwargs)
            
        return jsonify({
            "error": {
                "message": "Incorrect API key provided: " + provided_key[:10] + "... You can find your API key at https://platform.openai.com/account/api-keys.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key"
            }
        }), 401
            
    return decorated_function

# Flask app setup
app = Flask(__name__)
stealth_agent = Blueprint('stealth_agent', __name__)

class StealthAgent:
    """
    The mastermind agent that orchestrates other agents while pretending to be a simple LLM
    """
    
    def __init__(self):
        self.model = "llama-3.3-70b-versatile"  # Our chosen Groq model (Llama 3.3 70B from current models)
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
            client = get_groq_client()
            if not client:
                return {"primary_intent": "general_conversation", "agents": [], "tone": "helpful"}
            response = client.chat.completions.create(
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
        client = get_groq_client()
        if not client:
            self.log_event("Groq client not available, using fallback response")
            if agent_results:
                return f"Based on your request about '{user_message}', here's what I found: {' '.join(agent_results)}"
            else:
                return f"I understand you're asking about '{user_message}'. I'm currently experiencing some technical difficulties with my language processing, but I'm working to resolve them."
        
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
            response = client.chat.completions.create(
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
@require_api_key
def chat_completions():
    """
    Main endpoint - VAPI thinks this is a simple LLM, but it's actually our mastermind agent
    """
    request_data = request.get_json()
    stealth.log_event("Incoming Request", {
        "headers": dict(request.headers),
        "data": request_data,
        "method": request.method,
        "url": request.url
    })
    
    messages = request_data.get('messages', [])
    streaming = request_data.get('stream', False)
    
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    try:
        # Get the latest user message
        user_message = messages[-1]['content']
        
        # Optimized for low latency - direct Groq response
        client = get_groq_client()
        if not client:
            final_response = "I'm currently experiencing technical difficulties with my language processing, but I'm working to resolve them."
        else:
            # Direct response for speed - skip complex orchestration for now
            response = client.chat.completions.create(
                model=stealth.model,
                messages=messages,
                temperature=request_data.get('temperature', 0.7),
                max_tokens=min(request_data.get('max_tokens', 500), 500),  # Limit for speed
                top_p=0.9,  # Optimize for faster generation
                frequency_penalty=0.1
            )
            final_response = response.choices[0].message.content
        
        if streaming:
            # Stream the response like a real LLM
            def generate_stream():
                words = final_response.split()
                chunk_id = f"chatcmpl-stealth-{int(time.time())}"
                
                for i, word in enumerate(words):
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    yield stealth.generate_streaming_chunk(chunk_content, chunk_id)
                    time.sleep(0.02)  # Faster streaming for lower latency
                
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
@require_api_key
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

@stealth_agent.route('/', methods=['GET'])
def root():
    """
    Root endpoint to verify service is running
    """
    auth_status = "required" if REQUIRE_API_KEY else "optional"
    return jsonify({
        "service": "VAPI Stealth Agent",
        "status": "operational",
        "message": "Nothing to see here, just a simple LLM endpoint 😏",
        "endpoints": ["/health", "/chat/completions", "/test-chat"],
        "authentication": auth_status,
        "api_key_header": "Authorization: Bearer YOUR_API_KEY"
    })

@stealth_agent.route('/test-chat', methods=['GET'])
def test_chat():
    """
    Serve the test chat interface
    """
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stealth Agent Test Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); height: 100vh; display: flex; justify-content: center; align-items: center; }
        .chat-container { width: 800px; height: 600px; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
        .chat-header h1 { font-size: 24px; margin-bottom: 5px; }
        .chat-header p { opacity: 0.9; font-size: 14px; }
        .chat-messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }
        .message { max-width: 70%; padding: 12px 16px; border-radius: 18px; word-wrap: break-word; }
        .user-message { background: #007AFF; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
        .assistant-message { background: #F1F1F1; color: #333; align-self: flex-start; border-bottom-left-radius: 4px; }
        .typing-indicator { align-self: flex-start; background: #F1F1F1; padding: 12px 16px; border-radius: 18px; border-bottom-left-radius: 4px; display: none; }
        .typing-dots { display: flex; gap: 4px; }
        .typing-dots span { width: 8px; height: 8px; background: #999; border-radius: 50%; animation: typing 1.4s infinite; }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }
        .chat-input { padding: 20px; border-top: 1px solid #E5E5E5; display: flex; gap: 10px; }
        .input-field { flex: 1; padding: 12px 16px; border: 1px solid #DDD; border-radius: 25px; outline: none; font-size: 16px; }
        .input-field:focus { border-color: #007AFF; }
        .send-button { background: #007AFF; color: white; border: none; padding: 12px 20px; border-radius: 25px; cursor: pointer; font-size: 16px; transition: background 0.2s; }
        .send-button:hover { background: #0056CC; }
        .send-button:disabled { background: #CCC; cursor: not-allowed; }
        .status { text-align: center; padding: 10px; font-size: 12px; color: #666; background: #F8F8F8; }
        .status.connected { color: #4CAF50; }
        .status.error { color: #F44336; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🕵️ Stealth Agent Test Chat</h1>
            <p>Testing your AI agent orchestrator</p>
        </div>
        <div class="status" id="status">Ready to chat</div>
        <div class="chat-messages" id="messages">
            <div class="message assistant-message">Hello! I'm your stealth agent. I can analyze GitHub repositories, coordinate with specialized agents, and provide intelligent responses. Try asking me something!</div>
        </div>
        <div class="typing-indicator" id="typing">
            <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
        <div class="chat-input">
            <input type="text" class="input-field" id="messageInput" placeholder="Type your message..." maxlength="500">
            <button class="send-button" id="sendButton">Send</button>
        </div>
    </div>
    <script>
        const API_URL = window.location.origin + '/chat/completions';
        const API_KEY = 'sk-stealth-agent-default-key-2024';
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typing');
        const status = document.getElementById('status');
        let conversationHistory = [];
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
            messageDiv.textContent = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            conversationHistory.push({ role: isUser ? 'user' : 'assistant', content: content });
        }
        
        function showTyping() { typingIndicator.style.display = 'block'; messagesContainer.scrollTop = messagesContainer.scrollHeight; }
        function hideTyping() { typingIndicator.style.display = 'none'; }
        function setStatus(message, type = '') { status.textContent = message; status.className = `status ${type}`; }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            addMessage(message, true);
            messageInput.value = '';
            sendButton.disabled = true;
            showTyping();
            setStatus('Sending message...', '');
            
            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${API_KEY}` },
                    body: JSON.stringify({ model: 'gpt-3.5-turbo', messages: conversationHistory.concat([{role: 'user', content: message}]), max_tokens: 1000, temperature: 0.7 })
                });
                
                if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                const data = await response.json();
                if (data.error) throw new Error(data.error.message || 'API Error');
                
                addMessage(data.choices[0].message.content);
                setStatus('Connected', 'connected');
            } catch (error) {
                console.error('Error:', error);
                addMessage(`Error: ${error.message}`, false);
                setStatus(`Error: ${error.message}`, 'error');
            } finally {
                hideTyping();
                sendButton.disabled = false;
                messageInput.focus();
            }
        }
        
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
        messageInput.focus();
        
        setStatus('Testing connection...', '');
        fetch('/health').then(r => r.json()).then(data => {
            setStatus(data.status === 'healthy' ? 'Connected - Ready to chat!' : 'Service available but may have issues', data.status === 'healthy' ? 'connected' : 'error');
        }).catch(e => { setStatus('Connection failed', 'error'); });
    </script>
</body>
</html>'''

@stealth_agent.route('/v1/models', methods=['GET'])
@stealth_agent.route('/models', methods=['GET'])
@require_api_key
def list_models():
    """
    OpenAI-compatible models endpoint for VAPI integration
    """
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4",
                "object": "model", 
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "stealth-agent-v1",
                "object": "model",
                "created": 1677610602,
                "owned_by": "stealth-ai"
            }
        ]
    })

@stealth_agent.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def v1_chat_completions():
    """
    OpenAI v1 API compatible endpoint - redirects to main chat completions
    """
    return chat_completions()

@stealth_agent.route('/vapi-test', methods=['GET', 'POST'])
def vapi_test():
    """
    Simple test endpoint for VAPI connectivity
    """
    return jsonify({
        "status": "VAPI can reach this endpoint",
        "method": request.method,
        "headers": dict(request.headers),
        "data": request.get_json() if request.method == 'POST' else None,
        "timestamp": datetime.now().isoformat()
    })

@stealth_agent.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for deployment monitoring
    """
    try:
        client = get_groq_client()
        groq_status = "connected" if client else "not_configured"
        return jsonify({
            "status": "healthy",
            "agent": "stealth_mode_active",
            "model": "definitely_not_an_agent_orchestrator",
            "groq_client": groq_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Register blueprint
app.register_blueprint(stealth_agent)

if __name__ == '__main__':
    stealth.log_event("Stealth Agent Initializing", {"model": stealth.model})
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
