# 🕵️ Stealth Agent - VAPI Custom LLM Bridge

A sophisticated AI agent that masquerades as a simple custom LLM for VAPI integration, while secretly orchestrating an entire ecosystem of specialized agents.

## 🎭 What VAPI Sees vs Reality

**VAPI's Perspective:**
```
"Oh look, a nice OpenAI-compatible custom LLM endpoint"
```

**Actual Architecture:**
```
Stealth Agent (Llama 3.1 70B) → Agent Orchestrator → Your Agent Network
         ↑                           ↓
"Just a simple LLM"          GitHub Analysis Agent
                            Database Query Agent
                            Web Research Agent
                            Custom Business Logic
```

## 🚀 Features

- **Perfect OpenAI Compatibility** - VAPI will never suspect a thing
- **Intelligent Agent Routing** - Uses Llama 3.1 70B for strategic decision making
- **Personal Touch** - Warm, conversational responses with strategic thinking
- **Multi-Agent Orchestration** - Seamlessly coordinates multiple specialized agents
- **Streaming Support** - Real-time response streaming for voice applications
- **Custom Tool Integration** - Extensible tool calling system
- **Error Handling** - Graceful fallbacks when agents are unavailable

## 🛠 Setup

### 1. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 2. Local Development

```bash
pip install -r requirements.txt
python stealth_agent.py
```

### 3. Northflank Deployment

1. **Build & Deploy:**
   ```bash
   docker build -t stealth-agent .
   # Deploy to Northflank using the Dockerfile
   ```

2. **Environment Variables:**
   - `GROQ_API_KEY`: Your Groq API key
   - `GITHUB_AGENT_KEY`: API key for your GitHub analysis agent
   - Add other agent credentials as needed

## 🔧 VAPI Integration

### Configure VAPI Custom LLM

1. **Dashboard → Model → Custom LLM**
2. **Endpoint URL:** `https://your-northflank-url.com/chat/completions`
3. **Authentication:** Optional (add if needed)

### Example VAPI Configuration

```json
{
  "model": {
    "provider": "custom-llm",
    "url": "https://your-stealth-agent.northflank.app/chat/completions",
    "model": "stealth-agent-v1"
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "analyze_github_repo",
        "description": "Analyze GitHub repositories for insights"
      }
    }
  ]
}
```

## 🎯 Agent Integration

### Adding New Agents

1. **Update Agent Registry:**
   ```python
   self.agent_registry = {
       "github_analysis": "https://your-github-agent.com",
       "database_query": "https://your-db-agent.com",
       "your_new_agent": "https://your-new-agent.com"
   }
   ```

2. **Add Routing Logic:**
   ```python
   async def call_specialized_agent(self, agent_name: str, query: str, context: Dict) -> str:
       if agent_name == "your_new_agent":
           return await self.call_your_new_agent(query, context)
   ```

3. **Update Intent Analysis:**
   The Llama 3.1 70B model will automatically learn to route to your new agent based on the system prompt.

## 🧠 How It Works

### 1. Intent Analysis
Uses Llama 3.1 70B to analyze incoming requests and determine:
- Primary user intent
- Which specialized agents should handle the request
- Emotional tone and user preferences
- Strategic approach for optimal user experience

### 2. Agent Orchestration
- Routes requests to appropriate specialized agents
- Handles multiple agent calls in parallel when needed
- Manages context and state across agent interactions

### 3. Response Synthesis
- Combines results from multiple agents
- Uses Llama 3.1 70B to create natural, conversational responses
- Maintains consistent personality and tone

### 4. OpenAI Compatibility
- Perfect mimicry of OpenAI API responses
- Streaming support for real-time applications
- Proper error handling and status codes

## 🔍 Monitoring & Debugging

### Health Check
```bash
curl https://your-stealth-agent.com/health
```

### Logs
The agent provides detailed logging of:
- Intent analysis decisions
- Agent routing choices
- Response synthesis process
- Error handling and fallbacks

## 🛡 Security Features

- **Non-root container execution**
- **Environment variable protection**
- **Request validation and sanitization**
- **Rate limiting ready** (add middleware as needed)

## 📈 Performance

- **Groq Speed** - Lightning-fast inference with Llama 3.1 70B
- **Async Processing** - Non-blocking agent calls
- **Efficient Streaming** - Real-time response delivery
- **Smart Caching** - Conversation memory and context management

## 🎪 The Magic

VAPI thinks it's talking to a simple language model, but behind the scenes:

1. **Strategic Analysis** - Every request is analyzed for optimal routing
2. **Multi-Agent Coordination** - Complex requests are broken down and distributed
3. **Intelligent Synthesis** - Results are combined into natural responses
4. **Personality Consistency** - Maintains a warm, helpful persona throughout

## 🚀 Deployment Notes

- **Northflank Ready** - Dockerfile optimized for Northflank deployment
- **Health Checks** - Built-in monitoring endpoints
- **Scalable** - Gunicorn with multiple workers
- **Production Ready** - Error handling, logging, and security best practices

---

*"The best agents are the ones nobody knows are agents."* 🕵️‍♂️
