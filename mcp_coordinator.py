"""
MCP Expert/Coordinator Agent
Specializes in Model Context Protocol operations and coordination
Uses Rube API for advanced MCP capabilities
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)

class MCPCoordinator:
    """
    MCP Expert and Coordinator Agent
    """
    
    def __init__(self):
        self.rube_api_key = os.environ.get('RUBE_API_KEY')
        self.rube_api_url = 'https://rube.app/mcp'
        self.mcp_sessions = {}
        self.context_store = {}
        self.tool_registry = {}
        
        if not self.rube_api_key:
            print("⚠️ RUBE_API_KEY not found in environment")
        else:
            print("✅ MCP Coordinator initialized with Rube API")
    
    async def initialize_mcp_session(self, session_id: str, capabilities: Dict = None) -> Dict:
        """
        Initialize a new MCP session
        """
        try:
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'jsonrpc': '2.0',
                'id': f"init_{session_id}",
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': capabilities or {
                        'roots': {'listChanged': True},
                        'sampling': {},
                        'tools': {'listChanged': True}
                    },
                    'clientInfo': {
                        'name': 'MCP-Coordinator',
                        'version': '1.0.0'
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        self.mcp_sessions[session_id] = {
                            'initialized': True,
                            'capabilities': result.get('result', {}).get('capabilities', {}),
                            'server_info': result.get('result', {}).get('serverInfo', {}),
                            'created_at': datetime.now().isoformat()
                        }
                        
                        return {
                            'status': 'success',
                            'session_id': session_id,
                            'server_capabilities': self.mcp_sessions[session_id]['capabilities'],
                            'server_info': self.mcp_sessions[session_id]['server_info']
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f"MCP initialization failed {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def list_tools(self, session_id: str = None) -> Dict:
        """
        List available MCP tools
        """
        try:
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'jsonrpc': '2.0',
                'id': f"list_tools_{session_id or 'default'}",
                'method': 'tools/list',
                'params': {}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        tools = result.get('result', {}).get('tools', [])
                        
                        # Update tool registry
                        for tool in tools:
                            self.tool_registry[tool.get('name')] = tool
                        
                        return {
                            'status': 'success',
                            'tools': tools,
                            'tool_count': len(tools)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f"Tool listing failed {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def call_tool(self, tool_name: str, arguments: Dict, session_id: str = None) -> Dict:
        """
        Call an MCP tool
        """
        try:
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'jsonrpc': '2.0',
                'id': f"call_tool_{tool_name}_{int(datetime.now().timestamp())}",
                'method': 'tools/call',
                'params': {
                    'name': tool_name,
                    'arguments': arguments
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            'status': 'success',
                            'tool_name': tool_name,
                            'arguments': arguments,
                            'result': result.get('result', {}),
                            'session_id': session_id
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f"Tool call failed {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def manage_context(self, operation: str, context_id: str, data: Dict = None) -> Dict:
        """
        Manage MCP context (store, retrieve, update, delete)
        """
        try:
            if operation == 'store':
                self.context_store[context_id] = {
                    'data': data,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                return {
                    'status': 'success',
                    'operation': 'store',
                    'context_id': context_id,
                    'size': len(json.dumps(data)) if data else 0
                }
                
            elif operation == 'retrieve':
                if context_id in self.context_store:
                    return {
                        'status': 'success',
                        'operation': 'retrieve',
                        'context_id': context_id,
                        'data': self.context_store[context_id]['data'],
                        'metadata': {
                            'created_at': self.context_store[context_id]['created_at'],
                            'updated_at': self.context_store[context_id]['updated_at']
                        }
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f'Context {context_id} not found'
                    }
                    
            elif operation == 'update':
                if context_id in self.context_store:
                    self.context_store[context_id]['data'].update(data or {})
                    self.context_store[context_id]['updated_at'] = datetime.now().isoformat()
                    return {
                        'status': 'success',
                        'operation': 'update',
                        'context_id': context_id
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f'Context {context_id} not found'
                    }
                    
            elif operation == 'delete':
                if context_id in self.context_store:
                    del self.context_store[context_id]
                    return {
                        'status': 'success',
                        'operation': 'delete',
                        'context_id': context_id
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f'Context {context_id} not found'
                    }
                    
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown operation: {operation}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def github_operation(self, operation: str, params: Dict) -> Dict:
        """
        Perform GitHub operations via Rube MCP
        """
        try:
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'jsonrpc': '2.0',
                'id': f"github_{operation}_{int(datetime.now().timestamp())}",
                'method': f'github_{operation}',
                'params': params
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'status': 'success',
                            'operation': operation,
                            'result': result,
                            'api_used': 'rube_mcp_github'
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f"GitHub operation failed {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    async def coordinate_agents(self, task: str, agents: List[str], context: Dict = None) -> Dict:
        """
        Coordinate multiple agents using MCP
        """
        try:
            coordination_id = f"coord_{int(datetime.now().timestamp())}"
            
            # Store coordination context
            await self.manage_context('store', coordination_id, {
                'task': task,
                'agents': agents,
                'context': context or {},
                'status': 'initiated'
            })
            
            results = []
            
            # For each agent, make appropriate MCP calls
            for agent in agents:
                if agent == 'code_generation':
                    result = await self.call_tool('generate_code', {
                        'prompt': task,
                        'context': context
                    })
                elif agent == 'file_operations':
                    result = await self.call_tool('file_operations', {
                        'operation': 'analyze',
                        'context': context
                    })
                elif agent == 'database_query':
                    result = await self.call_tool('database_query', {
                        'query': task,
                        'context': context
                    })
                else:
                    result = {
                        'status': 'delegated',
                        'agent': agent,
                        'task': task
                    }
                
                results.append({
                    'agent': agent,
                    'result': result
                })
            
            # Update coordination context
            await self.manage_context('update', coordination_id, {
                'status': 'completed',
                'results': results
            })
            
            return {
                'status': 'success',
                'coordination_id': coordination_id,
                'task': task,
                'agents_coordinated': len(agents),
                'results': results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_status(self) -> Dict:
        """
        Get MCP coordinator status
        """
        return {
            'mcp_sessions': len(self.mcp_sessions),
            'context_store_size': len(self.context_store),
            'registered_tools': len(self.tool_registry),
            'rube_api_configured': bool(self.rube_api_key),
            'rube_api_url': self.rube_api_url,
            'active_sessions': list(self.mcp_sessions.keys()),
            'available_tools': list(self.tool_registry.keys())
        }

# Initialize MCP coordinator
mcp_coordinator = MCPCoordinator()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'MCP Expert/Coordinator Agent',
        'status': 'operational',
        'version': '1.0.0',
        'protocol': 'Model Context Protocol (MCP)',
        'capabilities': [
            'MCP Session Management',
            'Tool Discovery & Execution',
            'Context Management',
            'Agent Coordination',
            'Rube API Integration'
        ],
        'endpoints': [
            '/health',
            '/init-session',
            '/list-tools',
            '/call-tool',
            '/manage-context',
            '/coordinate-agents',
            '/github-operation',
            '/status'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'mcp_sessions': len(mcp_coordinator.mcp_sessions),
        'context_store': len(mcp_coordinator.context_store),
        'tools_registered': len(mcp_coordinator.tool_registry),
        'rube_api': 'configured' if mcp_coordinator.rube_api_key else 'not_configured',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/init-session', methods=['POST'])
def init_session():
    """Initialize MCP session"""
    data = request.get_json()
    
    if not data or 'session_id' not in data:
        return jsonify({'error': 'session_id required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            mcp_coordinator.initialize_mcp_session(data['session_id'], data.get('capabilities'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/list-tools', methods=['GET'])
def list_tools():
    """List available MCP tools"""
    session_id = request.args.get('session_id')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(mcp_coordinator.list_tools(session_id))
        return jsonify(result)
    finally:
        loop.close()

@app.route('/call-tool', methods=['POST'])
def call_tool():
    """Call an MCP tool"""
    data = request.get_json()
    
    if not data or 'tool_name' not in data or 'arguments' not in data:
        return jsonify({'error': 'tool_name and arguments required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            mcp_coordinator.call_tool(data['tool_name'], data['arguments'], data.get('session_id'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/manage-context', methods=['POST'])
def manage_context():
    """Manage MCP context"""
    data = request.get_json()
    
    if not data or 'operation' not in data or 'context_id' not in data:
        return jsonify({'error': 'operation and context_id required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            mcp_coordinator.manage_context(data['operation'], data['context_id'], data.get('data'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/coordinate-agents', methods=['POST'])
def coordinate_agents():
    """Coordinate multiple agents"""
    data = request.get_json()
    
    if not data or 'task' not in data or 'agents' not in data:
        return jsonify({'error': 'task and agents required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            mcp_coordinator.coordinate_agents(data['task'], data['agents'], data.get('context'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/github-operation', methods=['POST'])
def github_operation():
    """Perform GitHub operation via Rube MCP"""
    data = request.get_json()
    
    if not data or 'operation' not in data or 'params' not in data:
        return jsonify({'error': 'operation and params required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            mcp_coordinator.github_operation(data['operation'], data['params'])
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/status', methods=['GET'])
def status():
    """Get MCP coordinator status"""
    return jsonify(mcp_coordinator.get_status())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8004))
    app.run(host='0.0.0.0', port=port, debug=False)
