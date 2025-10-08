"""
Master Workflow Engine Agent
Orchestrates complex multi-agent workflows with task queuing, dependency management,
and event-driven coordination via RabbitMQ
"""

import os
import json
import asyncio
import aiohttp
import pika
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from enum import Enum
import threading
import time

app = Flask(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowEngine:
    """
    Master Workflow Engine for coordinating all agents
    """
    
    def __init__(self):
        self.rabbitmq_url = os.environ.get('RABBITMQ_URL', 'amqp://guest:guest@rabbitmq:5672/')
        self.task_queue = {}
        self.active_workflows = {}
        self.agent_status = {}
        self.workflow_templates = {}
        
        # Agent registry
        self.agents = {
            'database_orchestrator': {'url': 'http://database-orchestrator:8000', 'status': 'unknown'},
            'etl_processor': {'url': 'http://etl-processor:8001', 'status': 'unknown'},
            'vector_search': {'url': 'http://vector-search:8002', 'status': 'unknown'},
            'dev_coordinator': {'url': 'http://dev-coordinator:8003', 'status': 'unknown'},
            'mcp_coordinator': {'url': 'http://mcp-coordinator:8004', 'status': 'unknown'},
            'repo_analyzer': {'url': 'http://repo-analyzer:8005', 'status': 'unknown'},
            'lead_guy': {'url': 'http://lead-guy:8000', 'status': 'unknown'}
        }
        
        # Initialize RabbitMQ connection
        self._setup_rabbitmq()
        
        # Load workflow templates
        self._load_workflow_templates()
        
        # Start background tasks
        self._start_background_tasks()
        
        print("✅ Master Workflow Engine initialized")
    
    def _setup_rabbitmq(self):
        """Setup RabbitMQ connection and exchanges"""
        try:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
            self.channel = self.connection.channel()
            
            # Declare exchanges
            self.channel.exchange_declare(exchange='agent_events', exchange_type='topic')
            self.channel.exchange_declare(exchange='task_queue', exchange_type='direct')
            
            # Declare queues
            self.channel.queue_declare(queue='workflow_tasks', durable=True)
            self.channel.queue_declare(queue='agent_status', durable=True)
            
            print("✅ RabbitMQ connection established")
            
        except Exception as e:
            print(f"⚠️ RabbitMQ connection failed: {e}")
            self.connection = None
            self.channel = None
    
    def _load_workflow_templates(self):
        """Load predefined workflow templates"""
        self.workflow_templates = {
            'data_pipeline': {
                'name': 'Complete Data Pipeline',
                'steps': [
                    {'agent': 'etl_processor', 'action': 'process_data', 'depends_on': []},
                    {'agent': 'database_orchestrator', 'action': 'store_data', 'depends_on': ['etl_processor']},
                    {'agent': 'vector_search', 'action': 'index_data', 'depends_on': ['database_orchestrator']}
                ]
            },
            'code_deployment': {
                'name': 'Full Code Deployment',
                'steps': [
                    {'agent': 'repo_analyzer', 'action': 'analyze_repo', 'depends_on': []},
                    {'agent': 'dev_coordinator', 'action': 'generate_code', 'depends_on': ['repo_analyzer']},
                    {'agent': 'dev_coordinator', 'action': 'run_tests', 'depends_on': ['generate_code']},
                    {'agent': 'dev_coordinator', 'action': 'deploy', 'depends_on': ['run_tests']}
                ]
            },
            'business_analysis': {
                'name': 'Complete Business Analysis',
                'steps': [
                    {'agent': 'database_orchestrator', 'action': 'query_metrics', 'depends_on': []},
                    {'agent': 'vector_search', 'action': 'semantic_analysis', 'depends_on': []},
                    {'agent': 'lead_guy', 'action': 'coordinate_report', 'depends_on': ['database_orchestrator', 'vector_search']}
                ]
            }
        }
    
    def _start_background_tasks(self):
        """Start background monitoring and processing tasks"""
        # Start agent health monitoring
        health_thread = threading.Thread(target=self._monitor_agent_health, daemon=True)
        health_thread.start()
        
        # Start task processor
        task_thread = threading.Thread(target=self._process_task_queue, daemon=True)
        task_thread.start()
    
    def _monitor_agent_health(self):
        """Continuously monitor agent health"""
        while True:
            try:
                for agent_name, agent_info in self.agents.items():
                    asyncio.run(self._check_agent_health(agent_name, agent_info))
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Health monitoring error: {e}")
                time.sleep(60)
    
    async def _check_agent_health(self, agent_name: str, agent_info: Dict):
        """Check individual agent health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{agent_info['url']}/health", timeout=10) as response:
                    if response.status == 200:
                        self.agents[agent_name]['status'] = 'healthy'
                        self.agents[agent_name]['last_check'] = datetime.now().isoformat()
                    else:
                        self.agents[agent_name]['status'] = 'unhealthy'
        except Exception as e:
            self.agents[agent_name]['status'] = 'unreachable'
            self.agents[agent_name]['error'] = str(e)
    
    def _process_task_queue(self):
        """Process queued tasks"""
        while True:
            try:
                # Process pending tasks
                for task_id, task in list(self.task_queue.items()):
                    if task['status'] == TaskStatus.PENDING:
                        asyncio.run(self._execute_task(task_id, task))
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"Task processing error: {e}")
                time.sleep(10)
    
    async def create_workflow(self, workflow_type: str, parameters: Dict) -> Dict:
        """Create a new workflow from template"""
        try:
            if workflow_type not in self.workflow_templates:
                return {'status': 'error', 'error': f'Unknown workflow type: {workflow_type}'}
            
            workflow_id = str(uuid.uuid4())
            template = self.workflow_templates[workflow_type]
            
            workflow = {
                'id': workflow_id,
                'type': workflow_type,
                'name': template['name'],
                'parameters': parameters,
                'steps': template['steps'].copy(),
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'progress': 0,
                'current_step': 0,
                'results': {}
            }
            
            self.active_workflows[workflow_id] = workflow
            
            # Queue first step
            await self._queue_next_steps(workflow_id)
            
            return {
                'status': 'success',
                'workflow_id': workflow_id,
                'workflow': workflow
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def _queue_next_steps(self, workflow_id: str):
        """Queue the next available steps in workflow"""
        workflow = self.active_workflows[workflow_id]
        
        for i, step in enumerate(workflow['steps']):
            if step.get('status') == 'completed':
                continue
            
            if step.get('status') in ['pending', 'running']:
                continue
            
            # Check if dependencies are met
            dependencies_met = True
            for dep in step['depends_on']:
                dep_completed = False
                for prev_step in workflow['steps']:
                    if prev_step['agent'] == dep and prev_step.get('status') == 'completed':
                        dep_completed = True
                        break
                
                if not dep_completed:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                # Queue this step
                task_id = f"{workflow_id}_step_{i}"
                task = {
                    'id': task_id,
                    'workflow_id': workflow_id,
                    'step_index': i,
                    'agent': step['agent'],
                    'action': step['action'],
                    'parameters': workflow['parameters'],
                    'status': TaskStatus.PENDING,
                    'created_at': datetime.now().isoformat()
                }
                
                self.task_queue[task_id] = task
                step['status'] = 'pending'
                step['task_id'] = task_id
    
    async def _execute_task(self, task_id: str, task: Dict):
        """Execute a queued task"""
        try:
            task['status'] = TaskStatus.RUNNING
            task['started_at'] = datetime.now().isoformat()
            
            agent_name = task['agent']
            agent_info = self.agents.get(agent_name)
            
            if not agent_info or agent_info['status'] != 'healthy':
                task['status'] = TaskStatus.FAILED
                task['error'] = f'Agent {agent_name} not available'
                return
            
            # Execute task on agent
            async with aiohttp.ClientSession() as session:
                payload = {
                    'action': task['action'],
                    'parameters': task['parameters'],
                    'task_id': task_id
                }
                
                # Try different endpoints based on agent
                endpoint = self._get_agent_endpoint(agent_name, task['action'])
                
                async with session.post(f"{agent_info['url']}{endpoint}", json=payload, timeout=300) as response:
                    if response.status == 200:
                        result = await response.json()
                        task['status'] = TaskStatus.COMPLETED
                        task['result'] = result
                        task['completed_at'] = datetime.now().isoformat()
                        
                        # Update workflow step
                        workflow = self.active_workflows[task['workflow_id']]
                        workflow['steps'][task['step_index']]['status'] = 'completed'
                        workflow['steps'][task['step_index']]['result'] = result
                        
                        # Queue next steps
                        await self._queue_next_steps(task['workflow_id'])
                        
                    else:
                        task['status'] = TaskStatus.FAILED
                        task['error'] = f'Agent returned {response.status}'
        
        except Exception as e:
            task['status'] = TaskStatus.FAILED
            task['error'] = str(e)
    
    def _get_agent_endpoint(self, agent_name: str, action: str) -> str:
        """Get appropriate endpoint for agent action"""
        endpoint_map = {
            'database_orchestrator': {
                'query_metrics': '/query',
                'store_data': '/query',
                'default': '/status'
            },
            'etl_processor': {
                'process_data': '/process',
                'default': '/health'
            },
            'vector_search': {
                'index_data': '/add-document',
                'semantic_analysis': '/search',
                'default': '/stats'
            },
            'dev_coordinator': {
                'generate_code': '/generate-code',
                'run_tests': '/run-tests',
                'deploy': '/commit-push',
                'default': '/status'
            },
            'repo_analyzer': {
                'analyze_repo': '/analyze-repo',
                'default': '/status'
            }
        }
        
        agent_endpoints = endpoint_map.get(agent_name, {})
        return agent_endpoints.get(action, agent_endpoints.get('default', '/health'))
    
    def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get workflow status"""
        if workflow_id not in self.active_workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate progress
        completed_steps = len([s for s in workflow['steps'] if s.get('status') == 'completed'])
        total_steps = len(workflow['steps'])
        progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
        
        workflow['progress'] = progress
        
        return workflow
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            'agents': self.agents,
            'active_workflows': len(self.active_workflows),
            'queued_tasks': len([t for t in self.task_queue.values() if t['status'] == TaskStatus.PENDING]),
            'running_tasks': len([t for t in self.task_queue.values() if t['status'] == TaskStatus.RUNNING]),
            'workflow_templates': list(self.workflow_templates.keys()),
            'rabbitmq_connected': self.connection is not None and not self.connection.is_closed
        }

# Initialize workflow engine
workflow_engine = WorkflowEngine()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Master Workflow Engine',
        'status': 'operational',
        'version': '1.0.0',
        'capabilities': [
            'Multi-Agent Workflow Orchestration',
            'Task Queue Management',
            'Agent Health Monitoring',
            'Event-Driven Coordination',
            'Dependency Management',
            'Progress Tracking'
        ],
        'endpoints': [
            '/health',
            '/create-workflow',
            '/workflow-status',
            '/system-status',
            '/agents',
            '/templates'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'active_workflows': len(workflow_engine.active_workflows),
        'queued_tasks': len(workflow_engine.task_queue),
        'agents_healthy': len([a for a in workflow_engine.agents.values() if a['status'] == 'healthy']),
        'rabbitmq_connected': workflow_engine.connection is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/create-workflow', methods=['POST'])
def create_workflow():
    """Create new workflow"""
    data = request.get_json()
    
    if not data or 'workflow_type' not in data:
        return jsonify({'error': 'workflow_type required'}), 400
    
    workflow_type = data['workflow_type']
    parameters = data.get('parameters', {})
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            workflow_engine.create_workflow(workflow_type, parameters)
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/workflow-status', methods=['GET'])
def workflow_status():
    """Get workflow status"""
    workflow_id = request.args.get('workflow_id')
    
    if not workflow_id:
        return jsonify({'error': 'workflow_id parameter required'}), 400
    
    status = workflow_engine.get_workflow_status(workflow_id)
    return jsonify(status)

@app.route('/system-status', methods=['GET'])
def system_status():
    """Get system status"""
    status = workflow_engine.get_system_status()
    return jsonify(status)

@app.route('/agents', methods=['GET'])
def agents():
    """Get agent status"""
    return jsonify(workflow_engine.agents)

@app.route('/templates', methods=['GET'])
def templates():
    """Get workflow templates"""
    return jsonify(workflow_engine.workflow_templates)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8006))
    app.run(host='0.0.0.0', port=port, debug=False)
