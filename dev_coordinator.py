"""
AI Development Coordinator Agent
Replaces human coordinator with full IDE capabilities and team management
Based on Microsoft Agent Framework patterns
"""

import os
import json
import asyncio
import aiohttp
import subprocess
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from pathlib import Path
import git
import shutil

app = Flask(__name__)

class DevCoordinator:
    """
    AI Development Coordinator with full IDE capabilities
    """
    
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.rube_api_key = os.environ.get('RUBE_API_KEY', 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOiJ1c2VyXzAxSzRRSDI5R1pWQURROU1IQVhWWFdZUjZLIiwib3JnSWQiOiJvcmdfMDFLNFFIMlBIUzI2RzJBVkRWRkZNUE0zNjkiLCJpYXQiOjE3NTg1NzA1MTB9.3GoJYV-XcNDy32IAJh7hbzsP9I1-hRhJ1kYpWNncj30')
        self.rube_api_url = 'https://rube.app/mcp'
        self.main_repo_url = 'https://github.com/klogins-hash/vapi-stealth-agent.git'
        self.workspace_dir = '/tmp/workspaces'
        self.active_repos = {}
        self.team_agents = {
            'database_orchestrator': 'http://database-orchestrator:8000',
            'etl_processor': 'http://etl-processor:8001',
            'vector_search': 'http://vector-search:8002',
            'business_integrator': 'http://vapi-stealth-agent:8080'
        }
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)
        print("✅ AI Development Coordinator initialized")
    
    async def clone_repository(self, repo_url: str, branch: str = 'main') -> Dict:
        """
        Clone repository for development work
        """
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            local_path = os.path.join(self.workspace_dir, repo_name)
            
            # Remove existing if present
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            
            # Clone repository
            repo = git.Repo.clone_from(repo_url, local_path, branch=branch)
            
            self.active_repos[repo_name] = {
                'path': local_path,
                'repo': repo,
                'url': repo_url,
                'branch': branch,
                'last_updated': datetime.now().isoformat()
            }
            
            return {
                'status': 'success',
                'repo_name': repo_name,
                'local_path': local_path,
                'branch': branch,
                'files': self._list_files(local_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def create_file(self, repo_name: str, file_path: str, content: str) -> Dict:
        """
        Create or update a file in the repository
        """
        try:
            if repo_name not in self.active_repos:
                return {'status': 'error', 'error': 'Repository not found'}
            
            repo_info = self.active_repos[repo_name]
            full_path = os.path.join(repo_info['path'], file_path)
            
            # Create directories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'size': len(content),
                'action': 'created' if not os.path.exists(full_path) else 'updated'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def read_file(self, repo_name: str, file_path: str) -> Dict:
        """
        Read file contents from repository
        """
        try:
            if repo_name not in self.active_repos:
                return {'status': 'error', 'error': 'Repository not found'}
            
            repo_info = self.active_repos[repo_name]
            full_path = os.path.join(repo_info['path'], file_path)
            
            if not os.path.exists(full_path):
                return {'status': 'error', 'error': 'File not found'}
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'status': 'success',
                'file_path': file_path,
                'content': content,
                'size': len(content)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def commit_and_push(self, repo_name: str, commit_message: str, files: List[str] = None) -> Dict:
        """
        Commit changes and push to GitHub
        """
        try:
            if repo_name not in self.active_repos:
                return {'status': 'error', 'error': 'Repository not found'}
            
            repo_info = self.active_repos[repo_name]
            repo = repo_info['repo']
            
            # Add files
            if files:
                for file_path in files:
                    repo.index.add([file_path])
            else:
                repo.git.add('.')
            
            # Check if there are changes to commit
            if not repo.index.diff("HEAD"):
                return {'status': 'no_changes', 'message': 'No changes to commit'}
            
            # Commit
            commit = repo.index.commit(commit_message)
            
            # Push
            origin = repo.remote('origin')
            origin.push()
            
            return {
                'status': 'success',
                'commit_hash': commit.hexsha,
                'commit_message': commit_message,
                'files_changed': len(commit.stats.files)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def run_tests(self, repo_name: str, test_command: str = None) -> Dict:
        """
        Run tests in the repository
        """
        try:
            if repo_name not in self.active_repos:
                return {'status': 'error', 'error': 'Repository not found'}
            
            repo_info = self.active_repos[repo_name]
            repo_path = repo_info['path']
            
            # Default test commands
            if not test_command:
                if os.path.exists(os.path.join(repo_path, 'package.json')):
                    test_command = 'npm test'
                elif os.path.exists(os.path.join(repo_path, 'requirements.txt')):
                    test_command = 'python -m pytest'
                else:
                    test_command = 'echo "No default test command found"'
            
            # Run test command
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'status': 'success',
                'command': test_command,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'error': 'Test execution timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def delegate_task(self, task_type: str, task_data: Dict) -> Dict:
        """
        Delegate tasks to appropriate team agents
        """
        try:
            if task_type == 'database_operation':
                agent_url = self.team_agents['database_orchestrator']
                endpoint = '/query'
            elif task_type == 'data_processing':
                agent_url = self.team_agents['etl_processor']
                endpoint = '/process'
            elif task_type == 'vector_search':
                agent_url = self.team_agents['vector_search']
                endpoint = '/search'
            elif task_type == 'business_query':
                agent_url = self.team_agents['business_integrator']
                endpoint = '/business-status'
            else:
                return {'status': 'error', 'error': f'Unknown task type: {task_type}'}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{agent_url}{endpoint}", json=task_data) as response:
                    result = await response.json()
                    
                    return {
                        'status': 'success',
                        'task_type': task_type,
                        'agent_response': result,
                        'delegated_to': agent_url
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def generate_code(self, prompt: str, language: str = 'python', context: Dict = None) -> Dict:
        """
        Generate code using Rube API with MCP capabilities
        """
        try:
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'prompt': f"Generate {language} code for: {prompt}",
                'language': language,
                'context': context or {},
                'max_tokens': 2000,
                'temperature': 0.3  # Lower temperature for more consistent code
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract generated code from Rube response
                        generated_code = result.get('code', result.get('content', ''))
                        
                        return {
                            'status': 'success',
                            'prompt': prompt,
                            'language': language,
                            'generated_code': generated_code,
                            'lines': len(generated_code.split('\n')),
                            'rube_response': result,
                            'api_used': 'rube_mcp'
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Rube API error {response.status}: {error_text}")
                        
        except Exception as e:
            # Fallback to template-based generation if Rube API fails
            print(f"Rube API failed, using fallback: {e}")
            
            templates = {
                'python': f'''
def generated_function():
    """
    Generated based on: {prompt}
    """
    # TODO: Implement functionality based on: {prompt}
    pass
''',
                'javascript': f'''
function generatedFunction() {{
    /**
     * Generated based on: {prompt}
     */
    // TODO: Implement functionality based on: {prompt}
}}
''',
                'dockerfile': f'''
FROM python:3.11-slim

WORKDIR /app

# Generated based on: {prompt}
# TODO: Add specific requirements

COPY . .

CMD ["python", "app.py"]
'''
            }
            
            template = templates.get(language, templates['python'])
            
            return {
                'status': 'fallback',
                'prompt': prompt,
                'language': language,
                'generated_code': template,
                'lines': len(template.split('\n')),
                'error': str(e),
                'api_used': 'fallback_template'
            }
    
    async def update_main_repo(self, file_path: str, content: str, commit_message: str) -> Dict:
        """
        Update the main vapi-stealth-agent repo via Rube MCP
        """
        try:
            # First clone the main repo if not already present
            repo_result = await self.clone_repository(self.main_repo_url, 'main')
            if repo_result['status'] != 'success':
                return repo_result
            
            repo_name = 'vapi-stealth-agent'
            
            # Create/update the file
            file_result = await self.create_file(repo_name, file_path, content)
            if file_result['status'] != 'success':
                return file_result
            
            # Commit and push via Rube MCP for enhanced capabilities
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'method': 'github_commit',
                'params': {
                    'repo': 'klogins-hash/vapi-stealth-agent',
                    'file_path': file_path,
                    'content': content,
                    'message': commit_message,
                    'branch': 'main'
                },
                'jsonrpc': '2.0',
                'id': f"github_commit_{int(datetime.now().timestamp())}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'status': 'success',
                            'method': 'rube_mcp_github',
                            'file_path': file_path,
                            'commit_message': commit_message,
                            'result': result
                        }
                    else:
                        # Fallback to regular git operations
                        return await self.commit_and_push(repo_name, commit_message, [file_path])
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def mcp_request(self, method: str, params: Dict = None) -> Dict:
        """
        Make MCP (Model Context Protocol) request to Rube API
        """
        try:
            headers = {
                'Authorization': self.rube_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'method': method,
                'params': params or {},
                'jsonrpc': '2.0',
                'id': f"mcp_{int(datetime.now().timestamp())}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.rube_api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'status': 'success',
                            'method': method,
                            'result': result,
                            'api_used': 'rube_mcp'
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'status': 'error',
                            'error': f"MCP request failed {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _list_files(self, directory: str, max_depth: int = 3) -> List[str]:
        """
        List files in directory with depth limit
        """
        files = []
        try:
            for root, dirs, filenames in os.walk(directory):
                # Calculate depth
                depth = root[len(directory):].count(os.sep)
                if depth >= max_depth:
                    dirs[:] = []  # Don't go deeper
                    continue
                
                for filename in filenames:
                    if not filename.startswith('.'):  # Skip hidden files
                        rel_path = os.path.relpath(os.path.join(root, filename), directory)
                        files.append(rel_path)
        except Exception as e:
            print(f"Error listing files: {e}")
        
        return files[:50]  # Limit to 50 files
    
    def get_status(self) -> Dict:
        """
        Get coordinator status
        """
        return {
            'active_repositories': len(self.active_repos),
            'workspace_directory': self.workspace_dir,
            'team_agents': list(self.team_agents.keys()),
            'github_token_configured': bool(self.github_token),
            'repositories': {name: {
                'branch': info['branch'],
                'last_updated': info['last_updated']
            } for name, info in self.active_repos.items()}
        }

# Initialize development coordinator
dev_coordinator = DevCoordinator()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'AI Development Coordinator',
        'status': 'operational',
        'version': '1.0.0',
        'capabilities': [
            'Repository Management',
            'Code Generation',
            'File Operations',
            'Git Operations',
            'Test Execution',
            'Team Delegation',
            'IDE Functionality'
        ],
        'endpoints': [
            '/health',
            '/clone-repo',
            '/create-file',
            '/read-file',
            '/commit-push',
            '/run-tests',
            '/delegate-task',
            '/generate-code',
            '/mcp-request',
            '/update-main-repo',
            '/status'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'active_repos': len(dev_coordinator.active_repos),
        'workspace_dir': dev_coordinator.workspace_dir,
        'team_agents_available': len(dev_coordinator.team_agents),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/clone-repo', methods=['POST'])
def clone_repo():
    """Clone repository"""
    data = request.get_json()
    
    if not data or 'repo_url' not in data:
        return jsonify({'error': 'repo_url required'}), 400
    
    repo_url = data['repo_url']
    branch = data.get('branch', 'main')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(dev_coordinator.clone_repository(repo_url, branch))
        return jsonify(result)
    finally:
        loop.close()

@app.route('/create-file', methods=['POST'])
def create_file():
    """Create or update file"""
    data = request.get_json()
    
    required_fields = ['repo_name', 'file_path', 'content']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': f'Required fields: {required_fields}'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.create_file(data['repo_name'], data['file_path'], data['content'])
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/read-file', methods=['POST'])
def read_file():
    """Read file contents"""
    data = request.get_json()
    
    if not data or 'repo_name' not in data or 'file_path' not in data:
        return jsonify({'error': 'repo_name and file_path required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.read_file(data['repo_name'], data['file_path'])
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/commit-push', methods=['POST'])
def commit_push():
    """Commit and push changes"""
    data = request.get_json()
    
    if not data or 'repo_name' not in data or 'commit_message' not in data:
        return jsonify({'error': 'repo_name and commit_message required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.commit_and_push(data['repo_name'], data['commit_message'], data.get('files'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/run-tests', methods=['POST'])
def run_tests():
    """Run tests"""
    data = request.get_json()
    
    if not data or 'repo_name' not in data:
        return jsonify({'error': 'repo_name required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.run_tests(data['repo_name'], data.get('test_command'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/delegate-task', methods=['POST'])
def delegate_task():
    """Delegate task to team agent"""
    data = request.get_json()
    
    if not data or 'task_type' not in data or 'task_data' not in data:
        return jsonify({'error': 'task_type and task_data required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.delegate_task(data['task_type'], data['task_data'])
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/generate-code', methods=['POST'])
def generate_code():
    """Generate code from prompt"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'prompt required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.generate_code(
                data['prompt'], 
                data.get('language', 'python'),
                data.get('context')
            )
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/mcp-request', methods=['POST'])
def mcp_request():
    """Make MCP request to Rube API"""
    data = request.get_json()
    
    if not data or 'method' not in data:
        return jsonify({'error': 'method required'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.mcp_request(data['method'], data.get('params'))
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/update-main-repo', methods=['POST'])
def update_main_repo():
    """Update main vapi-stealth-agent repo via Rube MCP"""
    data = request.get_json()
    
    required_fields = ['file_path', 'content', 'commit_message']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': f'Required fields: {required_fields}'}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            dev_coordinator.update_main_repo(
                data['file_path'], 
                data['content'], 
                data['commit_message']
            )
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/status', methods=['GET'])
def status():
    """Get coordinator status"""
    return jsonify(dev_coordinator.get_status())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8003))
    app.run(host='0.0.0.0', port=port, debug=False)
