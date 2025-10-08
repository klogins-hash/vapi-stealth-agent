"""
Repository Analysis Agent
Expert at analyzing repos and translating them for Microsoft Agent Framework + Groq integration
Specializes in understanding architecture patterns and avoiding conflicts
"""

import os
import json
import asyncio
import aiohttp
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from pathlib import Path
import git

app = Flask(__name__)

class RepoAnalyzer:
    """
    Repository Analysis Agent - Expert at understanding and translating codebases
    """
    
    def __init__(self):
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        self.workspace_dir = '/tmp/repo_analysis'
        self.analyzed_repos = {}
        self.analysis_cache = {}
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)
        print("✅ Repository Analysis Agent initialized")
    
    async def clone_and_analyze(self, repo_url: str, target_framework: str = 'microsoft_agent') -> Dict:
        """
        Clone repository and perform comprehensive analysis
        """
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            local_path = os.path.join(self.workspace_dir, repo_name)
            
            # Remove existing if present
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            
            # Clone repository
            print(f"Cloning {repo_url}...")
            repo = git.Repo.clone_from(repo_url, local_path)
            
            # Perform analysis
            analysis = await self._analyze_repository(local_path, repo_name, target_framework)
            
            # Store analysis
            self.analyzed_repos[repo_name] = {
                'url': repo_url,
                'local_path': local_path,
                'analysis': analysis,
                'analyzed_at': datetime.now().isoformat()
            }
            
            return {
                'status': 'success',
                'repo_name': repo_name,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _analyze_repository(self, repo_path: str, repo_name: str, target_framework: str) -> Dict:
        """
        Comprehensive repository analysis
        """
        analysis = {
            'repo_name': repo_name,
            'target_framework': target_framework,
            'structure': await self._analyze_structure(repo_path),
            'dependencies': await self._analyze_dependencies(repo_path),
            'architecture': await self._analyze_architecture(repo_path),
            'ai_patterns': await self._analyze_ai_patterns(repo_path),
            'translation_strategy': {},
            'conflict_analysis': {},
            'integration_recommendations': []
        }
        
        # Generate translation strategy
        analysis['translation_strategy'] = await self._generate_translation_strategy(analysis, target_framework)
        
        # Analyze potential conflicts
        analysis['conflict_analysis'] = await self._analyze_conflicts(analysis, target_framework)
        
        # Generate integration recommendations
        analysis['integration_recommendations'] = await self._generate_recommendations(analysis, target_framework)
        
        return analysis
    
    async def _analyze_structure(self, repo_path: str) -> Dict:
        """
        Analyze repository structure and key files
        """
        structure = {
            'total_files': 0,
            'key_files': [],
            'directories': [],
            'file_types': {},
            'entry_points': []
        }
        
        try:
            for root, dirs, files in os.walk(repo_path):
                # Skip .git directory
                if '.git' in root:
                    continue
                
                structure['total_files'] += len(files)
                
                # Track directories
                rel_root = os.path.relpath(root, repo_path)
                if rel_root != '.':
                    structure['directories'].append(rel_root)
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # Track file types
                    ext = os.path.splitext(file)[1]
                    structure['file_types'][ext] = structure['file_types'].get(ext, 0) + 1
                    
                    # Identify key files
                    if file.lower() in ['readme.md', 'requirements.txt', 'package.json', 'dockerfile', 'main.py', 'app.py', '__init__.py']:
                        structure['key_files'].append(rel_path)
                    
                    # Identify entry points
                    if file in ['main.py', 'app.py', 'server.py', 'run.py']:
                        structure['entry_points'].append(rel_path)
        
        except Exception as e:
            structure['error'] = str(e)
        
        return structure
    
    async def _analyze_dependencies(self, repo_path: str) -> Dict:
        """
        Analyze dependencies and requirements
        """
        dependencies = {
            'python': [],
            'javascript': [],
            'docker': False,
            'ai_libraries': [],
            'web_frameworks': [],
            'database_deps': []
        }
        
        try:
            # Python dependencies
            req_file = os.path.join(repo_path, 'requirements.txt')
            if os.path.exists(req_file):
                with open(req_file, 'r') as f:
                    deps = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                    dependencies['python'] = deps
                    
                    # Categorize AI libraries
                    ai_libs = ['anthropic', 'openai', 'groq', 'transformers', 'langchain', 'llamaindex', 'semantic-kernel']
                    dependencies['ai_libraries'] = [dep for dep in deps if any(lib in dep.lower() for lib in ai_libs)]
                    
                    # Web frameworks
                    web_frameworks = ['flask', 'fastapi', 'django', 'streamlit', 'gradio']
                    dependencies['web_frameworks'] = [dep for dep in deps if any(fw in dep.lower() for fw in web_frameworks)]
                    
                    # Database dependencies
                    db_deps = ['psycopg2', 'pymongo', 'redis', 'sqlalchemy', 'asyncpg']
                    dependencies['database_deps'] = [dep for dep in deps if any(db in dep.lower() for db in db_deps)]
            
            # JavaScript dependencies
            package_file = os.path.join(repo_path, 'package.json')
            if os.path.exists(package_file):
                with open(package_file, 'r') as f:
                    package_data = json.load(f)
                    dependencies['javascript'] = list(package_data.get('dependencies', {}).keys())
            
            # Docker
            dockerfile = os.path.join(repo_path, 'Dockerfile')
            dependencies['docker'] = os.path.exists(dockerfile)
        
        except Exception as e:
            dependencies['error'] = str(e)
        
        return dependencies
    
    async def _analyze_architecture(self, repo_path: str) -> Dict:
        """
        Analyze architectural patterns
        """
        architecture = {
            'patterns': [],
            'agent_framework': None,
            'api_style': None,
            'data_flow': [],
            'key_components': []
        }
        
        try:
            # Look for common patterns in Python files
            for root, dirs, files in os.walk(repo_path):
                if '.git' in root:
                    continue
                
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().lower()
                                
                                # Detect patterns
                                if 'class agent' in content or 'agent(' in content:
                                    architecture['patterns'].append('agent_pattern')
                                
                                if 'anthropic' in content:
                                    architecture['agent_framework'] = 'anthropic_claude'
                                elif 'openai' in content:
                                    architecture['agent_framework'] = 'openai'
                                elif 'groq' in content:
                                    architecture['agent_framework'] = 'groq'
                                
                                if 'fastapi' in content or '@app.route' in content:
                                    architecture['api_style'] = 'rest_api'
                                
                                if 'async def' in content:
                                    architecture['patterns'].append('async_pattern')
                                
                                if 'tool' in content and 'function' in content:
                                    architecture['patterns'].append('tool_calling')
                        
                        except Exception:
                            continue
        
        except Exception as e:
            architecture['error'] = str(e)
        
        return architecture
    
    async def _analyze_ai_patterns(self, repo_path: str) -> Dict:
        """
        Analyze AI-specific patterns and approaches
        """
        ai_patterns = {
            'llm_provider': None,
            'prompt_patterns': [],
            'tool_usage': [],
            'context_management': [],
            'agent_coordination': []
        }
        
        try:
            # Scan for AI patterns
            for root, dirs, files in os.walk(repo_path):
                if '.git' in root:
                    continue
                
                for file in files:
                    if file.endswith(('.py', '.md', '.txt')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                content_lower = content.lower()
                                
                                # LLM Provider detection
                                if 'anthropic' in content_lower and 'claude' in content_lower:
                                    ai_patterns['llm_provider'] = 'anthropic_claude'
                                elif 'openai' in content_lower:
                                    ai_patterns['llm_provider'] = 'openai'
                                elif 'groq' in content_lower:
                                    ai_patterns['llm_provider'] = 'groq'
                                
                                # Prompt patterns
                                if 'system prompt' in content_lower or 'system_prompt' in content_lower:
                                    ai_patterns['prompt_patterns'].append('system_prompts')
                                if 'few shot' in content_lower or 'few_shot' in content_lower:
                                    ai_patterns['prompt_patterns'].append('few_shot')
                                if 'chain of thought' in content_lower or 'cot' in content_lower:
                                    ai_patterns['prompt_patterns'].append('chain_of_thought')
                                
                                # Tool usage
                                if 'function_call' in content_lower or 'tool_call' in content_lower:
                                    ai_patterns['tool_usage'].append('function_calling')
                                if 'computer use' in content_lower or 'computer_use' in content_lower:
                                    ai_patterns['tool_usage'].append('computer_use')
                                
                                # Context management
                                if 'conversation' in content_lower and 'history' in content_lower:
                                    ai_patterns['context_management'].append('conversation_history')
                                if 'memory' in content_lower and ('store' in content_lower or 'retrieve' in content_lower):
                                    ai_patterns['context_management'].append('memory_management')
                        
                        except Exception:
                            continue
        
        except Exception as e:
            ai_patterns['error'] = str(e)
        
        return ai_patterns
    
    async def _generate_translation_strategy(self, analysis: Dict, target_framework: str) -> Dict:
        """
        Generate strategy for translating to target framework
        """
        strategy = {
            'approach': 'hybrid_integration',
            'key_changes': [],
            'preserved_patterns': [],
            'new_components': [],
            'migration_steps': []
        }
        
        # Analyze current vs target
        current_provider = analysis['ai_patterns'].get('llm_provider')
        
        if current_provider == 'anthropic_claude' and target_framework == 'microsoft_agent':
            strategy['key_changes'] = [
                'Replace Anthropic client with Groq client',
                'Adapt Claude-specific prompts for Groq models',
                'Integrate with Microsoft Agent Framework patterns',
                'Maintain tool calling but adapt to Groq format'
            ]
            
            strategy['migration_steps'] = [
                '1. Set up Groq client with API key',
                '2. Create Microsoft Agent Framework wrapper',
                '3. Translate prompt templates',
                '4. Adapt tool calling mechanisms',
                '5. Test with existing workflows'
            ]
        
        # Preserve valuable patterns
        if 'async_pattern' in analysis['architecture']['patterns']:
            strategy['preserved_patterns'].append('Async/await patterns for performance')
        
        if 'tool_calling' in analysis['architecture']['patterns']:
            strategy['preserved_patterns'].append('Tool calling architecture')
        
        return strategy
    
    async def _analyze_conflicts(self, analysis: Dict, target_framework: str) -> Dict:
        """
        Analyze potential conflicts with target framework
        """
        conflicts = {
            'high_risk': [],
            'medium_risk': [],
            'low_risk': [],
            'mitigation_strategies': {}
        }
        
        # Check for framework conflicts
        if target_framework == 'microsoft_agent':
            if analysis['ai_patterns']['llm_provider'] == 'anthropic_claude':
                conflicts['medium_risk'].append({
                    'issue': 'Claude-specific prompt patterns',
                    'impact': 'Prompts may need significant adaptation for Groq models'
                })
            
            if 'computer_use' in analysis['ai_patterns']['tool_usage']:
                conflicts['high_risk'].append({
                    'issue': 'Computer Use API dependency',
                    'impact': 'Claude Computer Use may not translate directly to other providers'
                })
        
        return conflicts
    
    async def _generate_recommendations(self, analysis: Dict, target_framework: str) -> List[str]:
        """
        Generate integration recommendations
        """
        recommendations = []
        
        if target_framework == 'microsoft_agent':
            recommendations.extend([
                'Create a Groq-based agent wrapper that mimics the original interface',
                'Use Microsoft Agent Framework for orchestration and coordination',
                'Leverage existing async patterns for better performance',
                'Implement tool calling through Microsoft Agent Framework tools',
                'Use your existing private network for agent communication'
            ])
            
            if analysis['dependencies']['web_frameworks']:
                recommendations.append('Integrate with your existing Flask-based agent ecosystem')
            
            if analysis['dependencies']['database_deps']:
                recommendations.append('Route database operations through your Database Orchestrator agent')
        
        return recommendations
    
    def get_analysis_summary(self, repo_name: str) -> Dict:
        """
        Get analysis summary for a repository
        """
        if repo_name not in self.analyzed_repos:
            return {'error': 'Repository not analyzed'}
        
        analysis = self.analyzed_repos[repo_name]['analysis']
        
        return {
            'repo_name': repo_name,
            'summary': {
                'total_files': analysis['structure']['total_files'],
                'main_language': 'Python' if '.py' in analysis['structure']['file_types'] else 'Unknown',
                'ai_provider': analysis['ai_patterns']['llm_provider'],
                'architecture_patterns': analysis['architecture']['patterns'],
                'translation_complexity': 'Medium' if analysis['conflict_analysis']['high_risk'] else 'Low'
            },
            'quick_wins': analysis['integration_recommendations'][:3],
            'analyzed_at': self.analyzed_repos[repo_name]['analyzed_at']
        }

# Initialize repo analyzer
repo_analyzer = RepoAnalyzer()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Repository Analysis Agent',
        'status': 'operational',
        'version': '1.0.0',
        'specialization': 'Microsoft Agent Framework + Groq Integration',
        'capabilities': [
            'Repository Structure Analysis',
            'Dependency Analysis',
            'Architecture Pattern Detection',
            'AI Framework Translation',
            'Conflict Detection',
            'Integration Recommendations'
        ],
        'endpoints': [
            '/health',
            '/analyze-repo',
            '/get-analysis',
            '/translation-guide',
            '/status'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'analyzed_repos': len(repo_analyzer.analyzed_repos),
        'workspace_dir': repo_analyzer.workspace_dir,
        'groq_configured': bool(repo_analyzer.groq_api_key),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze-repo', methods=['POST'])
def analyze_repo():
    """Analyze a repository"""
    data = request.get_json()
    
    if not data or 'repo_url' not in data:
        return jsonify({'error': 'repo_url required'}), 400
    
    repo_url = data['repo_url']
    target_framework = data.get('target_framework', 'microsoft_agent')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            repo_analyzer.clone_and_analyze(repo_url, target_framework)
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/get-analysis', methods=['GET'])
def get_analysis():
    """Get analysis for a repository"""
    repo_name = request.args.get('repo_name')
    
    if not repo_name:
        return jsonify({'error': 'repo_name parameter required'}), 400
    
    summary = repo_analyzer.get_analysis_summary(repo_name)
    return jsonify(summary)

@app.route('/translation-guide', methods=['POST'])
def translation_guide():
    """Get detailed translation guide"""
    data = request.get_json()
    
    if not data or 'repo_name' not in data:
        return jsonify({'error': 'repo_name required'}), 400
    
    repo_name = data['repo_name']
    
    if repo_name not in repo_analyzer.analyzed_repos:
        return jsonify({'error': 'Repository not analyzed'}), 404
    
    analysis = repo_analyzer.analyzed_repos[repo_name]['analysis']
    
    return jsonify({
        'repo_name': repo_name,
        'translation_strategy': analysis['translation_strategy'],
        'conflict_analysis': analysis['conflict_analysis'],
        'recommendations': analysis['integration_recommendations'],
        'next_steps': analysis['translation_strategy']['migration_steps']
    })

@app.route('/status', methods=['GET'])
def status():
    """Get analyzer status"""
    return jsonify({
        'analyzed_repositories': list(repo_analyzer.analyzed_repos.keys()),
        'total_analyses': len(repo_analyzer.analyzed_repos),
        'workspace_directory': repo_analyzer.workspace_dir,
        'cache_size': len(repo_analyzer.analysis_cache)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8005))
    app.run(host='0.0.0.0', port=port, debug=False)
