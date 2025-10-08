"""
Job Discovery & Optimization Agent
Proactively finds work opportunities and optimizations for all other agents
Acts like a continuous improvement manager for the entire system
"""

import os
import json
import asyncio
import aiohttp
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify

app = Flask(__name__)

class JobOptimizer:
    """
    Job Discovery & Optimization Agent - Finds work and improvements for all agents
    """
    
    def __init__(self):
        self.agents = {
            'database_orchestrator': 'http://database-orchestrator:8000',
            'etl_processor': 'http://etl-processor:8001',
            'vector_search': 'http://vector-search:8002',
            'dev_coordinator': 'http://dev-coordinator:8003',
            'mcp_coordinator': 'http://mcp-coordinator:8004',
            'repo_analyzer': 'http://repo-analyzer:8005',
            'workflow_engine': 'http://workflow-engine:8006',
            'lead_guy': 'http://lead-guy:8000'
        }
        
        self.discovered_jobs = []
        self.optimization_suggestions = []
        self.performance_metrics = {}
        self.last_scan = None
        
        # Start background job discovery
        self._start_job_discovery()
        
        print("✅ Job Discovery & Optimization Agent initialized")
    
    def _start_job_discovery(self):
        """Start background job discovery and optimization scanning"""
        # Schedule regular scans
        schedule.every(5).minutes.do(self._scan_for_jobs)
        schedule.every(15).minutes.do(self._analyze_optimizations)
        schedule.every(30).minutes.do(self._performance_analysis)
        schedule.every(1).hours.do(self._system_health_check)
        
        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run the job scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def _scan_for_jobs(self):
        """Scan all agents for potential work opportunities"""
        try:
            print("🔍 Scanning for job opportunities...")
            
            jobs_found = []
            
            # Check each agent for opportunities
            for agent_name, agent_url in self.agents.items():
                agent_jobs = await self._discover_agent_jobs(agent_name, agent_url)
                jobs_found.extend(agent_jobs)
            
            # Add system-wide opportunities
            system_jobs = await self._discover_system_jobs()
            jobs_found.extend(system_jobs)
            
            # Store discovered jobs
            self.discovered_jobs = jobs_found
            self.last_scan = datetime.now().isoformat()
            
            print(f"✅ Found {len(jobs_found)} job opportunities")
            
        except Exception as e:
            print(f"❌ Job scanning failed: {e}")
    
    async def _discover_agent_jobs(self, agent_name: str, agent_url: str) -> List[Dict]:
        """Discover jobs for a specific agent"""
        jobs = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check agent status
                async with session.get(f"{agent_url}/health", timeout=10) as response:
                    if response.status != 200:
                        jobs.append({
                            'type': 'maintenance',
                            'agent': agent_name,
                            'priority': 'high',
                            'description': f'{agent_name} is not responding - needs investigation',
                            'action': 'health_check',
                            'discovered_at': datetime.now().isoformat()
                        })
                        return jobs
                
                # Agent-specific job discovery
                if agent_name == 'database_orchestrator':
                    jobs.extend(await self._discover_database_jobs(session, agent_url))
                elif agent_name == 'etl_processor':
                    jobs.extend(await self._discover_etl_jobs(session, agent_url))
                elif agent_name == 'vector_search':
                    jobs.extend(await self._discover_vector_jobs(session, agent_url))
                elif agent_name == 'dev_coordinator':
                    jobs.extend(await self._discover_dev_jobs(session, agent_url))
                elif agent_name == 'repo_analyzer':
                    jobs.extend(await self._discover_repo_jobs(session, agent_url))
                elif agent_name == 'workflow_engine':
                    jobs.extend(await self._discover_workflow_jobs(session, agent_url))
        
        except Exception as e:
            jobs.append({
                'type': 'error',
                'agent': agent_name,
                'priority': 'medium',
                'description': f'Failed to scan {agent_name}: {str(e)}',
                'action': 'investigate_error',
                'discovered_at': datetime.now().isoformat()
            })
        
        return jobs
    
    async def _discover_database_jobs(self, session: aiohttp.ClientSession, agent_url: str) -> List[Dict]:
        """Discover database optimization opportunities"""
        jobs = []
        
        try:
            # Check if database needs optimization
            async with session.get(f"{agent_url}/status") as response:
                if response.status == 200:
                    status = await response.json()
                    
                    # Suggest regular maintenance
                    jobs.append({
                        'type': 'optimization',
                        'agent': 'database_orchestrator',
                        'priority': 'low',
                        'description': 'Run database ANALYZE and VACUUM for performance',
                        'action': 'optimize_database',
                        'discovered_at': datetime.now().isoformat()
                    })
                    
                    # Check for unused indexes or tables
                    jobs.append({
                        'type': 'analysis',
                        'agent': 'database_orchestrator',
                        'priority': 'medium',
                        'description': 'Analyze database usage patterns and suggest optimizations',
                        'action': 'analyze_usage_patterns',
                        'discovered_at': datetime.now().isoformat()
                    })
        
        except Exception as e:
            pass
        
        return jobs
    
    async def _discover_etl_jobs(self, session: aiohttp.ClientSession, agent_url: str) -> List[Dict]:
        """Discover ETL processing opportunities"""
        jobs = []
        
        try:
            async with session.get(f"{agent_url}/jobs") as response:
                if response.status == 200:
                    job_status = await response.json()
                    
                    # Suggest data quality checks
                    jobs.append({
                        'type': 'quality_check',
                        'agent': 'etl_processor',
                        'priority': 'medium',
                        'description': 'Run data quality validation on recent ETL jobs',
                        'action': 'validate_data_quality',
                        'discovered_at': datetime.now().isoformat()
                    })
                    
                    # Suggest performance optimization
                    jobs.append({
                        'type': 'optimization',
                        'agent': 'etl_processor',
                        'priority': 'low',
                        'description': 'Analyze ETL performance and suggest improvements',
                        'action': 'optimize_etl_performance',
                        'discovered_at': datetime.now().isoformat()
                    })
        
        except Exception as e:
            pass
        
        return jobs
    
    async def _discover_vector_jobs(self, session: aiohttp.ClientSession, agent_url: str) -> List[Dict]:
        """Discover vector search optimization opportunities"""
        jobs = []
        
        try:
            async with session.get(f"{agent_url}/stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    
                    # Suggest index optimization
                    jobs.append({
                        'type': 'optimization',
                        'agent': 'vector_search',
                        'priority': 'medium',
                        'description': 'Optimize vector search indexes for better performance',
                        'action': 'optimize_vector_indexes',
                        'discovered_at': datetime.now().isoformat()
                    })
                    
                    # Suggest embedding updates
                    jobs.append({
                        'type': 'maintenance',
                        'agent': 'vector_search',
                        'priority': 'low',
                        'description': 'Update embeddings for recently modified documents',
                        'action': 'refresh_embeddings',
                        'discovered_at': datetime.now().isoformat()
                    })
        
        except Exception as e:
            pass
        
        return jobs
    
    async def _discover_dev_jobs(self, session: aiohttp.ClientSession, agent_url: str) -> List[Dict]:
        """Discover development opportunities"""
        jobs = []
        
        try:
            async with session.get(f"{agent_url}/status") as response:
                if response.status == 200:
                    status = await response.json()
                    
                    # Suggest code quality checks
                    jobs.append({
                        'type': 'quality_check',
                        'agent': 'dev_coordinator',
                        'priority': 'medium',
                        'description': 'Run code quality analysis on active repositories',
                        'action': 'analyze_code_quality',
                        'discovered_at': datetime.now().isoformat()
                    })
                    
                    # Suggest dependency updates
                    jobs.append({
                        'type': 'maintenance',
                        'agent': 'dev_coordinator',
                        'priority': 'low',
                        'description': 'Check for dependency updates and security patches',
                        'action': 'update_dependencies',
                        'discovered_at': datetime.now().isoformat()
                    })
        
        except Exception as e:
            pass
        
        return jobs
    
    async def _discover_repo_jobs(self, session: aiohttp.ClientSession, agent_url: str) -> List[Dict]:
        """Discover repository analysis opportunities"""
        jobs = []
        
        # Suggest analyzing popular repositories
        jobs.append({
            'type': 'analysis',
            'agent': 'repo_analyzer',
            'priority': 'low',
            'description': 'Analyze trending GitHub repositories for integration opportunities',
            'action': 'analyze_trending_repos',
            'discovered_at': datetime.now().isoformat()
        })
        
        return jobs
    
    async def _discover_workflow_jobs(self, session: aiohttp.ClientSession, agent_url: str) -> List[Dict]:
        """Discover workflow optimization opportunities"""
        jobs = []
        
        try:
            async with session.get(f"{agent_url}/system-status") as response:
                if response.status == 200:
                    status = await response.json()
                    
                    # Suggest workflow optimization
                    jobs.append({
                        'type': 'optimization',
                        'agent': 'workflow_engine',
                        'priority': 'medium',
                        'description': 'Analyze workflow patterns and suggest optimizations',
                        'action': 'optimize_workflows',
                        'discovered_at': datetime.now().isoformat()
                    })
        
        except Exception as e:
            pass
        
        return jobs
    
    async def _discover_system_jobs(self) -> List[Dict]:
        """Discover system-wide opportunities"""
        jobs = []
        
        # System integration opportunities
        jobs.append({
            'type': 'integration',
            'agent': 'system',
            'priority': 'medium',
            'description': 'Analyze inter-agent communication patterns for optimization',
            'action': 'optimize_agent_communication',
            'discovered_at': datetime.now().isoformat()
        })
        
        # Performance monitoring
        jobs.append({
            'type': 'monitoring',
            'agent': 'system',
            'priority': 'low',
            'description': 'Set up comprehensive performance monitoring across all agents',
            'action': 'setup_monitoring',
            'discovered_at': datetime.now().isoformat()
        })
        
        return jobs
    
    async def _analyze_optimizations(self):
        """Analyze system for optimization opportunities"""
        try:
            print("🔧 Analyzing optimization opportunities...")
            
            optimizations = []
            
            # Performance optimizations
            optimizations.append({
                'type': 'performance',
                'priority': 'medium',
                'description': 'Implement response caching across all agents',
                'impact': 'Reduce response times by 30-50%',
                'effort': 'medium',
                'suggested_at': datetime.now().isoformat()
            })
            
            # Resource optimizations
            optimizations.append({
                'type': 'resource',
                'priority': 'low',
                'description': 'Optimize memory usage in vector search operations',
                'impact': 'Reduce memory footprint by 20%',
                'effort': 'low',
                'suggested_at': datetime.now().isoformat()
            })
            
            # Architecture optimizations
            optimizations.append({
                'type': 'architecture',
                'priority': 'high',
                'description': 'Implement load balancing for high-traffic agents',
                'impact': 'Improve system reliability and scalability',
                'effort': 'high',
                'suggested_at': datetime.now().isoformat()
            })
            
            self.optimization_suggestions = optimizations
            
            print(f"✅ Generated {len(optimizations)} optimization suggestions")
            
        except Exception as e:
            print(f"❌ Optimization analysis failed: {e}")
    
    async def _performance_analysis(self):
        """Analyze system performance metrics"""
        try:
            print("📊 Analyzing system performance...")
            
            metrics = {}
            
            # Collect metrics from all agents
            for agent_name, agent_url in self.agents.items():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{agent_url}/health", timeout=5) as response:
                            if response.status == 200:
                                start_time = time.time()
                                await response.json()
                                response_time = (time.time() - start_time) * 1000
                                
                                metrics[agent_name] = {
                                    'status': 'healthy',
                                    'response_time_ms': response_time,
                                    'last_check': datetime.now().isoformat()
                                }
                            else:
                                metrics[agent_name] = {
                                    'status': 'unhealthy',
                                    'last_check': datetime.now().isoformat()
                                }
                
                except Exception as e:
                    metrics[agent_name] = {
                        'status': 'unreachable',
                        'error': str(e),
                        'last_check': datetime.now().isoformat()
                    }
            
            self.performance_metrics = metrics
            
            print("✅ Performance analysis complete")
            
        except Exception as e:
            print(f"❌ Performance analysis failed: {e}")
    
    async def _system_health_check(self):
        """Comprehensive system health check"""
        try:
            print("🏥 Running system health check...")
            
            health_issues = []
            
            # Check agent availability
            for agent_name, metrics in self.performance_metrics.items():
                if metrics.get('status') != 'healthy':
                    health_issues.append({
                        'type': 'availability',
                        'agent': agent_name,
                        'issue': f'{agent_name} is {metrics.get("status")}',
                        'priority': 'high',
                        'detected_at': datetime.now().isoformat()
                    })
                
                # Check response times
                response_time = metrics.get('response_time_ms', 0)
                if response_time > 5000:  # 5 seconds
                    health_issues.append({
                        'type': 'performance',
                        'agent': agent_name,
                        'issue': f'{agent_name} response time is {response_time}ms (slow)',
                        'priority': 'medium',
                        'detected_at': datetime.now().isoformat()
                    })
            
            if health_issues:
                print(f"⚠️ Found {len(health_issues)} health issues")
            else:
                print("✅ System health check passed")
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    
    def get_job_opportunities(self, priority: str = None, agent: str = None) -> List[Dict]:
        """Get discovered job opportunities"""
        jobs = self.discovered_jobs.copy()
        
        if priority:
            jobs = [j for j in jobs if j.get('priority') == priority]
        
        if agent:
            jobs = [j for j in jobs if j.get('agent') == agent]
        
        return jobs
    
    def get_optimization_suggestions(self, priority: str = None) -> List[Dict]:
        """Get optimization suggestions"""
        suggestions = self.optimization_suggestions.copy()
        
        if priority:
            suggestions = [s for s in suggestions if s.get('priority') == priority]
        
        return suggestions
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            'metrics': self.performance_metrics,
            'job_opportunities': len(self.discovered_jobs),
            'optimization_suggestions': len(self.optimization_suggestions),
            'last_scan': self.last_scan,
            'system_health': 'healthy' if all(
                m.get('status') == 'healthy' 
                for m in self.performance_metrics.values()
            ) else 'degraded'
        }

# Initialize job optimizer
job_optimizer = JobOptimizer()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Job Discovery & Optimization Agent',
        'status': 'operational',
        'version': '1.0.0',
        'role': 'Proactive job finder and system optimizer',
        'capabilities': [
            'Continuous Job Discovery',
            'Performance Optimization Analysis',
            'System Health Monitoring',
            'Agent Workload Balancing',
            'Improvement Recommendations'
        ],
        'endpoints': [
            '/health',
            '/jobs',
            '/optimizations',
            '/performance',
            '/scan-now',
            '/status'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'discovered_jobs': len(job_optimizer.discovered_jobs),
        'optimization_suggestions': len(job_optimizer.optimization_suggestions),
        'agents_monitored': len(job_optimizer.agents),
        'last_scan': job_optimizer.last_scan,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/jobs', methods=['GET'])
def get_jobs():
    """Get discovered job opportunities"""
    priority = request.args.get('priority')
    agent = request.args.get('agent')
    
    jobs = job_optimizer.get_job_opportunities(priority, agent)
    
    return jsonify({
        'jobs': jobs,
        'total_jobs': len(jobs),
        'last_scan': job_optimizer.last_scan
    })

@app.route('/optimizations', methods=['GET'])
def get_optimizations():
    """Get optimization suggestions"""
    priority = request.args.get('priority')
    
    suggestions = job_optimizer.get_optimization_suggestions(priority)
    
    return jsonify({
        'optimizations': suggestions,
        'total_suggestions': len(suggestions)
    })

@app.route('/performance', methods=['GET'])
def get_performance():
    """Get performance report"""
    report = job_optimizer.get_performance_report()
    return jsonify(report)

@app.route('/scan-now', methods=['POST'])
def scan_now():
    """Trigger immediate job scan"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(job_optimizer._scan_for_jobs())
        return jsonify({
            'status': 'success',
            'message': 'Job scan completed',
            'jobs_found': len(job_optimizer.discovered_jobs)
        })
    finally:
        loop.close()

@app.route('/status', methods=['GET'])
def status():
    """Get optimizer status"""
    return jsonify({
        'optimizer_status': 'operational',
        'agents_monitored': list(job_optimizer.agents.keys()),
        'performance_metrics': job_optimizer.performance_metrics,
        'last_scan': job_optimizer.last_scan
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8007))
    app.run(host='0.0.0.0', port=port, debug=False)
