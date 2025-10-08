"""
Master Database Orchestrator Agent
Coordinates all database operations across PostgreSQL, MongoDB, MiniIO, and Valkey
"""

import os
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
import asyncpg
import aiohttp
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)

class DatabaseOrchestrator:
    """
    Master database orchestrator that manages all database operations
    """
    
    def __init__(self):
        self.connections = {}
        self.health_status = {}
        self.operation_log = []
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize connections to all databases"""
        try:
            # PostgreSQL connection
            self.pg_config = {
                'host': os.environ.get('POSTGRES_HOST', 'postgresql'),
                'port': int(os.environ.get('POSTGRES_PORT', 5432)),
                'database': os.environ.get('POSTGRES_DATABASE', 'postgres'),
                'user': os.environ.get('POSTGRES_USERNAME', 'postgres'),
                'password': os.environ.get('POSTGRES_PASSWORD', '')
            }
            
            # MongoDB connection
            self.mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://mongodb:27017')
            
            # MiniIO connection
            self.minio_config = {
                'endpoint': os.environ.get('MINIO_ENDPOINT', 'miniio:9000'),
                'access_key': os.environ.get('MINIO_ACCESS_KEY', ''),
                'secret_key': os.environ.get('MINIO_SECRET_KEY', '')
            }
            
            # Valkey (Redis) connection
            self.valkey_config = {
                'host': os.environ.get('VALKEY_HOST', 'valkey-1'),
                'port': int(os.environ.get('VALKEY_PORT', 6379))
            }
            
            print("✅ Database orchestrator initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize database orchestrator: {e}")
    
    async def health_check_all(self) -> Dict:
        """Check health of all database services"""
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'databases': {}
        }
        
        # Check PostgreSQL
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            health_results['databases']['postgresql'] = {
                'status': 'healthy',
                'version': version[:50],
                'connection': 'successful'
            }
        except Exception as e:
            health_results['databases']['postgresql'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_results['overall_status'] = 'degraded'
        
        # Check MongoDB
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            server_info = client.server_info()
            client.close()
            
            health_results['databases']['mongodb'] = {
                'status': 'healthy',
                'version': server_info.get('version', 'unknown'),
                'connection': 'successful'
            }
        except Exception as e:
            health_results['databases']['mongodb'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_results['overall_status'] = 'degraded'
        
        # Check MiniIO (basic connectivity)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.minio_config['endpoint']}/minio/health/live", timeout=5) as response:
                    if response.status == 200:
                        health_results['databases']['minio'] = {
                            'status': 'healthy',
                            'connection': 'successful'
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            health_results['databases']['minio'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_results['overall_status'] = 'degraded'
        
        # Check Valkey (Redis)
        try:
            import redis
            r = redis.Redis(host=self.valkey_config['host'], port=self.valkey_config['port'], socket_timeout=5)
            r.ping()
            info = r.info()
            r.close()
            
            health_results['databases']['valkey'] = {
                'status': 'healthy',
                'version': info.get('redis_version', 'unknown'),
                'connection': 'successful'
            }
        except Exception as e:
            health_results['databases']['valkey'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_results['overall_status'] = 'degraded'
        
        self.health_status = health_results
        return health_results
    
    async def execute_query(self, database: str, query: str, params: List = None) -> Dict:
        """Execute a query on specified database"""
        operation_id = f"op_{int(time.time())}_{database}"
        start_time = time.time()
        
        try:
            if database.lower() == 'postgresql':
                conn = psycopg2.connect(**self.pg_config)
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    result_data = [dict(row) for row in results]
                else:
                    conn.commit()
                    result_data = {"affected_rows": cursor.rowcount}
                
                cursor.close()
                conn.close()
                
            elif database.lower() == 'mongodb':
                # MongoDB operations would go here
                result_data = {"message": "MongoDB operations not yet implemented"}
                
            else:
                raise ValueError(f"Unsupported database: {database}")
            
            execution_time = time.time() - start_time
            
            operation_log = {
                'operation_id': operation_id,
                'database': database,
                'query': query[:100] + '...' if len(query) > 100 else query,
                'execution_time_ms': int(execution_time * 1000),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            self.operation_log.append(operation_log)
            
            return {
                'operation_id': operation_id,
                'status': 'success',
                'data': result_data,
                'execution_time_ms': int(execution_time * 1000)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            operation_log = {
                'operation_id': operation_id,
                'database': database,
                'query': query[:100] + '...' if len(query) > 100 else query,
                'execution_time_ms': int(execution_time * 1000),
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.operation_log.append(operation_log)
            
            return {
                'operation_id': operation_id,
                'status': 'error',
                'error': str(e),
                'execution_time_ms': int(execution_time * 1000)
            }
    
    def get_operation_history(self, limit: int = 50) -> List[Dict]:
        """Get recent operation history"""
        return self.operation_log[-limit:]
    
    async def optimize_databases(self) -> Dict:
        """Run optimization tasks across all databases"""
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations': {}
        }
        
        # PostgreSQL optimizations
        try:
            pg_result = await self.execute_query('postgresql', 'ANALYZE;')
            optimization_results['optimizations']['postgresql'] = {
                'analyze_completed': pg_result['status'] == 'success',
                'execution_time_ms': pg_result['execution_time_ms']
            }
        except Exception as e:
            optimization_results['optimizations']['postgresql'] = {
                'error': str(e)
            }
        
        return optimization_results

# Initialize the orchestrator
db_orchestrator = DatabaseOrchestrator()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Master Database Orchestrator',
        'status': 'operational',
        'version': '1.0.0',
        'databases_managed': ['PostgreSQL', 'MongoDB', 'MiniIO', 'Valkey'],
        'endpoints': [
            '/health',
            '/query',
            '/optimize',
            '/operations',
            '/status'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        health_result = loop.run_until_complete(db_orchestrator.health_check_all())
        return jsonify(health_result)
    finally:
        loop.close()

@app.route('/query', methods=['POST'])
def execute_query():
    """Execute database query"""
    data = request.get_json()
    
    if not data or 'database' not in data or 'query' not in data:
        return jsonify({'error': 'Missing database or query parameter'}), 400
    
    database = data['database']
    query = data['query']
    params = data.get('params', [])
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(db_orchestrator.execute_query(database, query, params))
        return jsonify(result)
    finally:
        loop.close()

@app.route('/optimize', methods=['POST'])
def optimize():
    """Run database optimizations"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(db_orchestrator.optimize_databases())
        return jsonify(result)
    finally:
        loop.close()

@app.route('/operations', methods=['GET'])
def operations():
    """Get operation history"""
    limit = request.args.get('limit', 50, type=int)
    history = db_orchestrator.get_operation_history(limit)
    
    return jsonify({
        'operations': history,
        'total_operations': len(db_orchestrator.operation_log),
        'limit': limit
    })

@app.route('/status', methods=['GET'])
def status():
    """Get orchestrator status"""
    return jsonify({
        'orchestrator_status': 'operational',
        'last_health_check': db_orchestrator.health_status.get('timestamp', 'never'),
        'total_operations': len(db_orchestrator.operation_log),
        'databases_configured': ['postgresql', 'mongodb', 'minio', 'valkey']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
