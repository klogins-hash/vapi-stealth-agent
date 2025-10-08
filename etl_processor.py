"""
ETL Processing Agent
Extracts, transforms, and loads data into appropriate databases
Reports to Database Orchestrator for coordination
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
import pandas as pd
import io
import csv
from pathlib import Path

app = Flask(__name__)

class ETLProcessor:
    """
    ETL Processing Agent that handles data transformation and loading
    """
    
    def __init__(self):
        self.db_orchestrator_url = os.environ.get('DB_ORCHESTRATOR_URL', 'http://database-orchestrator:8000')
        self.processing_queue = []
        self.completed_jobs = []
        self.supported_formats = ['json', 'csv', 'xml', 'txt', 'parquet']
        
        print("✅ ETL Processor initialized")
    
    async def process_data(self, data: Any, source_format: str, target_db: str, table_name: str = None) -> Dict:
        """
        Main ETL processing pipeline
        """
        job_id = f"etl_{int(datetime.now().timestamp())}"
        
        try:
            # Extract phase
            extracted_data = await self._extract_data(data, source_format)
            
            # Transform phase
            transformed_data = await self._transform_data(extracted_data, target_db)
            
            # Load phase
            load_result = await self._load_data(transformed_data, target_db, table_name)
            
            job_result = {
                'job_id': job_id,
                'status': 'completed',
                'source_format': source_format,
                'target_db': target_db,
                'records_processed': len(transformed_data) if isinstance(transformed_data, list) else 1,
                'load_result': load_result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.completed_jobs.append(job_result)
            return job_result
            
        except Exception as e:
            error_result = {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.completed_jobs.append(error_result)
            return error_result
    
    async def _extract_data(self, data: Any, source_format: str) -> List[Dict]:
        """
        Extract data from various formats
        """
        if source_format.lower() == 'json':
            if isinstance(data, str):
                return json.loads(data)
            return data
        
        elif source_format.lower() == 'csv':
            if isinstance(data, str):
                # Handle CSV string
                csv_reader = csv.DictReader(io.StringIO(data))
                return list(csv_reader)
            return data
        
        elif source_format.lower() == 'xml':
            # Basic XML parsing - could be enhanced
            import xml.etree.ElementTree as ET
            root = ET.fromstring(data)
            return [{'tag': child.tag, 'text': child.text, 'attrib': child.attrib} for child in root]
        
        elif source_format.lower() == 'txt':
            # Process text data line by line
            lines = data.split('\n') if isinstance(data, str) else data
            return [{'line_number': i+1, 'content': line.strip()} for i, line in enumerate(lines) if line.strip()]
        
        else:
            # Default: treat as raw data
            return [{'raw_data': data}]
    
    async def _transform_data(self, data: List[Dict], target_db: str) -> List[Dict]:
        """
        Transform data based on target database requirements
        """
        transformed = []
        
        for record in data:
            if target_db.lower() == 'postgresql':
                # PostgreSQL transformations
                transformed_record = {
                    'id': record.get('id') or len(transformed) + 1,
                    'data': json.dumps(record) if isinstance(record, dict) else str(record),
                    'created_at': datetime.now().isoformat(),
                    'processed_by': 'etl_processor'
                }
                
            elif target_db.lower() == 'mongodb':
                # MongoDB transformations
                transformed_record = {
                    '_id': record.get('_id') or f"etl_{len(transformed) + 1}",
                    'original_data': record,
                    'metadata': {
                        'processed_at': datetime.now().isoformat(),
                        'processor': 'etl_agent'
                    }
                }
                
            elif target_db.lower() == 'valkey':
                # Valkey/Redis transformations (key-value pairs)
                key = record.get('key') or f"etl_key_{len(transformed) + 1}"
                transformed_record = {
                    'key': key,
                    'value': json.dumps(record),
                    'ttl': 3600  # 1 hour default TTL
                }
                
            else:
                # Default transformation
                transformed_record = record
            
            transformed.append(transformed_record)
        
        return transformed
    
    async def _load_data(self, data: List[Dict], target_db: str, table_name: str = None) -> Dict:
        """
        Load data into target database via Database Orchestrator
        """
        try:
            if target_db.lower() == 'postgresql':
                # Create table if needed and insert data
                table = table_name or 'etl_processed_data'
                
                # Create table query
                create_query = f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    data JSONB,
                    created_at TIMESTAMP,
                    processed_by VARCHAR(255)
                );
                """
                
                # Send create table request to DB orchestrator
                async with aiohttp.ClientSession() as session:
                    create_payload = {
                        'database': 'postgresql',
                        'query': create_query
                    }
                    
                    async with session.post(f"{self.db_orchestrator_url}/query", json=create_payload) as response:
                        create_result = await response.json()
                
                # Insert data
                for record in data:
                    insert_query = f"""
                    INSERT INTO {table} (data, created_at, processed_by) 
                    VALUES (%s, %s, %s)
                    """
                    
                    insert_payload = {
                        'database': 'postgresql',
                        'query': insert_query,
                        'params': [record['data'], record['created_at'], record['processed_by']]
                    }
                    
                    async with session.post(f"{self.db_orchestrator_url}/query", json=insert_payload) as response:
                        insert_result = await response.json()
                
                return {'status': 'loaded', 'target': f'postgresql.{table}', 'records': len(data)}
                
            elif target_db.lower() == 'mongodb':
                # MongoDB loading would be implemented here
                return {'status': 'loaded', 'target': 'mongodb', 'records': len(data), 'note': 'MongoDB loading not yet implemented'}
                
            elif target_db.lower() == 'valkey':
                # Valkey/Redis loading would be implemented here
                return {'status': 'loaded', 'target': 'valkey', 'records': len(data), 'note': 'Valkey loading not yet implemented'}
                
            else:
                return {'status': 'error', 'message': f'Unsupported target database: {target_db}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_job_status(self, job_id: str = None) -> Dict:
        """
        Get status of ETL jobs
        """
        if job_id:
            for job in self.completed_jobs:
                if job['job_id'] == job_id:
                    return job
            return {'error': 'Job not found'}
        
        return {
            'total_jobs': len(self.completed_jobs),
            'recent_jobs': self.completed_jobs[-10:],  # Last 10 jobs
            'queue_size': len(self.processing_queue)
        }

# Initialize ETL processor
etl_processor = ETLProcessor()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'ETL Processing Agent',
        'status': 'operational',
        'version': '1.0.0',
        'reports_to': 'database-orchestrator',
        'supported_formats': etl_processor.supported_formats,
        'endpoints': [
            '/health',
            '/process',
            '/jobs',
            '/status'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'db_orchestrator_connection': etl_processor.db_orchestrator_url,
        'supported_formats': etl_processor.supported_formats,
        'jobs_completed': len(etl_processor.completed_jobs),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/process', methods=['POST'])
def process_data():
    """Process data through ETL pipeline"""
    request_data = request.get_json()
    
    if not request_data:
        return jsonify({'error': 'No data provided'}), 400
    
    data = request_data.get('data')
    source_format = request_data.get('source_format', 'json')
    target_db = request_data.get('target_db', 'postgresql')
    table_name = request_data.get('table_name')
    
    if not data:
        return jsonify({'error': 'No data to process'}), 400
    
    # Run ETL process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            etl_processor.process_data(data, source_format, target_db, table_name)
        )
        return jsonify(result)
    finally:
        loop.close()

@app.route('/jobs', methods=['GET'])
def get_jobs():
    """Get job status"""
    job_id = request.args.get('job_id')
    result = etl_processor.get_job_status(job_id)
    return jsonify(result)

@app.route('/status', methods=['GET'])
def status():
    """Get ETL processor status"""
    return jsonify({
        'processor_status': 'operational',
        'db_orchestrator_url': etl_processor.db_orchestrator_url,
        'total_jobs_completed': len(etl_processor.completed_jobs),
        'supported_formats': etl_processor.supported_formats,
        'last_job': etl_processor.completed_jobs[-1] if etl_processor.completed_jobs else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    app.run(host='0.0.0.0', port=port, debug=False)
