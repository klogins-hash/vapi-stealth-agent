"""
PostgreSQL Extensions Manager for VAPI Stealth Agent
Handles installation and management of advanced PostgreSQL extensions
"""

import os
from typing import List, Dict, Any
from sqlalchemy import text
from database import db_manager

class ExtensionsManager:
    """Manage PostgreSQL extensions for advanced AI capabilities"""
    
    def __init__(self):
        self.recommended_extensions = {
            'vector': {
                'description': 'Vector similarity search for embeddings',
                'use_case': 'Semantic search, conversation similarity, AI embeddings',
                'priority': 'high'
            },
            'pg_trgm': {
                'description': 'Trigram matching for fuzzy text search',
                'use_case': 'Fuzzy string matching, typo tolerance',
                'priority': 'high'
            },
            'uuid-ossp': {
                'description': 'UUID generation functions',
                'use_case': 'Unique identifiers for sessions and users',
                'priority': 'medium'
            },
            'hstore': {
                'description': 'Key-value store within PostgreSQL',
                'use_case': 'Flexible metadata storage',
                'priority': 'medium'
            },
            'ltree': {
                'description': 'Hierarchical tree-like structures',
                'use_case': 'Conversation threads, topic hierarchies',
                'priority': 'medium'
            },
            'timescaledb': {
                'description': 'Time-series database capabilities',
                'use_case': 'Performance metrics, time-based analytics',
                'priority': 'low'
            },
            'pg_cron': {
                'description': 'Cron-based job scheduler',
                'use_case': 'Automated cleanup, periodic analytics',
                'priority': 'low'
            }
        }
    
    def install_extension(self, extension_name: str) -> bool:
        """Install a PostgreSQL extension"""
        try:
            with db_manager.get_session() as db:
                # Check if already installed
                result = db.execute(text("SELECT 1 FROM pg_extension WHERE extname = :name"), 
                                  {"name": extension_name})
                if result.fetchone():
                    print(f"✅ Extension '{extension_name}' already installed")
                    return True
                
                # Install the extension
                db.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension_name}"))
                db.commit()
                print(f"✅ Successfully installed extension '{extension_name}'")
                return True
                
        except Exception as e:
            print(f"❌ Failed to install extension '{extension_name}': {e}")
            return False
    
    def install_recommended_extensions(self, priority_filter: str = None) -> Dict[str, bool]:
        """Install all recommended extensions, optionally filtered by priority"""
        results = {}
        
        for ext_name, ext_info in self.recommended_extensions.items():
            if priority_filter and ext_info['priority'] != priority_filter:
                continue
                
            print(f"\n📦 Installing {ext_name} ({ext_info['priority']} priority)")
            print(f"   Use case: {ext_info['use_case']}")
            
            results[ext_name] = self.install_extension(ext_name)
        
        return results
    
    def get_installed_extensions(self) -> List[Dict[str, Any]]:
        """Get list of currently installed extensions"""
        try:
            with db_manager.get_session() as db:
                result = db.execute(text("""
                    SELECT 
                        e.extname as name,
                        e.extversion as version,
                        n.nspname as schema,
                        d.description
                    FROM pg_extension e
                    LEFT JOIN pg_namespace n ON n.oid = e.extnamespace
                    LEFT JOIN pg_description d ON d.objoid = e.oid
                    ORDER BY e.extname
                """))
                
                return [{
                    'name': row[0],
                    'version': row[1],
                    'schema': row[2],
                    'description': row[3]
                } for row in result.fetchall()]
                
        except Exception as e:
            print(f"❌ Failed to get installed extensions: {e}")
            return []
    
    def test_vector_extension(self) -> bool:
        """Test pgvector functionality"""
        try:
            with db_manager.get_session() as db:
                # Test vector operations
                db.execute(text("SELECT '[1,2,3]'::vector"))
                db.execute(text("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector"))
                print("✅ Vector extension is working correctly")
                return True
                
        except Exception as e:
            print(f"❌ Vector extension test failed: {e}")
            return False
    
    def create_vector_indexes(self):
        """Create vector indexes for optimal performance"""
        try:
            with db_manager.get_session() as db:
                # We'll add these after creating vector columns
                print("📊 Vector indexes will be created with vector columns")
                return True
                
        except Exception as e:
            print(f"❌ Failed to create vector indexes: {e}")
            return False

# Global extensions manager
extensions_manager = ExtensionsManager()
