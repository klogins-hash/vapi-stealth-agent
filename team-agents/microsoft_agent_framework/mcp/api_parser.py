"""API specification parser for converting APIs to MCP servers."""

import json
import yaml
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIParameter:
    """Represents an API parameter."""
    name: str
    type: str
    description: str = ""
    required: bool = False
    default: Any = None
    enum_values: Optional[List[str]] = None


@dataclass
class APIEndpoint:
    """Represents an API endpoint."""
    path: str
    method: str
    summary: str = ""
    description: str = ""
    parameters: List[APIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    operation_id: Optional[str] = None


@dataclass
class APIDefinition:
    """Complete API definition parsed from specification."""
    title: str
    description: str = ""
    version: str = "1.0.0"
    base_url: str = ""
    endpoints: List[APIEndpoint] = field(default_factory=list)
    schemas: Dict[str, Any] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)
    servers: List[Dict[str, str]] = field(default_factory=list)
    api_type: str = "openapi"


class APISpecificationParser:
    """Parse various API specifications into standardized format."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Microsoft-Agent-Framework-MCP-Parser/1.0'
        })
    
    def parse_openapi(self, spec_source: Union[str, Dict]) -> APIDefinition:
        """Parse OpenAPI/Swagger specification."""
        try:
            if isinstance(spec_source, str):
                if spec_source.startswith(('http://', 'https://')):
                    # URL source
                    response = self.session.get(spec_source)
                    response.raise_for_status()
                    
                    if spec_source.endswith('.yaml') or spec_source.endswith('.yml'):
                        spec = yaml.safe_load(response.text)
                    else:
                        spec = response.json()
                else:
                    # File path
                    with open(spec_source, 'r') as f:
                        if spec_source.endswith('.yaml') or spec_source.endswith('.yml'):
                            spec = yaml.safe_load(f)
                        else:
                            spec = json.load(f)
            else:
                spec = spec_source
            
            # Parse OpenAPI spec
            api_def = APIDefinition(
                title=spec.get('info', {}).get('title', 'Unknown API'),
                description=spec.get('info', {}).get('description', ''),
                version=spec.get('info', {}).get('version', '1.0.0'),
                api_type='openapi'
            )
            
            # Extract servers/base URL
            servers = spec.get('servers', [])
            if servers:
                api_def.base_url = servers[0].get('url', '')
                api_def.servers = servers
            
            # Extract security schemes
            api_def.security = spec.get('security', [])
            
            # Extract schemas
            api_def.schemas = spec.get('components', {}).get('schemas', {})
            
            # Parse endpoints
            paths = spec.get('paths', {})
            for path, path_item in paths.items():
                for method, operation in path_item.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                        endpoint = self._parse_openapi_operation(path, method.upper(), operation)
                        api_def.endpoints.append(endpoint)
            
            logger.info(f"Parsed OpenAPI spec: {api_def.title} with {len(api_def.endpoints)} endpoints")
            return api_def
            
        except Exception as e:
            logger.error(f"Error parsing OpenAPI spec: {e}")
            raise ValueError(f"Failed to parse OpenAPI specification: {e}")
    
    def _parse_openapi_operation(self, path: str, method: str, operation: Dict) -> APIEndpoint:
        """Parse individual OpenAPI operation."""
        endpoint = APIEndpoint(
            path=path,
            method=method,
            summary=operation.get('summary', ''),
            description=operation.get('description', ''),
            operation_id=operation.get('operationId'),
            tags=operation.get('tags', [])
        )
        
        # Parse parameters
        parameters = operation.get('parameters', [])
        for param in parameters:
            api_param = APIParameter(
                name=param.get('name', ''),
                type=param.get('schema', {}).get('type', 'string'),
                description=param.get('description', ''),
                required=param.get('required', False)
            )
            
            # Handle enum values
            if 'enum' in param.get('schema', {}):
                api_param.enum_values = param['schema']['enum']
            
            endpoint.parameters.append(api_param)
        
        # Parse request body
        if 'requestBody' in operation:
            endpoint.request_body = operation['requestBody']
        
        # Parse responses
        endpoint.responses = operation.get('responses', {})
        
        return endpoint
    
    def parse_graphql(self, schema_source: Union[str, Dict]) -> APIDefinition:
        """Parse GraphQL schema into API definition."""
        try:
            if isinstance(schema_source, str):
                if schema_source.startswith(('http://', 'https://')):
                    # GraphQL introspection query
                    introspection_query = """
                    query IntrospectionQuery {
                        __schema {
                            queryType { name }
                            mutationType { name }
                            subscriptionType { name }
                            types {
                                ...FullType
                            }
                        }
                    }
                    
                    fragment FullType on __Type {
                        kind
                        name
                        description
                        fields(includeDeprecated: true) {
                            name
                            description
                            args {
                                ...InputValue
                            }
                            type {
                                ...TypeRef
                            }
                        }
                    }
                    
                    fragment InputValue on __InputValue {
                        name
                        description
                        type { ...TypeRef }
                        defaultValue
                    }
                    
                    fragment TypeRef on __Type {
                        kind
                        name
                        ofType {
                            kind
                            name
                            ofType {
                                kind
                                name
                                ofType {
                                    kind
                                    name
                                }
                            }
                        }
                    }
                    """
                    
                    response = self.session.post(
                        schema_source,
                        json={'query': introspection_query},
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()
                    schema_data = response.json()
                else:
                    # File path - assume it's a GraphQL schema file
                    with open(schema_source, 'r') as f:
                        schema_text = f.read()
                    # For now, create a basic definition
                    # TODO: Implement proper GraphQL schema parsing
                    schema_data = {'data': {'__schema': {}}}
            else:
                schema_data = schema_source
            
            # Create API definition from GraphQL schema
            api_def = APIDefinition(
                title="GraphQL API",
                description="GraphQL API converted to MCP",
                api_type='graphql'
            )
            
            # Parse GraphQL types into endpoints
            schema = schema_data.get('data', {}).get('__schema', {})
            
            # Add query endpoint
            if schema.get('queryType'):
                query_endpoint = APIEndpoint(
                    path="/graphql",
                    method="POST",
                    summary="GraphQL Query",
                    description="Execute GraphQL queries",
                    parameters=[
                        APIParameter(
                            name="query",
                            type="string",
                            description="GraphQL query string",
                            required=True
                        ),
                        APIParameter(
                            name="variables",
                            type="object",
                            description="Query variables",
                            required=False
                        )
                    ]
                )
                api_def.endpoints.append(query_endpoint)
            
            # Add mutation endpoint if mutations are supported
            if schema.get('mutationType'):
                mutation_endpoint = APIEndpoint(
                    path="/graphql",
                    method="POST", 
                    summary="GraphQL Mutation",
                    description="Execute GraphQL mutations",
                    parameters=[
                        APIParameter(
                            name="query",
                            type="string", 
                            description="GraphQL mutation string",
                            required=True
                        ),
                        APIParameter(
                            name="variables",
                            type="object",
                            description="Mutation variables", 
                            required=False
                        )
                    ]
                )
                api_def.endpoints.append(mutation_endpoint)
            
            logger.info(f"Parsed GraphQL schema with {len(api_def.endpoints)} endpoints")
            return api_def
            
        except Exception as e:
            logger.error(f"Error parsing GraphQL schema: {e}")
            raise ValueError(f"Failed to parse GraphQL schema: {e}")
    
    def parse_rest_discovery(self, base_url: str) -> APIDefinition:
        """Discover REST API endpoints through common patterns."""
        try:
            api_def = APIDefinition(
                title=f"REST API at {base_url}",
                description=f"Auto-discovered REST API endpoints",
                base_url=base_url,
                api_type='rest'
            )
            
            # Common API discovery patterns
            discovery_paths = [
                '/.well-known/api',
                '/api/docs',
                '/swagger.json',
                '/openapi.json',
                '/api/v1',
                '/api',
                '/health',
                '/status'
            ]
            
            discovered_endpoints = []
            
            for path in discovery_paths:
                try:
                    url = urljoin(base_url, path)
                    response = self.session.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        endpoint = APIEndpoint(
                            path=path,
                            method="GET",
                            summary=f"Discovered endpoint: {path}",
                            description=f"Auto-discovered endpoint at {url}"
                        )
                        discovered_endpoints.append(endpoint)
                        
                        # Try to parse as OpenAPI if it looks like a spec
                        if path.endswith(('.json', '/swagger.json', '/openapi.json')):
                            try:
                                spec = response.json()
                                if 'openapi' in spec or 'swagger' in spec:
                                    return self.parse_openapi(spec)
                            except:
                                pass
                                
                except requests.RequestException:
                    continue
            
            api_def.endpoints = discovered_endpoints
            
            logger.info(f"Discovered {len(discovered_endpoints)} REST endpoints")
            return api_def
            
        except Exception as e:
            logger.error(f"Error during REST discovery: {e}")
            raise ValueError(f"Failed to discover REST API: {e}")
    
    def parse_postman_collection(self, collection_source: Union[str, Dict]) -> APIDefinition:
        """Parse Postman collection into API definition."""
        try:
            if isinstance(collection_source, str):
                if collection_source.startswith(('http://', 'https://')):
                    response = self.session.get(collection_source)
                    response.raise_for_status()
                    collection = response.json()
                else:
                    with open(collection_source, 'r') as f:
                        collection = json.load(f)
            else:
                collection = collection_source
            
            info = collection.get('info', {})
            api_def = APIDefinition(
                title=info.get('name', 'Postman Collection'),
                description=info.get('description', ''),
                version=info.get('version', '1.0.0'),
                api_type='postman'
            )
            
            # Parse collection items
            items = collection.get('item', [])
            for item in items:
                endpoints = self._parse_postman_item(item)
                api_def.endpoints.extend(endpoints)
            
            logger.info(f"Parsed Postman collection: {api_def.title} with {len(api_def.endpoints)} endpoints")
            return api_def
            
        except Exception as e:
            logger.error(f"Error parsing Postman collection: {e}")
            raise ValueError(f"Failed to parse Postman collection: {e}")
    
    def _parse_postman_item(self, item: Dict) -> List[APIEndpoint]:
        """Parse individual Postman collection item."""
        endpoints = []
        
        if 'request' in item:
            # This is a request item
            request = item['request']
            
            endpoint = APIEndpoint(
                path=self._extract_postman_path(request.get('url', {})),
                method=request.get('method', 'GET').upper(),
                summary=item.get('name', ''),
                description=item.get('description', '')
            )
            
            # Parse query parameters
            url_info = request.get('url', {})
            if isinstance(url_info, dict) and 'query' in url_info:
                for query_param in url_info.get('query', []):
                    param = APIParameter(
                        name=query_param.get('key', ''),
                        type='string',
                        description=query_param.get('description', ''),
                        required=not query_param.get('disabled', False)
                    )
                    endpoint.parameters.append(param)
            
            # Parse headers as parameters
            headers = request.get('header', [])
            for header in headers:
                if not header.get('disabled', False):
                    param = APIParameter(
                        name=f"header_{header.get('key', '')}",
                        type='string',
                        description=f"Header: {header.get('description', '')}",
                        required=False
                    )
                    endpoint.parameters.append(param)
            
            endpoints.append(endpoint)
            
        elif 'item' in item:
            # This is a folder, recurse into items
            for sub_item in item['item']:
                endpoints.extend(self._parse_postman_item(sub_item))
        
        return endpoints
    
    def _extract_postman_path(self, url_info: Union[str, Dict]) -> str:
        """Extract path from Postman URL info."""
        if isinstance(url_info, str):
            parsed = urlparse(url_info)
            return parsed.path or '/'
        elif isinstance(url_info, dict):
            path_segments = url_info.get('path', [])
            if isinstance(path_segments, list):
                return '/' + '/'.join(str(segment) for segment in path_segments)
            else:
                return str(path_segments) if path_segments else '/'
        
        return '/'
    
    def validate_api_definition(self, api_def: APIDefinition) -> bool:
        """Validate that API definition is complete and usable."""
        if not api_def.title:
            return False
        
        if not api_def.endpoints:
            return False
        
        for endpoint in api_def.endpoints:
            if not endpoint.path or not endpoint.method:
                return False
        
        return True
