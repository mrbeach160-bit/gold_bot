"""
Model Serving Infrastructure

FastAPI-based REST API and WebSocket service for real-time model predictions.
Provides high-performance endpoints for production trading systems with
robust error handling, rate limiting, and connection management.
"""

import asyncio
import json
import logging
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# Import enhanced utilities
try:
    from utils.logging_system import get_logger, get_performance_logger, log_function_call
    from utils.dependency_manager import is_available
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False
    # Fallback to basic logging
    import logging as basic_logging
    logging = basic_logging.getLogger(__name__)

# Setup enhanced logging
if ENHANCED_UTILS_AVAILABLE:
    logger = get_logger("model_server")
    perf_logger = get_performance_logger("server_performance")
else:
    logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_BURST_SIZE = 10
WEBSOCKET_MAX_CONNECTIONS = 100
CONNECTION_TIMEOUT_SECONDS = 300

# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    symbol: str = Field(..., description="Trading symbol (e.g., XAUUSD)")
    timeframe: str = Field(default="5m", description="Timeframe (e.g., 1m, 5m, 1h)")
    data: Dict[str, Any] = Field(..., description="Market data (OHLCV)")
    features: Optional[Dict[str, Any]] = Field(None, description="Pre-computed features")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "XAUUSD",
                "timeframe": "5m",
                "data": {
                    "open": 2000.0,
                    "high": 2005.0,
                    "low": 1998.0,
                    "close": 2003.0,
                    "volume": 1500
                }
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: Dict[str, Any] = Field(..., description="Model prediction")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")

class HealthResponse(BaseModel):
    """Enhanced health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of loaded models")
    uptime: float = Field(..., description="Service uptime in seconds")
    model_status: str = Field(default="unknown", description="Model manager status")
    dependency_status: str = Field(default="unknown", description="Dependency availability")
    websocket_connections: int = Field(default=0, description="Active WebSocket connections")

class RateLimiter:
    """Token bucket rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = RATE_LIMIT_REQUESTS_PER_MINUTE, 
                 burst_size: int = RATE_LIMIT_BURST_SIZE):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.clients = defaultdict(lambda: {
            'tokens': burst_size,
            'last_update': time.time()
        })
        self.cleanup_interval = 300  # Clean up old clients every 5 minutes
        self.last_cleanup = time.time()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()
        client = self.clients[client_id]
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - client['last_update']
        tokens_to_add = time_elapsed * (self.requests_per_minute / 60.0)
        
        # Update client tokens
        client['tokens'] = min(self.burst_size, client['tokens'] + tokens_to_add)
        client['last_update'] = now
        
        # Check if request is allowed
        if client['tokens'] >= 1:
            client['tokens'] -= 1
            return True
        return False
    
    def cleanup_old_clients(self):
        """Remove old client records to prevent memory leaks."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = now - 3600  # Remove clients inactive for 1 hour
        old_clients = [
            client_id for client_id, data in self.clients.items()
            if data['last_update'] < cutoff_time
        ]
        
        for client_id in old_clients:
            del self.clients[client_id]
        
        self.last_cleanup = now
        if old_clients:
            logger.info(f"Cleaned up {len(old_clients)} old rate limit records")


class EnhancedWebSocketManager:
    """Enhanced WebSocket manager with connection pooling and error handling."""
    
    def __init__(self, max_connections: int = WEBSOCKET_MAX_CONNECTIONS):
        self.max_connections = max_connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict] = {}
        self.connection_timestamps: Dict[str, float] = {}
        self.subscription_topics: Dict[str, Set[str]] = defaultdict(set)
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._connection_counter = 0
        
        # Use weak references to avoid memory leaks
        self._websocket_refs = weakref.WeakSet()
    
    async def connect(self, websocket: WebSocket, client_info: Dict = None) -> str:
        """Accept new WebSocket connection with enhanced error handling."""
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            raise HTTPException(status_code=503, detail="Maximum connections reached")
        
        try:
            await websocket.accept()
            
            # Generate unique connection ID
            connection_id = f"ws_{self._connection_counter}_{int(time.time())}"
            self._connection_counter += 1
            
            # Store connection info
            self.active_connections[connection_id] = websocket
            self.connection_info[connection_id] = client_info or {}
            self.connection_timestamps[connection_id] = time.time()
            self._websocket_refs.add(websocket)
            
            logger.info(f"WebSocket connected: {connection_id}. Total: {len(self.active_connections)}")
            
            # Send welcome message
            await self.send_personal_message(
                json.dumps({
                    'type': 'connected',
                    'connection_id': connection_id,
                    'timestamp': datetime.now().isoformat()
                }),
                connection_id
            )
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            try:
                await websocket.close(code=1011, reason="Connection error")
            except:
                pass
            raise
    
    def disconnect(self, connection_id: str, reason: str = "Normal closure"):
        """Remove WebSocket connection with cleanup."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            
            # Clean up all references
            del self.active_connections[connection_id]
            self.connection_info.pop(connection_id, None)
            self.connection_timestamps.pop(connection_id, None)
            self.message_queues.pop(connection_id, None)
            
            # Clean up subscriptions
            for topic_set in self.subscription_topics.values():
                topic_set.discard(connection_id)
            
            logger.info(f"WebSocket disconnected: {connection_id} ({reason}). Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, connection_id: str) -> bool:
        """Send message to specific WebSocket with error handling."""
        if connection_id not in self.active_connections:
            return False
        
        websocket = self.active_connections[connection_id]
        try:
            await websocket.send_text(message)
            return True
        except WebSocketDisconnect:
            self.disconnect(connection_id, "Client disconnected")
            return False
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            self.disconnect(connection_id, "Send error")
            return False
    
    async def broadcast(self, message: str, topic: str = None) -> int:
        """Broadcast message with error handling and optional topic filtering."""
        if topic:
            target_connections = self.subscription_topics.get(topic, set())
        else:
            target_connections = set(self.active_connections.keys())
        
        successful_sends = 0
        failed_connections = []
        
        for connection_id in list(target_connections):
            if await self.send_personal_message(message, connection_id):
                successful_sends += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            if connection_id in target_connections:
                target_connections.remove(connection_id)
        
        return successful_sends
    
    async def broadcast_prediction(self, prediction: Dict[str, Any], topic: str = "predictions"):
        """Broadcast prediction to subscribers."""
        message = json.dumps({
            "type": "prediction",
            "data": prediction,
            "timestamp": datetime.now().isoformat(),
            "topic": topic
        })
        
        sent_count = await self.broadcast(message, topic)
        if ENHANCED_UTILS_AVAILABLE:
            perf_logger.log_metric("websocket_broadcast_count", sent_count)
    
    def subscribe_to_topic(self, connection_id: str, topic: str):
        """Subscribe connection to a topic."""
        if connection_id in self.active_connections:
            self.subscription_topics[topic].add(connection_id)
            logger.debug(f"Connection {connection_id} subscribed to {topic}")
    
    def unsubscribe_from_topic(self, connection_id: str, topic: str):
        """Unsubscribe connection from a topic."""
        if topic in self.subscription_topics:
            self.subscription_topics[topic].discard(connection_id)
            logger.debug(f"Connection {connection_id} unsubscribed from {topic}")
    
    async def cleanup_stale_connections(self):
        """Clean up connections that have been inactive too long."""
        now = time.time()
        stale_connections = []
        
        for connection_id, timestamp in self.connection_timestamps.items():
            if now - timestamp > CONNECTION_TIMEOUT_SECONDS:
                stale_connections.append(connection_id)
        
        for connection_id in stale_connections:
            websocket = self.active_connections.get(connection_id)
            if websocket:
                try:
                    await websocket.close(code=1000, reason="Connection timeout")
                except:
                    pass
                self.disconnect(connection_id, "Timeout")
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale WebSocket connections")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        return {
            'active_connections': len(self.active_connections),
            'max_connections': self.max_connections,
            'subscription_topics': {
                topic: len(connections) for topic, connections in self.subscription_topics.items()
            },
            'connection_ids': list(self.active_connections.keys()),
            'total_created': self._connection_counter
        }

class ModelServer:
    """Main model serving infrastructure with enhanced error handling."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Gold Bot Model Server",
            description="Production-ready model serving API for gold trading predictions",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize enhanced components
        self.websocket_manager = EnhancedWebSocketManager()
        self.rate_limiter = RateLimiter()
        self.start_time = time.time()
        self.model_manager = None
        self.data_streamer = None
        
        # Enhanced metrics tracking
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'rate_limited_requests': 0,
            'avg_response_time': 0.0,
            'websocket_connections': 0,
            'websocket_messages_sent': 0,
            'errors_by_type': defaultdict(int),
            'model_prediction_times': deque(maxlen=1000)
        }
        
        # Load models with error handling
        self._load_models()
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("ModelServer initialized successfully")
    
    def _load_models(self):
        """Load trading models with enhanced error handling."""
        try:
            from models.manager import ModelManager
            self.model_manager = ModelManager(symbol="XAUUSD", timeframe="5m")
            logger.info("Model manager loaded successfully")
            
            # Log available models
            if hasattr(self.model_manager, 'models'):
                model_count = len(self.model_manager.models)
                logger.info(f"Loaded {model_count} trading models")
            
        except Exception as e:
            logger.error(f"Could not load model manager: {e}")
            self.model_manager = None
            self.metrics['errors_by_type']['model_loading'] += 1
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        @self.app.on_event("startup")
        async def startup_tasks():
            # Schedule periodic cleanup
            asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of stale connections and old data."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up stale WebSocket connections
                await self.websocket_manager.cleanup_stale_connections()
                
                # Clean up rate limiter
                self.rate_limiter.cleanup_old_clients()
                
                # Update metrics
                self.metrics['websocket_connections'] = len(self.websocket_manager.active_connections)
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client ID for rate limiting."""
        # Use IP address as client ID, could be enhanced with API keys
        return request.client.host if request.client else "unknown"
    
    def _setup_routes(self):
        """Setup FastAPI routes with enhanced error handling."""
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            """Rate limiting middleware."""
            client_id = self._get_client_id(request)
            
            # Skip rate limiting for health checks
            if request.url.path == "/health":
                response = await call_next(request)
                return response
            
            if not self.rate_limiter.is_allowed(client_id):
                self.metrics['rate_limited_requests'] += 1
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            response = await call_next(request)
            return response
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Enhanced health check endpoint."""
            try:
                # Check model manager status
                models_loaded = 0
                model_status = "unknown"
                
                if self.model_manager:
                    if hasattr(self.model_manager, 'models'):
                        models_loaded = len(self.model_manager.models)
                        model_status = "loaded"
                    else:
                        model_status = "manager_available"
                else:
                    model_status = "unavailable"
                
                # Check dependencies
                dependency_status = "unknown"
                if ENHANCED_UTILS_AVAILABLE and is_available('sklearn'):
                    dependency_status = "available"
                else:
                    dependency_status = "limited"
                
                return HealthResponse(
                    status="healthy",
                    timestamp=datetime.now(),
                    version="2.0.0",
                    models_loaded=models_loaded,
                    uptime=time.time() - self.start_time,
                    model_status=model_status,
                    dependency_status=dependency_status,
                    websocket_connections=len(self.websocket_manager.active_connections)
                )
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail="Health check failed")
        
        @self.app.post("/predict", response_model=PredictionResponse)
        @log_function_call("model_server") if ENHANCED_UTILS_AVAILABLE else lambda x: x
        async def predict(request: PredictionRequest, http_request: Request):
            """Generate prediction with enhanced error handling."""
            start_time = time.time()
            client_id = self._get_client_id(http_request)
            
            self.metrics['total_requests'] += 1
            
            try:
                # Validate request data
                if not request.data or 'close' not in request.data:
                    raise HTTPException(status_code=400, detail="Invalid market data")
                
                # Process the prediction
                prediction_result = await self._generate_prediction(request)
                
                processing_time = time.time() - start_time
                self.metrics['successful_predictions'] += 1
                self.metrics['model_prediction_times'].append(processing_time)
                
                # Update average response time
                if self.metrics['successful_predictions'] > 0:
                    self.metrics['avg_response_time'] = (
                        (self.metrics['avg_response_time'] * (self.metrics['successful_predictions'] - 1) +
                         processing_time) / self.metrics['successful_predictions']
                    )
                
                response = PredictionResponse(
                    prediction=prediction_result['prediction'],
                    confidence=prediction_result['confidence'],
                    processing_time=processing_time,
                    timestamp=datetime.now(),
                    model_info={
                        'ensemble_models': prediction_result.get('models_used', 1),
                        'features_count': prediction_result.get('features_count', 0),
                        'version': '2.0.0',
                        'client_id': client_id
                    }
                )
                
                # Broadcast to WebSocket subscribers
                await self.websocket_manager.broadcast_prediction(
                    prediction_result, 
                    topic=f"predictions_{request.symbol}"
                )
                
                # Log performance metrics
                if ENHANCED_UTILS_AVAILABLE:
                    perf_logger.log_metric("prediction_processing_time", processing_time)
                    perf_logger.log_metric("prediction_confidence", prediction_result['confidence'])
                
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                self.metrics['failed_predictions'] += 1
                self.metrics['errors_by_type']['prediction_error'] += 1
                
                logger.error(f"Prediction error for client {client_id}: {e}")
                
                # Return fallback prediction instead of failing completely
                fallback_result = self._get_fallback_prediction(request)
                
                return PredictionResponse(
                    prediction=fallback_result['prediction'],
                    confidence=0.0,  # Low confidence for fallback
                    processing_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    model_info={
                        'ensemble_models': 0,
                        'features_count': 0,
                        'version': '2.0.0',
                        'fallback': True,
                        'error': str(e)
                    }
                )
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get comprehensive service metrics."""
            current_metrics = self.metrics.copy()
            current_metrics.update({
                'uptime': time.time() - self.start_time,
                'models_loaded': 1 if self.model_manager else 0,
                'websocket_stats': self.websocket_manager.get_connection_stats(),
                'rate_limiter_stats': {
                    'active_clients': len(self.rate_limiter.clients),
                    'requests_per_minute_limit': self.rate_limiter.requests_per_minute
                }
            })
            
            # Add performance statistics
            if self.metrics['model_prediction_times']:
                times = list(self.metrics['model_prediction_times'])
                current_metrics['performance_stats'] = {
                    'min_response_time': min(times),
                    'max_response_time': max(times),
                    'median_response_time': sorted(times)[len(times)//2]
                }
            
            return current_metrics
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Enhanced WebSocket endpoint with better error handling."""
            connection_id = None
            try:
                # Extract client info
                client_info = {
                    'user_agent': websocket.headers.get('user-agent', 'unknown'),
                    'origin': websocket.headers.get('origin', 'unknown'),
                    'connect_time': datetime.now().isoformat()
                }
                
                connection_id = await self.websocket_manager.connect(websocket, client_info)
                
                # Send connection confirmation
                await self.websocket_manager.send_personal_message(
                    json.dumps({
                        'type': 'connection_established',
                        'connection_id': connection_id,
                        'available_topics': ['predictions_XAUUSD', 'system_status']
                    }),
                    connection_id
                )
                
                # Message handling loop
                while True:
                    try:
                        # Set a timeout for receiving messages
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        
                        # Update connection timestamp
                        self.websocket_manager.connection_timestamps[connection_id] = time.time()
                        
                        # Handle different message types
                        try:
                            message = json.loads(data)
                            await self._handle_websocket_message(connection_id, message)
                            
                        except json.JSONDecodeError:
                            await self.websocket_manager.send_personal_message(
                                json.dumps({
                                    'type': 'error',
                                    'message': 'Invalid JSON format',
                                    'timestamp': datetime.now().isoformat()
                                }),
                                connection_id
                            )
                    
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await self.websocket_manager.send_personal_message(
                            json.dumps({
                                'type': 'ping',
                                'timestamp': datetime.now().isoformat()
                            }),
                            connection_id
                        )
                        
            except WebSocketDisconnect:
                if connection_id:
                    self.websocket_manager.disconnect(connection_id, "Client disconnected")
                    
            except Exception as e:
                logger.error(f"WebSocket error for connection {connection_id}: {e}")
                if connection_id:
                    self.websocket_manager.disconnect(connection_id, f"Error: {str(e)}")
                try:
                    await websocket.close(code=1011, reason="Internal error")
                except:
                    pass
        
        @self.app.post("/start_stream")
        async def start_real_time_stream(background_tasks: BackgroundTasks):
            """Start real-time data streaming with enhanced error handling."""
            if self.data_streamer and getattr(self.data_streamer, 'running', False):
                return {"message": "Stream already running", "status": "active"}
            
            try:
                from .data_streamer import RealTimeDataStreamer
                self.data_streamer = RealTimeDataStreamer()
                
                # Subscribe to stream updates
                self.data_streamer.subscribe(self._on_stream_update)
                
                # Start streaming in background
                background_tasks.add_task(self.data_streamer.stream_predictions)
                
                logger.info("Real-time data stream started")
                return {"message": "Real-time stream started", "status": "started"}
                
            except ImportError as e:
                logger.error(f"Data streamer not available: {e}")
                raise HTTPException(status_code=501, detail="Data streaming not implemented")
            except Exception as e:
                logger.error(f"Error starting stream: {e}")
                self.metrics['errors_by_type']['stream_start'] += 1
                raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")
        
        @self.app.post("/stop_stream")
        async def stop_real_time_stream():
            """Stop real-time data streaming."""
            if self.data_streamer:
                try:
                    self.data_streamer.stop()
                    logger.info("Real-time data stream stopped")
                    return {"message": "Stream stopped", "status": "stopped"}
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                    return {"message": f"Error stopping stream: {str(e)}", "status": "error"}
            return {"message": "No active stream", "status": "none"}
    
    async def _handle_websocket_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = message.get('type')
        
        try:
            if message_type == 'ping':
                await self.websocket_manager.send_personal_message(
                    json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }),
                    connection_id
                )
                
            elif message_type == 'subscribe':
                topic = message.get('topic', 'predictions_XAUUSD')
                self.websocket_manager.subscribe_to_topic(connection_id, topic)
                await self.websocket_manager.send_personal_message(
                    json.dumps({
                        'type': 'subscribed',
                        'topic': topic,
                        'timestamp': datetime.now().isoformat()
                    }),
                    connection_id
                )
                
            elif message_type == 'unsubscribe':
                topic = message.get('topic', 'predictions_XAUUSD')
                self.websocket_manager.unsubscribe_from_topic(connection_id, topic)
                await self.websocket_manager.send_personal_message(
                    json.dumps({
                        'type': 'unsubscribed',
                        'topic': topic,
                        'timestamp': datetime.now().isoformat()
                    }),
                    connection_id
                )
                
            elif message_type == 'status':
                stats = self.websocket_manager.get_connection_stats()
                await self.websocket_manager.send_personal_message(
                    json.dumps({
                        'type': 'status_response',
                        'data': stats,
                        'timestamp': datetime.now().isoformat()
                    }),
                    connection_id
                )
                
            else:
                await self.websocket_manager.send_personal_message(
                    json.dumps({
                        'type': 'error',
                        'message': f'Unknown message type: {message_type}',
                        'timestamp': datetime.now().isoformat()
                    }),
                    connection_id
                )
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.websocket_manager.send_personal_message(
                json.dumps({
                    'type': 'error',
                    'message': 'Internal server error',
                    'timestamp': datetime.now().isoformat()
                }),
                connection_id
            )
    
    def _get_fallback_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate fallback prediction when models fail."""
        data = request.data
        
        # Simple momentum-based prediction with enhanced logic
        try:
            price_change = (data['close'] - data['open']) / data['open']
            volume_factor = data.get('volume', 1000) / 1000  # Normalize volume
            
            # More sophisticated fallback logic
            if price_change > 0.002 and volume_factor > 1.5:  # Strong bullish with volume
                direction = 'BUY'
                confidence = 0.6
            elif price_change < -0.002 and volume_factor > 1.5:  # Strong bearish with volume
                direction = 'SELL'
                confidence = 0.6
            elif abs(price_change) > 0.001:  # Moderate movement
                direction = 'BUY' if price_change > 0 else 'SELL'
                confidence = 0.4
            else:  # Weak movement
                direction = 'HOLD'
                confidence = 0.3
            
            return {
                'prediction': {
                    'direction': direction,
                    'confidence': confidence,
                    'price': data['close'],
                    'symbol': request.symbol,
                    'fallback_reason': 'model_unavailable'
                },
                'confidence': confidence,
                'models_used': 0,
                'features_count': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return {
                'prediction': {
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'price': data.get('close', 0),
                    'symbol': request.symbol,
                    'fallback_reason': 'fallback_error'
                },
                'confidence': 0.0,
                'models_used': 0,
                'features_count': 0
            }
    
    async def _generate_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate prediction using loaded models with enhanced error handling."""
        try:
            if self.model_manager:
                # Use real model manager
                df = pd.DataFrame([request.data])
                
                # Validate dataframe
                if df.empty or 'close' not in df.columns:
                    raise ValueError("Invalid market data format")
                
                # Get ensemble prediction with timeout
                prediction = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.model_manager.get_ensemble_prediction,
                        df, 
                        'dynamic_ensemble'
                    ),
                    timeout=5.0  # 5 second timeout
                )
                
                return {
                    'prediction': prediction,
                    'confidence': prediction.get('confidence', 0.5),
                    'models_used': len(getattr(self.model_manager, 'models', [])),
                    'features_count': len(request.features) if request.features else len(request.data)
                }
            else:
                # Use fallback prediction
                return self._get_fallback_prediction(request)
                
        except asyncio.TimeoutError:
            logger.warning("Model prediction timeout, using fallback")
            self.metrics['errors_by_type']['prediction_timeout'] += 1
            return self._get_fallback_prediction(request)
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            self.metrics['errors_by_type']['model_error'] += 1
            return self._get_fallback_prediction(request)
    
    def _mock_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate mock prediction for demo purposes (deprecated - use _get_fallback_prediction)."""
        return self._get_fallback_prediction(request)
    
    async def _on_stream_update(self, prediction: Dict[str, Any]):
        """Handle updates from data streamer with error handling."""
        try:
            # Broadcast prediction via WebSocket
            await self.websocket_manager.broadcast_prediction(prediction)
            self.metrics['websocket_messages_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error handling stream update: {e}")
            self.metrics['errors_by_type']['stream_update'] += 1
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI server with enhanced configuration."""
        logger.info(f"Starting Enhanced Model Server on {host}:{port}")
        
        # Enhanced uvicorn configuration for production
        config = {
            'host': host,
            'port': port,
            'log_level': 'info',
            'access_log': True,
            'use_colors': True,
            'reload': False,  # Disable reload in production
            'workers': 1,  # Single worker for WebSocket support
            **kwargs
        }
        
        uvicorn.run(self.app, **config)

# Global server instance
server = ModelServer()

# FastAPI app for external access
app = server.app

# Example usage
if __name__ == "__main__":
    # Run with: python -m production.model_server
    server.run(host="0.0.0.0", port=8000, reload=False)