"""
Model Serving Infrastructure

FastAPI-based REST API and WebSocket service for real-time model predictions.
Provides high-performance endpoints for production trading systems.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

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
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of loaded models")
    uptime: float = Field(..., description="Service uptime in seconds")

class WebSocketManager:
    """Manages WebSocket connections for real-time predictions"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_info: Dict = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = client_info or {}
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_info.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_prediction(self, prediction: Dict[str, Any]):
        """Broadcast prediction to all subscribers"""
        message = json.dumps({
            "type": "prediction",
            "data": prediction,
            "timestamp": datetime.now().isoformat()
        })
        await self.broadcast(message)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

class ModelServer:
    """Main model serving infrastructure"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Gold Bot Model Server",
            description="Real-time model serving API for gold trading predictions",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.websocket_manager = WebSocketManager()
        self.start_time = time.time()
        self.request_count = 0
        self.model_manager = None
        self.data_streamer = None
        
        # Load models
        self._load_models()
        
        # Setup routes
        self._setup_routes()
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_response_time': 0.0,
            'websocket_connections': 0
        }
    
    def _load_models(self):
        """Load trading models"""
        try:
            from models.manager import ModelManager
            self.model_manager = ModelManager(symbol="XAUUSD", timeframe="5m")
            logger.info("Model manager loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model manager: {e}")
            self.model_manager = None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="1.0.0",
                models_loaded=1 if self.model_manager else 0,
                uptime=time.time() - self.start_time
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Generate prediction from market data"""
            start_time = time.time()
            self.metrics['total_requests'] += 1
            
            try:
                # Process the prediction
                prediction_result = await self._generate_prediction(request)
                
                processing_time = time.time() - start_time
                self.metrics['successful_predictions'] += 1
                
                # Update average response time
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
                        'version': '1.0.0'
                    }
                )
                
                # Broadcast to WebSocket subscribers
                await self.websocket_manager.broadcast_prediction(prediction_result)
                
                return response
                
            except Exception as e:
                self.metrics['failed_predictions'] += 1
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics"""
            self.metrics['websocket_connections'] = self.websocket_manager.get_connection_count()
            return {
                **self.metrics,
                'uptime': time.time() - self.start_time,
                'models_loaded': 1 if self.model_manager else 0
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time predictions"""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle client messages
                    data = await websocket.receive_text()
                    
                    # Handle different message types
                    try:
                        message = json.loads(data)
                        message_type = message.get('type')
                        
                        if message_type == 'ping':
                            await websocket.send_text(json.dumps({
                                'type': 'pong',
                                'timestamp': datetime.now().isoformat()
                            }))
                        elif message_type == 'subscribe':
                            # Handle subscription to specific symbols/timeframes
                            await websocket.send_text(json.dumps({
                                'type': 'subscribed',
                                'symbol': message.get('symbol', 'XAUUSD'),
                                'timestamp': datetime.now().isoformat()
                            }))
                        
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON format'
                        }))
                        
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
        
        @self.app.post("/start_stream")
        async def start_real_time_stream(background_tasks: BackgroundTasks):
            """Start real-time data streaming"""
            if self.data_streamer and self.data_streamer.running:
                return {"message": "Stream already running"}
            
            try:
                from .data_streamer import RealTimeDataStreamer
                self.data_streamer = RealTimeDataStreamer()
                
                # Subscribe to stream updates
                self.data_streamer.subscribe(self._on_stream_update)
                
                # Start streaming in background
                background_tasks.add_task(self.data_streamer.stream_predictions)
                
                return {"message": "Real-time stream started"}
                
            except Exception as e:
                logger.error(f"Error starting stream: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")
        
        @self.app.post("/stop_stream")
        async def stop_real_time_stream():
            """Stop real-time data streaming"""
            if self.data_streamer:
                self.data_streamer.stop()
                return {"message": "Stream stopped"}
            return {"message": "No active stream"}
    
    async def _generate_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate prediction using loaded models"""
        try:
            if self.model_manager:
                # Use real model manager
                df = pd.DataFrame([request.data])
                
                # Get ensemble prediction
                prediction = self.model_manager.get_ensemble_prediction(
                    df, method='dynamic_ensemble'
                )
                
                return {
                    'prediction': prediction,
                    'confidence': prediction.get('confidence', 0.5),
                    'models_used': len(self.model_manager.models) if hasattr(self.model_manager, 'models') else 1,
                    'features_count': len(request.features) if request.features else len(request.data)
                }
            else:
                # Fallback mock prediction
                return self._mock_prediction(request)
                
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self._mock_prediction(request)
    
    def _mock_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Generate mock prediction for demo purposes"""
        data = request.data
        
        # Simple momentum-based prediction
        price_change = (data['close'] - data['open']) / data['open']
        
        if price_change > 0.001:
            direction = 'BUY'
            confidence = 0.7
        elif price_change < -0.001:
            direction = 'SELL'
            confidence = 0.7
        else:
            direction = 'HOLD'
            confidence = 0.5
        
        return {
            'prediction': {
                'direction': direction,
                'confidence': confidence,
                'price': data['close'],
                'symbol': request.symbol
            },
            'confidence': confidence,
            'models_used': 1,
            'features_count': len(data)
        }
    
    async def _on_stream_update(self, prediction: Dict[str, Any]):
        """Handle updates from data streamer"""
        try:
            # Broadcast prediction via WebSocket
            await self.websocket_manager.broadcast_prediction(prediction)
        except Exception as e:
            logger.error(f"Error handling stream update: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI server"""
        logger.info(f"Starting Model Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)

# Global server instance
server = ModelServer()

# FastAPI app for external access
app = server.app

# Example usage
if __name__ == "__main__":
    # Run with: python -m production.model_server
    server.run(host="0.0.0.0", port=8000, reload=False)