"""
Production Server Integration

Main production server that integrates all Phase 4 & 5 components:
- Real-time data streaming
- Model serving with FastAPI
- Risk management
- Monitoring and alerting
- Adaptive retraining
- Market microstructure analysis
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Production components
try:
    from production import (
        RealTimeDataStreamer, 
        ModelServer, 
        ProductionRiskManager, 
        MonitoringManager, 
        CacheManager
    )
    from optimization import (
        AdaptiveRetrainer, 
        MicrostructureAnalyzer
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Production components not available: {e}")
    COMPONENTS_AVAILABLE = False

class ProductionSystem:
    """Main production trading system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Core components
        self.data_streamer = None
        self.model_server = None
        self.risk_manager = None
        self.monitoring_manager = None
        self.cache_manager = None
        self.adaptive_retrainer = None
        self.microstructure_analyzer = None
        
        # System state
        self.running = False
        self.startup_time = None
        
        logger.info("Production System initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1
            },
            'trading': {
                'symbol': 'XAUUSD',
                'timeframe': '5m',
                'update_interval': 1.0,
                'enable_live_trading': False
            },
            'risk': {
                'max_risk_per_trade': 0.02,
                'max_portfolio_risk': 0.10,
                'max_drawdown': 0.15
            },
            'monitoring': {
                'performance_interval': 60,
                'system_interval': 30,
                'enable_alerts': True
            },
            'cache': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'default_ttl': 3600
            },
            'optimization': {
                'enable_adaptive_retraining': True,
                'enable_microstructure_analysis': True,
                'retrain_check_interval': 3600
            }
        }
    
    async def initialize(self):
        """Initialize all production components"""
        try:
            logger.info("Initializing production system...")
            
            if not COMPONENTS_AVAILABLE:
                raise RuntimeError("Production components not available")
            
            # Initialize cache manager
            self.cache_manager = CacheManager(
                redis_host=self.config['cache']['redis_host'],
                redis_port=self.config['cache']['redis_port'],
                default_ttl=self.config['cache']['default_ttl']
            )
            
            # Initialize risk manager
            from production.risk_manager import RiskLimits
            risk_limits = RiskLimits(
                max_risk_per_trade=self.config['risk']['max_risk_per_trade'],
                max_portfolio_risk=self.config['risk']['max_portfolio_risk'],
                max_drawdown=self.config['risk']['max_drawdown']
            )
            self.risk_manager = ProductionRiskManager(risk_limits)
            
            # Initialize monitoring
            self.monitoring_manager = MonitoringManager()
            self.monitoring_manager.add_alert_callback(self._handle_system_alert)
            
            # Initialize model server
            self.model_server = ModelServer()
            
            # Initialize data streamer
            self.data_streamer = RealTimeDataStreamer(
                symbol=self.config['trading']['symbol'],
                update_interval=self.config['trading']['update_interval']
            )
            
            # Subscribe data streamer to model server
            self.data_streamer.subscribe(self._on_data_stream_update)
            
            # Initialize optimization components
            if self.config['optimization']['enable_adaptive_retraining']:
                self.adaptive_retrainer = AdaptiveRetrainer()
                self.adaptive_retrainer.add_retrain_callback(self._on_retrain_event)
            
            if self.config['optimization']['enable_microstructure_analysis']:
                self.microstructure_analyzer = MicrostructureAnalyzer()
            
            logger.info("Production system initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    async def start(self):
        """Start all production services"""
        try:
            logger.info("Starting production system...")
            self.startup_time = datetime.now()
            self.running = True
            
            # Start monitoring
            self.monitoring_manager.start_all_monitoring(
                performance_interval=self.config['monitoring']['performance_interval'],
                system_interval=self.config['monitoring']['system_interval']
            )
            
            # Start adaptive retraining
            if self.adaptive_retrainer:
                self.adaptive_retrainer.start_continuous_optimization()
            
            # Start data streaming
            asyncio.create_task(self.data_streamer.stream_predictions())
            
            logger.info("Production system started successfully")
            
        except Exception as e:
            logger.error(f"Startup error: {e}")
            raise
    
    async def stop(self):
        """Stop all production services"""
        try:
            logger.info("Stopping production system...")
            self.running = False
            
            # Stop data streaming
            if self.data_streamer:
                self.data_streamer.stop()
            
            # Stop adaptive retraining
            if self.adaptive_retrainer:
                self.adaptive_retrainer.stop_optimization()
            
            # Stop monitoring
            if self.monitoring_manager:
                self.monitoring_manager.stop_all_monitoring()
            
            logger.info("Production system stopped")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    async def _on_data_stream_update(self, prediction: Dict[str, Any]):
        """Handle real-time data stream updates"""
        try:
            # Record prediction for monitoring
            if self.monitoring_manager:
                self.monitoring_manager.performance_monitor.record_prediction(
                    prediction,
                    latency_ms=prediction.get('processing_time', 0) * 1000
                )
            
            # Record for adaptive retraining
            if self.adaptive_retrainer:
                features = prediction.get('features', {})
                if features:
                    self.adaptive_retrainer.record_prediction(prediction, features=features)
            
            # Microstructure analysis
            if self.microstructure_analyzer and prediction.get('tick_data'):
                microstructure_analysis = self.microstructure_analyzer.analyze_order_flow(
                    prediction['tick_data']
                )
                prediction['microstructure'] = microstructure_analysis
            
            # Risk assessment
            if self.risk_manager and prediction['direction'] != 'HOLD':
                risk_evaluation = self.risk_manager.evaluate_trade(prediction)
                prediction['risk_assessment'] = risk_evaluation
                
                # Log significant risk decisions
                if risk_evaluation['action'] in ['SKIP', 'STOP_TRADING']:
                    logger.warning(f"Risk management action: {risk_evaluation}")
            
        except Exception as e:
            logger.error(f"Error processing data stream update: {e}")
    
    def _handle_system_alert(self, alert):
        """Handle system alerts"""
        try:
            # Log alert
            logger.log(
                getattr(logging, alert.level), 
                f"[{alert.category}] {alert.message}"
            )
            
            # Take automated actions for critical alerts
            if alert.level == 'CRITICAL':
                if 'trading' in alert.category.lower():
                    # Stop trading on critical trading alerts
                    logger.critical("Stopping trading due to critical alert")
                    # Implement emergency stop logic here
                
                elif 'system' in alert.category.lower():
                    # System health critical - consider graceful shutdown
                    logger.critical("Critical system alert - monitoring system health")
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def _on_retrain_event(self, event: Dict[str, Any]):
        """Handle retraining events"""
        try:
            logger.info(f"Retraining event: {event}")
            
            # Log to monitoring system
            if self.monitoring_manager:
                # Create alert for successful retraining
                from production.monitor import Alert
                alert = Alert(
                    level='INFO',
                    message=f"Model retrained successfully in {event.get('training_time', 0):.2f}s",
                    category='MODEL'
                )
                self.monitoring_manager._handle_alert(alert)
            
        except Exception as e:
            logger.error(f"Error handling retrain event: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
                'components': {}
            }
            
            # Data streamer status
            if self.data_streamer:
                status['components']['data_streamer'] = self.data_streamer.get_status()
            
            # Risk manager status
            if self.risk_manager:
                status['components']['risk_manager'] = self.risk_manager.get_portfolio_summary()
            
            # Monitoring status
            if self.monitoring_manager:
                status['components']['monitoring'] = self.monitoring_manager.get_comprehensive_status()
            
            # Cache status
            if self.cache_manager:
                status['components']['cache'] = self.cache_manager.get_stats()
            
            # Adaptive retrainer status
            if self.adaptive_retrainer:
                status['components']['adaptive_retrainer'] = self.adaptive_retrainer.get_status()
            
            # Microstructure analyzer status
            if self.microstructure_analyzer:
                status['components']['microstructure'] = self.microstructure_analyzer.get_comprehensive_analysis()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Global production system instance
production_system = None

@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager"""
    global production_system
    
    try:
        # Startup
        logger.info("Starting production server...")
        production_system = ProductionSystem()
        await production_system.initialize()
        await production_system.start()
        
        yield
        
    finally:
        # Shutdown
        if production_system:
            await production_system.stop()
        logger.info("Production server stopped")

# Create FastAPI app with production system integration
def create_production_app():
    """Create production FastAPI application"""
    if not COMPONENTS_AVAILABLE:
        logger.warning("Creating minimal FastAPI app - production components not available")
        from fastapi import FastAPI
        app = FastAPI(title="Gold Bot Production Server (Limited)")
        
        @app.get("/health")
        async def health():
            return {"status": "limited", "message": "Production components not available"}
        
        return app
    
    # Get the FastAPI app from model server
    from production.model_server import server
    app = server.app
    
    # Add production system integration
    app.router.lifespan_context = lifespan
    
    # Add production-specific endpoints
    @app.get("/system/status")
    async def get_system_status():
        """Get comprehensive production system status"""
        if production_system:
            return production_system.get_system_status()
        return {"error": "Production system not initialized"}
    
    @app.post("/system/restart")
    async def restart_system():
        """Restart production system components"""
        if production_system:
            await production_system.stop()
            await production_system.start()
            return {"message": "System restarted successfully"}
        return {"error": "Production system not available"}
    
    @app.get("/system/config")
    async def get_system_config():
        """Get system configuration"""
        if production_system:
            return production_system.config
        return {"error": "Production system not available"}
    
    return app

def run_production_server():
    """Run the production server"""
    try:
        app = create_production_app()
        
        # Handle shutdown signals
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        config = ProductionSystem()._get_default_config()
        uvicorn.run(
            app,
            host=config['server']['host'],
            port=config['server']['port'],
            workers=config['server']['workers'],
            loop="asyncio"
        )
        
    except Exception as e:
        logger.error(f"Failed to start production server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_production_server()