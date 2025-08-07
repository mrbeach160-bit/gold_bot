"""
Real-Time Data Streaming System

Provides real-time data streaming capabilities for live trading environments
with data validation, anomaly detection, and feature generation.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates incoming market data for quality and completeness"""
    
    def __init__(self):
        self.required_fields = ['open', 'high', 'low', 'close', 'volume']
        self.min_volume = 0
        self.max_price_change = 0.10  # 10% max price change per period
    
    def is_valid(self, data: Dict[str, Any]) -> bool:
        """Validate data quality"""
        try:
            # Check required fields
            for field in self.required_fields:
                if field not in data or data[field] is None:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check for reasonable values
            ohlc = [data['open'], data['high'], data['low'], data['close']]
            if any(val <= 0 for val in ohlc):
                logger.warning("Invalid OHLC values (negative or zero)")
                return False
            
            # Check high >= low logic
            if data['high'] < data['low']:
                logger.warning("High price is less than low price")
                return False
            
            # Check volume
            if data['volume'] < self.min_volume:
                logger.warning(f"Volume too low: {data['volume']}")
                return False
            
            # Check for extreme price movements (potential data error)
            if len(ohlc) >= 2:
                price_change = abs(max(ohlc) - min(ohlc)) / min(ohlc)
                if price_change > self.max_price_change:
                    logger.warning(f"Extreme price movement detected: {price_change:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

class AnomalyDetector:
    """Detects market anomalies using statistical methods"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.price_history = []
        self.volume_history = []
        self.volatility_threshold = 3.0  # 3 standard deviations
    
    def is_anomaly(self, features: Dict[str, Any]) -> bool:
        """Detect if current market conditions are anomalous"""
        try:
            current_price = features.get('close', 0)
            current_volume = features.get('volume', 0)
            
            if not current_price or not current_volume:
                return False
            
            # Add to history
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            
            # Keep only recent history
            if len(self.price_history) > self.lookback_periods:
                self.price_history.pop(0)
                self.volume_history.pop(0)
            
            # Need enough history for analysis
            if len(self.price_history) < 20:
                return False
            
            # Calculate price volatility
            price_returns = np.diff(np.log(self.price_history))
            if len(price_returns) > 0:
                volatility = np.std(price_returns)
                recent_return = price_returns[-1] if len(price_returns) > 0 else 0
                
                # Check for extreme price movements
                if abs(recent_return) > self.volatility_threshold * volatility:
                    logger.warning(f"Price anomaly detected: return={recent_return:.4f}, volatility={volatility:.4f}")
                    return True
            
            # Check volume anomalies
            if len(self.volume_history) > 5:
                avg_volume = np.mean(self.volume_history[:-1])
                if current_volume > 5 * avg_volume:  # 5x average volume
                    logger.warning(f"Volume anomaly detected: current={current_volume}, avg={avg_volume:.0f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return False

class MT5Connector:
    """Mock MT5 connector for demonstration (replace with real MT5 integration)"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.connected = False
        self.last_price = 2000.0  # Starting price for simulation
    
    async def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            # In real implementation, initialize MT5 connection
            # import MetaTrader5 as mt5
            # return mt5.initialize()
            
            # Mock connection for demo
            self.connected = True
            logger.info(f"Connected to MT5 for symbol {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    async def get_latest_ohlcv(self) -> Optional[Dict[str, Any]]:
        """Get latest OHLCV data"""
        try:
            if not self.connected:
                await self.connect()
            
            # In real implementation:
            # import MetaTrader5 as mt5
            # rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)
            
            # Mock data generation for demo
            price_change = np.random.normal(0, 0.001) * self.last_price
            new_price = self.last_price + price_change
            
            # Generate realistic OHLCV
            volatility = abs(price_change) * 2
            data = {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'open': self.last_price,
                'high': new_price + abs(volatility),
                'low': new_price - abs(volatility), 
                'close': new_price,
                'volume': np.random.randint(100, 1000),
                'spread': 0.5,
                'tick_volume': np.random.randint(50, 200)
            }
            
            self.last_price = new_price
            return data
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            return None

class RealTimeDataStreamer:
    """Main real-time data streaming system"""
    
    def __init__(self, symbol: str = "XAUUSD", update_interval: float = 1.0):
        self.symbol = symbol
        self.update_interval = update_interval
        self.mt5_connector = MT5Connector(symbol)
        self.data_validator = DataValidator()
        self.anomaly_detector = AnomalyDetector()
        
        # Import feature pipeline from existing models
        try:
            from models.advanced_features import AdvancedDataPipeline
            self.feature_pipeline = AdvancedDataPipeline()
        except ImportError:
            logger.warning("Advanced feature pipeline not available, using basic features")
            self.feature_pipeline = None
        
        self.running = False
        self.subscribers = []
        self.last_prediction = None
        self.performance_stats = {
            'total_updates': 0,
            'valid_updates': 0,
            'anomalies_detected': 0,
            'avg_processing_time': 0.0
        }
    
    def subscribe(self, callback):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback):
        """Unsubscribe from updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def broadcast_prediction(self, prediction: Dict[str, Any]):
        """Broadcast prediction to all subscribers"""
        try:
            self.last_prediction = prediction
            
            # Notify all subscribers
            for callback in self.subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(prediction)
                    else:
                        callback(prediction)
                except Exception as e:
                    logger.error(f"Subscriber notification error: {e}")
        
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
    
    async def process_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process raw data into features and predictions"""
        try:
            start_time = time.time()
            
            # Validate data quality
            if not self.data_validator.is_valid(raw_data):
                return None
            
            # Generate features
            if self.feature_pipeline:
                # Convert to DataFrame for feature pipeline
                df = pd.DataFrame([raw_data])
                features = self.feature_pipeline.transform(df)
                feature_dict = features.iloc[0].to_dict() if not features.empty else raw_data
            else:
                # Basic features
                feature_dict = raw_data.copy()
                feature_dict['price_change'] = (raw_data['close'] - raw_data['open']) / raw_data['open']
                feature_dict['high_low_ratio'] = raw_data['high'] / raw_data['low']
            
            # Detect anomalies
            is_anomaly = self.anomaly_detector.is_anomaly(feature_dict)
            if is_anomaly:
                self.performance_stats['anomalies_detected'] += 1
                feature_dict['anomaly_confidence'] = 0.5  # Reduce confidence during anomalies
            else:
                feature_dict['anomaly_confidence'] = 1.0
            
            # Add metadata
            feature_dict.update({
                'timestamp': raw_data.get('timestamp', datetime.now()),
                'symbol': self.symbol,
                'is_anomaly': is_anomaly,
                'processing_time': time.time() - start_time
            })
            
            # Update performance stats
            self.performance_stats['valid_updates'] += 1
            self.performance_stats['avg_processing_time'] = (
                (self.performance_stats['avg_processing_time'] * (self.performance_stats['valid_updates'] - 1) +
                 feature_dict['processing_time']) / self.performance_stats['valid_updates']
            )
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            return None
    
    async def stream_predictions(self):
        """Main streaming loop - generates continuous predictions"""
        logger.info(f"Starting real-time data stream for {self.symbol}")
        self.running = True
        
        # Connect to data source
        if not await self.mt5_connector.connect():
            logger.error("Failed to connect to data source")
            return
        
        try:
            while self.running:
                # Get real-time market data
                current_data = await self.mt5_connector.get_latest_ohlcv()
                self.performance_stats['total_updates'] += 1
                
                if current_data is None:
                    await asyncio.sleep(self.update_interval)
                    continue
                
                # Process data into features
                processed_data = await self.process_data(current_data)
                
                if processed_data is None:
                    await asyncio.sleep(self.update_interval)
                    continue
                
                # Generate ensemble prediction (mock for now)
                prediction = await self.generate_prediction(processed_data)
                
                # Broadcast to trading engine
                await self.broadcast_prediction(prediction)
                
                # Log periodic status
                if self.performance_stats['total_updates'] % 60 == 0:  # Every minute
                    logger.info(f"Stream status: {self.performance_stats}")
                
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            self.running = False
            logger.info("Data streaming stopped")
    
    async def generate_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction from features (placeholder)"""
        try:
            # In real implementation, this would use the ensemble manager
            # from models import ModelManager
            # prediction = self.ensemble_manager.predict(features)
            
            # Mock prediction for demo
            confidence = features.get('anomaly_confidence', 1.0)
            
            # Simple momentum-based prediction
            price_change = features.get('price_change', 0)
            if price_change > 0.001:
                direction = 'BUY'
                confidence *= 0.7
            elif price_change < -0.001:
                direction = 'SELL'
                confidence *= 0.7
            else:
                direction = 'HOLD'
                confidence *= 0.5
            
            prediction = {
                'timestamp': features['timestamp'],
                'symbol': features['symbol'],
                'direction': direction,
                'confidence': min(confidence, 1.0),
                'price': features.get('close', 0),
                'features_count': len(features),
                'is_anomaly': features.get('is_anomaly', False),
                'processing_time': features.get('processing_time', 0)
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction generation error: {e}")
            return {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'direction': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def stop(self):
        """Stop the data stream"""
        self.running = False
        logger.info("Stopping data stream...")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        return {
            'running': self.running,
            'symbol': self.symbol,
            'subscribers': len(self.subscribers),
            'last_prediction': self.last_prediction,
            'performance_stats': self.performance_stats.copy()
        }

# Example usage
if __name__ == "__main__":
    async def demo():
        # Create streamer
        streamer = RealTimeDataStreamer("XAUUSD", update_interval=2.0)
        
        # Add a subscriber
        async def on_prediction(prediction):
            print(f"Received prediction: {prediction['direction']} "
                  f"(confidence: {prediction['confidence']:.2f})")
        
        streamer.subscribe(on_prediction)
        
        # Start streaming for 30 seconds
        stream_task = asyncio.create_task(streamer.stream_predictions())
        await asyncio.sleep(30)
        
        # Stop streaming
        streamer.stop()
        await stream_task
        
        # Print final status
        print("Final status:", streamer.get_status())
    
    # Run demo
    asyncio.run(demo())