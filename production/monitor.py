"""
Monitoring and Alerting System

Comprehensive monitoring for model performance, system health, trading performance,
and real-time alerting for production environments.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import statistics
import numpy as np
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert data structure"""
    level: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message: str
    category: str  # 'PERFORMANCE', 'SYSTEM', 'TRADING', 'MODEL'
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class MetricThreshold:
    """Metric threshold configuration"""
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    comparison: str = 'gt'  # 'gt', 'lt', 'eq'
    duration_minutes: int = 1  # Duration before triggering alert

class PerformanceMonitor:
    """Monitor model and trading performance"""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.alert_callback = alert_callback or self._default_alert_handler
        
        # Performance metrics
        self.prediction_history = []
        self.accuracy_history = []
        self.latency_history = []
        self.confidence_history = []
        
        # Trading metrics
        self.trade_history = []
        self.pnl_history = []
        self.drawdown_history = []
        
        # Model drift detection
        self.feature_distributions = {}
        self.prediction_distributions = {}
        
        # Thresholds
        self.thresholds = {
            'accuracy': MetricThreshold(warning_threshold=0.55, error_threshold=0.50, comparison='lt'),
            'latency': MetricThreshold(warning_threshold=100, error_threshold=1000, comparison='gt'),  # ms
            'confidence': MetricThreshold(warning_threshold=0.6, error_threshold=0.4, comparison='lt'),
            'drawdown': MetricThreshold(warning_threshold=0.10, error_threshold=0.15, comparison='gt'),
            'daily_loss': MetricThreshold(warning_threshold=0.05, error_threshold=0.10, comparison='gt')
        }
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        logger.info("Performance Monitor initialized")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                self._check_all_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def record_prediction(self, prediction: Dict[str, Any], actual_result: Optional[bool] = None,
                         latency_ms: Optional[float] = None):
        """Record prediction for performance tracking"""
        try:
            timestamp = datetime.now()
            
            record = {
                'timestamp': timestamp,
                'direction': prediction.get('direction'),
                'confidence': prediction.get('confidence', 0),
                'symbol': prediction.get('symbol'),
                'actual_result': actual_result,
                'latency_ms': latency_ms or 0
            }
            
            self.prediction_history.append(record)
            
            # Keep only recent history (last 1000 predictions)
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            # Update metrics
            if latency_ms:
                self.latency_history.append(latency_ms)
                if len(self.latency_history) > 100:
                    self.latency_history = self.latency_history[-100:]
            
            confidence = prediction.get('confidence', 0)
            if confidence:
                self.confidence_history.append(confidence)
                if len(self.confidence_history) > 100:
                    self.confidence_history = self.confidence_history[-100:]
            
            # Calculate accuracy if we have actual results
            if actual_result is not None:
                recent_predictions = [p for p in self.prediction_history[-50:] 
                                    if p['actual_result'] is not None]
                if recent_predictions:
                    correct_predictions = sum(1 for p in recent_predictions 
                                            if p['actual_result'] == (p['direction'] != 'HOLD'))
                    accuracy = correct_predictions / len(recent_predictions)
                    self.accuracy_history.append(accuracy)
                    
                    if len(self.accuracy_history) > 50:
                        self.accuracy_history = self.accuracy_history[-50:]
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def record_trade(self, trade_result: Dict[str, Any]):
        """Record trading result for performance tracking"""
        try:
            timestamp = datetime.now()
            
            trade_record = {
                'timestamp': timestamp,
                'symbol': trade_result.get('symbol'),
                'direction': trade_result.get('direction'),
                'pnl': trade_result.get('pnl', 0),
                'duration_hours': trade_result.get('duration_hours', 0),
                'entry_price': trade_result.get('entry_price'),
                'exit_price': trade_result.get('exit_price')
            }
            
            self.trade_history.append(trade_record)
            
            # Keep only recent history
            if len(self.trade_history) > 500:
                self.trade_history = self.trade_history[-500:]
            
            # Update P&L history
            pnl = trade_result.get('pnl', 0)
            if pnl != 0:
                self.pnl_history.append({'timestamp': timestamp, 'pnl': pnl})
                if len(self.pnl_history) > 200:
                    self.pnl_history = self.pnl_history[-200:]
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def detect_model_drift(self, features: Dict[str, Any]) -> bool:
        """Detect concept drift in feature distributions"""
        try:
            drift_detected = False
            
            for feature_name, value in features.items():
                if not isinstance(value, (int, float)):
                    continue
                
                if feature_name not in self.feature_distributions:
                    self.feature_distributions[feature_name] = []
                
                self.feature_distributions[feature_name].append(value)
                
                # Keep only recent values
                if len(self.feature_distributions[feature_name]) > 200:
                    self.feature_distributions[feature_name] = self.feature_distributions[feature_name][-200:]
                
                # Check for drift if we have enough data
                if len(self.feature_distributions[feature_name]) >= 100:
                    recent_data = self.feature_distributions[feature_name][-50:]
                    historical_data = self.feature_distributions[feature_name][-100:-50]
                    
                    # Simple drift detection using mean and std comparison
                    recent_mean = statistics.mean(recent_data)
                    historical_mean = statistics.mean(historical_data)
                    
                    recent_std = statistics.stdev(recent_data) if len(recent_data) > 1 else 0
                    historical_std = statistics.stdev(historical_data) if len(historical_data) > 1 else 0
                    
                    # Detect significant change in mean (> 2 standard deviations)
                    if historical_std > 0:
                        z_score = abs(recent_mean - historical_mean) / historical_std
                        if z_score > 2.0:
                            drift_detected = True
                            self._trigger_alert(
                                'WARNING',
                                f'Model drift detected in feature {feature_name}: z-score={z_score:.2f}',
                                'MODEL',
                                {'feature': feature_name, 'z_score': z_score}
                            )
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Model drift detection error: {e}")
            return False
    
    def _check_all_metrics(self):
        """Check all metrics against thresholds"""
        try:
            # Check accuracy
            if self.accuracy_history:
                recent_accuracy = statistics.mean(self.accuracy_history[-10:])
                self._check_threshold('accuracy', recent_accuracy)
            
            # Check latency
            if self.latency_history:
                avg_latency = statistics.mean(self.latency_history[-20:])
                self._check_threshold('latency', avg_latency)
            
            # Check confidence
            if self.confidence_history:
                avg_confidence = statistics.mean(self.confidence_history[-20:])
                self._check_threshold('confidence', avg_confidence)
            
            # Check drawdown
            if self.pnl_history:
                cumulative_pnl = [sum(p['pnl'] for p in self.pnl_history[:i+1]) 
                                 for i in range(len(self.pnl_history))]
                if cumulative_pnl:
                    peak = max(cumulative_pnl)
                    current = cumulative_pnl[-1]
                    drawdown = (peak - current) / peak if peak > 0 else 0
                    self._check_threshold('drawdown', drawdown)
            
            # Check daily losses
            today = datetime.now().date()
            today_trades = [t for t in self.pnl_history 
                           if t['timestamp'].date() == today]
            if today_trades:
                daily_pnl = sum(t['pnl'] for t in today_trades)
                if daily_pnl < 0:
                    daily_loss_pct = abs(daily_pnl) / 10000  # Assume 10k starting balance
                    self._check_threshold('daily_loss', daily_loss_pct)
            
        except Exception as e:
            logger.error(f"Metrics check error: {e}")
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check metric against configured thresholds"""
        try:
            threshold = self.thresholds.get(metric_name)
            if not threshold:
                return
            
            comparison = threshold.comparison
            warning_threshold = threshold.warning_threshold
            error_threshold = threshold.error_threshold
            
            # Check error threshold first
            if error_threshold is not None:
                trigger_error = False
                if comparison == 'gt' and value > error_threshold:
                    trigger_error = True
                elif comparison == 'lt' and value < error_threshold:
                    trigger_error = True
                elif comparison == 'eq' and abs(value - error_threshold) < 0.001:
                    trigger_error = True
                
                if trigger_error:
                    self._trigger_alert(
                        'ERROR',
                        f'{metric_name} threshold exceeded: {value:.4f} (threshold: {error_threshold})',
                        'PERFORMANCE',
                        {'metric': metric_name, 'value': value, 'threshold': error_threshold}
                    )
                    return
            
            # Check warning threshold
            if warning_threshold is not None:
                trigger_warning = False
                if comparison == 'gt' and value > warning_threshold:
                    trigger_warning = True
                elif comparison == 'lt' and value < warning_threshold:
                    trigger_warning = True
                elif comparison == 'eq' and abs(value - warning_threshold) < 0.001:
                    trigger_warning = True
                
                if trigger_warning:
                    self._trigger_alert(
                        'WARNING',
                        f'{metric_name} warning threshold reached: {value:.4f} (threshold: {warning_threshold})',
                        'PERFORMANCE',
                        {'metric': metric_name, 'value': value, 'threshold': warning_threshold}
                    )
            
        except Exception as e:
            logger.error(f"Threshold check error for {metric_name}: {e}")
    
    def _trigger_alert(self, level: str, message: str, category: str, data: Dict[str, Any] = None):
        """Trigger alert"""
        alert = Alert(
            level=level,
            message=message,
            category=category,
            data=data or {}
        )
        
        try:
            self.alert_callback(alert)
        except Exception as e:
            logger.error(f"Alert callback error: {e}")
    
    def _default_alert_handler(self, alert: Alert):
        """Default alert handler - just log the alert"""
        log_level = getattr(logging, alert.level, logging.INFO)
        logger.log(log_level, f"[{alert.category}] {alert.message}")
    
    def get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics"""
        try:
            metrics = {}
            
            # Accuracy metrics
            if self.accuracy_history:
                metrics['accuracy'] = {
                    'current': self.accuracy_history[-1] if self.accuracy_history else 0,
                    'average': statistics.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else 0,
                    'trend': 'improving' if len(self.accuracy_history) >= 2 and 
                            self.accuracy_history[-1] > self.accuracy_history[-2] else 'declining'
                }
            
            # Latency metrics
            if self.latency_history:
                metrics['latency'] = {
                    'current_ms': self.latency_history[-1] if self.latency_history else 0,
                    'average_ms': statistics.mean(self.latency_history[-20:]) if len(self.latency_history) >= 20 else 0,
                    'p95_ms': np.percentile(self.latency_history[-50:], 95) if len(self.latency_history) >= 50 else 0
                }
            
            # Confidence metrics
            if self.confidence_history:
                metrics['confidence'] = {
                    'current': self.confidence_history[-1] if self.confidence_history else 0,
                    'average': statistics.mean(self.confidence_history[-20:]) if len(self.confidence_history) >= 20 else 0
                }
            
            # Trading metrics
            if self.trade_history:
                recent_trades = self.trade_history[-20:]
                winning_trades = [t for t in recent_trades if t['pnl'] > 0]
                metrics['trading'] = {
                    'total_trades': len(recent_trades),
                    'winning_trades': len(winning_trades),
                    'win_rate': len(winning_trades) / len(recent_trades) if recent_trades else 0,
                    'total_pnl': sum(t['pnl'] for t in recent_trades),
                    'avg_trade_duration': statistics.mean([t['duration_hours'] for t in recent_trades]) if recent_trades else 0
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting recent metrics: {e}")
            return {}

class SystemMonitor:
    """Monitor system resources and health"""
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.alert_callback = alert_callback or self._default_alert_handler
        
        # System metrics
        self.cpu_history = []
        self.memory_history = []
        self.disk_history = []
        
        # Service health
        self.service_health = {}
        self.last_health_check = {}
        
        # Error tracking
        self.error_counts = {}
        self.last_error_reset = datetime.now()
        
        # Thresholds
        self.thresholds = {
            'cpu_percent': MetricThreshold(warning_threshold=70, error_threshold=90, comparison='gt'),
            'memory_percent': MetricThreshold(warning_threshold=80, error_threshold=95, comparison='gt'),
            'disk_percent': MetricThreshold(warning_threshold=85, error_threshold=95, comparison='gt'),
            'error_rate': MetricThreshold(warning_threshold=10, error_threshold=25, comparison='gt')  # errors per minute
        }
        
        self._monitoring_active = False
        self._monitoring_thread = None
        
        logger.info("System Monitor initialized")
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"System monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """System monitoring loop"""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_system_health()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"System monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_history.append({'timestamp': datetime.now(), 'value': cpu_percent})
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_history.append({'timestamp': datetime.now(), 'value': memory.percent})
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_history.append({'timestamp': datetime.now(), 'value': disk_percent})
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.cpu_history = [m for m in self.cpu_history if m['timestamp'] > cutoff_time]
            self.memory_history = [m for m in self.memory_history if m['timestamp'] > cutoff_time]
            self.disk_history = [m for m in self.disk_history if m['timestamp'] > cutoff_time]
            
            # Check thresholds
            self._check_threshold('cpu_percent', cpu_percent)
            self._check_threshold('memory_percent', memory.percent)
            self._check_threshold('disk_percent', disk_percent)
            
        except ImportError:
            # psutil not available, use basic monitoring
            logger.warning("psutil not available, using basic system monitoring")
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check error rates
            current_time = datetime.now()
            
            # Reset error counts hourly
            if (current_time - self.last_error_reset).total_seconds() > 3600:
                self.error_counts = {}
                self.last_error_reset = current_time
            
            # Calculate error rate (errors per minute)
            minute_ago = current_time - timedelta(minutes=1)
            recent_errors = sum(
                count for timestamp, count in self.error_counts.items()
                if timestamp > minute_ago
            )
            
            self._check_threshold('error_rate', recent_errors)
            
        except Exception as e:
            logger.error(f"System health check error: {e}")
    
    def record_error(self, error_type: str, error_message: str):
        """Record system error for monitoring"""
        try:
            timestamp = datetime.now()
            
            if timestamp not in self.error_counts:
                self.error_counts[timestamp] = 0
            
            self.error_counts[timestamp] += 1
            
            # Log the error
            logger.error(f"[{error_type}] {error_message}")
            
            # Trigger alert for critical errors
            if error_type.upper() in ['CRITICAL', 'FATAL']:
                self._trigger_alert(
                    'CRITICAL',
                    f'Critical system error: {error_message}',
                    'SYSTEM',
                    {'error_type': error_type, 'timestamp': timestamp.isoformat()}
                )
            
        except Exception as e:
            logger.error(f"Error recording error: {e}")
    
    def check_service_health(self, service_name: str, health_check_func: Callable) -> bool:
        """Check health of a specific service"""
        try:
            start_time = time.time()
            is_healthy = health_check_func()
            response_time = time.time() - start_time
            
            self.service_health[service_name] = {
                'healthy': is_healthy,
                'last_check': datetime.now(),
                'response_time': response_time
            }
            
            self.last_health_check[service_name] = datetime.now()
            
            if not is_healthy:
                self._trigger_alert(
                    'ERROR',
                    f'Service health check failed: {service_name}',
                    'SYSTEM',
                    {'service': service_name, 'response_time': response_time}
                )
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check error for {service_name}: {e}")
            self.service_health[service_name] = {
                'healthy': False,
                'last_check': datetime.now(),
                'error': str(e)
            }
            return False
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check metric against thresholds (same as PerformanceMonitor)"""
        try:
            threshold = self.thresholds.get(metric_name)
            if not threshold:
                return
            
            comparison = threshold.comparison
            warning_threshold = threshold.warning_threshold
            error_threshold = threshold.error_threshold
            
            # Check error threshold
            if error_threshold is not None:
                trigger_error = False
                if comparison == 'gt' and value > error_threshold:
                    trigger_error = True
                elif comparison == 'lt' and value < error_threshold:
                    trigger_error = True
                
                if trigger_error:
                    self._trigger_alert(
                        'ERROR',
                        f'System {metric_name} critical: {value:.1f}% (threshold: {error_threshold}%)',
                        'SYSTEM',
                        {'metric': metric_name, 'value': value, 'threshold': error_threshold}
                    )
                    return
            
            # Check warning threshold
            if warning_threshold is not None:
                trigger_warning = False
                if comparison == 'gt' and value > warning_threshold:
                    trigger_warning = True
                elif comparison == 'lt' and value < warning_threshold:
                    trigger_warning = True
                
                if trigger_warning:
                    self._trigger_alert(
                        'WARNING',
                        f'System {metric_name} high: {value:.1f}% (threshold: {warning_threshold}%)',
                        'SYSTEM',
                        {'metric': metric_name, 'value': value, 'threshold': warning_threshold}
                    )
            
        except Exception as e:
            logger.error(f"System threshold check error for {metric_name}: {e}")
    
    def _trigger_alert(self, level: str, message: str, category: str, data: Dict[str, Any] = None):
        """Trigger system alert"""
        alert = Alert(
            level=level,
            message=message,
            category=category,
            data=data or {}
        )
        
        try:
            self.alert_callback(alert)
        except Exception as e:
            logger.error(f"System alert callback error: {e}")
    
    def _default_alert_handler(self, alert: Alert):
        """Default alert handler"""
        log_level = getattr(logging, alert.level, logging.INFO)
        logger.log(log_level, f"[{alert.category}] {alert.message}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self._monitoring_active
            }
            
            # Recent resource usage
            if self.cpu_history:
                recent_cpu = [m['value'] for m in self.cpu_history[-10:]]
                status['cpu'] = {
                    'current': recent_cpu[-1] if recent_cpu else 0,
                    'average': statistics.mean(recent_cpu) if recent_cpu else 0,
                    'max': max(recent_cpu) if recent_cpu else 0
                }
            
            if self.memory_history:
                recent_memory = [m['value'] for m in self.memory_history[-10:]]
                status['memory'] = {
                    'current': recent_memory[-1] if recent_memory else 0,
                    'average': statistics.mean(recent_memory) if recent_memory else 0,
                    'max': max(recent_memory) if recent_memory else 0
                }
            
            if self.disk_history:
                recent_disk = [m['value'] for m in self.disk_history[-10:]]
                status['disk'] = {
                    'current': recent_disk[-1] if recent_disk else 0,
                    'average': statistics.mean(recent_disk) if recent_disk else 0,
                    'max': max(recent_disk) if recent_disk else 0
                }
            
            # Service health
            status['services'] = self.service_health.copy()
            
            # Error summary
            recent_errors = sum(
                count for timestamp, count in self.error_counts.items()
                if timestamp > datetime.now() - timedelta(minutes=5)
            )
            status['errors_last_5min'] = recent_errors
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Combined monitoring manager
class MonitoringManager:
    """Combined monitoring system manager"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor(self._handle_alert)
        self.system_monitor = SystemMonitor(self._handle_alert)
        self.alerts_history = []
        self.alert_callbacks = []
        
        logger.info("Monitoring Manager initialized")
    
    def add_alert_callback(self, callback: Callable):
        """Add custom alert callback"""
        self.alert_callbacks.append(callback)
    
    def start_all_monitoring(self, performance_interval: int = 60, system_interval: int = 30):
        """Start all monitoring components"""
        self.performance_monitor.start_monitoring(performance_interval)
        self.system_monitor.start_monitoring(system_interval)
        logger.info("All monitoring started")
    
    def stop_all_monitoring(self):
        """Stop all monitoring components"""
        self.performance_monitor.stop_monitoring()
        self.system_monitor.stop_monitoring()
        logger.info("All monitoring stopped")
    
    def _handle_alert(self, alert: Alert):
        """Central alert handler"""
        # Store in history
        self.alerts_history.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alerts_history) > 1000:
            self.alerts_history = self.alerts_history[-1000:]
        
        # Call custom callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # Default handling - log critical alerts
        if alert.level in ['ERROR', 'CRITICAL']:
            logger.error(f"[{alert.category}] {alert.message}")
        elif alert.level == 'WARNING':
            logger.warning(f"[{alert.category}] {alert.message}")
        else:
            logger.info(f"[{alert.category}] {alert.message}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts_history if a.timestamp > cutoff_time]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_level': {},
            'by_category': {},
            'recent_alerts': recent_alerts[-10:]  # Last 10 alerts
        }
        
        for alert in recent_alerts:
            # Count by level
            if alert.level not in summary['by_level']:
                summary['by_level'][alert.level] = 0
            summary['by_level'][alert.level] += 1
            
            # Count by category
            if alert.category not in summary['by_category']:
                summary['by_category'][alert.category] = 0
            summary['by_category'][alert.category] += 1
        
        return summary
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            'performance_metrics': self.performance_monitor.get_recent_metrics(),
            'system_status': self.system_monitor.get_system_status(),
            'alert_summary': self.get_alert_summary(hours=1),
            'monitoring_active': {
                'performance': self.performance_monitor._monitoring_active,
                'system': self.system_monitor._monitoring_active
            }
        }

# Example usage
if __name__ == "__main__":
    # Create monitoring manager
    monitoring = MonitoringManager()
    
    # Add custom alert handler
    def custom_alert_handler(alert: Alert):
        print(f"CUSTOM ALERT: [{alert.level}] {alert.message}")
    
    monitoring.add_alert_callback(custom_alert_handler)
    
    # Start monitoring
    monitoring.start_all_monitoring()
    
    # Simulate some metrics
    import time
    for i in range(5):
        # Record some performance data
        monitoring.performance_monitor.record_prediction({
            'direction': 'BUY',
            'confidence': 0.8 - (i * 0.1),  # Decreasing confidence
            'symbol': 'XAUUSD'
        }, actual_result=True, latency_ms=50 + (i * 20))  # Increasing latency
        
        time.sleep(2)
    
    # Get status
    status = monitoring.get_comprehensive_status()
    print("Monitoring status:", json.dumps(status, indent=2, default=str))
    
    # Stop monitoring
    monitoring.stop_all_monitoring()