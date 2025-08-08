"""
Centralized Path Management System

Provides cross-platform path handling with environment-specific configurations.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class PathManager:
    """Centralized path management for cross-platform compatibility."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """Initialize path manager with project root."""
        if project_root is None:
            # Auto-detect project root by finding key files
            self._project_root = self._find_project_root()
        else:
            self._project_root = Path(project_root).resolve()
        
        self._ensure_directory_structure()
        logger.info(f"PathManager initialized with project root: {self._project_root}")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for key files."""
        current = Path(__file__).resolve()
        
        # Look for project indicators
        indicators = ['requirements.txt', 'main.py', 'README.md', '.git']
        
        # Start from current file and go up
        for parent in [current.parent.parent] + list(current.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        # Fallback to parent of utils directory
        return current.parent.parent
    
    def _ensure_directory_structure(self):
        """Create necessary directory structure."""
        directories = [
            'model',
            'logs',
            'data',
            'config',
            'tmp',
            'cache'
        ]
        
        for dir_name in directories:
            dir_path = self._project_root / dir_name
            dir_path.mkdir(exist_ok=True)
    
    @property
    def project_root(self) -> Path:
        """Get project root path."""
        return self._project_root
    
    def get_model_path(self, filename: str = None) -> Path:
        """Get model directory or specific model file path."""
        model_dir = self._project_root / 'model'
        return model_dir / filename if filename else model_dir
    
    def get_data_path(self, filename: str = None) -> Path:
        """Get data directory or specific data file path."""
        data_dir = self._project_root / 'data'
        return data_dir / filename if filename else data_dir
    
    def get_config_path(self, filename: str = None) -> Path:
        """Get config directory or specific config file path."""
        config_dir = self._project_root / 'config'
        return config_dir / filename if filename else config_dir
    
    def get_log_path(self, filename: str = None) -> Path:
        """Get logs directory or specific log file path."""
        log_dir = self._project_root / 'logs'
        return log_dir / filename if filename else log_dir
    
    def get_cache_path(self, filename: str = None) -> Path:
        """Get cache directory or specific cache file path."""
        cache_dir = self._project_root / 'cache'
        return cache_dir / filename if filename else cache_dir
    
    def get_tmp_path(self, filename: str = None) -> Path:
        """Get temporary directory or specific temp file path."""
        tmp_dir = self._project_root / 'tmp'
        return tmp_dir / filename if filename else tmp_dir
    
    def get_safe_filename(self, name: str) -> str:
        """Convert name to safe filename."""
        import re
        # Remove or replace unsafe characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'[^\w\-_.]', '_', safe_name)
        return safe_name
    
    def get_model_file_path(self, symbol: str, timeframe: str, model_type: str, extension: str = '.pkl') -> Path:
        """Generate standardized model file path."""
        safe_symbol = self.get_safe_filename(symbol)
        safe_timeframe = self.get_safe_filename(timeframe)
        safe_model_type = self.get_safe_filename(model_type)
        
        filename = f"{safe_model_type}_{safe_symbol}_{safe_timeframe}{extension}"
        return self.get_model_path(filename)
    
    def get_log_file_path(self, component: str, date_suffix: bool = True) -> Path:
        """Generate standardized log file path."""
        from datetime import datetime
        
        safe_component = self.get_safe_filename(component)
        
        if date_suffix:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"{safe_component}_{date_str}.log"
        else:
            filename = f"{safe_component}.log"
        
        return self.get_log_path(filename)
    
    def ensure_path_exists(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists for given path."""
        path = Path(path)
        if not path.suffix:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
        else:  # It's a file, create parent directory
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def relative_to_project(self, path: Union[str, Path]) -> Path:
        """Get path relative to project root."""
        path = Path(path)
        try:
            return path.relative_to(self._project_root)
        except ValueError:
            # Path is not relative to project root
            return path
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files."""
        import time
        from datetime import datetime, timedelta
        
        tmp_dir = self.get_tmp_path()
        if not tmp_dir.exists():
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        for file_path in tmp_dir.iterdir():
            try:
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not clean up temp file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def get_path_info(self) -> Dict[str, str]:
        """Get comprehensive path information."""
        return {
            'project_root': str(self._project_root),
            'model_dir': str(self.get_model_path()),
            'data_dir': str(self.get_data_path()),
            'config_dir': str(self.get_config_path()),
            'log_dir': str(self.get_log_path()),
            'cache_dir': str(self.get_cache_path()),
            'tmp_dir': str(self.get_tmp_path()),
            'platform': os.name,
            'separator': os.sep
        }


# Global path manager instance
path_manager = PathManager()

# Convenience functions
def get_project_root() -> Path:
    """Get project root path."""
    return path_manager.project_root

def get_model_path(filename: str = None) -> Path:
    """Get model path."""
    return path_manager.get_model_path(filename)

def get_data_path(filename: str = None) -> Path:
    """Get data path."""
    return path_manager.get_data_path(filename)

def get_config_path(filename: str = None) -> Path:
    """Get config path."""
    return path_manager.get_config_path(filename)

def get_log_path(filename: str = None) -> Path:
    """Get log path."""
    return path_manager.get_log_path(filename)

def get_safe_filename(name: str) -> str:
    """Get safe filename."""
    return path_manager.get_safe_filename(name)

def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensure path exists."""
    return path_manager.ensure_path_exists(path)

# Export convenience functions
__all__ = [
    'PathManager',
    'path_manager',
    'get_project_root',
    'get_model_path',
    'get_data_path', 
    'get_config_path',
    'get_log_path',
    'get_safe_filename',
    'ensure_path_exists'
]