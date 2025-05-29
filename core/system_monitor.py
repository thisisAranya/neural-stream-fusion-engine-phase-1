import psutil
import time
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger()

class SystemMonitor:
    """Monitor system resources and health"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_system_health(self, model_manager) -> Dict[str, Any]:
        """Get comprehensive system health information"""
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # System uptime
        uptime = time.time() - self.start_time
        
        # Model status
        model_info = model_manager.get_model_info()
        
        return {
            "status": "healthy" if memory.percent < 90 and cpu_percent < 90 else "warning",
            "model_loaded": model_info["is_loaded"],
            "memory_usage": memory_info,
            "cpu_usage": cpu_percent,
            "uptime": uptime,
            "model_info": model_info
        }
    
    def check_resources(self) -> bool:
        """Check if system has enough resources"""
        memory = psutil.virtual_memory()
        
        # Check if we have at least 2GB available
        available_gb = memory.available / (1024**3)
        
        if available_gb < 2.0:
            logger.warning(f"Low memory: {available_gb:.2f}GB available")
            return False
        
        return True
