from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # System settings
    system_mode: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Model settings
    model_path: str = "./models/phi-3-mini-4k-instruct-q4.gguf"
    model_context_length: int = 4096
    model_max_tokens: int = 512
    model_temperature: float = 0.7
    
    # Resource settings
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
