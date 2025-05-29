from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class TextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192, description="Input text prompt")
    max_tokens: Optional[int] = Field(None, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling")
    stream: bool = Field(False, description="Stream response")

class TextResponse(BaseModel):
    id: str = Field(..., description="Unique response ID")
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    tokens_used: int = Field(..., description="Total tokens consumed")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class SystemHealth(BaseModel):
    status: str = Field(..., description="System status")
    model_loaded: bool = Field(..., description="Model load status")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage statistics")
    cpu_usage: float = Field(..., description="CPU usage percentage")
    uptime: float = Field(..., description="System uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
