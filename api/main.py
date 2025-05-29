import asyncio
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.schemas import TextRequest, TextResponse, SystemHealth, ErrorResponse
from core.model_manager import ModelManager
from core.system_monitor import SystemMonitor
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger()

# Global instances
model_manager = ModelManager()
system_monitor = SystemMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Neural Stream Fusion Engine...")
    
    # Check system resources
    if not system_monitor.check_resources():
        logger.error("Insufficient system resources")
        raise RuntimeError("Insufficient system resources")
    
    # Load model
    success = await model_manager.load_model()
    if not success:
        logger.error("Failed to load model")
        raise RuntimeError("Failed to load model")
    
    logger.success("Neural Stream Fusion Engine started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Neural Stream Fusion Engine...")
    await model_manager.unload_model()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Neural Stream Fusion Engine",
    description="Unified Real-Time Intelligence Infrastructure with Adaptive Multi-Modal Processing",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "name": "Neural Stream Fusion Engine",
        "version": "0.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check endpoint"""
    try:
        health_data = system_monitor.get_system_health(model_manager)
        return SystemHealth(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    """Generate text using the loaded model"""
    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())
        
        logger.info(f"Processing request {request_id}: {request.prompt[:50]}...")
        
        # Generate text
        result = await model_manager.generate_text(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream
        )
        
        # Create response
        response = TextResponse(
            id=request_id,
            text=result["text"],
            model=result["model"],
            tokens_used=result["tokens_used"],
            processing_time=result["processing_time"]
        )
        
        logger.success(f"Request {request_id} completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Text generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

@app.get("/model/info", response_model=dict)
async def model_info():
    """Get information about the loaded model"""
    try:
        info = model_manager.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.system_mode == "development",
        log_level=settings.log_level.lower()
    )
