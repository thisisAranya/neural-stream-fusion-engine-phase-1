import asyncio
import time
import psutil
from typing import Optional, Dict, Any
from llama_cpp import Llama
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger()

class ModelManager:
    """Manages llama.cpp model lifecycle and inference"""
    
    def __init__(self):
        self.model: Optional[Llama] = None
        self.model_path = settings.model_path
        self.is_loaded = False
        self.load_time: Optional[float] = None
        self.inference_count = 0
        
    async def load_model(self) -> bool:
        """Load the model asynchronously"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            start_time = time.time()
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                self._load_model_sync
            )
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            logger.success(f"Model loaded successfully in {self.load_time:.2f}s")
            logger.info(f"Model context length: {settings.model_context_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            return False
    
    def _load_model_sync(self) -> Llama:
        """Synchronous model loading"""
        return Llama(
            model_path=self.model_path,
            n_ctx=settings.model_context_length,
            n_threads=psutil.cpu_count(logical=False),  # Physical cores only
            verbose=False,
            use_mmap=True,
            use_mlock=False,
        )
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the loaded model"""
        
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Use defaults from settings if not provided
        max_tokens = max_tokens or settings.model_max_tokens
        temperature = temperature or settings.model_temperature
        top_p = top_p or 0.9
        
        try:
            start_time = time.time()
            
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt, max_tokens, temperature, top_p, stream
            )
            
            processing_time = time.time() - start_time
            self.inference_count += 1
            
            logger.info(f"Generated {len(result['text'])} chars in {processing_time:.2f}s")
            
            return {
                "text": result["text"],
                "tokens_used": result["tokens_used"],
                "processing_time": processing_time,
                "model": "phi-3-mini"
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool
    ) -> Dict[str, Any]:
        """Synchronous text generation"""
        
        # Create the full prompt with proper formatting
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Generate response
        response = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stop=["<|end|>", "<|user|>"],
            stream=stream
        )
        
        if stream:
            # Handle streaming response
            full_text = ""
            for chunk in response:
                if 'choices' in chunk and chunk['choices']:
                    text_chunk = chunk['choices'][0].get('text', '')
                    full_text += text_chunk
            
            return {
                "text": full_text.strip(),
                "tokens_used": len(full_text.split())  # Approximate token count
            }
        else:
            # Handle non-streaming response
            generated_text = response['choices'][0]['text'].strip()
            
            return {
                "text": generated_text,
                "tokens_used": response['usage']['total_tokens']
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "load_time": self.load_time,
            "inference_count": self.inference_count,
            "context_length": settings.model_context_length
        }
    
    async def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded successfully")
