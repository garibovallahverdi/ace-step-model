from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import torch
import os
import sys
import logging
import traceback
import uvicorn
from contextlib import asynccontextmanager

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import handler
from handler import handler as runpod_handler

app = FastAPI(title="Music Generation API", version="1.0.0")

# Request model
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Music description text")
    duration: int = Field(default=10, ge=1, le=30, description="Duration in seconds")

# Response model
class GenerateResponse(BaseModel):
    status: str
    url: Optional[str] = None
    filename: Optional[str] = None
    prompt: Optional[str] = None
    duration: Optional[int] = None
    message: Optional[str] = None
    error_type: Optional[str] = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate music from text prompt
    """
    logger.info(f"Received request: prompt={request.prompt[:50]}... duration={request.duration}")
    
    try:
        # Create job input
        job_input = {
            "id": f"req_{os.urandom(4).hex()}",
            "input": {
                "prompt": request.prompt,
                "duration": request.duration
            }
        }
        
        # Call the handler
        result = runpod_handler(job_input)
        
        if result.get("status") == "success":
            return GenerateResponse(**result)
        else:
            return GenerateResponse(
                status="error",
                message=result.get("message", "Unknown error"),
                error_type=result.get("error_type")
            )
            
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return GenerateResponse(
            status="error",
            message=str(e),
            error_type=type(e).__name__
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
