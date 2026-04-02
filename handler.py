import os
import io
import uuid
import torch
import torchaudio
from diffusers import AudioLDM2Pipeline
import boto3
from botocore.config import Config
import logging
import traceback
from dotenv import load_dotenv
import numpy as np

load_dotenv()

logger = logging.getLogger(__name__)

# Environment variables
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
BUCKET_NAME = os.getenv("BUCKET_NAME", "music")
REGION_NAME = os.getenv("REGION_NAME", "ap-southeast-2")

# S3 client
s3_client = None
model_pipe = None

def initialize():
    """Initialize model and S3 client (called once at startup)"""
    global s3_client, model_pipe
    
    logger.info("Initializing S3 client...")
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=REGION_NAME,
        config=Config(signature_version='s3v4')
    )
    
    logger.info("Loading AudioLDM2 model...")
    repo_id = "cvssp/audioldm2-music"
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        model_pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        model_pipe.to("cuda")
    else:
        logger.warning("No GPU, using CPU")
        model_pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
        model_pipe.to("cpu")
    
    logger.info("✅ Model and S3 ready!")

def save_to_storage(audio_tensor, sample_rate, filename):
    """Upload audio to Supabase Storage"""
    buffer = io.BytesIO()
    audio_tensor = audio_tensor.float()
    
    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / audio_tensor.abs().max()
    
    torchaudio.save(buffer, audio_tensor, sample_rate, format="wav", backend="soundfile")
    buffer.seek(0)
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=filename,
        Body=buffer.getvalue(),
        ContentType='audio/wav',
        ACL='public-read'
    )
    
    return f"https://{BUCKET_NAME}.supabase.co/storage/v1/object/public/{BUCKET_NAME}/{filename}"

def handler(job):
    """Main handler function"""
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("prompt", job_input.get("text", "Lofi hip hop beat"))
        duration = min(job_input.get("duration", 10), 30)
        
        logger.info(f"🎵 Generating: {prompt[:50]}...")
        
        with torch.inference_mode():
            result = model_pipe(
                prompt,
                audio_length_in_s=duration,
                num_inference_steps=30,
                guidance_scale=7.5
            )
            audio_output = result.audios[0]
        
        if isinstance(audio_output, np.ndarray):
            audio_tensor = torch.from_numpy(audio_output).float()
        else:
            audio_tensor = audio_output.float()
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        filename = f"music_{uuid.uuid4().hex[:8]}.wav"
        public_url = save_to_storage(audio_tensor, 16000, filename)
        
        return {
            "status": "success",
            "url": public_url,
            "filename": filename,
            "prompt": prompt,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

# Initialize on module load
initialize()
