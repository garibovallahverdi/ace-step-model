# handler.py - Dtype hatası düzəldilmiş versiya
import os
import io
import uuid
import runpod
import torch
import torchaudio
from diffusers import AudioLDM2Pipeline
import boto3
from botocore.config import Config
import logging
import sys
import traceback
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/handler_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
BUCKET_NAME = os.getenv("BUCKET_NAME", "music")
REGION_NAME = os.getenv("REGION_NAME", "ap-southeast-2")

# Log environment
logger.info("=== ENVIRONMENT VARIABLES ===")
logger.info(f"S3_ENDPOINT: {S3_ENDPOINT}")
logger.info(f"BUCKET_NAME: {BUCKET_NAME}")
logger.info(f"REGION_NAME: {REGION_NAME}")
logger.info("S3_ACCESS_KEY: SET" if S3_ACCESS_KEY else "S3_ACCESS_KEY: NOT SET")
logger.info("S3_SECRET_KEY: SET" if S3_SECRET_KEY else "S3_SECRET_KEY: NOT SET")
logger.info("==============================")

# Initialize S3 client
try:
    logger.info("Initializing S3 client...")
    s3_client = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=REGION_NAME,
        config=Config(signature_version='s3v4')
    )
    logger.info("✅ S3 client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize S3 client: {e}")
    raise

def ensure_bucket_exists():
    """Check and create bucket if needed"""
    try:
        logger.info(f"Checking bucket: {BUCKET_NAME}")
        existing_buckets = s3_client.list_buckets()
        bucket_names = [b['Name'] for b in existing_buckets['Buckets']]
        
        if BUCKET_NAME not in bucket_names:
            logger.info(f"Creating bucket: {BUCKET_NAME}")
            s3_client.create_bucket(Bucket=BUCKET_NAME)
            s3_client.put_bucket_acl(Bucket=BUCKET_NAME, ACL='public-read')
            logger.info(f"✅ Created bucket: {BUCKET_NAME}")
        else:
            logger.info(f"✅ Bucket already exists: {BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Error with bucket: {e}")
        raise

def load_model():
    """Load model with proper dtype handling"""
    try:
        logger.info("🚀 Loading AudioLDM2 model...")
        
        repo_id = "cvssp/audioldm2-music"
        
        # Force float32 to avoid dtype issues
        torch_dtype = torch.float32
        
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            # Use float16 on GPU for better performance
            torch_dtype = torch.float16
            pipe = AudioLDM2Pipeline.from_pretrained(
                repo_id, 
                torch_dtype=torch_dtype
            )
            pipe.to("cuda")
            logger.info("✅ Model loaded on GPU")
        else:
            logger.warning("No GPU available, using CPU with float32")
            pipe = AudioLDM2Pipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float32
            )
            pipe.to("cpu")
            logger.info("✅ Model loaded on CPU")
        
        # Ensure all sub-models use same dtype
        pipe.vae = pipe.vae.to(dtype=torch_dtype)
        pipe.unet = pipe.unet.to(dtype=torch_dtype)
        
        # Enable memory efficiency
        pipe.enable_model_cpu_offload() if not torch.cuda.is_available() else None
        
        logger.info("✅ Model ready!")
        return pipe
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise

def save_to_supabase_storage(audio_tensor, sample_rate, filename):
    """Upload audio to Supabase Storage"""
    try:
        logger.info(f"Saving audio: {filename}")
        
        # Convert to proper dtype and range
        audio_tensor = audio_tensor.float()  # Ensure float32
        
        # Normalize to [-1, 1] range if needed
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav", backend="soundfile")
        buffer.seek(0)
        audio_bytes = buffer.getvalue()
        
        logger.info(f"Audio size: {len(audio_bytes)} bytes")
        
        # Upload to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=audio_bytes,
            ContentType='audio/wav',
            ACL='public-read'
        )
        
        # Generate public URL
        public_url = f"https://{BUCKET_NAME}.supabase.co/storage/v1/object/public/{BUCKET_NAME}/{filename}"
        
        logger.info(f"✅ Uploaded: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        logger.error(traceback.format_exc())
        raise

# Initialize
logger.info("="*60)
logger.info("Starting RunPod worker initialization...")
logger.info("="*60)

try:
    logger.info("Step 1: Ensuring bucket exists...")
    ensure_bucket_exists()
    
    logger.info("Step 2: Loading model...")
    model_pipe = load_model()
    
    logger.info("✅ Initialization complete!")
    
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    raise

def handler(job):
    """RunPod handler"""
    job_id = job.get('id', 'unknown')
    logger.info(f"="*60)
    logger.info(f"Processing job: {job_id}")
    logger.info(f"="*60)
    
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("prompt", job_input.get("text", "Lofi hip hop beat, calm and relaxing"))
        duration = min(job_input.get("duration", 10), 30)  # Max 30 seconds
        
        logger.info(f"📝 Prompt: {prompt[:100]}")
        logger.info(f"⏱️ Duration: {duration}s")
        
        # Generate music
        logger.info("🎵 Generating music...")
        
        with torch.inference_mode():
            # Ensure inputs are correct dtype
            result = model_pipe(
                prompt,
                audio_length_in_s=duration,
                num_inference_steps=30,  # Reduced for speed
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42)  # Consistent results
            )
            
            audio_output = result.audios[0]
            
            # Convert to tensor with proper dtype
            if isinstance(audio_output, np.ndarray):
                audio_tensor = torch.from_numpy(audio_output).float()
            else:
                audio_tensor = audio_output.float() if torch.is_tensor(audio_output) else torch.from_numpy(audio_output).float()
            
            # Ensure correct shape [channels, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            logger.info(f"Audio shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")
        
        sample_rate = 16000
        
        # Create filename
        file_name = f"music_{uuid.uuid4().hex[:8]}.wav"
        
        # Upload
        public_url = save_to_supabase_storage(audio_tensor, sample_rate, file_name)
        
        result = {
            "status": "success",
            "url": public_url,
            "filename": file_name,
            "prompt": prompt,
            "duration": duration,
            "sample_rate": sample_rate
        }
        
        logger.info(f"✅ Job completed: {job_id}")
        logger.info(f"🔊 Audio URL: {public_url}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in job {job_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
