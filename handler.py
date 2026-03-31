# handler.py - DEBUG VERSION
import os
import io
import uuid
import runpod
import torch
import torchaudio
from diffusers import AudioLDM2Pipeline
from supabase import create_client, Client
import logging
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/handler_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Log environment variables (without sensitive data)
logger.info("=== ENVIRONMENT VARIABLES ===")
logger.info(f"SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')}")
logger.info(f"BUCKET_NAME: {os.getenv('BUCKET_NAME', 'NOT SET')}")
logger.info(f"REGION_NAME: {os.getenv('REGION_NAME', 'NOT SET')}")
logger.info(f"S3_ENDPOINT: {os.getenv('S3_ENDPOINT', 'NOT SET')}")
logger.info("SUPABASE_KEY: SET" if os.getenv('SUPABASE_KEY') else "SUPABASE_KEY: NOT SET")
logger.info("S3_ACCESS_KEY: SET" if os.getenv('S3_ACCESS_KEY') else "S3_ACCESS_KEY: NOT SET")
logger.info("S3_SECRET_KEY: SET" if os.getenv('S3_SECRET_KEY') else "S3_SECRET_KEY: NOT SET")
logger.info("==============================")

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "music")

# Supabase client
try:
    logger.info("Initializing Supabase client...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Supabase client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase client: {e}")
    logger.error(traceback.format_exc())
    raise

def ensure_bucket_exists():
    """Storage bucket-in mövcud olduğunu yoxla"""
    try:
        logger.info(f"Checking bucket: {BUCKET_NAME}")
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        
        if BUCKET_NAME not in bucket_names:
            logger.info(f"Creating bucket: {BUCKET_NAME}")
            supabase.storage.create_bucket(BUCKET_NAME, {'public': True})
            logger.info(f"✅ Created bucket: {BUCKET_NAME}")
        else:
            logger.info(f"✅ Bucket already exists: {BUCKET_NAME}")
            
    except Exception as e:
        logger.error(f"Error checking/creating bucket: {e}")
        logger.error(traceback.format_exc())
        raise

def load_model():
    """Modeli yüklə"""
    try:
        logger.info("🚀 ACE-Step (AudioLDM2) Model yüklənir...")
        
        repo_id = "cvssp/audioldm2-music"
        logger.info(f"Loading model from: {repo_id}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            pipe = AudioLDM2Pipeline.from_pretrained(
                repo_id, 
                torch_dtype=torch.float16
            )
            pipe.to("cuda")
            logger.info("✅ Model loaded on GPU (float16)")
        else:
            logger.warning("No GPU available, using CPU (this will be slow)")
            pipe = AudioLDM2Pipeline.from_pretrained(repo_id)
            pipe.to("cpu")
            logger.info("✅ Model loaded on CPU (float32)")
        
        logger.info("✅ Model loaded successfully!")
        return pipe
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise

def save_to_supabase_storage(audio_tensor, sample_rate, filename):
    """Audionu Supabase Storage-a yüklə"""
    try:
        logger.info(f"Saving audio to Supabase Storage: {filename}")
        
        # Audio-nu WAV formatında bytes-a çevir
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav", backend="soundfile")
        buffer.seek(0)
        audio_bytes = buffer.getvalue()
        
        logger.info(f"Audio size: {len(audio_bytes)} bytes")
        
        # Upload to Supabase Storage
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            audio_bytes,
            {"content-type": "audio/wav"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        
        logger.info(f"✅ Uploaded successfully: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Failed to upload to Supabase Storage: {e}")
        logger.error(traceback.format_exc())
        raise

# Initialize model and bucket
logger.info("=" * 60)
logger.info("Starting RunPod worker initialization...")
logger.info("=" * 60)

try:
    logger.info("Step 1: Ensuring bucket exists...")
    ensure_bucket_exists()
    
    logger.info("Step 2: Loading model...")
    model_pipe = load_model()
    
    logger.info("✅ All initialization completed successfully!")
    
except Exception as e:
    logger.error(f"❌ Initialization failed: {e}")
    logger.error(traceback.format_exc())
    raise

def handler(job):
    """RunPod handler function"""
    job_id = job.get('id', 'unknown')
    logger.info(f"=" * 60)
    logger.info(f"Received job: {job_id}")
    logger.info(f"=" * 60)
    
    try:
        job_input = job.get('input', {})
        prompt = job_input.get("text", "Lofi hip hop beat, calm and relaxing")
        duration = job_input.get("duration", 10)
        
        logger.info(f"📝 Prompt: {prompt[:100]}...")
        logger.info(f"⏱️ Duration: {duration} seconds")
        
        # Generate music
        logger.info("🎵 Generating music...")
        
        with torch.inference_mode():
            logger.info("Running model inference...")
            audio_output = model_pipe(
                prompt, 
                audio_length_in_s=duration, 
                num_inference_steps=50,
                guidance_scale=7.5
            ).audios[0]
        
        logger.info(f"Audio generated, shape: {audio_output.shape}")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_output).unsqueeze(0)
        sample_rate = 16000
        
        logger.info(f"Audio tensor shape: {audio_tensor.shape}, sample rate: {sample_rate}")
        
        # Create filename
        file_name = f"ace_{uuid.uuid4()}.wav"
        logger.info(f"Generated filename: {file_name}")
        
        # Upload to Supabase Storage
        public_url = save_to_supabase_storage(audio_tensor, sample_rate, file_name)
        
        # Response
        result = {
            "status": "success",
            "url": public_url,
            "filename": file_name,
            "prompt": prompt,
            "duration": duration,
            "sample_rate": sample_rate
        }
        
        logger.info(f"✅ Job completed successfully: {job_id}")
        logger.info(f"🔊 Audio URL: {public_url}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in handler for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Xəta: {str(e)}",
            "error_type": type(e).__name__
        }

# RunPod serverless start
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
