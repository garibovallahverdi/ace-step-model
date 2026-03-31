# handler.py
import os
import io
import uuid
import runpod
import torch
import torchaudio
from diffusers import AudioLDM2Pipeline
from supabase import create_client, Client
import logging
from dotenv import load_dotenv

# Environment variables-ı yüklə
load_dotenv()

# Logging konfiqurasiyası
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "music")

# Supabase Storage üçün S3 uyğunluğu
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
REGION_NAME = os.getenv("REGION_NAME", "ap-southeast-2")

# Supabase client (verilənlər bazası üçün lazım deyil, ancaq storage üçün)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def ensure_bucket_exists():
    """Storage bucket-in mövcud olduğunu yoxla"""
    try:
        # Bucket-ləri siyahıla
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        
        if BUCKET_NAME not in bucket_names:
            # Bucket yoxdursa yarat (public olaraq)
            supabase.storage.create_bucket(BUCKET_NAME, {'public': True})
            logger.info(f"✅ Created bucket: {BUCKET_NAME}")
        else:
            logger.info(f"✅ Bucket already exists: {BUCKET_NAME}")
            
    except Exception as e:
        logger.error(f"Error checking/creating bucket: {e}")
        raise

# 1. Modelin Yüklənməsi (GPU-ya köçürülür)
def load_model():
    logger.info("🚀 ACE-Step (AudioLDM2) Model yüklənir... Bu bir az vaxt ala bilər.")
    
    # ACE-Step 1.5 üçün ən stabil model
    repo_id = "cvssp/audioldm2-music"
    
    # GPU varsa float16 istifadə et
    if torch.cuda.is_available():
        pipe = AudioLDM2Pipeline.from_pretrained(
            repo_id, 
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
        logger.info("✅ Model GPU-ya yükləndi (float16)")
    else:
        pipe = AudioLDM2Pipeline.from_pretrained(repo_id)
        pipe.to("cpu")
        logger.info("✅ Model CPU-ya yükləndi (float32)")
    
    logger.info("✅ Model tam yükləndi!")
    return pipe

# Modeli bir dəfə global olaraq yükləyirik
logger.info("Initializing model...")
model_pipe = load_model()

# Bucket-in mövcudluğunu yoxla
ensure_bucket_exists()

def save_to_supabase_storage(audio_buffer, filename):
    """Audionu Supabase Storage-a yüklə"""
    try:
        # Buffer-ı bytes-a çevir
        audio_bytes = audio_buffer.getvalue()
        
        # Supabase Storage-a yüklə
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            audio_bytes,
            {"content-type": "audio/wav"}
        )
        
        # Public URL generasiya et
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        
        logger.info(f"✅ Uploaded to Supabase Storage: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"Error uploading to Supabase Storage: {e}")
        raise

def handler(job):
    try:
        job_input = job['input']
        prompt = job_input.get("text", "Lofi hip hop beat, calm and relaxing")
        duration = job_input.get("duration", 10)  # Saniyə
        
        logger.info(f"🎵 Generating music for prompt: {prompt[:50]}... (duration: {duration}s)")
        
        # 2. Musiqi Yaratma Prosesi (Real Model İşi)
        with torch.inference_mode():
            # audio_length_in_s modelin neçə saniyəlik səs yaradacağını təyin edir
            audio_output = model_pipe(
                prompt, 
                audio_length_in_s=duration, 
                num_inference_steps=50,  # Keyfiyyət üçün 50 addım
                guidance_scale=7.5  # Prompt-a uyğunluq
            ).audios[0]

        # Numpy massivini Tensor-a çeviririk
        audio_tensor = torch.from_numpy(audio_output).unsqueeze(0)
        sample_rate = 16000  # AudioLDM2 adətən 16kHz çıxış verir

        # 3. Səsi RAM-da WAV kimi hazırlamaq
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav", backend="soundfile")
        buffer.seek(0)

        # 4. Unikal Fayl Adı
        file_name = f"ace_{uuid.uuid4()}.wav"
        
        # 5. Supabase Storage-a Yükləmə
        public_url = save_to_supabase_storage(buffer, file_name)
        
        logger.info(f"✅ Audio generated and saved successfully: {file_name}")
        
        # Response
        return {
            "status": "success",
            "url": public_url,
            "filename": file_name,
            "prompt": prompt,
            "duration": duration,
            "sample_rate": sample_rate
        }

    except Exception as e:
        logger.error(f"❌ Error in handler: {str(e)}")
        return {
            "status": "error", 
            "message": f"Xəta: {str(e)}"
        }

# RunPod serverless start
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})