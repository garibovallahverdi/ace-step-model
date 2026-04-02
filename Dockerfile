# Dockerfile - Official NVIDIA CUDA 11.8
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Python və digər paketlər
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10-u default et
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# pip-i yenilə
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CUDA 11.8 versiyasını quraşdır
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Dependency-ləri kopyala və quraşdır
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Handler faylını kopyala
COPY handler.py .

EXPOSE 8000

CMD ["python", "-u", "handler.py"]
