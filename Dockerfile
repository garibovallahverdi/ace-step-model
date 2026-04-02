FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Öncə pip-i yenilə
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA 11.8 üçün PyTorch (ayrıca)
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Qalan dependency-lər
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

EXPOSE 8000

CMD ["python", "-u", "handler.py"]
