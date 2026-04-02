# Dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Python quraşdır
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10-u default edin
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# İş qovluğu
WORKDIR /app

# Dependency-ləri kopyala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Faylları kopyala
COPY handler.py .

# Port aç
EXPOSE 8000

# RunPod handler
CMD ["python", "-u", "handler.py"]
