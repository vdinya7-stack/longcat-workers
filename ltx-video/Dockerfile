FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# ffmpeg нужен для записи mp4 (diffusers export_to_video через imageio-ffmpeg).
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Зависимости (diffusers, transformers, accelerate, runpod, boto3, requests + ...).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Веса LTX-2.3 НЕ копируются в образ — монтируются с RunPod Network Volume
# (каталог задаётся env WEIGHTS_DIR, по умолчанию /workspace/LTX-2.3).

CMD ["python", "handler.py"]
