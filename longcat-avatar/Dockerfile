FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# Системные зависимости и git.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Официальный репозиторий LongCat Video (содержит avatar-демоскрипты).
RUN git clone https://github.com/meituan-longcat/LongCat-Video

# Зависимости проекта.
RUN pip install --no-cache-dir -r LongCat-Video/requirements.txt

# Зависимости хендлера (RunPod, R2, HTTP).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# FlashAttention-2 ставим из официального ПРЕДСОБРАННОГО wheel.
# runtime-образ не содержит nvcc, поэтому сборка из исходников невозможна;
# wheel под torch2.4 / cu12x / cp311 (ABI=FALSE, как у pip-сборок torch) не
# требует ни компилятора, ни nvcc и держит образ компактным.
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

COPY handler.py /app/handler.py

# Веса НЕ копируются в образ — монтируются с Network Volume
# (RunPod Serverless: /runpod-volume; путь задаётся env WEIGHTS_DIR).

CMD ["python", "handler.py"]
