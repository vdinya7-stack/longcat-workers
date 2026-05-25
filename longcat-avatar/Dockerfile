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

# FlashAttention-2 собирается из исходников и прожорлив по памяти.
# ninja ускоряет сборку, MAX_JOBS ограничивает параллелизм, чтобы не словить OOM
# на CI-раннере (docker-in-docker).
ENV MAX_JOBS=4
RUN pip install --no-cache-dir packaging ninja wheel
RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY handler.py /app/handler.py

# Веса НЕ копируются в образ — монтируются с Network Volume
# (RunPod Serverless: /runpod-volume; путь задаётся env WEIGHTS_DIR).

CMD ["python", "handler.py"]
