"""RunPod Serverless handler для LTX-2.3 (Lightricks) image-to-video.

Контракт ввода/вывода СОВПАДАЕТ с провайдером MaxGravity
(backend/src/services/providers/ltx.ts), который шлёт payload внутри `input`:

Input:
    {
        "image_url": str,            # исходный кадр (i2v)
        "prompt": str,
        "duration": int,             # секунды (бэкенд шлёт 5 или 10)
        "resolution": "480p" | "720p" | "1080p"
    }

Output:
    { "video_url": str }

Веса LTX-2.3 читаются из WEIGHTS_DIR (по умолчанию /workspace/LTX-2.3 —
смонтированный RunPod Network Volume). Если каталога нет — diffusers скачает
их с HF Hub (env LTX_MODEL_REPO). Пайплайн грузится ОДИН раз на холодном
старте и переиспользуется между запросами.

ВНИМАНИЕ: вход содержит image_url (image-to-video), поэтому используется
documented diffusers-класс `LTXImageToVideoPipeline`. (В diffusers нет класса
с именем `LTXVideoPipeline`; t2v-путь — это `LTXPipeline`, condition-путь
LTX-2 — `LTXConditionPipeline`.) Контракт handler'а от выбора класса не зависит.
"""

import os
import time
import uuid
import logging

import boto3
import requests
import torch
import PIL.Image

import runpod
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ltx-video")

# Каталог весов. По умолчанию /workspace/LTX-2.3 — каталог на смонтированном
# RunPod Network Volume (на serverless том обычно монтируется в /runpod-volume,
# поэтому при необходимости переопределите WEIGHTS_DIR через env).
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/workspace/LTX-2.3")
# HF-репозиторий весов (fallback, если каталога нет). Задайте точный id для 2.3/22B.
MODEL_REPO = os.environ.get("LTX_MODEL_REPO", "Lightricks/LTX-Video")
NUM_STEPS = int(os.environ.get("LTX_STEPS", "40"))
FPS = int(os.environ.get("LTX_FPS", "24"))
GLOBAL_SEED = int(os.environ.get("LTX_SEED", "42"))

# LTX требует ширину/высоту кратными 32 и num_frames == 8*k + 1.
RES_MAP = {
    "480p": (832, 480),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}

# --- R2 / Cloudflare (тот же контракт вывода, что у longcat-video) -----------
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")

# Лениво инициализируемый синглтон пайплайна (загрузка один раз на воркер).
_PIPE = None


def _model_path() -> str:
    """Локальные веса (WEIGHTS_DIR) в приоритете; иначе тянем с HF Hub."""
    return WEIGHTS_DIR if os.path.isdir(WEIGHTS_DIR) else MODEL_REPO


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def download_image(url: str, dst: str) -> str:
    log.info("Скачивание изображения: %s", url)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dst


def upload_to_r2(local_path: str, key: str) -> str:
    log.info("Загрузка в R2: bucket=%s key=%s", R2_BUCKET, key)
    client = _r2_client()
    client.upload_file(local_path, R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/{key}"
    return f"{R2_ENDPOINT}/{R2_BUCKET}/{key}"


def get_pipeline() -> LTXImageToVideoPipeline:
    """Загружает пайплайн один раз; повторные вызовы возвращают кэш."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    log.info("Холодный старт: загрузка LTX из %s", _model_path())
    t0 = time.time()
    pipe = LTXImageToVideoPipeline.from_pretrained(_model_path(), torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    # Тайлинг VAE снижает пик VRAM на высоких разрешениях (22B — крупная модель).
    pipe.vae.enable_tiling()
    _PIPE = pipe
    log.info("Пайплайн готов за %.1f c", time.time() - t0)
    return _PIPE


def _num_frames(duration_s: float) -> int:
    """duration (сек) → num_frames, приведённое к виду 8*k + 1 (требование LTX)."""
    n = max(1, int(round(duration_s * FPS)))
    n = ((n - 1) // 8) * 8 + 1
    return max(9, n)


def generate_video(image: "PIL.Image.Image", prompt: str, resolution: str,
                   duration_s: float, output_path: str) -> int:
    pipe = get_pipeline()
    width, height = RES_MAP[resolution]
    num_frames = _num_frames(duration_s)
    generator = torch.Generator(device="cuda").manual_seed(GLOBAL_SEED)

    result = pipe(
        image=image,
        prompt=prompt or "high quality, detailed, cinematic",
        negative_prompt="worst quality, blurry, distorted, low resolution",
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=NUM_STEPS,
        generator=generator,
    )
    frames = result.frames[0]
    export_to_video(frames, output_path, fps=FPS)
    return num_frames


def handler(event):
    job_id = event.get("id", uuid.uuid4().hex)
    inp = event.get("input", {}) or {}

    image_url = inp.get("image_url")
    prompt = inp.get("prompt", "")
    duration = float(inp.get("duration", 5))
    resolution = inp.get("resolution", "720p")

    if not image_url:
        return {"error": "image_url is required"}
    if resolution not in RES_MAP:
        return {"error": "resolution must be one of 480p / 720p / 1080p"}
    if duration <= 0:
        return {"error": "duration must be > 0"}

    image_path = f"/tmp/{job_id}_input.jpg"
    output_path = f"/tmp/{job_id}.mp4"

    try:
        download_image(image_url, image_path)
        image = PIL.Image.open(image_path).convert("RGB")

        start = time.time()
        num_frames = generate_video(image, prompt, resolution, duration, output_path)
        log.info("Инференс завершён за %.1f c (%d кадров)", time.time() - start, num_frames)

        if not os.path.exists(output_path):
            return {"error": f"output not found at {output_path}"}

        video_url = upload_to_r2(output_path, f"{job_id}.mp4")
        return {"video_url": video_url}
    except Exception as e:  # noqa: BLE001
        log.exception("Ошибка обработки задачи")
        return {"error": str(e)}
    finally:
        for p in (image_path, output_path):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
