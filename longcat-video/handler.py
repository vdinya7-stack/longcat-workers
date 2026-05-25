"""RunPod Serverless handler для LongCat Video (image-to-video).

Production-вариант: импортирует LongCatVideoPipeline напрямую и вызывает
методы пайплайна, без запуска/патчинга демоскрипта.

Логика инференса воспроизводит run_demo_image_to_video.py, но опускает
первую (нон-дистилл, 50 шагов) стадию: в демо её результат записывается в
output_i2v.mp4 и НЕ используется для refine — refine потребляет вывод distill.
Поэтому в проде остаётся честный путь: distill (16 шагов) -> refine (50 шагов).

Веса монтируются через RunPod Network Volume в /weights/LongCat-Video.
Пайплайн и LoRA загружаются ОДИН раз на холодном старте и переиспользуются
между запросами.

Input:
    {
        "image_url": str,
        "prompt": str,
        "num_frames": int,        # опционально, по умолчанию 93 (нативное значение)
        "resolution": "480p" | "720p"
    }

Output:
    {
        "video_url": str,
        "num_frames": int
    }
"""

import os
import sys
import time
import uuid
import logging

import boto3
import requests

# Пакет longcat_video лежит в директории репозитория, а не рядом с handler.py.
REPO_DIR = "/app/LongCat-Video"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import datetime
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

import runpod

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("longcat-video")

# Корень Network Volume. На RunPod Serverless том монтируется в /runpod-volume
# (на Pod — обычно /workspace). Переопределяется через env WEIGHTS_DIR.
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/runpod-volume")
CHECKPOINT_DIR = os.path.join(WEIGHTS_DIR, "LongCat-Video")
ENABLE_COMPILE = os.environ.get("LONGCAT_COMPILE", "1") == "1"
GLOBAL_SEED = 42
DEFAULT_NUM_FRAMES = 93  # нативное значение демо (~6с); кратно 4n+1

# --- R2 / Cloudflare ---------------------------------------------------------
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")

# Лениво инициализируемый синглтон пайплайна (загрузка один раз на воркер).
_PIPE = None
_LOCAL_RANK = 0


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def download_image(url: str, dst: str) -> str:
    """Скачивает входное изображение по URL в локальный файл."""
    log.info("Скачивание изображения: %s", url)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return dst


def upload_to_r2(local_path: str, key: str) -> str:
    """Загружает файл в R2 и возвращает публичный URL."""
    log.info("Загрузка в R2: bucket=%s key=%s", R2_BUCKET, key)
    client = _r2_client()
    client.upload_file(
        local_path,
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/{key}"
    return f"{R2_ENDPOINT}/{R2_BUCKET}/{key}"


def _init_distributed():
    """Single-process single-GPU окружение (handler запускается как `python`,
    а не через torchrun, поэтому переменные RANK/WORLD_SIZE задаём сами)."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=3600 * 24),
        )
    return local_rank


def get_pipeline():
    """Загружает пайплайн и LoRA один раз; повторные вызовы возвращают кэш."""
    global _PIPE, _LOCAL_RANK
    if _PIPE is not None:
        return _PIPE, _LOCAL_RANK

    log.info("Холодный старт: инициализация пайплайна LongCat-Video")
    t0 = time.time()

    local_rank = _init_distributed()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    init_context_parallel(
        context_parallel_size=1,
        global_rank=global_rank,
        world_size=world_size,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(
        CHECKPOINT_DIR, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_DIR, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        CHECKPOINT_DIR, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        CHECKPOINT_DIR, subfolder="dit", cp_split_hw=cp_split_hw,
        torch_dtype=torch.bfloat16)

    if ENABLE_COMPILE:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    # LoRA загружаем один раз; переключаем enable/disable по стадиям.
    pipe.dit.load_lora(
        os.path.join(CHECKPOINT_DIR, "lora/cfg_step_lora.safetensors"),
        "cfg_step_lora")
    pipe.dit.load_lora(
        os.path.join(CHECKPOINT_DIR, "lora/refinement_lora.safetensors"),
        "refinement_lora")
    pipe.dit.disable_all_loras()

    _PIPE, _LOCAL_RANK = pipe, local_rank
    log.info("Пайплайн готов за %.1f c", time.time() - t0)
    return _PIPE, _LOCAL_RANK


def generate_video(image: "PIL.Image.Image", prompt: str, resolution: str,
                   num_frames: int, output_path: str) -> None:
    """Distill (16 шагов) -> refine (50 шагов), запись в output_path."""
    pipe, local_rank = get_pipeline()

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(GLOBAL_SEED + dist.get_rank())
    target_size = image.size  # (width, height)

    # --- distill: даёт stage1_video для refine -------------------------------
    pipe.dit.disable_all_loras()
    pipe.dit.enable_loras(["cfg_step_lora"])
    output_distill = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=16,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
    )[0]
    pipe.dit.disable_all_loras()

    stage1_video = [(output_distill[i] * 255).astype(np.uint8)
                    for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # --- refine (720p) -------------------------------------------------------
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()
    output_refine = pipe.generate_refine(
        image=image,
        prompt=prompt,
        stage1_video=stage1_video,
        num_cond_frames=1,
        num_inference_steps=50,
        generator=generator,
        spatial_refine_only=False,
    )[0]
    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        frames = [(output_refine[i] * 255).astype(np.uint8)
                  for i in range(output_refine.shape[0])]
        frames = [PIL.Image.fromarray(img) for img in frames]
        frames = [f.resize(target_size, PIL.Image.BICUBIC) for f in frames]
        output_tensor = torch.from_numpy(np.array(frames))
        write_video(output_path, output_tensor, fps=30,
                    video_codec="libx264", options={"crf": "10"})


def handler(event):
    job_id = event.get("id", uuid.uuid4().hex)
    inp = event.get("input", {}) or {}

    image_url = inp.get("image_url")
    prompt = inp.get("prompt", "")
    num_frames = int(inp.get("num_frames", DEFAULT_NUM_FRAMES))
    resolution = inp.get("resolution", "480p")

    if not image_url:
        return {"error": "image_url is required"}
    if resolution not in ("480p", "720p"):
        return {"error": "resolution must be '480p' or '720p'"}
    if num_frames < 1:
        return {"error": "num_frames must be >= 1"}

    image_path = f"/tmp/{job_id}_input.jpg"
    output_path = f"/tmp/{job_id}.mp4"

    try:
        download_image(image_url, image_path)
        # load_image из diffusers принимает путь/URL; используем PIL напрямую.
        image = PIL.Image.open(image_path).convert("RGB")

        start = time.time()
        generate_video(image, prompt, resolution, num_frames, output_path)
        log.info("Инференс завершён за %.1f c", time.time() - start)

        if not os.path.exists(output_path):
            return {"error": f"output not found at {output_path}"}

        video_url = upload_to_r2(output_path, f"{job_id}.mp4")
        return {"video_url": video_url, "num_frames": num_frames}
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
