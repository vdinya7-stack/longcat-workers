"""RunPod Serverless handler для LongCat Avatar 1.5 (audio + image → video).

Модель: meituan-longcat/LongCat-Video-Avatar-1.5
Веса монтируются через RunPod Network Volume в /weights/LongCat-Video-Avatar-1.5.

ВАЖНО (сверено с официальным репозиторием):
Скрипт run_demo_avatar_single_audio_to_video.py НЕ принимает image/audio/prompt
как отдельные CLI-аргументы. Входные данные читаются из JSON-файла, путь к
которому передаётся через --input_json. Структура JSON:
    {
        "prompt": str,
        "cond_image": "<путь к изображению>",
        "cond_audio": {"person1": "<путь к аудио>"}
    }
Результат сохраняется в --output_dir с фиксированным именем "ai2v_demo_1.mp4"
(stage_1 по умолчанию 'ai2v'). Аргумента --output_path / --audio_path нет.

Длительность задаётся напрямую числом сегментов (--num_segments — реальный
флаг скрипта). Каждый сегмент при avatar-v1.5 (fps=25, 93 кадра, 13 cond) ≈
3.7с для первого и ≈3.2с для каждого следующего. Никакого приближённого
пересчёта из секунд не делаем.

Input:
    {
        "image_url": str,
        "audio_url": str,
        "prompt": str,
        "num_segments": int,             # число сегментов (>=1), по умолчанию 1
        "resolution": "480p" | "720p"    # опционально, по умолчанию 480p
    }

Output:
    {
        "video_url": str
    }
"""

import os
import json
import time
import uuid
import shutil
import logging
import subprocess

import boto3
import requests
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("longcat-avatar")

REPO_DIR = "/app/LongCat-Video"
# Корень Network Volume. На RunPod Serverless том монтируется в /runpod-volume
# (на Pod — обычно /workspace). Переопределяется через env WEIGHTS_DIR.
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/runpod-volume")
CHECKPOINT_DIR = os.path.join(WEIGHTS_DIR, "LongCat-Video-Avatar-1.5")
DEMO_SCRIPT = "run_demo_avatar_single_audio_to_video.py"

# --- R2 / Cloudflare ---------------------------------------------------------
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY")
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        region_name="auto",
    )


def download_file(url: str, dst: str) -> str:
    """Скачивает файл по URL в локальный путь."""
    log.info("Скачивание: %s", url)
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


def run_inference(input_json: str, output_dir: str, resolution: str,
                  num_segments: int) -> None:
    """Запускает torchrun с avatar-демоскриптом на одной GPU.

    Имена флагов и их наличие сверены с _parse_args() официального скрипта.
    """
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        DEMO_SCRIPT,
        f"--checkpoint_dir={CHECKPOINT_DIR}",
        f"--input_json={input_json}",
        f"--output_dir={output_dir}",
        f"--resolution={resolution}",
        f"--num_segments={num_segments}",
        "--use_distill",
        "--model_type", "avatar-v1.5",
        "--use_int8",
        "--context_parallel_size=1",
    ]
    log.info("Запуск инференса: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def handler(event):
    job_id = event.get("id", uuid.uuid4().hex)
    inp = event.get("input", {}) or {}

    image_url = inp.get("image_url")
    audio_url = inp.get("audio_url")
    prompt = inp.get("prompt", "")
    num_segments = int(inp.get("num_segments", 1))
    resolution = inp.get("resolution", "480p")

    if not image_url:
        return {"error": "image_url is required"}
    if not audio_url:
        return {"error": "audio_url is required"}
    if resolution not in ("480p", "720p"):
        return {"error": "resolution must be '480p' or '720p'"}
    if num_segments < 1:
        return {"error": "num_segments must be >= 1"}

    image_path = f"/tmp/{job_id}_input.jpg"
    audio_path = f"/tmp/{job_id}_input.wav"
    input_json = f"/tmp/{job_id}_input.json"
    output_dir = f"/tmp/{job_id}_out"
    # save_video_ffmpeg сохраняет в <output_dir>/ai2v_demo_1.mp4 (stage_1='ai2v').
    output_path = os.path.join(output_dir, "ai2v_demo_1.mp4")
    os.makedirs(output_dir, exist_ok=True)

    try:
        download_file(image_url, image_path)
        download_file(audio_url, audio_path)

        # Демоскрипт читает входные данные из JSON, а не из CLI.
        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "cond_image": image_path,
                    "cond_audio": {"person1": audio_path},
                },
                f,
                ensure_ascii=False,
            )

        log.info("num_segments=%s", num_segments)

        start = time.time()
        run_inference(input_json, output_dir, resolution, num_segments)
        elapsed = time.time() - start
        log.info("Инференс завершён за %.1f c", elapsed)

        if not os.path.exists(output_path):
            return {"error": f"output not found at {output_path}"}

        video_url = upload_to_r2(output_path, f"{job_id}.mp4")
        return {"video_url": video_url}
    except subprocess.CalledProcessError as e:
        log.exception("Ошибка инференса")
        return {"error": f"inference failed: {e}"}
    except Exception as e:  # noqa: BLE001
        log.exception("Ошибка обработки задачи")
        return {"error": str(e)}
    finally:
        for p in (image_path, audio_path, input_json):
            try:
                os.remove(p)
            except OSError:
                pass
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
