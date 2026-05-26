"""Разовое скачивание весов LTX-2.3 на RunPod Network Volume.

Запускается ОДИН РАЗ на временном RunPod Pod, к которому примонтирован
Network Volume. После завершения Pod можно удалить — веса останутся на томе
и будут монтироваться в serverless-воркеры.

Использование:
    pip install -U "huggingface_hub[cli]" hf_transfer
    python download_weights.py

ВАЖНО: каталог назначения должен совпадать с WEIGHTS_DIR воркера
(по умолчанию /workspace/LTX-2.3 на Pod → /runpod-volume/... на serverless,
если том монтируется иначе — задайте WEIGHTS_DIR одинаково в обоих местах).
"""

import os
import subprocess

# Точный HF-репозиторий весов LTX-2.3 (22B). Переопределяется через env.
MODEL_REPO = os.environ.get("LTX_MODEL_REPO", "Lightricks/LTX-Video")
# Каталог назначения на томе (тот же, что читает handler через WEIGHTS_DIR).
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/workspace/LTX-2.3")


def download(repo_id: str, local_dir: str) -> None:
    print(f"==> Скачивание {repo_id} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    subprocess.run(
        [
            "huggingface-cli", "download", repo_id,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks", "False",
        ],
        check=True,
    )


if __name__ == "__main__":
    download(MODEL_REPO, WEIGHTS_DIR)
    print("Готово. Веса LTX-2.3 загружены на Network Volume.")
