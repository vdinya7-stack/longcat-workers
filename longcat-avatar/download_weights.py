"""Разовое скачивание весов LongCat на RunPod Network Volume.

Запускается ОДИН РАЗ на временном RunPod Pod, к которому примонтирован
Network Volume в /weights. После завершения Pod можно удалить — веса
останутся на томе и будут монтироваться в serverless-воркеры.

Использование:
    pip install -U "huggingface_hub[cli]" hf_transfer
    python download_weights.py
"""

import os
import subprocess

# Корень Network Volume. На временном Pod том монтируется в /workspace.
# ВАЖНО: это должен быть ТОТ ЖЕ том, что монтируется в serverless-воркеры
# (там он появляется как /runpod-volume) — тогда подпапки LongCat-* совпадут.
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/workspace")

MODELS = [
    ("meituan-longcat/LongCat-Video",
     os.path.join(WEIGHTS_DIR, "LongCat-Video")),
    ("meituan-longcat/LongCat-Video-Avatar-1.5",
     os.path.join(WEIGHTS_DIR, "LongCat-Video-Avatar-1.5")),
]


def download(repo_id: str, local_dir: str) -> None:
    print(f"==> Скачивание {repo_id} -> {local_dir}")
    subprocess.run(
        [
            "huggingface-cli", "download", repo_id,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks", "False",
        ],
        check=True,
    )


if __name__ == "__main__":
    for repo_id, local_dir in MODELS:
        download(repo_id, local_dir)
    print("Готово. Все веса загружены на Network Volume.")
