# LIG Docker image (GPU)

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# PyTorch (CUDA) then package
COPY pyproject.toml README.md LICENSE ./
COPY lig/ lig/
COPY utils/ utils/
COPY scripts/ops/ scripts/ops/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio \
         --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -e .

# Hugging Face cache (mount a volume here in production)
ENV HF_HOME=/cache/huggingface

ENTRYPOINT ["lig"]
CMD ["--help"]
