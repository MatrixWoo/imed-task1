FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch_cache \
    XDG_CACHE_HOME=/opt/xdg_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade build tooling — Ubuntu 22.04's apt-shipped setuptools is too old
# to parse modern pyproject.toml [project] tables (causes "UNKNOWN-0.0.0").
RUN echo "numpy==1.26.4" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.0 setuptools==69.5.1 wheel==0.43.0

RUN python -m pip install --no-cache-dir \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        opencv-python-headless==4.9.0.80 \
        tqdm==4.66.2

RUN git clone --depth 1 https://github.com/cvg/LightGlue.git /tmp/LightGlue \
    && python -m pip install --no-cache-dir /tmp/LightGlue \
    && python -m pip show lightglue \
    && python -c "import numpy; print('numpy version after lightglue:', numpy.__version__)" \
    && rm -rf /tmp/LightGlue

# Pre-cache model weights so runtime can run with --network=none.
RUN python -c "from lightglue import ALIKED, LightGlue; \
ALIKED(max_num_keypoints=2048).eval(); \
LightGlue(features='aliked').eval(); \
print('Weights cached.')"

WORKDIR /app
COPY src /app/src
COPY scripts/entrypoint.py /app/entrypoint.py

ENV PYTHONPATH=/app/src \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output

ENTRYPOINT ["python", "/app/entrypoint.py"]