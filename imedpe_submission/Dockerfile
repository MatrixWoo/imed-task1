FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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

# Pin numpy globally to avoid scipy/opencv ABI breakage from transitive deps.
RUN echo "numpy==1.26.4" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Torch (cu118 wheels) -- matches the evaluation host.
RUN python -m pip install --no-cache-dir \
        torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# IMPORTANT: any model weights your method needs must be downloaded here at
# build time. The evaluation container runs with --network=none.
# Example:
#   RUN python -c "from your_package import YourModel; YourModel().eval()"

COPY predict.py /app/predict.py

ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output

ENTRYPOINT ["python", "/app/predict.py"]