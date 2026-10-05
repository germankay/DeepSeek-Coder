# Dockerfile for CyberCode Studio - DeepSeek-Coder V1 Platform
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip and install wheel
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt requirements-serve.txt requirements-finetune.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-serve.txt

# Copy source code
COPY . /app

EXPOSE 8000 7860

CMD ["python3", "-m", "serve.api_server"]
