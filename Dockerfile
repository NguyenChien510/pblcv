# SpaceTime-DSCA web demo (CPU). Model weights are NOT baked in: mount ./checkpoints and ./yolov8n.pt at runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first, from the CPU index by default (otherwise pip pulls the multi-GB CUDA wheels).
# GPU image: --build-arg TORCH_INDEX=cu126 (deploy/docker-compose.gpu.yml does this); the engine picks CUDA by itself.
# torch and ultralytics are pinned to the versions the demo was measured with: with torch 2.15 + ultralytics
# 8.4.214 the same video gave different tracks (TESTVIDEO: 16 tracks / top-1 7/9 instead of 18 / 8/9).
ARG TORCH_INDEX=cpu
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt ultralytics==8.4.178

# Bake the two Hugging Face models into the image so startup never needs the network.
# If BERT failed to load, TextEncoder would silently fall back to an untrained encoder and the checkpoint would be refused.
RUN python -c "from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor; \
AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased'); \
CLIPModel.from_pretrained('openai/clip-vit-base-patch16'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')" \
 && chmod -R a+rX /opt/hf

COPY . .

RUN useradd -u 1000 -m app && mkdir -p uploads samples && chown -R app:app /app/uploads /app/samples
USER app

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "web_demo.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser", "--samples_dir", "samples", "--max_upload_mb", "300"]
