# Diffusion for Krita: ComfyUI image for Runpod GPU pods.
# Build from the repository root:
#   docker build -f scripts/runpod/Dockerfile -t ghcr.io/shoogstad/diffusion-for-krita-runpod:<version> .
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ARG COMFYUI_REF=426f5eea52a7ece1dd599fff4cb6cd1f352ddd24
ARG RUNPODCTL_VERSION=3.0.1
ARG H3_STUDIO_REF=9834e55976cd1d0572b55a3453de0e1b3c5d3dad

# torch lives in the system Python of this image; keep its CUDA build when installing nodes
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 PIP_CONSTRAINT=/opt/torch-constraints.txt
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /opt/torch-constraints.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /usr/local/bin/runpodctl \
        "https://github.com/runpod/runpodctl/releases/download/v${RUNPODCTL_VERSION}/runpodctl-linux-amd64" \
    && chmod +x /usr/local/bin/runpodctl

RUN git clone https://github.com/comfyanonymous/ComfyUI /opt/comfyui \
    && cd /opt/comfyui && git checkout "$COMFYUI_REF" \
    && pip install -r requirements.txt

WORKDIR /opt/comfyui/custom_nodes
RUN git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux \
    && git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus \
    && git clone --depth 1 https://github.com/Acly/comfyui-inpaint-nodes \
    && git clone --depth 1 https://github.com/Acly/comfyui-tooling-nodes \
    && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
    && git clone https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio \
    && git -C ComfyUI-MiniMax-H3-Image-Studio checkout "$H3_STUDIO_REF" \
    && for d in */; do \
        if [ -f "$d/requirements.txt" ]; then pip install -r "$d/requirements.txt" || exit 1; fi; \
    done

# H3 Edit: still decode without the stock decode's 16 px banding
COPY scripts/runpod/custom_nodes/ComfyUI-H3StudioStill /opt/comfyui/custom_nodes/ComfyUI-H3StudioStill

RUN pip install aiohttp tqdm runpod

# Plugin code is only used for its model list and download script
COPY ai_diffusion /opt/krita/ai_diffusion
COPY scripts/download_models.py /opt/krita/scripts/download_models.py
COPY scripts/runpod /opt/krita/runpod
RUN rm -rf /opt/krita/ai_diffusion/debugpy && chmod +x /opt/krita/runpod/start.sh

WORKDIR /opt/comfyui
EXPOSE 8188
CMD ["/opt/krita/runpod/start.sh"]
