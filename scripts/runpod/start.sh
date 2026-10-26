#!/bin/bash
# Entry point of the Diffusion for Krita Runpod image.
#   KRITA_TOKEN      secret the plugin sends with every request (required)
#   KRITA_STORAGE    "network": keep models on the network volume at /workspace
#                    "container": download into the container disk on every start
#   KRITA_WORKLOADS, KRITA_CHECKPOINTS: which models to download (see krita_download.py)
set -uo pipefail

export KRITA_STATUS_FILE=/tmp/krita_status.json
export COMFY_PORT=8189
status() {
    printf '{"phase": "%s", "message": "%s", "progress": [0, 0]}' "$1" "$2" > "$KRITA_STATUS_FILE.tmp"
    mv "$KRITA_STATUS_FILE.tmp" "$KRITA_STATUS_FILE"
}

if [ -z "${KRITA_TOKEN:-}" ]; then
    echo "KRITA_TOKEN is not set, refusing to expose ComfyUI without authentication" >&2
    exit 1
fi

status starting "Starting"
python /opt/krita/runpod/krita_proxy.py &

if [ "${KRITA_STORAGE:-container}" = "network" ]; then
    # Models persist on the network volume, ComfyUI's models folder points there
    root=/workspace/krita
    mkdir -p "$root/models"
    cp -rn /opt/comfyui/models/. "$root/models/"
    rm -rf /opt/comfyui/models
    ln -s "$root/models" /opt/comfyui/models
else
    root=/opt/comfyui
fi

if ! python /opt/krita/runpod/krita_download.py "$root"; then
    wait  # status is "error", the proxy reports it and terminates the pod after a while
fi

status starting "Starting ComfyUI"
cd /opt/comfyui
python main.py --listen 127.0.0.1 --port "$COMFY_PORT" ${COMFY_ARGS:-} &
comfy_pid=$!

until curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" > /dev/null; do
    if ! kill -0 "$comfy_pid" 2> /dev/null; then
        status error "ComfyUI failed to start, see the pod logs"
        wait
    fi
    sleep 2
done
status ready "Ready"

wait "$comfy_pid"
status error "ComfyUI stopped unexpectedly, see the pod logs"
wait
