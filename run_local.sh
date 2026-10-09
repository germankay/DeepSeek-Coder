#!/usr/bin/env bash
# Prueba local: crea el venv, instala dependencias, corre los tests y levanta la API.
# Uso: ./run_local.sh [--solo-tests] [--demo]
#   --solo-tests  instala, corre los tests y termina
#   --demo        al final levanta la demo Gradio con una biblioteca de modelos en ./.hf_cache
set -euo pipefail

cd "$(dirname "$0")"

SOLO_TESTS=0
DEMO=0
for arg in "$@"; do
  case "$arg" in
    --solo-tests) SOLO_TESTS=1 ;;
    --demo) DEMO=1 ;;
    *) echo "Argumento desconocido: $arg"; exit 1 ;;
  esac
done

API_KEY="${CYBERCODE_API_KEY:-clave-local-de-prueba}"
PORT="${PORT:-8000}"

# Los modelos se guardan dentro del proyecto (ignorado por git)
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

echo "==> [1/4] Entorno virtual"
if [ -d .venv ]; then
  echo "    .venv ya existe, se reutiliza"
else
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [2/4] Dependencias"
if python -c "import pytest, httpx, fastapi, pydantic, uvicorn, numpy, transformers, datasets, accelerate, bitsandbytes, torch" 2>/dev/null; then
  echo "    dependencias ya instaladas, se omite"
else
  pip install -U pip
  pip install pytest httpx fastapi pydantic "uvicorn[standard]" numpy transformers datasets accelerate bitsandbytes
  python -c "import torch" 2>/dev/null || pip install torch
fi

# Con GPU NVIDIA presente, asegura torch con CUDA (reemplaza una build solo-CPU)
if command -v nvidia-smi >/dev/null 2>&1 && ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "    GPU NVIDIA detectada pero torch no tiene CUDA: reinstalando torch con CUDA (puede tardar, ~2-3 GB)"
  pip uninstall -y torch
  pip install torch
fi
python -c "import torch; print('    torch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "==> [3/4] Tests"
PYTHONPATH=. pytest tests/ -v

if [ "$SOLO_TESTS" -eq 1 ]; then
  echo "Tests OK."
  exit 0
fi

if [ "$DEMO" -eq 1 ]; then
  DEMO_MODEL="${MODEL_ID:-deepseek-ai/deepseek-coder-6.7b-instruct}"
  echo "==> [4/4] Demo Gradio"
  echo "    Instalando dependencias de la demo..."
  pip install -r demo/requirement.txt
  echo "    Iniciando demo en http://localhost:7860. Gestioná las descargas desde Modelos locales."
  MODEL_ID="$DEMO_MODEL" PYTHONUNBUFFERED=1 python -u demo/app.py
  exit 0
fi

echo "==> [4/4] API en http://localhost:${PORT} (Ctrl+C para salir)"
CYBERCODE_API_KEY="$API_KEY" python -m uvicorn serve.api_server:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  curl -fs "http://127.0.0.1:${PORT}/healthz" >/dev/null && break
  sleep 1
done

echo "--- /healthz"
curl -s "http://127.0.0.1:${PORT}/healthz"; echo
echo "--- /v1/security/audit"
curl -s -X POST "http://127.0.0.1:${PORT}/v1/security/audit" \
  -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
  -d '{"code":"password = \"abc12345\"\nresult = eval(user_input)"}'; echo
echo "--- sin clave (debe dar 401)"
curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:${PORT}/v1/models"

echo
echo "API corriendo. Clave: ${API_KEY}"
wait "$SERVER_PID"
