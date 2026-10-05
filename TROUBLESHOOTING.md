# Troubleshooting Guide

## Common Issues & Solutions

### 1. `CUDA_HOME is not set` or GPU missing
- **Cause**: PyTorch or vLLM installed without CUDA environment variables.
- **Solution**: Ensure NVIDIA CUDA drivers are installed or use CPU-fallback PyTorch installation (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

### 2. `HTTP 401 Unauthorized` on API Requests
- **Cause**: Missing or incorrect `Authorization` header.
- **Solution**: Include `Authorization: Bearer <API_KEY>` in your HTTP request header matching the `API_KEY` defined in `.env`.

### 3. Out of Memory (OOM) during Fine-Tuning
- **Cause**: Sequence length or batch size too large for available GPU VRAM.
- **Solution**: Enable QLoRA 4-bit quantization (`--use_peft --load_in_4bit`) and reduce `--max_seq_length` to `2048`.
