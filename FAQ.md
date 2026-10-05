# Frequently Asked Questions (FAQ)

### Q1: What model family does CyberCode Studio use?
CyberCode Studio strictly utilizes **DeepSeek-Coder V1** (1.3B, 6.7B, and 33B parameters). No V2, V3, or R1 models are used.

### Q2: How do I run the API server locally?
You can launch the server using Docker Compose:
```bash
docker-compose up -d
```
Or directly via Python:
```bash
uvicorn serve.api_server:app --host 0.0.0.0 --port 8000
```

### Q3: Is GPU required for inference?
A GPU (e.g. RTX 3060 12GB or higher) is recommended for fast response times, but QLoRA 4-bit CPU/GPU execution is supported for smaller models.
