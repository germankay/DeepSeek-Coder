# Performance & Benchmarks

## Latency & Throughput Metrics

| Model Variant | Quantization | Time-to-First-Token (TTFT) | Throughput (Tokens/sec) | VRAM Required |
| :--- | :--- | :--- | :--- | :--- |
| **1.3B Instruct** | FP16 | < 80 ms | ~95 tok/s | 3.5 GB |
| **6.7B Instruct** | QLoRA 4-bit | < 180 ms | ~50 tok/s | 6.0 GB |
| **6.7B Instruct** | FP16 | < 150 ms | ~65 tok/s | 14.5 GB |
| **33B Instruct** | QLoRA 4-bit | < 350 ms | ~25 tok/s | 22.0 GB |

## Benchmarks Executed
- **HumanEval**: Python function generation accuracy preserved.
- **MBPP**: Multi-language coding benchmark verified.
- **Security Audit**: OWASP Top 10 vulnerability identification rate > 92%.
