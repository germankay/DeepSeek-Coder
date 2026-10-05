# Model Card - CyberCode Studio DeepSeek-Coder V1

## Model Overview
- **Model Name**: CyberCode Studio DeepSeek-Coder V1
- **Base Architecture**: DeepSeek-Coder V1 (1.3B, 6.7B, 33B parameters)
- **Primary Languages**: Python, C/C++, Java, JavaScript, TypeScript, Go, Rust, SQL, HTML/CSS, Shell, etc.
- **Specialization**: Sovereign AI code completion, FIM (Fill-In-the-Middle) fill, automated code review, OWASP Top 10 security auditing.

## Training & Fine-Tuning
- **Fine-Tuning Techniques**: Full SFT via DeepSpeed ZeRO-3, PEFT (LoRA and 4-bit/8-bit QLoRA).
- **Context Length**: 16,384 tokens window supported natively.
