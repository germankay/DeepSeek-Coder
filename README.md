<p align="center">
<img width="1000px" alt="CyberCode Studio" src="pictures/logo.png">
</p>
<p align="center">
  <a href="https://cybercode.studio">[🌐 CyberCode Studio]</a> |
  <a href="https://linkedin.com/company/cybercode-studio">[💼 LinkedIn]</a> |
  <a href="https://wa.me/24206000000">[💬 WhatsApp Business]</a> |
  <a href="mailto:contact@cybercode.studio">[✉️ Direct Email]</a> |
  <a href="https://github.com/CyberCode-Studio/deepseek-coder-cybercode/discussions">[💬 GitHub Discussions]</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CI%2FCD-Passing-brightgreen" alt="CI Status">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Model%20Lock-DeepSeek--Coder%20V1-orange" alt="Model Lock">
  <img src="https://img.shields.io/badge/Docker-Ready-blue" alt="Docker Ready">
</p>
<hr>

## CyberCode Studio Custom Edition (2025)

> **Notice**: This repository is a customized and modernized industrial version of DeepSeek-Coder V1 built specifically for **CyberCode Studio**. All original baseline features, scripts, and evaluation benchmarks remain 100% retrocompatible. The model engine is strictly constrained to DeepSeek-Coder V1 weights (`1.3b-instruct`, `6.7b-instruct`, `33b-instruct`). Usage of V2, V3, or R1 models is prohibited.

### Key Custom Features:
1. **LoRA / QLoRA Fine-Tuning Pipeline** (`finetune/finetune_deepseekcoder.py`, `finetune/finetune_cybercode.py`)
2. **LoRA Adapter Merging** (`finetune/merge_peft_adapters.py`)
3. **Automated Security Audit & Code Review Server** (`serve/security_audit.py`, `serve/api_server.py`)
4. **OpenAI V1 Compatible API Server** (`serve/api_server.py`)
5. **High-Performance vLLM Support** (`serve/vllm_server.py`)
6. **Containerized Production & Docker Compose** (`Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`)
7. **CI/CD & Security Review Pipeline** (`.github/workflows/ci.yml`, `tests/`)

---

### Quickstart Docker (5 Minutes)

Lancer l'API sécurisée compatible OpenAI et la démo Web CyberCode Studio avec Docker :

```bash
# 1. Configurer les variables d'environnement
cp .env.example .env

# 2. Lancer la pile complète via Docker Compose
docker-compose up -d --build

# 3. Tester le point de terminaison d'audit de sécurité
curl -X POST http://localhost:8000/v1/security/audit \
  -H "Authorization: Bearer cybercode-secret-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "SELECT * FROM users WHERE username = '\''" + input_user + "'\'';",
    "langage": "python"
  }'
```

---

### Estimation des Coûts en FCFA (XAF) / EUR

| Postes de Dépenses | Estimation en Euros (€) | Estimation en FCFA (XAF) |
| :--- | :--- | :--- |
| **VPS GPU Cloud Prod (NVIDIA A10G / RTX 4090)** | 150 € - 300 € / mois | ~100 000 FCFA - 200 000 FCFA / mois |
| **Stockage NVMe Sécurisé (500 Go Chiffré)** | ~30 € / mois | ~20 000 FCFA / mois |
| **Bande passante, Trafic API & Domaines** | ~20 € / mois | ~13 000 FCFA / mois |
| **Total Mensuel Estimé** | **200 € - 350 € / mois** | **~133 000 FCFA - 233 000 FCFA / mois** |

---

### Gouvernance, Conformité & Documentation

- **Gouvernance**: [CONTRIBUTING.md](CONTRIBUTING.md) | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | [SECURITY.md](SECURITY.md) | [CHANGELOG.md](CHANGELOG.md) | [ROADMAP.md](ROADMAP.md)
- **Conformité & Éthique**: [PRIVACY.md](PRIVACY.md) | [TERMS.md](TERMS.md) | [COMPLIANCE.md](COMPLIANCE.md) | [MODEL_CARD.md](MODEL_CARD.md) | [DATASHEET.md](DATASHEET.md) | [ETHICS.md](ETHICS.md) | [SBOM.json](SBOM.json)
- **Opérationnel**: [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) | [FAQ.md](FAQ.md) | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | [PERFORMANCE.md](PERFORMANCE.md) | [SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md) | [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) | [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) | [SLA.md](SLA.md) | [SUPPORT.md](SUPPORT.md)

---

### 1. Introduction of DeepSeek Coder

DeepSeek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. We provide various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and an extra fill-in-the-blank task, to support project-level code completion and infilling. For coding capabilities, DeepSeek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks.

<p align="center">
<img src="pictures/result.png" alt="result" width="70%">
</p>

- **Massive Training Data**: Trained from scratch on 2T tokens, including 87% code and 13% linguistic data in both English and Chinese languages.

- **Highly Flexible & Scalable**: Offered in model sizes of 1B, 5.7B, 6.7B and 33B, enabling users to choose the setup most suitable for their requirements.

- **Superior Model Performance**: State-of-the-art performance among publicly available code models on HumanEval, MultiPL-E, MBPP, DS-1000, and APPS benchmarks.

- **Advanced Code Completion Capabilities**: A window size of 16K and a fill-in-the-blank task, supporting project-level code completion and infilling tasks.

#### Supported Programming Languages
`['ada', 'agda', 'alloy', 'antlr', 'applescript', 'assembly', 'augeas', 'awk', 'batchfile', 'bluespec', 'c', 'c-sharp', 'clojure', 'cmake', 'coffeescript', 'common-lisp', 'cpp', 'css', 'cuda', 'dart', 'dockerfile', 'elixir', 'elm', 'emacs-lisp', 'erlang', 'f-sharp', 'fortran', 'glsl', 'go', 'groovy', 'haskell', 'html', 'idris', 'isabelle', 'java', 'java-server-pages', 'javascript', 'json', 'julia', 'jupyter-notebook', 'kotlin', 'lean', 'literate-agda', 'literate-coffeescript', 'literate-haskell', 'lua', 'makefile', 'maple', 'markdown', 'mathematica', 'matlab', 'ocaml', 'pascal', 'perl', 'php', 'powershell', 'prolog', 'protocol-buffer', 'python', 'r', 'racket', 'restructuredtext', 'rmarkdown', 'ruby', 'rust', 'sas', 'scala', 'scheme', 'shell', 'smalltalk', 'solidity', 'sparql', 'sql', 'stan', 'standard-ml', 'stata', 'systemverilog', 'tcl', 'tcsh', 'tex', 'thrift', 'typescript', 'verilog', 'vhdl', 'visual-basic', 'xslt', 'yacc', 'yaml', 'zig']`

### 2. Evaluation Results
We evaluate DeepSeek Coder on various coding-related benchmarks.
Only `pass@1` results on HumanEval (Python and Multilingual), MBPP, and DS-1000 are reported here:

<p align="center">
<img src="pictures/table.png" alt="table" width="70%">
</p>


The result shows that DeepSeek-Coder-Base-33B significantly outperforms existing open-source code LLMs. Compared with CodeLlama-34B, it leads by 7.9%, 9.3%, 10.8% and 5.9% respectively on HumanEval Python, HumanEval Multilingual, MBPP and DS-1000.
Surprisingly, our DeepSeek-Coder-Base-7B reaches the performance of CodeLlama-34B.
The DeepSeek-Coder-Instruct-33B model after instruction tuning outperforms GPT35-turbo on HumanEval and achieves comparable results with GPT35-turbo on MBPP.

More evaluation details can be found in the [Detailed Evaluation](#6-detailed-evaluation-results).


### 3. Procedure of Data Creation and Model Training

#### Data Creation

- Step 1: Collect code data from GitHub and apply the same filtering rules as [StarCoder Data](https://github.com/bigcode-project/bigcode-dataset) to filter data.
- Step 2: Parsing the dependencies of files within the same repository to rearrange the file positions based on their dependencies.
- Step 3: Concatenating dependent files to form a single example and employ repo-level minhash for deduplication.
- Step 4: Further filtering out low-quality code, such as codes with syntax errors or poor readability.

<img src="pictures/data_clean.png" alt="data_creation" width="100%">

#### Model Training

- Step 1: Initially pre-trained with a dataset consisting of 87% code, 10% code-related language (Github Markdown and StackExchange), and 3% non-code-related Chinese language. Models are pre-trained using 1.8T tokens and a 4K window size in this step.
- Step 2: Further Pre-training using an extended 16K window size on an additional 200B tokens, resulting in foundational models (**DeepSeek-Coder-Base**).
- Step 3: Instruction Fine-tuning on 2B tokens of instruction data, resulting in instruction-tuned models (**DeepSeek-Coder-Instruct**).

<img src="pictures/model_pretraining.png" alt="model_pretraining" width="100%">


### 4. How to Use
Before proceeding, you'll need to install the necessary dependencies. You can do this by running the following command:
```
pip install -r requirements.txt
```
A demo is also available on the [🤗 Hugging Face Space](https://huggingface.co/spaces/deepseek-ai/deepseek-coder-33b-instruct), and you can run the demo locally using `app.py` in the [demo](https://github.com/deepseek-ai/deepseek-coder/tree/main/demo) folder.  (Thanks to all the HF team for their support)

Here are some examples of how to use our model.

#### 1) Code Completion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### 2) Code Insertion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
input_text = """<｜fim▁begin｜>def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<｜fim▁hole｜>
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)<｜fim▁end｜>"""
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):])
```

#### 3) Chat Model Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
```

---

### 5. How to Fine-tune DeepSeek-Coder (Full / LoRA / QLoRA)

We provide `finetune/finetune_deepseekcoder.py` and `finetune/finetune_cybercode.py` for finetuning our models on downstream tasks.

```bash
pip install -r finetune/requirements.txt
```

#### QLoRA Fine-Tuning
```bash
python finetune/finetune_deepseekcoder.py \
    --model_name_or_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --data_path ./data/dataset.json \
    --output_dir ./models/cybercode-lora \
    --use_peft --load_in_4bit --lora_r 16 --lora_alpha 32
```

#### Continuous Fine-Tuning with Anonymization
```bash
python finetune/finetune_cybercode.py \
    --data_path ./data/raw_data.json \
    --output_dir ./models/cybercode-lora
```

---

### 6. Running the API Server & Docker Deployment

#### Local FastAPI Launch
```bash
pip install -r requirements-serve.txt
python -m serve.api_server
```

#### Docker Compose Launch
```bash
docker compose up --build -d
```

For complete documentation, security incident procedures, and cost analyses, see `DOCS_CYBERCODE.md` and `INCIDENT_RESPONSE.md`.

---

### 7. License
This code repository is licensed under the MIT License. The use of DeepSeek Coder models is subject to the Model License. DeepSeek Coder supports commercial use.

See the [LICENSE-CODE](LICENSE-CODE) and [LICENSE-MODEL](LICENSE-MODEL) for more details.
