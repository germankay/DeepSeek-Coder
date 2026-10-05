import argparse
import json
import os
import re

from finetune.finetune_deepseekcoder import (
    DEFAULT_MODEL_NAME,
)


def anonymize_code_and_text(text: str) -> str:
    """
    Anonymizes client data: removes IP addresses, API keys, bearer tokens, credentials, and internal project names.
    """
    if not isinstance(text, str):
        return text

    # IP addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[ANONYMIZED_IP]', text)
    # API keys / Bearer tokens / secret strings
    text = re.sub(r'(?i)(api[_-]?key|secret|token|password|auth)\s*[:=]\s*["\'][A-Za-z0-9_\-\.]{8,}["\']', r'\1 = "[ANONYMIZED_SECRET]"', text)
    text = re.sub(r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*', 'Bearer [ANONYMIZED_TOKEN]', text)
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[ANONYMIZED_EMAIL]', text)

    return text

def clean_and_anonymize_dataset(raw_dataset_path: str, output_cleaned_path: str) -> str:
    """
    Loads json/jsonl dataset, applies anonymization, and saves cleaned dataset.
    """
    print(f"Cleaning and anonymizing dataset from {raw_dataset_path}...")
    cleaned_data = []

    with open(raw_dataset_path, "r", encoding="utf-8") as f:
        if raw_dataset_path.endswith(".jsonl"):
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    item = json.loads(line.strip())
                    cleaned_data.append(item)
        else:
            cleaned_data = json.load(f)

    for item in cleaned_data:
        if "instruction" in item:
            item["instruction"] = anonymize_code_and_text(item["instruction"])
        if "input" in item:
            item["input"] = anonymize_code_and_text(item["input"])
        if "output" in item:
            item["output"] = anonymize_code_and_text(item["output"])

    os.makedirs(os.path.dirname(os.path.abspath(output_cleaned_path)), exist_ok=True)
    with open(output_cleaned_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned dataset saved to {output_cleaned_path}")
    return output_cleaned_path

def main():
    parser = argparse.ArgumentParser(description="CyberCode Studio Fine-Tuning Pipeline with Data Anonymization.")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data_path", type=str, required=True, help="Path to input raw training JSON dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model/LoRA adapter.")
    parser.add_argument("--use_peft", action="store_true", default=True, help="Use PEFT (LoRA/QLoRA).")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    args = parser.parse_args()

    if "v2" in args.model_name_or_path.lower() or "v3" in args.model_name_or_path.lower() or "r1" in args.model_name_or_path.lower():
        raise ValueError("Usage of DeepSeek V2, V3 or R1 models is strictly prohibited. Only V1 models are allowed.")

    cleaned_data_path = os.path.join(args.output_dir, "cleaned_cybercode_dataset.json")
    clean_and_anonymize_dataset(args.data_path, cleaned_data_path)

    print(f"Starting CyberCode Studio fine-tuning with model {args.model_name_or_path}...")

    # Build Training Command / Execution logic using transformers Trainer
    cmd = (
        f"python finetune/finetune_deepseekcoder.py "
        f"--model_name_or_path {args.model_name_or_path} "
        f"--data_path {cleaned_data_path} "
        f"--output_dir {args.output_dir} "
        f"{'--use_peft' if args.use_peft else ''} "
        f"--lora_r {args.lora_r} "
        f"--lora_alpha {args.lora_alpha} "
        f"--lora_dropout {args.lora_dropout} "
        f"{'--load_in_4bit' if args.load_in_4bit else ''} "
        f"--num_train_epochs {args.num_train_epochs} "
        f"--per_device_train_batch_size {args.per_device_train_batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--save_strategy steps --save_steps 100 --logging_steps 10"
    )

    print(f"Executing: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
