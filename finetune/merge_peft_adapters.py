import argparse
import os

import torch
import transformers

DEFAULT_BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"

def merge_peft_adapters(base_model_path: str, adapter_path: str, output_dir: str):
    """
    Merges LoRA adapter weights with the DeepSeek-Coder V1 base model.
    """
    if "v2" in base_model_path.lower() or "v3" in base_model_path.lower() or "r1" in base_model_path.lower():
        raise ValueError("Usage of DeepSeek V2, V3 or R1 models is strictly prohibited.")

    print(f"Loading base model: {base_model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading PEFT adapter from: {adapter_path}")
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft library is required to merge adapters.") from e

    model = PeftModel.from_pretrained(model, adapter_path)
    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge successfully completed.")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into DeepSeek-Coder V1 base model.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL, help="Path or HuggingFace ID of base model.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained PEFT LoRA adapter.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where merged model will be saved.")
    args = parser.parse_args()

    merge_peft_adapters(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
