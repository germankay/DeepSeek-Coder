import argparse
import sys

DEFAULT_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"

def start_vllm_server(
    model_name: str = DEFAULT_MODEL,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
):
    """
    Launches vLLM OpenAI-compatible server for high throughput DeepSeek-Coder V1 inference.
    """
    if "v2" in model_name.lower() or "v3" in model_name.lower() or "r1" in model_name.lower():
        raise ValueError("Usage of DeepSeek V2, V3 or R1 models is strictly prohibited. Only V1 models are allowed.")

    print(f"Starting high-performance vLLM server with model: {model_name}")
    print(f"Host: {host}:{port} | Tensor Parallel Size: {tensor_parallel_size} | Max Model Len: {max_model_len}")

    try:
        from vllm.entrypoints.openai.api_server import main as vllm_api_main
    except ImportError:
        print("Error: vLLM is not installed. Please install vllm (`pip install vllm`).")
        sys.exit(1)

    # Set CLI arguments for vllm.entrypoints.openai.api_server
    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code"
    ]

    vllm_api_main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM OpenAI API Server for CyberCode Studio")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HF ID")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)

    args = parser.parse_args()

    start_vllm_server(
        model_name=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )
