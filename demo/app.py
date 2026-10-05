import os
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/deepseek-coder-6.7b-instruct")

# Enforce DeepSeek-Coder V1 models restriction
if "v2" in MODEL_ID.lower() or "v3" in MODEL_ID.lower() or "r1" in MODEL_ID.lower():
    raise ValueError("Usage of DeepSeek V2, V3 or R1 models is strictly prohibited. Only V1 models are allowed.")

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """\
# DeepSeek-6.7B-Chat - CyberCode Studio Edition

This application demonstrates the [DeepSeek-Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) model fine-tuned for code generation and technical discussions.
"""

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

if device == "cpu":
    DESCRIPTION += "\n<p><strong>Note:</strong> Running on CPU mode. Generation speed may be limited.</p>"
elif device == "mps":
    DESCRIPTION += "\n<p>Running on Apple Silicon MPS hardware acceleration.</p>"

model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return
    print(f"Loading model {MODEL_ID} on device {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.use_default_system_prompt = False

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        if device_map is None:
            model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@spaces.GPU
def generate(
    message: str,
    chat_history: list,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
) -> Iterator[str]:
    load_model_and_tokenizer()
    if model is None or tokenizer is None:
        yield "Model is not loaded. Please verify configuration and server memory."
        return

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    try:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            gr.Warning(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            num_beams=1,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
        # Remove None values
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield "".join(outputs).replace("<|EOT|>", "")

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield "An Out of Memory (OOM) error occurred on the GPU. Please try again with shorter context or fewer generation tokens."
    except Exception as e:
        yield f"Generation error: {e!s}"


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1.0,
        ),
    ],
    examples=[
        ["implement snake game using pygame"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["write a program to find the factorial of a number"],
    ],
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    demo.queue().launch(share=False)
