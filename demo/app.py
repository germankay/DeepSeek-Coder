import os
from html import escape
import time
import uuid
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

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

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful programming assistant. Always reply in the language of the user's "
    "LAST message, even if earlier messages used another language: Spanish if the last "
    "message is in Spanish, English if it is in English. "
    "Eres un asistente de programación: responde siempre en el idioma del ÚLTIMO mensaje "
    "del usuario, aunque los anteriores estén en otro idioma. "
    "Keep code, identifiers and technical terms in their original form."
)
TIMING_MARK = "\n\n⏱"

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

    # QUANTIZATION: auto (4bit en GPU salvo el 1.3b) | 4bit | 8bit | none
    quant = os.getenv("QUANTIZATION", "auto").lower()
    if quant == "auto":
        quant = "4bit" if device == "cuda" and "1.3b" not in MODEL_ID.lower() else "none"
    quantization_config = None
    if device == "cuda" and quant == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif device == "cuda" and quant == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"Quantization: {quant if quantization_config else 'none'}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
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

    conversation = [{"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}]
    for turn in chat_history:
        # Gradio entrega el historial como dicts {"role","content"} o como pares (user, assistant)
        turns = [turn] if isinstance(turn, dict) else [
            {"role": "user", "content": turn[0]},
            {"role": "assistant", "content": turn[1]},
        ]
        for item in turns:
            content = item["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            if content is None:
                continue
            # No reenviar al modelo la línea de tiempo que agrega la interfaz
            content = content.split(TIMING_MARK)[0]
            conversation.append({"role": item["role"], "content": content})
    conversation.append({"role": "user", "content": message})

    try:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
        # transformers >= 5 devuelve un BatchEncoding en lugar de un tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
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

        start_time = time.perf_counter()
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            elapsed = time.perf_counter() - start_time
            yield "".join(outputs).replace("<|EOT|>", "") + f"{TIMING_MARK} {elapsed:.1f} s…"

        elapsed = time.perf_counter() - start_time
        final_text = "".join(outputs).replace("<|EOT|>", "")
        n_tokens = len(tokenizer.encode(final_text, add_special_tokens=False))
        speed = n_tokens / elapsed if elapsed > 0 else 0.0
        yield final_text + f"{TIMING_MARK} {elapsed:.1f} s · {n_tokens} tokens · {speed:.1f} tokens/s"

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield "An Out of Memory (OOM) error occurred on the GPU. Please try again with shorter context or fewer generation tokens."
    except Exception as e:
        yield f"Generation error: {type(e).__name__}: {e!s}"


EXAMPLES = [
    "Explícame brevemente qué es Python",
    "Escribe una función en Python que calcule el factorial",
    "Write a program to find the factorial of a number",
    "Implement snake game using pygame",
]


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def chat_choices(chats: dict):
    return [(c["title"], cid) for cid, c in reversed(list(chats.items()))]


def respond(message, history, chats, cur, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_pen):
    message = (message or "").strip()
    if not message:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return
    if not cur or cur not in chats:
        cur = new_id()
        chats[cur] = {"title": message[:40] + ("…" if len(message) > 40 else ""), "history": []}
    prev = list(history)
    history = prev + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    hide_welcome = gr.update(visible=False)
    for partial in generate(message, prev, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_pen):
        history[-1] = {"role": "assistant", "content": partial}
        yield gr.update(value=history, visible=True), "", hide_welcome, chats, cur, gr.skip()
    chats[cur]["history"] = history
    yield gr.update(value=history, visible=True), "", hide_welcome, chats, cur, gr.update(choices=chat_choices(chats), value=cur)


def start_chat(text):
    return text


def new_chat(chats):
    return gr.update(value=[], visible=False), "", gr.update(visible=True), None, gr.update(choices=chat_choices(chats), value=None)


def open_chat(cid, chats):
    if not cid or cid not in chats:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    h = chats[cid]["history"]
    return gr.update(value=h, visible=bool(h)), gr.update(visible=not h), cid, gr.skip()


with gr.Blocks(title="DeepSeek-Coder", fill_width=True, fill_height=True) as demo:
    chats_state = gr.State({})
    cur_state = gr.State(None)

    with gr.Sidebar(open=True, width=260, elem_id="sidebar"):
        gr.HTML('<div class="sidebar-brand">DeepSeek <span>Coder</span></div>')
        new_btn = gr.Button("✎  Nuevo chat", variant="secondary", elem_id="new-chat")
        gr.Markdown("Tus chats", elem_id="history-heading")
        chat_list = gr.Radio(choices=[], label="", show_label=False, elem_id="chat-list", interactive=True)
        with gr.Accordion("Ajustes del modelo", open=False, elem_id="settings"):
            system_box = gr.Textbox(label="System prompt", lines=6, value=DEFAULT_SYSTEM_PROMPT)
            max_tok = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
            temp = gr.Slider(label="Temperature (0 = determinista)", minimum=0.0, maximum=2.0, step=0.05, value=0.0)
            top_p = gr.Slider(label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
            top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
            rep = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.0)
        gr.HTML(f'<div class="local-profile"><span class="profile-icon">D</span><div>IA local<small title="{escape(MODEL_ID, quote=True)}">{escape(MODEL_ID.split("/")[-1])}</small></div><span class="status-dot" title="Ejecución local"></span></div>', elem_id="local-profile")

    with gr.Column(elem_id="main"):
        gr.HTML('<div class="topbar"><span>DeepSeek Coder</span><span class="local-badge">En tu equipo</span></div>', elem_id="topbar")
        with gr.Column(visible=True, elem_id="welcome") as welcome:
            gr.HTML('<div class="welcome-symbol" aria-hidden="true">&gt;_</div>')
            gr.Markdown("# ¿Qué quieres desarrollar hoy?\n\nTu asistente de programación local.")
            with gr.Row(elem_id="examples"):
                ex_btns = [gr.Button(t, size="sm", variant="secondary") for t in EXAMPLES]
        chatbot = gr.Chatbot(visible=False, elem_id="chatbot", height="100%", show_label=False, render_markdown=True, buttons=["copy"], layout="bubble")
        with gr.Column(elem_id="composer", scale=0):
            with gr.Row(elem_id="inputbar"):
                msg = gr.Textbox(
                    placeholder="Pregunta lo que quieras o describe tu idea",
                    label="Mensaje",
                    show_label=False,
                    container=False,
                    lines=2,
                    max_lines=8,
                    scale=12,
                    elem_id="msg",
                )
                send = gr.Button("↑", variant="primary", scale=0, min_width=38, elem_id="send")
            gr.HTML('<div class="composer-note">DeepSeek Coder · 100% local<span>Enter para enviar</span></div>')

    settings = [system_box, max_tok, temp, top_p, top_k, rep]
    ins = [msg, chatbot, chats_state, cur_state, *settings]
    outs = [chatbot, msg, welcome, chats_state, cur_state, chat_list]
    msg.submit(respond, ins, outs)
    send.click(respond, ins, outs)
    for b in ex_btns:
        b.click(start_chat, b, msg).then(respond, ins, outs)
    new_btn.click(new_chat, chats_state, [chatbot, msg, welcome, cur_state, chat_list])
    chat_list.input(open_chat, [chat_list, chats_state], [chatbot, welcome, cur_state, chat_list])

if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        theme=gr.themes.Monochrome(font=["Arial", "sans-serif"]),
        css_paths=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")],
        footer_links=[],
    )
