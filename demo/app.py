import os
from html import escape
from pathlib import Path
import time
import uuid
from collections.abc import Iterator
from threading import Thread

if __package__:
    from .model_manager import ModelManager, MODEL_DETAILS
    from .chat_store import ChatStore
    from .chat_input import GenerationControls, StopGeneration, with_context
else:
    from model_manager import ModelManager, MODEL_DETAILS
    from chat_store import ChatStore
    from chat_input import GenerationControls, StopGeneration, with_context

import gradio as gr
import torch
from transformers import TextIteratorStreamer, StoppingCriteriaList

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

manager = ModelManager(MODEL_ID, device)


def active_model_badge():
    active = manager.loaded_id
    state = "is-loaded" if active else "is-idle"
    name = escape(active.split("/")[-1]) if active else "Ningún modelo cargado"
    label = "En uso · cargado en memoria" if active else "Memoria libre"
    return (
        f'<div class="active-model {state}" role="status" aria-live="polite">'
        f'<span class="model-indicator" aria-hidden="true"></span>'
        f'<div><strong>{label}</strong><span class="model-name">{name}</span></div></div>'
    )


def composer_model_label():
    name = manager.loaded_id.split("/")[-1] if manager.loaded_id else "Modelo local"
    return f'<span title="{escape(name, quote=True)}">{escape(name)}</span>'


def model_profile():
    return active_model_badge()


def model_status(selected):
    saved = "Guardado en disco" if selected and manager.local_path(selected) else "Pendiente de descarga"
    details = MODEL_DETAILS.get(selected)
    description = ""
    if details:
        description = (
            '<div class="model-description"><strong>Recomendado para</strong>'
            f'<p>{escape(details["purpose"])}</p>'
            f'<p>{escape(details["tradeoff"])}</p>'
            f'<small>VRAM estimada a 4 bits: {escape(details["vram"])}. '
            'Varía según el contexto y la configuración; no es el tamaño de descarga.</small></div>'
        )
    return description + f'<div class="model-storage">Selección: {saved.lower()}</div>' + active_model_badge()


def model_action(action, selected):
    yield gr.skip(), "Procesando… Esto puede tardar unos minutos.", gr.skip()
    try:
        if action == "download":
            manager.download(selected)
            note = "Modelo guardado. Ya podés cargarlo sin Internet."
        elif action == "load":
            manager.load(selected)
            note = "Modelo cargado y listo para conversar."
        else:
            manager.unload()
            note = "Memoria liberada. Los archivos siguen en el disco."
    except Exception as exc:
        note = f"No se pudo completar la operación: {exc}"
    yield gr.update(choices=manager.choices(), value=selected), '<p class="model-notice">' + escape(note) + "</p>" + model_status(selected), model_profile()


def download_model(selected):
    yield from model_action("download", selected)


def switch_model(selected):
    yield from model_action("load", selected)


def unload_model(selected):
    yield from model_action("unload", selected)


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
    stop_event=None,
) -> Iterator[str]:
    with manager.lock:
        try:
            manager._load(manager.selected_id)
        except Exception as exc:
            yield f"No se pudo cargar el modelo: {exc}"
            return
        if stop_event is not None and stop_event.is_set():
            yield "Respuesta detenida."
            return
        yield from generate_loaded(message, chat_history, system_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stop_event)


def generate_loaded(message, chat_history, system_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stop_event=None):
    model, tokenizer = manager.model, manager.tokenizer
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

    t = None
    try:
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
        # transformers >= 5 devuelve un BatchEncoding en lugar de un tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            raise ValueError("La conversación y el contexto exceden el límite de tokens. Usá un chat nuevo o un contexto más corto.")
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
            stopping_criteria=StoppingCriteriaList([StopGeneration(stop_event)]) if stop_event is not None else None,
            eos_token_id=tokenizer.eos_token_id
        )
        # Remove None values
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

        start_time = time.perf_counter()
        errors = []
        def run_generation():
            try:
                model.generate(**generate_kwargs)
            except Exception as exc:
                errors.append(exc)
                streamer.end()
        t = Thread(target=run_generation)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            elapsed = time.perf_counter() - start_time
            yield "".join(outputs).replace("<|EOT|>", "") + f"{TIMING_MARK} {elapsed:.1f} s…"

        t.join()
        if errors:
            raise errors[0]
        elapsed = time.perf_counter() - start_time
        final_text = "".join(outputs).replace("<|EOT|>", "")
        n_tokens = len(tokenizer.encode(final_text, add_special_tokens=False))
        speed = n_tokens / elapsed if elapsed > 0 else 0.0
        yield final_text + f"{TIMING_MARK} {elapsed:.1f} s · {n_tokens} tokens · {speed:.1f} tokens/s" + (" · Detenida" if stop_event is not None and stop_event.is_set() else "")

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield "An Out of Memory (OOM) error occurred on the GPU. Please try again with shorter context or fewer generation tokens."
    except Exception as e:
        yield f"Generation error: {type(e).__name__}: {e!s}"
    finally:
        # A timed-out stream must not allow unloading a still-running GPU worker.
        if t is not None:
            if stop_event is not None:
                stop_event.set()
            t.join()


EXAMPLES = [
    "Explícame brevemente qué es Python",
    "Escribe una función en Python que calcule el factorial",
    "Write a program to find the factorial of a number",
    "Implement snake game using pygame",
]


chat_store = ChatStore(Path(__file__).resolve().parents[1] / ".local_chats")


def disk_usage(selected=None):
    def size(value):
        for unit in ("B", "KiB", "MiB", "GiB"):
            if value < 1024 or unit == "GiB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{value} B"
            value /= 1024
    count, total, allocated, current = chat_store.usage(selected)
    detail = f" · Este chat: {size(current)}" if selected else ""
    return f"{count} chats · Archivos: {size(total)}{detail}\n\nOcupado en disco: {size(allocated)}"


def new_id() -> str:
    return uuid.uuid4().hex


def chat_choices(chats: dict):
    return [(c["title"], cid) for cid, c in reversed(list(chats.items()))]


def respond(message, history, chats, cur, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_pen, stop_event=None):
    message = (message or "").strip()
    if not message:
        yield (gr.skip(),) * 7
        return
    chats = chat_store.load()
    if not cur or cur not in chats:
        cur = new_id()
        chats[cur] = {"title": message[:40] + ("…" if len(message) > 40 else ""), "history": []}
    prev = list(chats[cur]["history"])
    history = prev + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    chats[cur]["history"] = history
    chat_store.save(cur, chats[cur])
    hide_welcome = gr.update(visible=False)
    yield gr.update(value=history, visible=True), "", hide_welcome, chats, cur, gr.update(choices=chat_choices(chats), value=cur), disk_usage(cur)
    saved_at = time.monotonic()
    try:
        for partial in generate(message, prev, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_pen, stop_event):
            history[-1] = {"role": "assistant", "content": partial}
            if time.monotonic() - saved_at >= 1:
                chat_store.save(cur, chats[cur])
                saved_at = time.monotonic()
            yield gr.update(value=history, visible=True), "", hide_welcome, chats, cur, gr.skip(), gr.skip()
    finally:
        chat_store.save(cur, chats[cur])
    chats = chat_store.load()
    yield gr.update(value=history, visible=True), "", hide_welcome, chats, cur, gr.update(choices=chat_choices(chats), value=cur), disk_usage(cur)


controls = GenerationControls()


def stop_response(request: gr.Request):
    controls.stop(request.session_hash)
    return gr.update(value="■", interactive=False)


def respond_ui(message, history, chats, cur, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_pen, context, files, request: gr.Request):
    if not (message or "").strip():
        gr.Warning("Escribí un mensaje para acompañar el contexto.")
        yield (gr.skip(),) * 11
        return
    try:
        prepared = with_context(message, context, files)
    except (ValueError, OSError) as exc:
        gr.Warning(str(exc))
        yield (gr.skip(),) * 11
        return
    event = controls.begin(request.session_hash)
    try:
        yield *((gr.skip(),) * 7), gr.update(visible=False, interactive=False), gr.update(value="■", visible=True, interactive=True), "", gr.update(value=None, visible=False)
        for result in respond(prepared, history, chats, cur, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_pen, event):
            yield *result, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    except Exception as exc:
        gr.Warning(f"No se pudo completar la respuesta: {exc}")
    finally:
        event.set()
        controls.finish(request.session_hash, event)
    yield *((gr.skip(),) * 7), gr.update(visible=True, interactive=True), gr.update(visible=False), gr.skip(), gr.skip()


def start_chat(text):
    return text


def new_chat(chats=None):
    chats = chat_store.load()
    return gr.update(value=[], visible=False), "", gr.update(visible=True), chats, None, gr.update(choices=chat_choices(chats), value=None), disk_usage()


def open_chat(cid, chats=None):
    chats = chat_store.load()
    if not cid or cid not in chats:
        return new_chat()
    history = chats[cid]["history"]
    return gr.update(value=history, visible=bool(history)), "", gr.update(visible=not history), chats, cid, gr.update(choices=chat_choices(chats), value=cid), disk_usage(cid)


def delete_chat(cid):
    if cid:
        chat_store.delete(cid)
    return new_chat()


def remove_attachment(filename):
    def remove(files):
        remaining = [file for file in files or [] if file != filename]
        return gr.update(value=remaining or None, visible=bool(remaining)), None
    return remove


with gr.Blocks(title="DeepSeek-Coder", fill_width=True, fill_height=True) as demo:
    demo.load(fn=None, js=Path(__file__).with_name("sidebar.js").read_text())
    chats_state = gr.State({})
    cur_state = gr.State(None)

    with gr.Sidebar(open=True, width=320, elem_id="sidebar"):
        gr.HTML('<div class="sidebar-brand">DeepSeek <span>Coder</span></div>')
        new_btn = gr.Button("✎  Nuevo chat", variant="secondary", elem_id="new-chat")
        gr.Markdown("Tus chats", elem_id="history-heading")
        chat_list = gr.Radio(choices=[], label="", show_label=False, elem_id="chat-list", interactive=True)
        with gr.Accordion("Almacenamiento de chats", open=False, elem_id="chat-storage"):
            delete_chat_btn = gr.Button("Borrar chat abierto", size="sm", elem_id="delete-chat")
            chat_usage = gr.Markdown(disk_usage(), elem_id="chat-usage")
        with gr.Accordion("Modelos locales", open=False, elem_id="model-library"):
            model_picker = gr.Dropdown(choices=manager.choices(), value=manager.selected_id, label="Modelo", interactive=True, elem_id="model-picker")
            model_info = gr.HTML(model_status(manager.selected_id), elem_id="model-info")
            download_btn = gr.Button("Descargar al disco", size="sm")
            load_btn = gr.Button("Cargar modelo", size="sm", variant="primary")
            unload_btn = gr.Button("Liberar memoria", size="sm")
            gr.Markdown("Descargá una vez. Cargá desde el disco cuando lo necesites. Al enviar un mensaje se carga el último modelo utilizado.")
        with gr.Accordion("Ajustes del modelo", open=False, elem_id="settings"):
            system_box = gr.Textbox(label="System prompt", lines=6, value=DEFAULT_SYSTEM_PROMPT)
            max_tok = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
            temp = gr.Slider(label="Temperature (0 = determinista)", minimum=0.0, maximum=2.0, step=0.05, value=0.0)
            top_p = gr.Slider(label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
            top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
            rep = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.0)
        profile = gr.HTML(model_profile(), elem_id="local-profile")

    with gr.Column(elem_id="main"):
        gr.HTML('<div class="topbar"><span>DeepSeek Coder</span><span class="local-badge">En tu equipo</span></div>', elem_id="topbar")
        with gr.Column(visible=True, elem_id="welcome") as welcome:
            gr.HTML('<div class="welcome-symbol" aria-hidden="true">&gt;_</div>')
            gr.Markdown("# ¿Qué quieres desarrollar hoy?\n\nTu asistente de programación local.")
            with gr.Row(elem_id="examples"):
                ex_btns = [gr.Button(t, size="sm", variant="secondary") for t in EXAMPLES]
        chatbot = gr.Chatbot(visible=False, elem_id="chatbot", height="100%", show_label=False, render_markdown=True, buttons=["copy"], layout="bubble")
        with gr.Column(elem_id="composer", scale=0):
            context_open = gr.State(False)
            with gr.Column(visible=False, elem_id="context-panel") as context_panel:
                gr.HTML('<div class="context-menu-title">Añadir</div>')
                upload_context = gr.UploadButton("📎  Archivos de texto, código o PDF", file_count="multiple", type="filepath", elem_id="context-upload")
                with gr.Accordion("✎  Pegar texto o instrucciones", open=False, elem_id="context-text-option"):
                    context_box = gr.Textbox(label="Texto adicional", show_label=False, placeholder="Pegá código o información para esta consulta…", lines=3, max_lines=5)
                context_files = gr.File(label="Archivos adjuntos", file_count="multiple", type="filepath", interactive=True, visible=False, elem_id="attached-files")
                gr.HTML('<div class="context-menu-title">Documentos</div><div class="pdf-capability"><span class="pdf-icon">PDF</span><div>Leer PDF<span>Documentos con texto seleccionable</span></div></div><div class="context-menu-help">Hasta 10 archivos · 20 MiB por archivo · 50 MiB en total.<br>En documentos largos se seleccionan fragmentos según tu consulta.</div>')
                context_done = gr.Button("Listo", size="sm", elem_id="context-done")
            with gr.Column(elem_id="inputbar", scale=0):
                with gr.Row(elem_id="context-summary"):
                    @gr.render(inputs=[context_box, context_files])
                    def attachment_chips(text, files):
                        if (text or "").strip():
                            clear_text = gr.Button("×  Texto añadido", size="sm", min_width=0, scale=0, elem_classes="context-chip", key="text-context-chip")
                            clear_text.click(lambda: "", outputs=context_box, queue=False)
                        for filename in files or []:
                            remove = gr.Button(f"×  {Path(filename).name}", size="sm", min_width=0, scale=0, elem_classes="context-chip", key=f"attachment-{filename}")
                            remove.click(remove_attachment(filename), inputs=context_files, outputs=[context_files, upload_context], queue=False)

                msg = gr.Textbox(
                    placeholder="Pregunta lo que quieras o describe tu idea",
                    label="Mensaje", show_label=False, container=False,
                    lines=1, max_lines=8, elem_id="msg",
                )
                with gr.Row(elem_id="composer-toolbar"):
                    context_toggle = gr.Button("+", scale=0, min_width=30, elem_id="add-context")
                    composer_model = gr.HTML(composer_model_label(), elem_id="composer-model")
                    send = gr.Button("↑", variant="primary", scale=0, min_width=30, elem_id="send")
                    stop_btn = gr.Button("■", visible=False, scale=0, min_width=30, elem_id="stop-generation")

    def toggle_context(opened):
        return not opened, gr.update(visible=not opened)

    def add_attachments(uploaded, existing):
        combined = list(dict.fromkeys([*(existing or []), *(uploaded or [])]))
        if len(combined) > 10 or sum(Path(f).stat().st_size for f in combined) > 50 * 1024 * 1024:
            gr.Warning("Límite: 10 archivos y 50 MiB en total.")
            return gr.skip()
        if any(Path(f).stat().st_size > 20 * 1024 * 1024 for f in combined):
            gr.Warning("El límite por archivo es 20 MiB.")
            return gr.skip()
        return gr.update(value=combined, visible=bool(combined))

    upload_context.upload(add_attachments, [upload_context, context_files], context_files)
    context_toggle.click(toggle_context, context_open, [context_open, context_panel], queue=False)
    context_done.click(lambda: (False, gr.update(visible=False)), outputs=[context_open, context_panel], queue=False)
    # Hide the floating panel on submit without changing the attached content.
    send.click(lambda: (False, gr.update(visible=False)), outputs=[context_open, context_panel], queue=False)
    msg.submit(lambda: (False, gr.update(visible=False)), outputs=[context_open, context_panel], queue=False)

    settings = [system_box, max_tok, temp, top_p, top_k, rep]
    ins = [msg, chatbot, chats_state, cur_state, *settings, context_box, context_files]
    outs = [chatbot, msg, welcome, chats_state, cur_state, chat_list, chat_usage]
    # One shared queue for all operations touching the process-wide model.
    queue_options = dict(concurrency_id="local-model", concurrency_limit=1)
    model_picker.input(model_status, model_picker, model_info)
    gr.Timer(2).tick(lambda: (model_profile(), composer_model_label()), outputs=[profile, composer_model], queue=False, show_progress="hidden")
    model_outs = [model_picker, model_info, profile]
    download_btn.click(download_model, model_picker, model_outs, **queue_options)
    load_btn.click(switch_model, model_picker, model_outs, **queue_options)
    unload_btn.click(unload_model, model_picker, model_outs, **queue_options)
    def refresh_model_info(selected):
        return model_status(selected), model_profile(), gr.update(choices=manager.choices(), value=selected)
    response_outs = [*outs, send, stop_btn, context_box, context_files]
    stop_btn.click(stop_response, outputs=stop_btn, queue=False)
    msg.submit(respond_ui, ins, response_outs, **queue_options).then(refresh_model_info, model_picker, [model_info, profile, model_picker])
    send.click(respond_ui, ins, response_outs, **queue_options).then(refresh_model_info, model_picker, [model_info, profile, model_picker])
    for b in ex_btns:
        b.click(start_chat, b, msg).then(respond_ui, ins, response_outs, **queue_options).then(refresh_model_info, model_picker, [model_info, profile, model_picker])
    demo.load(new_chat, outputs=outs, **queue_options)
    new_btn.click(new_chat, outputs=outs, **queue_options)
    chat_list.input(open_chat, chat_list, outs, **queue_options)
    delete_chat_btn.click(delete_chat, cur_state, outs, **queue_options)

if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        theme=gr.themes.Monochrome(font=["Arial", "sans-serif"]),
        css_paths=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")],
        footer_links=[],
        max_file_size="20mb",
    )
