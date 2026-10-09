"""Text attachments and cooperative, per-session generation cancellation."""
from pathlib import Path
import re
from threading import Event, Lock
from transformers import StoppingCriteria


class StopGeneration(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class GenerationControls:
    def __init__(self):
        self.lock = Lock()
        self.active = {}

    def begin(self, session):
        event = Event()
        with self.lock:
            self.active[session] = event
        return event

    def stop(self, session):
        with self.lock:
            event = self.active.get(session)
            if event:
                event.set()

    def finish(self, session, event):
        with self.lock:
            if self.active.get(session) is event:
                del self.active[session]


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_FILES = 10
MAX_EXTRACTED_CHARS = 1_000_000


def read_attachment(path):
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path.name}: el límite es 20 MiB por archivo.")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("Instalá las dependencias de la demo para leer PDF: pip install -r demo/requirement.txt") from None
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("El PDF tiene contraseña. Adjuntá una copia sin protección.")
            if len(reader.pages) > 300:
                raise ValueError("El PDF supera las 300 páginas. Dividilo en documentos más pequeños.")
            pages, count = [], 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                count += len(text)
                if count > MAX_EXTRACTED_CHARS:
                    raise ValueError("El PDF contiene demasiado texto. Dividilo en documentos más pequeños.")
                if text.strip():
                    pages.append(f"[Página {number}]\n{text}")
            if not pages:
                raise ValueError("El PDF no contiene texto extraíble. Los documentos escaneados necesitan OCR.")
            return "\n\n".join(pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"No se pudo leer {path.name}: el PDF está dañado o no es compatible.") from exc
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path.name}: usá texto UTF-8 o PDF. No se admiten otros archivos binarios.") from None
    if "\x00" in text:
        raise ValueError(f"{path.name}: no se admiten archivos binarios.")
    return text


def relevant_excerpt(text, question, budget):
    if len(text) <= budget:
        return text
    # Bound the prompt, not the upload: choose matching passages throughout the file.
    terms = set(re.findall(r"\w{3,}", question.lower())) - {"que", "del", "los", "las", "una", "para", "con", "the", "and"}
    chunks = [(index, text[index:index + 900]) for index in range(0, len(text), 800)]
    ranked = sorted(chunks, key=lambda item: (-sum(min(item[1].lower().count(term), 5) for term in terms), item[0]))
    picked = sorted(ranked[:max(1, (budget - 100) // 920)])
    return "[Fragmentos seleccionados; no es el documento completo]\n" + "\n[…]\n".join(chunk for _, chunk in picked)[:budget - 80]


def with_context(message, context, files):
    context = (context or "").strip()
    if len(context) > 6000:
        raise ValueError("El texto adicional supera los 6.000 caracteres.")
    files = files or []
    if len(files) > MAX_FILES:
        raise ValueError("Adjuntá como máximo 10 archivos.")
    if sum(Path(file).stat().st_size for file in files) > MAX_TOTAL_BYTES:
        raise ValueError("Los archivos superan los 50 MiB en total.")
    parts = [context] if context else []
    budget = max(250, (10000 - len(context)) // max(1, len(files)))
    for filename in files:
        path = Path(filename)
        text = read_attachment(path)
        excerpt = relevant_excerpt(text, message, budget)
        parts.append(f"Archivo: {path.name}\n```\n{excerpt}\n```")
    if not parts:
        return message
    return f"{message}\n\n--- Contexto adjunto ---\n" + "\n\n".join(parts)
