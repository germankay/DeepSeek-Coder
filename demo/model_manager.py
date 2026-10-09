"""Local model library shared by the demo; downloads are always explicit."""
import gc
import json
import os
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
CACHE_DIR = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")))

from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, BitsAndBytesConfig

CATALOG = {
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek-Coder 6.7B",
    "deepseek-ai/deepseek-coder-1.3b-instruct": "DeepSeek-Coder 1.3B",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder 7B",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen2.5-Coder 3B",
}
# UI descriptions live with the application, independently of cached weights.
MODEL_DETAILS = {
    "deepseek-ai/deepseek-coder-6.7b-instruct": {
        "purpose": "Programación general, explicación de código y ayuda con errores.",
        "tradeoff": "Más capacidad que el 1.3B, con mayor consumo de memoria.",
        "vram": "5–7 GB",
    },
    "deepseek-ai/deepseek-coder-1.3b-instruct": {
        "purpose": "Consultas sencillas, ejemplos cortos y respuestas rápidas.",
        "tradeoff": "Consume menos memoria; tiene más limitaciones en tareas complejas.",
        "vram": "1,5–3 GB",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "purpose": "Priorizar calidad de código: generación, revisión y corrección de funciones.",
        "tradeoff": "La opción para dedicar más memoria a tareas de programación exigentes.",
        "vram": "5–7 GB",
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "purpose": "Programación cotidiana con más velocidad y memoria disponible.",
        "tradeoff": "Un equilibrio entre rapidez y capacidad; menos margen para problemas complejos que el 7B.",
        "vram": "3–4 GB",
    },
}

DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"]


def validate_model_id(model_id):
    if not model_id or any(word in model_id.lower() for word in ("v2", "v3", "r1")):
        raise ValueError("Seleccioná un modelo compatible. DeepSeek V2, V3 y R1 no están habilitados.")


def complete_snapshot(path):
    """A snapshot directory can exist even when a download was interrupted."""
    path = Path(path)
    try:
        json.loads((path / "config.json").read_text())
        json.loads((path / "tokenizer_config.json").read_text())
        if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
            return False
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weights = set(json.loads(index.read_text())["weight_map"].values())
        else:
            weights = {"model.safetensors"}
        return bool(weights) and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in weights)
    except (OSError, ValueError, KeyError, TypeError):
        return False


class ModelManager:
    def __init__(self, default_id, device, cache_dir=CACHE_DIR):
        validate_model_id(default_id)
        self.selected_id = default_id
        self.loaded_id = None
        self.model = None
        self.tokenizer = None
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.lock = Lock()

    def local_path(self, model_id):
        validate_model_id(model_id)
        try:
            path = snapshot_download(model_id, cache_dir=self.cache_dir, local_files_only=True, allow_patterns=DOWNLOAD_PATTERNS)
            return Path(path) if complete_snapshot(path) else None
        except (OSError, ValueError):
            return None

    def choices(self):
        ids = dict(CATALOG)
        ids.setdefault(self.selected_id, self.selected_id.split("/")[-1])
        for folder in sorted(self.cache_dir.glob("models--*")):
            model_id = folder.name.removeprefix("models--").replace("--", "/")
            try:
                validate_model_id(model_id)
                ids.setdefault(model_id, model_id.split("/")[-1])
            except ValueError:
                continue
        return [(f"{name} · {'🟢 en uso' if mid == self.loaded_id else ('guardado' if self.local_path(mid) else 'por descargar')}", mid) for mid, name in ids.items()]

    def download(self, model_id):
        validate_model_id(model_id)
        with self.lock:
            path = self.local_path(model_id)
            if path:
                return path
            path = snapshot_download(model_id, cache_dir=self.cache_dir, allow_patterns=DOWNLOAD_PATTERNS)
            if not complete_snapshot(path):
                raise ValueError("La descarga no está completa o el formato no es compatible. Reintentá la descarga.")
            return Path(path)

    def unload(self):
        with self.lock:
            self._unload()

    def _unload(self):
        self.model = None
        self.tokenizer = None
        self.loaded_id = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def load(self, model_id):
        with self.lock:
            self._load(model_id)

    def _load(self, model_id):
        """Caller holds lock, including throughout streaming generation."""
        if self.loaded_id == model_id and self.model is not None:
            return
        path = self.local_path(model_id)
        if path is None:
            raise ValueError("Este modelo todavía no está guardado. Usá «Descargar al disco» primero.")
        # Validate before releasing the working model; never hold two models in VRAM.
        self._unload()
        self.selected_id = model_id
        quant = os.getenv("QUANTIZATION", "auto").lower()
        if quant == "auto":
            quant = "4bit" if self.device == "cuda" else "none"
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        config = None
        if self.device == "cuda" and quant == "4bit":
            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        elif self.device == "cuda" and quant == "8bit":
            config = BitsAndBytesConfig(load_in_8bit=True)
        try:
            # Preserve the saved tokenizer pipeline. Transformers 5 can dispatch
            # legacy DeepSeek configs to LlamaTokenizer and replace ByteLevel.
            tokenizer_class = PreTrainedTokenizerFast if (path / "tokenizer.json").is_file() else AutoTokenizer
            self.tokenizer = tokenizer_class.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            self.tokenizer.use_default_system_prompt = False
            self.model = AutoModelForCausalLM.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False,
                dtype=dtype, device_map="auto" if self.device == "cuda" else None,
                quantization_config=config,
            )
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            self.loaded_id = model_id
        except Exception:
            self._unload()
            raise
