"""One UTF-8 JSON file per conversation, replaced atomically."""
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock


class ChatStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = RLock()

    def _path(self, chat_id):
        if not isinstance(chat_id, str) or not re.fullmatch(r'[a-f0-9]{8,32}', chat_id):
            raise ValueError('Identificador de chat inválido')
        return self.directory / f'{chat_id}.json'

    def load(self):
        with self.lock:
            chats = {}
            for path in sorted(self.directory.glob('*.json'), key=lambda p: (p.stat().st_mtime_ns, p.name)):
                self._path(path.stem)
                chat = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(chat.get('title'), str) or not isinstance(chat.get('history'), list):
                    raise ValueError(f'Chat inválido: {path.name}')
                chats[path.stem] = chat
            return chats

    def save(self, chat_id, chat):
        with self.lock:
            target = self._path(chat_id)
            self.directory.mkdir(parents=True, exist_ok=True)
            data = json.dumps(chat, ensure_ascii=False, indent=2).encode('utf-8')
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def delete(self, chat_id):
        with self.lock:
            self._path(chat_id).unlink(missing_ok=True)

    def usage(self, selected=None):
        with self.lock:
            files = list(self.directory.glob('*.json'))
            sizes = {p.stem: p.stat().st_size for p in files}
            allocated = sum(p.stat().st_blocks * 512 for p in files)
            return len(files), sum(sizes.values()), allocated, sizes.get(selected, 0)
