"""Offline model-library tests: no downloads or GPU allocations."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from demo import model_manager as mm


@pytest.fixture
def library(tmp_path, monkeypatch):
    snapshots = {}
    def snapshot_download(model_id, **kwargs):
        assert kwargs['cache_dir'] == tmp_path
        if model_id not in snapshots:
            raise OSError('not cached')
        return str(snapshots[model_id])
    monkeypatch.setattr(mm, 'snapshot_download', Mock(side_effect=snapshot_download))
    manager = mm.ModelManager('deepseek-ai/deepseek-coder-1.3b-instruct', 'cpu', tmp_path)
    def add(model_id):
        path = tmp_path / model_id.split('/')[-1]
        path.mkdir()
        (path / 'config.json').write_text('{}')
        (path / 'tokenizer_config.json').write_text('{}')
        (path / 'tokenizer.json').write_text('{}')
        (path / 'model.safetensors').write_bytes(b'fixture')
        snapshots[model_id] = path
        return path
    return manager, add


def test_reuse_saved_download_without_network(library):
    manager, add = library
    path = add(manager.selected_id)
    assert manager.download(manager.selected_id) == path
    assert all(c.kwargs.get('local_files_only') is True for c in mm.snapshot_download.call_args_list)


def test_sharded_snapshot_requires_every_shard(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'tokenizer_config.json').write_text('{}')
    (tmp_path / 'tokenizer.json').write_text('{}')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
    (tmp_path / 'one.safetensors').write_bytes(b'a')
    assert not mm.complete_snapshot(tmp_path)
    (tmp_path / 'two.safetensors').write_bytes(b'b')
    assert mm.complete_snapshot(tmp_path)


def test_load_releases_previous_and_uses_only_local_files(library, monkeypatch):
    manager, add = library
    selected = 'Qwen/Qwen2.5-Coder-3B-Instruct'
    path = add(selected)
    manager.model = object()
    manager.tokenizer = object()
    manager.loaded_id = manager.selected_id
    def load_weights(*args, **kwargs):
        assert manager.model is None
        assert manager.loaded_id is None
        assert args == (str(path),)
        assert kwargs['local_files_only'] is True
        result = Mock()
        result.to.return_value = result
        return result
    tokenizer_loader = Mock(return_value=SimpleNamespace())
    weights_loader = Mock(side_effect=load_weights)
    monkeypatch.setattr(mm.PreTrainedTokenizerFast, 'from_pretrained', tokenizer_loader)
    monkeypatch.setattr(mm.AutoModelForCausalLM, 'from_pretrained', weights_loader)
    manager.load(selected)
    manager.load(selected)
    assert weights_loader.call_count == 1
    assert tokenizer_loader.call_args.kwargs['local_files_only'] is True
    assert manager.loaded_id == selected
    manager.unload()
    assert manager.model is None and manager.tokenizer is None
    assert manager.loaded_id is None
    assert manager.local_path(selected) == path


def test_missing_model_keeps_active_model(library):
    manager, _ = library
    original = object()
    manager.model = original
    manager.loaded_id = manager.selected_id
    with pytest.raises(ValueError, match='Descargar'):
        manager.load('Qwen/Qwen2.5-Coder-7B-Instruct')
    assert manager.model is original
    assert all(c.kwargs.get('local_files_only') is True for c in mm.snapshot_download.call_args_list)


def test_failed_load_clears_partial_state(library, monkeypatch):
    manager, add = library
    add(manager.selected_id)
    monkeypatch.setattr(mm.PreTrainedTokenizerFast, 'from_pretrained', Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(mm.AutoModelForCausalLM, 'from_pretrained', Mock(side_effect=RuntimeError('out of memory')))
    with pytest.raises(RuntimeError, match='out of memory'):
        manager.load(manager.selected_id)
    assert manager.model is None and manager.tokenizer is None
    assert manager.loaded_id is None


def test_explicit_download_fetches_missing_snapshot(library, monkeypatch):
    manager, add = library
    def download(model_id, **kwargs):
        if kwargs.get('local_files_only'):
            raise OSError('missing')
        return add(model_id)
    mocked = Mock(side_effect=download)
    monkeypatch.setattr(mm, 'snapshot_download', mocked)
    assert manager.download(manager.selected_id).is_dir()
    assert mocked.call_count == 2
    assert mocked.call_args.kwargs['allow_patterns'] == mm.DOWNLOAD_PATTERNS


def test_model_change_waits_for_generation_lock(library):
    from threading import Event, Thread
    manager, _ = library
    manager.model = object()
    started, finished = Event(), Event()
    def unload():
        started.set()
        manager.unload()
        finished.set()
    with manager.lock:
        worker = Thread(target=unload)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(.05)
        assert manager.model is not None
    worker.join(timeout=2)
    assert finished.is_set()
    assert manager.model is None
