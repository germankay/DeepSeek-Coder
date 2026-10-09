import pytest
from demo.chat_store import ChatStore


def test_persistence_usage_and_delete(tmp_path):
    store = ChatStore(tmp_path / 'chats')
    chat = {'title': '¿Cómo estás?', 'history': [{'role': 'user', 'content': 'Hola\n¡Sí!'}]}
    assert store.load() == {}
    store.save('abcdef12', chat)
    reopened = ChatStore(store.directory)
    assert reopened.load() == {'abcdef12': chat}
    count, total, allocated, selected = reopened.usage('abcdef12')
    assert count == 1
    assert total == selected == (store.directory / 'abcdef12.json').stat().st_size
    assert allocated > 0
    reopened.delete('abcdef12')
    assert reopened.load() == {}
    assert reopened.usage() == (0, 0, 0, 0)


def test_updating_one_chat_preserves_others(tmp_path):
    store = ChatStore(tmp_path)
    store.save('abcdef12', {'title': 'Uno', 'history': []})
    store.save('abcdef13', {'title': 'Dos', 'history': []})
    store.save('abcdef12', {'title': 'Uno', 'history': [{'role': 'assistant', 'content': 'Respuesta'}]})
    assert len(store.load()) == 2
    assert store.load()['abcdef12']['history'][0]['content'] == 'Respuesta'
    assert not list(tmp_path.glob('*.tmp'))


def test_rejects_paths_outside_storage(tmp_path):
    store = ChatStore(tmp_path)
    with pytest.raises(ValueError):
        store.delete('../file')
