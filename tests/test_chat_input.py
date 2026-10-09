from threading import Event
import pytest
from demo.chat_input import GenerationControls, StopGeneration, with_context


def test_stop_is_per_session_and_new_generation_is_reset():
    controls = GenerationControls()
    a, b = controls.begin('a'), controls.begin('b')
    criteria = StopGeneration(a)
    assert not criteria(None, None)
    controls.stop('a')
    assert criteria(None, None)
    assert not b.is_set()
    controls.finish('a', a)
    assert not controls.begin('a').is_set()


def test_text_context_is_preserved(tmp_path):
    file = tmp_path / 'sample.py'
    file.write_text('print("¡Hola!")', encoding='utf-8')
    result = with_context('Explicá el código', 'En español', [str(file)])
    assert 'Explicá el código' in result
    assert 'En español' in result
    assert 'sample.py' in result
    assert 'print("¡Hola!")' in result
    assert with_context('Hola', '', []) == 'Hola'


def test_reject_binary_and_large_context(tmp_path):
    file = tmp_path / 'data.bin'
    file.write_bytes(b'\x00\x01')
    with pytest.raises(ValueError, match='binarios'):
        with_context('Hola', '', [str(file)])
    with pytest.raises(ValueError, match='6.000'):
        with_context('Hola', 'x' * 6001, [])
    file.write_bytes(b'x' * (20 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match='20 MiB'):
        with_context('Hola', '', [str(file)])


def test_long_file_selects_relevant_passage(tmp_path):
    file = tmp_path / 'large.txt'
    file.write_text(('irrelevant filler ' * 10000) + '\nLa contraseña del servidor es un ejemplo de contexto objetivo.', encoding='utf-8')
    result = with_context('Explicá la contraseña del servidor', '', [str(file)])
    assert 'Fragmentos seleccionados' in result
    assert 'contraseña del servidor' in result
    assert len(result) < 11000


def test_pdf_text_and_scanned_pdf(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    from demo.chat_input import read_attachment
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 30 200 Td (PDF attachment text) Tj ET')
    page[NameObject('/Contents')] = stream
    path = tmp_path / 'test.pdf'
    writer.write(path)
    result = read_attachment(path)
    assert 'PDF attachment text' in result
    assert '[Página 1]' in result
    blank = PdfWriter()
    blank.add_blank_page(width=300, height=300)
    blank.write(path)
    with pytest.raises(ValueError, match='OCR'):
        read_attachment(path)
