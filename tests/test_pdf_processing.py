import os
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv
from pdf_processors import PdfProcessingFactory, TextLayerPdfProcessor, TokenStorage


class PdfProcessingTests(unittest.TestCase):
    def test_factory_selects_text_layer_processor_for_cdw_pdf(self):
        processor = PdfProcessingFactory.create("test.cdw.PDF")
        self.assertEqual(processor.__class__.__name__, "TextLayerPdfProcessor")

    def test_factory_selects_text_layer_processor_for_cwd_pdf(self):
        processor = PdfProcessingFactory.create("test.cwd.pdf")
        self.assertEqual(processor.__class__.__name__, "TextLayerPdfProcessor")

    def test_factory_selects_ocr_processor_for_dwg_pdf(self):
        processor = PdfProcessingFactory.create("test.dwg.pdf")
        self.assertEqual(processor.__class__.__name__, "OCRPdfProcessor")

    def test_token_storage_keeps_unique_blocks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "tokens.db")
            store = TokenStorage(db_path=db_path)
            block = "ГОСТ 1234\nГОСТ 5678\nГОСТ 1234"
            store.store_block_tokens(block, source_file="example.pdf")
            tokens = store.get_unique_tokens()
            self.assertIn(block, tokens)
            self.assertEqual(len(tokens), 1)

    def test_factory_respects_allowed_file_types_env(self):
        env_path = Path(__file__).resolve().parents[1] / ".env"
        original = env_path.read_text() if env_path.exists() else None
        try:
            env_path.write_text("ALLOWED_FILE_TYPES=cdw.pdf\n")
            load_dotenv(env_path, override=True)
            self.assertIsNone(PdfProcessingFactory.create("test.dwg.pdf"))
            self.assertEqual(PdfProcessingFactory.create("test.cdw.pdf").__class__.__name__, "TextLayerPdfProcessor")
        finally:
            if original is None:
                if env_path.exists():
                    env_path.unlink()
            else:
                env_path.write_text(original)
            load_dotenv(env_path, override=True)

    def test_runtime_env_reload_applies_new_allowed_types(self):
        env_path = Path(__file__).resolve().parents[1] / ".env"
        original = env_path.read_text() if env_path.exists() else None
        try:
            env_path.write_text("ALLOWED_FILE_TYPES=dwg.pdf\n")
            load_dotenv(env_path, override=True)
            self.assertIsNone(PdfProcessingFactory.create("test.cdw.pdf"))
            self.assertEqual(PdfProcessingFactory.create("test.dwg.pdf").__class__.__name__, "OCRPdfProcessor")
        finally:
            if original is None:
                if env_path.exists():
                    env_path.unlink()
            else:
                env_path.write_text(original)
            load_dotenv(env_path, override=True)

    def test_text_layer_filters_noise(self):
        raw = """Документ подписан усиленной квалифицированной ЭЦП
Сертификат : 03с35176006Бае7544ас4 066969 013а
1 Гр. 1-Б ОСТ 95-29-72.
2 Заготовка должна поставляться с ограничением по массе доли:
- фосфор не более 0,035 %;
- медь не более 0,50 %.
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertIn("1 Гр. 1-Б ОСТ 95-29-72.", cleaned)
        self.assertIn("2 Заготовка должна поставляться", cleaned)
        self.assertNotIn("Документ подписан", cleaned)
        self.assertNotIn("Сертификат", cleaned)


if __name__ == "__main__":
    unittest.main()
