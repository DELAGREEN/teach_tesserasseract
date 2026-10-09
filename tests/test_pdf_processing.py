import os
import tempfile
import unittest

from pdf_processors import PdfProcessingFactory, TokenStorage


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

    def test_token_storage_keeps_unique_tokens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "tokens.db")
            store = TokenStorage(db_path=db_path)
            store.store_block_tokens("ГОСТ 1234 ГОСТ 5678", source_file="example.pdf")
            tokens = store.get_unique_tokens()
            self.assertIn("ГОСТ", tokens)
            self.assertIn("1234", tokens)
            self.assertIn("5678", tokens)
            self.assertEqual(len(tokens), 3)


if __name__ == "__main__":
    unittest.main()
