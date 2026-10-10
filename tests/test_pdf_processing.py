import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv
from pdf_processors import OCRPdfProcessor, PdfProcessingFactory, TextLayerPdfProcessor, TokenStorage


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

    def test_ocr_processing_does_not_store_dwg_results_in_token_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "tokens.db")
            store = TokenStorage(db_path=db_path)
            processor = PdfProcessingFactory.create("test.dwg.pdf", token_storage=store)
            self.assertEqual(processor.__class__.__name__, "OCRPdfProcessor")

            import types
            import pdf_processors

            pdf_processors.cv2 = types.SimpleNamespace(
                cvtColor=lambda image, code: image,
                GaussianBlur=lambda image, kernel, sigma: image,
                threshold=lambda image, thresh, maxval, flags: (None, image),
                COLOR_BGR2GRAY=0,
                THRESH_BINARY=0,
                THRESH_OTSU=0,
            )
            pdf_processors.np = types.SimpleNamespace(array=lambda value: value[0])
            pdf_processors.pytesseract = types.SimpleNamespace(image_to_string=lambda *args, **kwargs: "Распознанный текст DWG")
            pdf_processors.convert_from_bytes = lambda pdf_bytes, dpi=300: [b"page-image"]

            result = processor.process(b"dummy-pdf", source_name="test.dwg.pdf")

            self.assertIn("Распознанный текст DWG", result["text"])
            with sqlite3.connect(db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM unique_tokens").fetchone()[0]
            self.assertEqual(total, 0)

    def test_text_layer_keeps_numbered_requirements_and_rejects_doc_metadata(self):
        raw = """.: С Че веносауих А Б поберхности ЁЕ опоры поз. 1 0,5 мм на длине 1 м. Допускается пробка. 1 СВарку Выполнить по произбодстбенно-технологической
6 После сварки контролировать прохождение имитаторов документации предприятия-изготобителя, разработанной с учетом
КЛАБ.7 16 331.012 В трубах поз. 2, 3. Прохождение должно быть свободным, требований разделоб 2-4, 6-13, 15, 18 ОСТ 5Р.9653-2015:
- сВарные шбы №1, 4 аргонодугобым способом проволокой
7 Покрытие, кроме опоры поз. 1 и труб поз. 2, 3: эмаль ПФ-115 (2)
- сварной шоб № 2 электродом 3ЭА-395/9 ОСТ В5Р.9574-81,
- сбарные шбы № 5 аргонодуговым способом проболокоц
2 Контроль качество сбарных соединении быполнить по ХУ!
комплексу контроля ОСТ 5Р.9654-2015. Нормы оценки качества по
3 * Размеры для спрабок.
Взаимное росположение поза Д трубы поз. 5, лыски Ж опоры поз. 1 ц
плуты поз. 4& по чертежу.
5 Допуск перпендекулярности труд поз. 2, 5 относительно
Г (1:1)
Д й квалиф ЭЦП | Д й квалиф й ЭЦП
документ подписан усиленнои квалифицированнои Сертификат : 032а1960006сае3297464а? а а1а59ееа
Владелец подписи : Морозова Дарья Валерьевна Действительно с 22.06.2022 до 22.09.2023
Лист. №докцм. |Подп. |Дата П
10 Версии: 093 6-1256-6407-0272-1013<#57055е Копировал СВС: 041А2687 Формат АёхЁ
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertIn("1 СВарку Выполнить", cleaned)
        self.assertIn("6 После сварки контролировать", cleaned)
        self.assertIn("7 Покрытие, кроме опоры", cleaned)
        self.assertNotIn("документ подписан", cleaned.lower())
        self.assertNotIn("Владелец подписи", cleaned.lower())
        self.assertNotIn("Лист. №докцм.", cleaned)
        self.assertNotIn("Формат АёхЁ", cleaned)

    def test_ocr_extracts_requirement_blocks_and_skips_metadata(self):
        sample = {
            "text": [
                "", "1", "СВарку", "Выполнить", "по", "произв.",
                "6", "После", "сварки", "контролировать",
                "", "Лист", "№", "докум.",
                "", "Формат", "A3"
            ],
            "conf": ["-1", "93", "92", "92", "90", "82", "94", "93", "92", "90", "-1", "0", "0", "0", "-1", "0", "0"],
            "left": [0, 10, 20, 70, 100, 150, 0, 10, 80, 150, 0, 20, 40, 60, 0, 10, 30],
            "top": [0, 10, 10, 10, 10, 10, 40, 40, 40, 40, 200, 200, 200, 200, 260, 260, 260],
            "width": [0, 20, 40, 60, 20, 40, 20, 60, 120, 100, 20, 30, 40, 50, 30, 40, 30],
            "height": [0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        }
        blocks = OCRPdfProcessor._extract_requirement_blocks(sample)
        self.assertTrue(any("1 СВарку Выполнить" in block for block in blocks))
        self.assertFalse(any("Лист" in block for block in blocks))
        self.assertFalse(any("Формат" in block for block in blocks))

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
