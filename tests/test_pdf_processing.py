import os
import tempfile
import unittest

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

    def test_text_layer_removes_short_identifier_noise(self):
        raw = """8H12
20h14
2 отв. Ç2H14
4 радиуса
Стеклотекстолит марки СТЭФ-У (тип 221) Вс-1,0
ТУ 16-89 И79.0066.002ТУ
1 Радиусы внутренних скруглений не более 0,4 мм.
2 * Размер для справок.
3 Маркировать обозначение изделия по основному конструкторскому документу на
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertNotIn("8H12", cleaned)
        self.assertNotIn("20h14", cleaned)
        self.assertNotIn("2 отв. Ç2H14", cleaned)
        self.assertNotIn("4 радиуса", cleaned)
        self.assertIn("Стеклотекстолит марки СТЭФ-У (тип 221) Вс-1,0", cleaned)
        self.assertIn("1 Радиусы внутренних скруглений не более 0,4 мм.", cleaned)
        self.assertIn("3 Маркировать обозначение изделия по основному конструкторскому документу на", cleaned)

    def test_numbered_requirement_lines_are_kept_in_sequence(self):
        raw = """1 Гр. II КТ 60 ОСТ 95-10-72.
2 Общие допуски по ГОСТ 30893.2-2002-mK.
3 Неуказанные радиусы внутренних скруглений не более 0.4 мм. Обеспеч. инстр.
4 * Размеры обеспеч. инстр.
5 Маркировать обозначение изделия по основному конструкторскому документу на бирке.
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertIn("1 Гр. II КТ 60 ОСТ 95-10-72.", cleaned)
        self.assertIn("2 Общие допуски по ГОСТ 30893.2-2002-mK.", cleaned)
        self.assertIn("3 Неуказанные радиусы внутренних скруглений не более 0.4 мм. Обеспеч. инстр.", cleaned)
        self.assertIn("4 * Размеры обеспеч. инстр.", cleaned)
        self.assertIn("5 Маркировать обозначение изделия по основному конструкторскому документу на бирке.", cleaned)

    def test_numbered_requirement_items_with_continuation_lines(self):
        raw = """4 отв. Ç18

Б-Б
(R3)
2 радиуса

Сталь 12Х18Н10Т ГОСТ 5632-2014

Копировал
Формат
CRC:
A4•4
6C06FA15
ID версии: 74b68533-b381-40eb-a595-9eabbe2c6469

1 Гр. II-Б ОСТ 95-29-72
Заготовка должна поставляться с ограничением по массовой доле:
- фосфор не более 0,035 %;
- медь не более 0,30 %.
2 Материал-заменитель: Сталь 08Х18Н10Т ГОСТ 5632-2014.
3 Общие допуски по ГОСТ 30893.2-2002-mK.
4 Отверстие Г допускается производить в сборочной еде.
5 Неуказанный радиус внутреннего скругления не более 0,4 мм. Обеспеч. инстр.
6 Размер в скобках - после сборки.
7 Маркировать шрифтом 3...5-Пр3 ГОСТ 26.020-80.
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertNotIn("Сталь 12Х18Н10Т ГОСТ 5632-2014", cleaned)
        self.assertNotIn("Копировал", cleaned)
        self.assertNotIn("Формат", cleaned)
        self.assertNotIn("A4", cleaned)
        self.assertNotIn("6C06FA15", cleaned)
        self.assertIn("1 Гр. II-Б ОСТ 95-29-72", cleaned)
        self.assertIn("Заготовка должна поставляться с ограничением по массовой доле:", cleaned)
        self.assertIn("2 Материал-заменитель: Сталь 08Х18Н10Т ГОСТ 5632-2014.", cleaned)
        self.assertIn("3 Общие допуски по ГОСТ 30893.2-2002-mK.", cleaned)
        self.assertIn("5 Неуказанный радиус внутреннего скругления не более 0,4 мм. Обеспеч. инстр.", cleaned)
        self.assertIn("7 Маркировать шрифтом 3...5-Пр3 ГОСТ 26.020-80.", cleaned)


    def test_user_noise_regression(self):
        raw = """Допуск на торцовое биение
 базовой поверхности

Допуск на радиальное биение 
 зубчатого венца
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertNotIn("Допуск на торцовое биение", cleaned)
        self.assertNotIn("ГОСТ 13755-2015", cleaned)


if __name__ == "__main__":
    unittest.main()

    def test_user_noise_regression(self):
        raw = """Допуск на торцовое биение
 базовой поверхности

Допуск на радиальное биение 
 зубчатого венца
"""
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(raw)
        self.assertNotIn("Допуск на торцовое биение", cleaned)
        self.assertNotIn("ГОСТ 13755-2015", cleaned)

