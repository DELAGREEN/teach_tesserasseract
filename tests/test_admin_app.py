import asyncio
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from admin_app import build_requirement_preview_html, get_admin_credentials, get_unique_token_stats, process_uploaded_pdf, render_pdf_preview, render_test_page


class AdminAppTests(unittest.TestCase):
    def test_admin_credentials_are_loaded_from_env(self):
        os.environ["ADMIN_LOGIN"] = "dev-admin"
        os.environ["ADMIN_PASSWORD"] = "dev-pass"
        self.assertEqual(get_admin_credentials()[0], "dev-admin")
        self.assertEqual(get_admin_credentials()[1], "dev-pass")

    def test_unique_token_stats_reads_sqlite_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "unique_tokens.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE unique_tokens (token TEXT PRIMARY KEY, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, source_file TEXT, seen_count INTEGER NOT NULL DEFAULT 1)"
            )
            conn.execute(
                "INSERT INTO unique_tokens (token, first_seen_at, last_seen_at, source_file, seen_count) VALUES (?, ?, ?, ?, ?)",
                ("ГОСТ", "2024-01-01", "2024-01-01", "example.pdf", 2),
            )
            conn.execute(
                "INSERT INTO unique_tokens (token, first_seen_at, last_seen_at, source_file, seen_count) VALUES (?, ?, ?, ?, ?)",
                ("1234", "2024-01-02", "2024-01-02", "example.pdf", 3),
            )
            conn.commit()
            conn.close()

            stats = get_unique_token_stats(db_path)
            self.assertEqual(stats["total_unique_tokens"], 2)
            self.assertIn("ГОСТ", [row["token"] for row in stats["tokens"]])


    def test_render_test_page_includes_preview_and_fullscreen_modal(self):
        result = {
            "text": "1 Требование\n2 Требование два",
            "blocks": [
                {"text": "1 Требование", "source": "sample.pdf"},
                {"text": "2 Требование два", "source": "sample.pdf"},
            ],
        }
        html = render_test_page(result=result)
        body = html.body.decode() if hasattr(html.body, "decode") else str(html.body)
        self.assertIn("thumbnail-preview", body)
        self.assertIn("fullscreen-preview", body)
        self.assertIn("requirement-box", body)

    def test_build_requirement_preview_uses_pdf_render_data(self):
        preview = build_requirement_preview_html(
            preview_data_url="data:image/png;base64,AAA",
            blocks=[{"text": "Блок 1"}, {"text": "Блок 2"}],
        )
        self.assertIn("thumbnail-preview", preview)
        self.assertIn("requirement-box", preview)
        self.assertIn("fullscreen-preview", preview)

    def test_render_pdf_preview_uses_block_bbox_metadata(self):
        class FakePix:
            def __init__(self):
                self.calls = []

            def draw_rect(self, rect, color=None, fill=None, overlay=None):
                self.calls.append((rect, color, fill, overlay))

            def tobytes(self, fmt):
                return b"png-bytes"

        class FakePage:
            rect = type("Rect", (), {"width": 1000, "height": 800})()

            def get_pixmap(self, matrix=None, alpha=None):
                return FakePix()

        class FakeDoc:
            def __init__(self):
                self.page = FakePage()

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.page

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_pix = FakePix()
        fake_doc = FakeDoc()
        fake_doc.page.get_pixmap = lambda matrix=None, alpha=None: fake_pix

        with patch("admin_app.fitz") as fitz_mock:
            fitz_mock.open.return_value.__enter__.return_value = fake_doc
            fitz_mock.Matrix.return_value = object()
            fitz_mock.Rect = lambda x0, y0, x1, y1: type("Rect", (), {"x0": x0, "y0": y0, "x1": x1, "y1": y1})()

            result = render_pdf_preview(b"fake-pdf", [{"text": "Требование", "bbox": {"x": 22, "y": 18, "w": 48, "h": 12}}])

        self.assertTrue(result.startswith("data:image/png;base64,"))
        self.assertTrue(len(fake_pix.calls) >= 1)
        rect = fake_pix.calls[0][0]
        self.assertGreater(rect.x0, 0)
        self.assertGreater(rect.y0, 0)


    def test_process_uploaded_pdf_accepts_async_upload_file(self):
        class DummyUploadFile:
            filename = "sample.pdf"
            def __init__(self):
                self.calls = 0

            async def read(self):
                self.calls += 1
                return b"fake-pdf-bytes"

        fake_result = {"text": "processed", "blocks": ["block-1"]}
        fake_processor = Mock()
        fake_processor.process.return_value = fake_result

        with patch("admin_app.PdfProcessingFactory.create", return_value=fake_processor) as create_mock:
            result = asyncio.run(process_uploaded_pdf(DummyUploadFile()))

        self.assertEqual(result, fake_result)
        create_mock.assert_called_once()
        fake_processor.process.assert_called_once_with(b"fake-pdf-bytes", source_name="sample.pdf")


if __name__ == "__main__":
    unittest.main()
