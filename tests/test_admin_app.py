import asyncio
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from admin_app import build_requirement_preview_html, get_admin_credentials, get_unique_token_stats, process_uploaded_pdf, render_test_page


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
