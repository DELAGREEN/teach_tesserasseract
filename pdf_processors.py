import os
import re
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
    convert_from_bytes = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

from logger import logger


class TokenStorage:
    """Хранилище уникальных токенов текста для последующего обучения."""

    def __init__(self, db_path="results/unique_tokens.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unique_tokens (
                    token TEXT PRIMARY KEY,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    source_file TEXT,
                    seen_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT,
                    block_hash TEXT UNIQUE,
                    block_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_unique_tokens_seen_count ON unique_tokens(seen_count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_blocks_source_file ON processed_blocks(source_file)"
            )

    @staticmethod
    def _normalize_token(token):
        if token is None:
            return ""
        return str(token).replace("\xa0", " ").strip()

    @staticmethod
    def _tokenize_block_text(text):
        if text is None:
            return []
        normalized = str(text).replace("\xa0", " ")
        return [tok for tok in normalized.split() if tok.strip()]

    def store_block_tokens(self, block_text, source_file=""):
        if not block_text or not str(block_text).strip():
            return 0

        tokens = self._tokenize_block_text(block_text)
        if not tokens:
            return 0

        block_hash = self._hash_block(block_text)
        now = datetime.utcnow().isoformat(timespec="seconds")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO processed_blocks (source_file, block_hash, block_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (source_file, block_hash, str(block_text), now),
            )

            for raw_token in tokens:
                clean = self._normalize_token(raw_token)
                if not clean:
                    continue
                conn.execute(
                    """
                    INSERT INTO unique_tokens (token, first_seen_at, last_seen_at, source_file, seen_count)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(token) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        source_file = excluded.source_file,
                        seen_count = unique_tokens.seen_count + 1
                    """,
                    (clean, now, now, source_file),
                )

        return len(set(tokens))

    @staticmethod
    def _hash_block(block_text):
        text = "" if block_text is None else str(block_text)
        return str(abs(hash(text.strip())))

    def get_unique_tokens(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT token FROM unique_tokens ORDER BY token").fetchall()
            return [row["token"] for row in rows]

    def get_token_stats(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT token, seen_count, source_file, last_seen_at FROM unique_tokens ORDER BY token"
            ).fetchall()
            return [dict(row) for row in rows]


class BasePdfProcessor(ABC):
    def __init__(self, token_storage=None):
        self.token_storage = token_storage

    @classmethod
    @abstractmethod
    def supports(cls, file_name):
        raise NotImplementedError

    @abstractmethod
    def process(self, pdf_bytes, source_name=""):
        raise NotImplementedError

    def _store_block_text(self, block_text, source_name=""):
        if self.token_storage is not None:
            self.token_storage.store_block_tokens(block_text, source_name)


class OCRPdfProcessor(BasePdfProcessor):
    """OCR-процессор для .dwg.pdf."""

    @classmethod
    def supports(cls, file_name):
        name = str(file_name or "").lower()
        return ".dwg.pdf" in name or name.endswith(".dwg.pdf")

    def process(self, pdf_bytes, source_name=""):
        if cv2 is None or np is None or pytesseract is None or convert_from_bytes is None:
            raise RuntimeError("Для OCR требуется cv2 / numpy / pytesseract / pdf2image")

        logger.info(f"OCRPdfProcessor: обработка файла {source_name or 'без имени'}")
        pages = convert_from_bytes(pdf_bytes, dpi=300)
        if not pages:
            raise RuntimeError("Не удалось конвертировать PDF в изображения")

        image = np.array(pages[0])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        custom_config = r"--psm 6 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(binary, lang="rus", config=custom_config).strip()
        cleaned = self._clean_ocr_text(text)
        if cleaned:
            self._store_block_text(cleaned, source_name)
        return {
            "text": cleaned,
            "blocks": [{"text": cleaned, "source": source_name}],
            "blocks_data": {"block_1": {"text": cleaned, "source": source_name}},
        }

    @staticmethod
    def _clean_ocr_text(text):
        if not text:
            return ""
        filtered = re.sub(r"\r+", "\n", text)
        filtered = filtered.replace("\xa0", " ")
        filtered = re.sub(r"\n{3,}", "\n\n", filtered)
        return filtered.strip()


class TextLayerPdfProcessor(BasePdfProcessor):
    """Text-layer extractor для .cdw/.cwd PDF, с базовой фильтрацией технических требований."""

    STRONG_KEYWORDS = (
        "ГОСТ",
        "ОСТ",
        "ТУ",
        "РД",
        "МАТЕРИАЛ",
        "ПОКРЫТИЕ",
        "РАЗМЕР",
        "РАДИУС",
        "СВАРКА",
        "КОНТРОЛЬ",
        "ДОПУСК",
        "ИНСТРУМЕНТ",
        "ПРИМЕЧАНИЕ",
    )

    @classmethod
    def supports(cls, file_name):
        name = str(file_name or "").lower()
        return (
            ".cdw.pdf" in name or name.endswith(".cdw.pdf") or
            ".cwd.pdf" in name or name.endswith(".cwd.pdf") or
            ".cdw" in name or ".cwd" in name
        )

    @staticmethod
    def _normalize_block_text(value):
        text = str(value or "").replace("\r", "\n")
        text = text.replace("\xa0", " ")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()

    @staticmethod
    def _looks_like_requirement_block(text):
        t = text.strip()
        if not t:
            return False
        upper = t.upper()
        if any(token in upper for token in ("ЛИСТ", "ПОДП.", "ДАТА", "РАЗРАБ.", "ПРОВ.", "УТВ.")):
            return False
        if any(token in upper for token in ("СЕРТИФИКАТ", "ЭЦП", "ПОДПИСАН")):
            return False
        if re.match(r"^\d{1,2}[\s\.)-]+", t):
            return True
        if any(word in upper for word in TextLayerPdfProcessor.STRONG_KEYWORDS):
            return True
        return len(t) >= 22 and re.search(r"\d", t) is not None

    def process(self, pdf_bytes, source_name=""):
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) не установлен. Невозможно использовать текстовый слой.")

        logger.info(f"TextLayerPdfProcessor: обработка файла {source_name or 'без имени'}")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        blocks = []
        seen = set()

        try:
            for page in doc:
                for block in page.get_text("blocks"):
                    text = self._normalize_block_text(block[4])
                    if not text:
                        continue
                    if not self._looks_like_requirement_block(text):
                        continue
                    line_key = text.lower().strip()
                    if line_key in seen:
                        continue
                    seen.add(line_key)
                    blocks.append(text)
                    self._store_block_text(text, source_name)
        finally:
            doc.close()

        full_text = "\n\n".join(blocks)
        return {
            "text": full_text,
            "blocks": blocks,
            "blocks_data": {f"block_{idx + 1}": {"text": block, "source": source_name} for idx, block in enumerate(blocks)},
        }


class PdfProcessingFactory:
    """Фабрика процессоров для выбора реализации по имени файла."""

    @staticmethod
    def create(file_name, token_storage=None):
        normalized = str(file_name or "").lower()
        if OCRPdfProcessor.supports(normalized):
            return OCRPdfProcessor(token_storage=token_storage)
        if TextLayerPdfProcessor.supports(normalized):
            return TextLayerPdfProcessor(token_storage=token_storage)
        if fitz is not None:
            return TextLayerPdfProcessor(token_storage=token_storage)
        return OCRPdfProcessor(token_storage=token_storage)


__all__ = [
    "BasePdfProcessor",
    "OCRPdfProcessor",
    "TextLayerPdfProcessor",
    "PdfProcessingFactory",
    "TokenStorage",
]
