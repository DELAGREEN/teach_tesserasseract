import os
import re
import sqlite3

from dotenv import load_dotenv
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
        return str(token).replace(" ", " ").strip()

    @staticmethod
    def _tokenize_block_text(text):
        if text is None:
            return []
        normalized = str(text).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        return [normalized]

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

    @staticmethod
    def _is_noise_ocr_fragment(text):
          value = re.sub(r"\s+", " ", str(text or "")).strip()
          if not value:
              return True
          upper = value.upper()
          metadata_tokens = (
              "ДОКУМЕНТ ПОДПИСАН",
              "СЕРТИФИКАТ",
              "ЭЦП",
              "ВЛАДЕЛЕЦ ПОДПИСИ",
              "ДЕЙСТВИТЕЛЬНО С",
              "КОПИРОВАЛ",
              "ФОРМАТ",
              "ЛИСТ",
              "ПОДП.",
              "ДАТА",
              "СВС:",
              "КЛАБ.",
              "ВЕРСИИ",
          )
          if any(token in upper for token in metadata_tokens):
              return True
          if re.fullmatch(r"[\W_]+", value):
              return True
          if re.fullmatch(r"[-=|/\\]+", value):
              return True
          if re.fullmatch(r"[A-ZА-ЯЁ]{1,3}", value):
              return True
          return False

    @staticmethod
    def _split_numbered_segments(text):
          value = re.sub(r"\s+", " ", str(text or "")).strip()
          if not value:
              return []
          matches = list(re.finditer(r"(?<!\w)(\d{1,2})(?:[\.\)\-]|[*•])?\s*(?=[A-ZА-ЯЁ])", value, flags=re.IGNORECASE))
          if not matches:
              return [value]

          parts = []
          for idx, match in enumerate(matches):
              start_index = match.start()
              end_index = matches[idx + 1].start() if idx + 1 < len(matches) else len(value)
              part = value[start_index:end_index].strip()
              if len(part) < 8:
                  continue
              if OCRPdfProcessor._is_noise_ocr_fragment(part):
                  continue
              parts.append(re.sub(r"\s+", " ", part).strip())
          return parts or [value]

    @staticmethod
    def _split_line_columns(entries):
        if not entries:
            return []

        sorted_entries = sorted(entries, key=lambda item: (item[3], item[2]))
        groups = []
        current = [sorted_entries[0]]

        for entry in sorted_entries[1:]:
            previous = current[-1]
            previous_top = previous[3]
            previous_left = previous[2]
            current_top = entry[3]
            current_left = entry[2]

            is_new_row = abs(current_top - previous_top) > 12
            is_new_column = current_left - previous_left > 180 and re.match(
                r"^(?:\d{1,2}|[A-ZА-ЯЁ])(?:[\.\)\-]|[*•])?\s*$",
                str(entry[0]).strip(),
            )

            if is_new_row or is_new_column:
                groups.append(current)
                current = [entry]
            else:
                current.append(entry)

        groups.append(current)

        segments = []
        for group in groups:
            line = " ".join(part for part, _, _, _ in sorted(group, key=lambda item: item[2]))
            clean = re.sub(r"\s+", " ", line).strip()
            if clean:
                segments.append(clean)
        return segments

    @staticmethod
    def _merge_visual_rows(entries):
        if not entries:
            return []

        rows = []
        for item in entries:
            cleaned = str(item[0]).strip()
            if not cleaned:
                continue
            rows.append({
                "text": cleaned,
                "top": int(item[3]),
                "left": int(item[2]),
            })

        if not rows:
            return []

        rows = sorted(rows, key=lambda row: (row["top"], row["left"]))
        blocks = []
        current = [rows[0]]

        for row in rows[1:]:
            previous = current[-1]
            top_gap = abs(row["top"] - previous["top"])
            left_gap = abs(row["left"] - previous["left"])
            if top_gap <= 20 or (top_gap <= 36 and left_gap <= 220):
                current.append(row)
            else:
                blocks.append(current)
                current = [row]
        blocks.append(current)

        merged = []
        for block_rows in blocks:
            text = " ".join(row["text"] for row in sorted(block_rows, key=lambda item: item["left"]))
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                merged.append(cleaned)
        return merged

    @staticmethod
    def _block_detect_text(image):
        if cv2 is None or np is None or pytesseract is None:
            return []
        if not all(hasattr(cv2, name) for name in ("cvtColor", "GaussianBlur", "filter2D", "adaptiveThreshold", "getStructuringElement", "dilate", "findContours", "boundingRect", "contourArea")):
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        try:
            data = pytesseract.image_to_data(image, lang="rus", output_type=pytesseract.Output.DICT, config=r"--psm 6")
        except Exception:
            return []

        mask = np.zeros_like(binary)
        for idx in range(len(data.get("level") or [])):
            text_value = str(data.get("text", [])[idx] if idx < len(data.get("text", [])) else "").strip()
            if not text_value:
                continue
            x = int(data.get("left", [])[idx] if idx < len(data.get("left", [])) else 0)
            y = int(data.get("top", [])[idx] if idx < len(data.get("top", [])) else 0)
            w = int(data.get("width", [])[idx] if idx < len(data.get("width", [])) else 0)
            h = int(data.get("height", [])[idx] if idx < len(data.get("height", [])) else 0)
            if w <= 0 or h <= 0:
                continue
            mask[y:y + h, x:x + w] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dilated = cv2.dilate(mask, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_area = image.shape[0] * image.shape[1] * 0.005
        blocks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            blocks.append({"bbox": (x, y, w, h), "area": area, "width": w, "height": h})

        if not blocks:
            return []

        largest = max(blocks, key=lambda item: item["area"])
        target_width = largest["width"]
        min_width = target_width * 0.7
        max_width = target_width * 1.3
        main_blocks = [block for block in blocks if min_width <= block["width"] <= max_width]
        if not main_blocks:
            main_blocks = blocks

        main_blocks = sorted(main_blocks, key=lambda item: (item["bbox"][1], -item["bbox"][0]))
        texts = []
        for block in main_blocks:
            x, y, w, h = block["bbox"]
            if w <= 0 or h <= 0:
                continue
            crop = image[y:y + h, x:x + w]
            raw = pytesseract.image_to_string(crop, lang="rus", config=r"--psm 6 -c preserve_interword_spaces=1").strip()
            cleaned = OCRPdfProcessor._clean_ocr_text(raw)
            if cleaned:
                texts.append(cleaned)
        return texts

    @classmethod
    def _extract_requirement_blocks(cls, ocr_data):
          if not isinstance(ocr_data, dict):
              return []

          text_values = list(ocr_data.get("text") or [])
          conf_values = list(ocr_data.get("conf") or [])
          left_values = list(ocr_data.get("left") or [])
          top_values = list(ocr_data.get("top") or [])
          block_values = list(ocr_data.get("block_num") or [])
          par_values = list(ocr_data.get("par_num") or [])
          line_values = list(ocr_data.get("line_num") or [])
          if not text_values:
              return []

          grouped = {}
          for idx, raw_text in enumerate(text_values):
              cleaned = str(raw_text or "").strip()
              if not cleaned:
                  continue
              try:
                  conf = int(str(conf_values[idx]) if idx < len(conf_values) else "0")
              except Exception:
                  conf = 0
              if conf < 0:
                  continue
              top = int(str(top_values[idx]) if idx < len(top_values) else "0")
              left = int(str(left_values[idx]) if idx < len(left_values) else "0")
              block_num = int(str(block_values[idx]) if idx < len(block_values) else "0")
              par_num = int(str(par_values[idx]) if idx < len(par_values) else "0")
              line_num = int(str(line_values[idx]) if idx < len(line_values) else "0")
              line_key = (block_num, par_num, line_num)
              grouped.setdefault(line_key, []).append((cleaned, conf, left, top))

          blocks = []
          for _, entries in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
              for candidate in cls._split_line_columns(entries):
                  candidate = re.sub(r"\s+", " ", candidate).strip()
                  if not candidate or cls._is_noise_ocr_fragment(candidate):
                      continue
                  upper = candidate.upper()
                  if any(token in upper for token in ("ДОКУМЕНТ ПОДПИСАН", "СЕРТИФИКАТ", "ЭЦП", "ЛИСТ", "ФОРМАТ", "ВЕРСИИ", "КОПИРОВАЛ")):
                      continue
                  if not (any(ch.isdigit() for ch in candidate) and any(ch.isalpha() for ch in candidate)):
                      continue
                  for part in cls._split_numbered_segments(candidate):
                      if not part or cls._is_noise_ocr_fragment(part):
                          continue
                      if not (any(ch.isdigit() for ch in part) and any(ch.isalpha() for ch in part)):
                          continue
                      blocks.append(part)

          if not blocks:
              candidate = "\n".join(
                  re.sub(r"\s+", " ", str(item or "")).strip()
                  for item in text_values if str(item or "").strip()
              )
              candidate = OCRPdfProcessor._clean_ocr_text(candidate)
              if candidate:
                  blocks = [candidate]
          return blocks

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

          blocks = []
          for page in pages:
              image = np.array(page)
              region_blocks = self._block_detect_text(image)
              cleaned_region = []
              if region_blocks:
                  for block in region_blocks:
                      candidate = self._clean_ocr_text(block)
                      if not candidate or self._is_noise_ocr_fragment(candidate):
                          continue
                      cleaned_region.append(candidate)
              if cleaned_region:
                  blocks.extend(cleaned_region)
                  continue

              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              gray = cv2.GaussianBlur(gray, (3, 3), 0)
              _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

              best_blocks = []
              best_score = -1
              for psm in (3, 11, 4, 6):
                  custom_config = rf"--psm {psm} -c preserve_interword_spaces=1"
                  try:
                      ocr_data = pytesseract.image_to_data(binary, lang="rus", config=custom_config, output_type=pytesseract.Output.DICT)
                  except Exception:
                      ocr_data = {"text": [pytesseract.image_to_string(binary, lang="rus", config=custom_config).strip()]}
                  page_blocks = self._extract_requirement_blocks(ocr_data)
                  score = sum(1 for block in page_blocks if re.search(r"\d", block) and len(block) > 20)
                  score += len(page_blocks) * 2
                  if score > best_score:
                      best_score = score
                      best_blocks = page_blocks

              if best_blocks:
                  blocks.extend(best_blocks)

          if not blocks:
              image = np.array(pages[0])
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              gray = cv2.GaussianBlur(gray, (3, 3), 0)
              _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              text = pytesseract.image_to_string(binary, lang="rus", config=r"--psm 3 -c preserve_interword_spaces=1").strip()
              cleaned = self._clean_ocr_text(text)
              if cleaned:
                  blocks = [cleaned]

          unique = []
          seen = set()
          for block in blocks:
              normalized = re.sub(r"\s+", " ", self._clean_ocr_text(block)).strip()
              if not normalized or self._is_noise_ocr_fragment(normalized):
                  continue
              key = normalized.lower()
              if key in seen:
                  continue
              seen.add(key)
              unique.append(normalized)
          blocks = unique

          cleaned = self._clean_ocr_text("\n\n".join(blocks)).strip()
          if cleaned:
              logger.info(
                  "OCRPdfProcessor: пропускаем запись OCR-результата в TokenStorage для %s; dwg.pdf должен уходить в API без записи в БД.",
                  source_name or 'без имени',
              )
          return {
              "text": cleaned,
              "blocks": [{"text": block, "source": source_name} for block in blocks if block],
              "blocks_data": {f"block_{idx + 1}": {"text": block, "source": source_name} for idx, block in enumerate(blocks) if block},
          }

    @staticmethod
    def _clean_ocr_text(text):
        if not text:
            return ""
        filtered = re.sub(r"\n+", "\n", text)
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
    def _is_material_descriptor(line):
        normalized = re.sub(r"\s+", " ", str(line or "")).strip()
        if not normalized:
            return False
        pattern = r"^(?:\d{1,2}\s*(?:[\.\)\-]|[*•])?\s*)?(?:МАТЕРИАЛ(?:-ЗАМЕНИТЕЛЬ)?|МАТЕРИАЛЫ?)\s*[:\-]"
        return bool(re.match(pattern, normalized, flags=re.IGNORECASE))

    @staticmethod
    def _is_noise_line(line):
        text = re.sub(r"\s+", " ", str(line or "")).strip()
        if not text:
            return True
        upper = text.upper()
        if len(text) < 3:
            return True
        if TextLayerPdfProcessor._is_material_descriptor(text):
            return True
        requirement_like = bool(re.search(r"(?<!\w)\d{1,2}\s*(?:[\.\)\-]|[*•])?\s*[A-ZА-ЯЁ]", text, flags=re.IGNORECASE))
        if requirement_like and any(token in upper for token in ("СВАРК", "ВЫПОЛНИТЬ", "ПОКРЫТИЕ", "КОНТРОЛЬ", "ДОПУСК", "РАЗМЕР", "ПРОХОЖДЕНИЕ", "ГОСТ", "ОСТ", "ТУ", "МАТЕРИАЛ")):
            return False
        if "СТЕКЛОТЕКСТОЛИТ" in upper and "МАРКИ" in upper:
            return True
        if upper.startswith("СТЕКЛОТЕКСТОЛИТ"):
            return True
        metadata_tokens = (
            "ДОКУМЕНТ ПОДПИСАН",
            "СЕРТИФИКАТ",
            "ЭЦП",
            "ВЛАДЕЛЕЦ ПОДПИСИ",
            "ДЕЙСТВИТЕЛЬНО С",
            "КОПИРОВАЛ",
            "ФОРМАТ",
            "CRC",
            "ID ВЕРСИИ",
            "ИНФОРМАЦИЯ",
            "ПОДТ.",
            "РАЗРАБ.",
            "ПРОВ.",
            "Н.КОНТР.",
            "Т.КОНТР.",
            "ИЗМ.",
            "ЛИСТ",
            "ПОДП.",
            "ДАТА",
            "СВС:",
            "КЛАБ.",
            "КЛАБ ",
            "ВЕРСИИ",
            "КРАТКИЙ",
        )
        if any(token in upper for token in metadata_tokens):
            return True
        if upper.startswith("A4") or upper.startswith("A3") or upper.startswith("A5"):
            return True
        if re.fullmatch(r"[0-9A-F]{8,}", upper):
            return True
        if re.fullmatch(r"[\W_]+", text):
            return True
        if re.fullmatch(r"[-=|/\\]+", text):
            return True
        if re.fullmatch(r"[A-ZА-ЯЁ]{1,3}", text):
            return True
        if re.match(r"^[A-ZА-ЯЁ]\s*\(\d+:\d+\)", upper):
            return True
        if re.match(r"^[:.;/|\\-]+", text):
            return True
        return False

    @staticmethod
    def _filter_requirement_lines(text):
        raw_lines = str(text or "").replace("\r", "\n").split("\n")
        kept = []
        metadata_tokens = (
            "ДОКУМЕНТ ПОДПИСАН",
            "СЕРТИФИКАТ",
            "ЭЦП",
            "ВЛАДЕЛЕЦ ПОДПИСИ",
            "ДЕЙСТВИТЕЛЬНО С",
            "КОПИРОВАЛ",
            "ФОРМАТ",
            "ЛИСТ",
            "ПОДП.",
            "ДАТА",
            "СВС:",
            "КЛАБ.",
            "ВЕРСИИ",
            "ПОДПИСИ",
        )
        keyword_patterns = (
            "ГОСТ", "ОСТ", "ТУ", "МАТЕРИАЛ", "ПОКРЫТИЕ", "РАДИУС", "ДИАМЕТР",
            "ШЕРОХОВАТОСТЬ", "СВАРКА", "СВАРК", "СВАРКУ", "СВАРНЫ", "СВАРНО", "РЕЗЬБА", "ПРИМЕЧАНИЕ",
            "ФОСФОР", "МЕДЬ", "ДОЛЯ", "НЕ БОЛЕЕ", "НЕ МЕНЕЕ", "МАРКИРОВАТЬ",
            "МАССОВОЙ", "МАССОВАЯ", "ПОСТАВЛЯТЬСЯ", "ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
            "ДОПУСКАЕТСЯ", "СКОБКАХ", "ШРИФТОМ", "СБОРКЕ", "СБОРКИ", "ДОПУСК",
            "ОТВЕРСТИЕ", "КОНТРОЛЬ", "ПРОХОЖДЕНИЕ", "РАЗМЕР", "МАТЕРИАЛ-ЗАМЕНИТЕЛЬ",
            "ВЫПОЛНИТЬ", "КОНТРОЛИРОВАТЬ", "ИСПОЛНИТЬ", "УСТАНОВИТЬ"
        )
        strong_requirement_markers = (
            "НЕ БОЛЕЕ", "НЕ МЕНЕЕ", "ДОПУСКАЕТСЯ", "МАРКИРОВАТЬ",
            "РАЗМЕРЫ", "РАЗМЕР", "РАДИУС", "ДОПУСКИ", "ДОПУСК",
            "ПРИМЕЧАНИЕ", "ПОСТАВЛЯТЬСЯ", "ЗАМЕНИТЕЛЬ", "СБОРКЕ",
            "СБОРКИ", "ОБЕСПЕЧ", "ОБЩИЕ", "МАТЕРИАЛ-ЗАМЕНИТЕЛЬ",
            "СВАРК", "СВАРКА", "ПОКРЫТИЕ", "ОТВЕРСТИЕ", "КОНТРОЛЬ",
            "ПРОХОЖДЕНИЕ", "ВЫПОЛНИТЬ"
        )

        for raw in raw_lines:
            candidate = re.sub(r"\s+", " ", raw).strip()
            if not candidate:
                continue
            upper = candidate.upper()
            if upper.startswith("СТЕКЛОТЕКСТОЛИТ") or ("СТЕКЛОТЕКСТОЛИТ" in upper and "МАРКИ" in upper):
                continue
            if TextLayerPdfProcessor._is_noise_line(candidate):
                continue
            if TextLayerPdfProcessor._is_material_descriptor(candidate):
                continue
            if any(token in upper for token in metadata_tokens):
                continue

            extracted = candidate
            numbered_match = re.search(
                r"(?P<num>\d{1,2})\s*(?:[\.\)\-]|[*•])?\s*(?P<body>[A-ZА-ЯЁ].+)$",
                candidate,
                flags=re.IGNORECASE,
            )
            if numbered_match:
                body = (numbered_match.group("body") or "").strip()
                if body and not any(token in body.upper() for token in metadata_tokens):
                    extracted = f"{numbered_match.group('num')} {body}".strip()

            if not re.match(r"^(?:[-*•\s]*\d)", extracted):
                continue

            upper_extracted = extracted.upper()
            if any(token in upper_extracted for token in metadata_tokens):
                continue
            if not (any(keyword in upper_extracted for keyword in keyword_patterns) or any(marker in upper_extracted for marker in strong_requirement_markers)):
                continue

            kept.append(extracted)

        return "\n".join(kept)
    @staticmethod
    def _looks_like_requirement_block(text):
        if not text:
            return False
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(text)
        return bool(cleaned)
    @staticmethod
    def _looks_like_requirement_block(text):
        if not text:
            return False
        cleaned = TextLayerPdfProcessor._filter_requirement_lines(text)
        return bool(cleaned)

    def process(self, pdf_bytes, source_name=''):
        if fitz is None:
            raise RuntimeError('PyMuPDF (fitz) не установлен. Невозможно использовать текстовый слой.')

        logger.info(f"TextLayerPdfProcessor: обработка файла {source_name or 'без имени'}")
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        blocks = []
        seen = set()

        try:
            for page in doc:
                for block in page.get_text('blocks'):
                    text = self._normalize_block_text(block[4])
                    if not text:
                        continue
                    cleaned_block = self._filter_requirement_lines(text)
                    if not cleaned_block:
                        continue
                    normalized_block = re.sub(r"[ \t]+\n", "\n", cleaned_block)
                    normalized_block = re.sub(r"\n{3,}", "\n\n", normalized_block).strip()
                    if not normalized_block:
                        continue
                    block_key = normalized_block.lower()
                    if block_key in seen:
                        continue
                    seen.add(block_key)
                    blocks.append(normalized_block)
                    self._store_block_text(normalized_block, source_name)
        finally:
            doc.close()

        full_text = "\n\n".join(blocks)
        return {
            'text': full_text,
            'blocks': blocks,
            'blocks_data': {f'block_{idx + 1}': {'text': block, 'source': source_name} for idx, block in enumerate(blocks)},
        }


class PdfProcessingFactory:
    """Фабрика процессоров для выбора реализации по имени файла."""

    @staticmethod
    def _reload_env():
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)

    @staticmethod
    def _env_flag(name, default=True):
        PdfProcessingFactory._reload_env()
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return str(raw_value).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}

    @staticmethod
    def _normalize_type_token(value):
        token = str(value or "").strip().lower().replace("\\", "/")
        token = token.split("#", 1)[0].strip()
        token = token.replace(" ", "").replace("_", "")
        token = token.strip(".")
        return token
    @classmethod
    def _allowed_file_types(cls):
        cls._reload_env()
        allowed = set()
        for env_name in ("ALLOWED_FILE_TYPES", "ALLOWED_PDF_TYPES", "FILE_TYPES_ALLOWED"):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            for item in re.split(r"[,;\s]+", str(raw_value)):
                token = cls._normalize_type_token(item)
                if token:
                    allowed.add(token)
        return allowed

    @classmethod
    def _disabled_file_types(cls):
        cls._reload_env()
        disabled = set()
        for env_name in ("DISABLED_FILE_TYPES", "DISABLED_PDF_TYPES", "DISABLED_TYPES", "FILE_TYPES_DISABLED"):
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            for item in re.split(r"[,;\s]+", str(raw_value)):
                token = cls._normalize_type_token(item)
                if token:
                    disabled.add(token)
        return disabled

    @classmethod
    def _file_type_tokens(cls, file_name):
        normalized = str(file_name or "").lower()
        tokens = set()
        for suffix in (".dwg.pdf", ".cdw.pdf", ".cwd.pdf", ".dwg", ".cdw", ".cwd"):
            if suffix in normalized:
                tokens.update({
                    cls._normalize_type_token(suffix),
                    cls._normalize_type_token(suffix.lstrip(".")),
                    cls._normalize_type_token(suffix.replace(".pdf", "")),
                })
                break
        if not tokens:
            return tokens
        if ".dwg.pdf" in normalized or normalized.endswith(".dwg.pdf") or ".dwg" in normalized:
            tokens.update({"dwg", "dwg.pdf"})
        if ".cdw.pdf" in normalized or normalized.endswith(".cdw.pdf") or ".cdw" in normalized:
            tokens.update({"cdw", "cdw.pdf"})
        if ".cwd.pdf" in normalized or normalized.endswith(".cwd.pdf") or ".cwd" in normalized:
            tokens.update({"cwd", "cwd.pdf"})
        return tokens

    @classmethod
    def is_processing_enabled(cls, file_name):
        cls._reload_env()
        normalized = str(file_name or "").lower()
        file_type_tokens = cls._file_type_tokens(normalized)
        allowed_types = cls._allowed_file_types()
        if allowed_types:
            if not file_type_tokens:
                return False
            return bool(file_type_tokens & allowed_types)

        disabled_types = cls._disabled_file_types()
        if file_type_tokens and (file_type_tokens & disabled_types):
            return False

        if ".dwg.pdf" in normalized or normalized.endswith(".dwg.pdf") or ".dwg" in normalized:
            return cls._env_flag("ENABLE_DWG_PDF_PROCESSING", True)
        if ".cdw.pdf" in normalized or normalized.endswith(".cdw.pdf") or ".cdw" in normalized:
            return cls._env_flag("ENABLE_CDW_PDF_PROCESSING", True)
        if ".cwd.pdf" in normalized or normalized.endswith(".cwd.pdf") or ".cwd" in normalized:
            return cls._env_flag("ENABLE_CWD_PDF_PROCESSING", True)
        return cls._env_flag("ENABLE_UNKNOWN_PDF_PROCESSING", True)

    @staticmethod
    def create(file_name, token_storage=None):
        normalized = str(file_name or '').lower()
        if OCRPdfProcessor.supports(normalized):
            if not PdfProcessingFactory.is_processing_enabled(normalized):
                return None
            return OCRPdfProcessor(token_storage=token_storage)
        if TextLayerPdfProcessor.supports(normalized):
            if not PdfProcessingFactory.is_processing_enabled(normalized):
                return None
            return TextLayerPdfProcessor(token_storage=token_storage)
        if fitz is not None:
            if not PdfProcessingFactory.is_processing_enabled(normalized):
                return None
            return TextLayerPdfProcessor(token_storage=token_storage)
        if not PdfProcessingFactory.is_processing_enabled(normalized):
            return None
        return OCRPdfProcessor(token_storage=token_storage)

__all__ = [
    'BasePdfProcessor',
    'OCRPdfProcessor',
    'TextLayerPdfProcessor',
    'PdfProcessingFactory',
    'TokenStorage',
]
