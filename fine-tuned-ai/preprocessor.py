"""
dataset/preprocessor.py
-----------------------
Text cleaning, filtering, and deduplication pipeline.
Takes raw JSONL scraped data and produces clean, high-quality text.

Steps:
  1. Load raw JSONL records
  2. Clean HTML artifacts, normalize whitespace
  3. Filter by language (English only)
  4. Filter by length and quality heuristics
  5. Deduplicate using MinHash LSH (approximate near-duplicate removal)
  6. Save cleaned JSONL

Usage:
    python dataset/preprocessor.py --input data/raw --output data/cleaned
"""

import sys
import re
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import ftfy
from tqdm import tqdm

# Optional: langdetect for language filtering
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# Optional: datasketch for MinHash deduplication
try:
    from datasketch import MinHash, MinHashLSH
    HAS_MINHASH = True
except ImportError:
    HAS_MINHASH = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config, chunk_text

log = get_logger("preprocessor")
fm = FileManager()


# ============================================================
# Text Cleaning
# ============================================================

class TextCleaner:
    """
    Cleans raw scraped text:
    - Fixes unicode/encoding issues (ftfy)
    - Removes HTML artifacts
    - Normalizes whitespace
    - Removes boilerplate patterns
    """

    # Boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r"Cookie Policy.*?Accept",
        r"Subscribe to our newsletter.*",
        r"Share this article.*",
        r"Tags:.*",
        r"Posted by.*on.*",
        r"Last updated:.*",
        r"Was this helpful\?.*",
        r"Rate this page.*",
        r"©\s*\d{4}.*?reserved.*",
        r"Advertisement.*",
        r"Table of Contents\s*",
    ]

    def __init__(self):
        self._boilerplate_re = re.compile(
            "|".join(self.BOILERPLATE_PATTERNS),
            re.IGNORECASE | re.DOTALL
        )

    def clean(self, text: str) -> str:
        """Apply full cleaning pipeline to text."""
        if not text:
            return ""

        # Fix encoding issues
        text = ftfy.fix_text(text)

        # Remove HTML entities
        text = self._remove_html_artifacts(text)

        # Remove boilerplate
        text = self._boilerplate_re.sub("", text)

        # Normalize whitespace (collapse multiple blank lines to max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t+", "    ", text)

        # Strip leading/trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _remove_html_artifacts(self, text: str) -> str:
        """Remove residual HTML tags and entities."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Common HTML entities
        replacements = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&quot;": '"', "&apos;": "'", "&nbsp;": " ",
            "&#39;": "'", "&#34;": '"',
        }
        for entity, char in replacements.items():
            text = text.replace(entity, char)
        # Numeric entities
        text = re.sub(r"&#\d+;", "", text)
        text = re.sub(r"&\w+;", "", text)
        return text

    def clean_code(self, code: str) -> str:
        """Clean a code block (lighter touch - preserve structure)."""
        if not code:
            return ""
        # Fix encoding only - don't touch code structure
        code = ftfy.fix_text(code)
        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in code.split("\n")]
        # Remove excessive blank lines
        code = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return code.strip()


# ============================================================
# Quality Filters
# ============================================================

class QualityFilter:
    """
    Filters records based on quality heuristics.
    Removes records that are too short, too long, non-English, etc.
    """

    def __init__(self, config: Dict):
        preproc = config.get("preprocessing", {})
        self.min_length = preproc.get("min_text_length", 100)
        self.max_length = preproc.get("max_text_length", 50000)
        self.allowed_langs = set(preproc.get("language_filter", ["en"]))
        self.filter_language = bool(self.allowed_langs) and HAS_LANGDETECT

        self.stats = defaultdict(int)

    def is_valid(self, record: Dict) -> Tuple[bool, str]:
        """
        Check if a record passes all quality filters.

        Returns:
            (is_valid, reason_if_rejected)
        """
        text = record.get("text", "")

        # Length check
        if len(text) < self.min_length:
            self.stats["too_short"] += 1
            return False, f"too_short ({len(text)} chars)"

        if len(text) > self.max_length:
            # Don't reject - we'll chunk these later
            pass

        # Language check
        if self.filter_language:
            lang = self._detect_language(text[:2000])  # Check first 2000 chars
            if lang and lang not in self.allowed_langs:
                self.stats["wrong_language"] += 1
                return False, f"wrong_language ({lang})"

        # Code content check (at least some useful content)
        if not self._has_useful_content(text):
            self.stats["no_useful_content"] += 1
            return False, "no_useful_content"

        # Gibberish/spam check
        if self._is_likely_spam(text):
            self.stats["spam"] += 1
            return False, "spam"

        self.stats["passed"] += 1
        return True, ""

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect text language. Returns None if detection fails."""
        if not HAS_LANGDETECT:
            return "en"
        try:
            return detect(text)
        except Exception:
            return None

    def _has_useful_content(self, text: str) -> bool:
        """
        Check if text has enough programming-related content.
        Uses keyword heuristics.
        """
        programming_keywords = [
            "function", "class", "method", "variable", "return",
            "import", "def ", "const ", "let ", "var ",
            "if ", "else", "for ", "while ", "loop",
            "array", "string", "integer", "object", "module",
            "api", "http", "database", "library", "package",
            "error", "exception", "debug", "test", "print",
            "code", "program", "script", "syntax", "compile",
        ]
        text_lower = text.lower()
        matches = sum(1 for kw in programming_keywords if kw in text_lower)
        return matches >= 3

    def _is_likely_spam(self, text: str) -> bool:
        """Detect spam/low-quality content."""
        # Excessive URLs
        url_count = len(re.findall(r"https?://\S+", text))
        if url_count > len(text.split()) * 0.1:
            return True

        # Very high ratio of special characters
        special_chars = len(re.findall(r"[^a-zA-Z0-9\s\n.,;:!?(){}[\]<>/=+\-*#@_'\"\\]", text))
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True

        return False

    def print_stats(self):
        """Print filter statistics."""
        total = sum(self.stats.values())
        log.info("Quality Filter Stats:")
        for reason, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            log.info(f"  {reason}: {count} ({pct:.1f}%)")


# ============================================================
# Deduplication
# ============================================================

class Deduplicator:
    """
    Near-duplicate removal using MinHash LSH.
    Falls back to exact-hash dedup if datasketch not installed.
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.exact_hashes: Set[str] = set()

        if HAS_MINHASH:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self.minhashes: Dict[str, MinHash] = {}
            log.info(f"Deduplication: MinHash LSH (threshold={threshold})")
        else:
            self.lsh = None
            log.warning("datasketch not available. Using exact hash dedup only.")

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Get character k-grams from text."""
        text = re.sub(r"\s+", " ", text.lower())
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _make_minhash(self, text: str) -> "MinHash":
        """Create MinHash signature for text."""
        m = MinHash(num_perm=self.num_perm)
        for shingle in self._get_shingles(text):
            m.update(shingle.encode("utf-8"))
        return m

    def is_duplicate(self, record_id: str, text: str) -> bool:
        """
        Check if text is a duplicate.
        Also registers the text if it's not a duplicate.

        Returns True if this is a duplicate of something already seen.
        """
        # Exact hash check first (fast)
        exact_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if exact_hash in self.exact_hashes:
            return True
        self.exact_hashes.add(exact_hash)

        if not HAS_MINHASH or not self.lsh:
            return False

        # MinHash near-duplicate check
        minhash = self._make_minhash(text)

        try:
            candidates = self.lsh.query(minhash)
            if candidates:
                return True
        except Exception:
            pass

        # Register this document
        try:
            self.lsh.insert(record_id, minhash)
            self.minhashes[record_id] = minhash
        except Exception as e:
            log.debug(f"MinHash insert error (probably duplicate key): {e}")

        return False

    @property
    def num_unique(self) -> int:
        return len(self.exact_hashes)


# ============================================================
# Preprocessing Pipeline
# ============================================================

class PreprocessingPipeline:
    """
    Orchestrates cleaning, filtering, and deduplication.
    Input: raw JSONL files
    Output: cleaned JSONL files
    """

    def __init__(self, config: Dict):
        self.config = config
        preproc = config.get("preprocessing", {})
        self.max_length = preproc.get("max_text_length", 50000)
        self.chunk_long_texts = True

        self.cleaner = TextCleaner()
        self.filter = QualityFilter(config)
        self.dedup = Deduplicator(threshold=preproc.get("dedup_threshold", 0.85))

        self.stats = defaultdict(int)

    def process_record(self, record: Dict) -> List[Dict]:
        """
        Process a single raw record.

        Returns:
            List of cleaned records (may be split into chunks).
            Empty list if record is filtered out.
        """
        self.stats["total"] += 1

        # Clean text
        text = self.cleaner.clean(record.get("text", ""))
        record["text"] = text

        # Clean code blocks
        code_blocks = record.get("code_blocks", [])
        record["code_blocks"] = [
            self.cleaner.clean_code(cb) for cb in code_blocks
            if cb and len(cb.strip()) > 10
        ]

        # Quality filter
        valid, reason = self.filter.is_valid(record)
        if not valid:
            self.stats[f"filtered_{reason.split('(')[0].strip()}"] += 1
            return []

        # Deduplication
        text_for_dedup = text[:3000]  # Use first 3000 chars for dedup
        if self.dedup.is_duplicate(record.get("id", ""), text_for_dedup):
            self.stats["deduplicated"] += 1
            return []

        # Chunk if too long
        if len(text) > self.max_length and self.chunk_long_texts:
            chunks = chunk_text(text, max_chars=self.max_length)
            results = []
            for i, chunk in enumerate(chunks):
                chunk_record = record.copy()
                chunk_record["text"] = chunk
                chunk_record["id"] = f"{record.get('id', '')}_{i}"
                chunk_record["chunk_index"] = i
                chunk_record["total_chunks"] = len(chunks)
                results.append(chunk_record)
            self.stats["chunked"] += len(results)
            return results

        self.stats["kept"] += 1
        return [record]

    def process_file(self, input_path: Path, output_path: Path) -> int:
        """Process a single JSONL file."""
        count_out = 0

        with open(input_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in tqdm(lines, desc=f"Preprocessing {input_path.name}", leave=False):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = __import__("json").loads(line)
                except Exception:
                    continue

                cleaned = self.process_record(record)
                for r in cleaned:
                    f_out.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")
                    count_out += 1

        return count_out

    def process_directory(self, input_dir: Path, output_dir: Path) -> int:
        """Process all JSONL files in a directory."""
        fm._ensure_dir(output_dir)
        total_out = 0

        jsonl_files = list(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            log.warning(f"No JSONL files found in {input_dir}")
            return 0

        log.info(f"Processing {len(jsonl_files)} files from {input_dir}")

        for input_file in jsonl_files:
            output_file = output_dir / input_file.name
            count = self.process_file(input_file, output_file)
            log.info(f"  {input_file.name}: {count} records kept")
            total_out += count

        log.info(f"\nPreprocessing complete: {total_out} records kept")
        log.info(f"Unique documents: {self.dedup.num_unique}")

        self.filter.print_stats()

        log.info("Processing stats:")
        for k, v in sorted(self.stats.items()):
            log.info(f"  {k}: {v}")

        return total_out


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Input directory with raw JSONL files")
    parser.add_argument("--output", required=True, help="Output directory for cleaned files")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    pipeline = PreprocessingPipeline(config)
    total = pipeline.process_directory(Path(args.input), Path(args.output))
    log.info(f"Done. {total} records written to {args.output}")


if __name__ == "__main__":
    main()
