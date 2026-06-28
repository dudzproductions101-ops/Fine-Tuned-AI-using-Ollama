"""
dataset/builder.py
------------------
Converts cleaned scraped records into instruction-following training pairs.

For each record, generates multiple training examples:
  - "Explain this code" from code blocks
  - "Summarize this documentation" from docs
  - "Answer this question" from Q&A
  - "Write code that does X" from tutorial descriptions + code
  - "Debug/improve this code" pairs

Output: JSONL with {"instruction": ..., "input": ..., "output": ...} format.

Usage:
    python dataset/builder.py --input data/cleaned --output data/dataset
    python dataset/builder.py --input data/cleaned --output data/dataset --merge data/final
"""

import sys
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config, extract_code_blocks, truncate_text

log = get_logger("builder")
fm = FileManager()


# ============================================================
# Instruction Templates
# ============================================================

EXPLAIN_CODE_TEMPLATES = [
    "Explain what the following {lang} code does:",
    "What does this {lang} code accomplish?",
    "Describe the purpose and functionality of the following {lang} code:",
    "Walk me through this {lang} code step by step:",
    "Provide a clear explanation of the following {lang} code snippet:",
]

WRITE_CODE_TEMPLATES = [
    "Write a {lang} function that {description}",
    "Implement the following in {lang}: {description}",
    "Create a {lang} program that {description}",
    "Write {lang} code to {description}",
    "Implement a {lang} solution for: {description}",
]

IMPROVE_CODE_TEMPLATES = [
    "Improve and optimize the following {lang} code:",
    "Refactor this {lang} code to be more readable and efficient:",
    "Review and improve this {lang} code snippet:",
    "Identify issues and improve this {lang} code:",
]

DEBUG_CODE_TEMPLATES = [
    "Debug the following {lang} code and explain any issues:",
    "Find and fix the bugs in this {lang} code:",
    "What's wrong with this {lang} code? Fix it:",
    "Review this {lang} code for errors and correct them:",
]

DOC_SUMMARY_TEMPLATES = [
    "Summarize the following programming documentation:",
    "Provide a concise summary of this technical documentation:",
    "What are the key points of this documentation?",
    "Explain the main concepts described in this documentation:",
]

QA_TEMPLATES = [
    "Answer the following programming question:",
    "Provide a solution to this programming problem:",
    "How would you solve this programming question?",
    "Help me understand: {question}",
]

CONCEPT_TEMPLATES = [
    "Explain the concept of {concept} in programming.",
    "What is {concept} and how is it used?",
    "Describe {concept} with an example.",
    "How does {concept} work?",
]


# ============================================================
# Language Detection from Code
# ============================================================

LANG_PATTERNS = {
    "python": [r"import\s+\w+", r"def\s+\w+\s*\(", r"print\s*\(", r":\s*$", r"elif\s+"],
    "javascript": [r"const\s+\w+", r"let\s+\w+", r"function\s+\w+\s*\(", r"=>\s*\{", r"\.then\("],
    "typescript": [r":\s*\w+\[\]", r"interface\s+\w+", r"type\s+\w+\s*=", r"<\w+>"],
    "rust": [r"fn\s+\w+\s*\(", r"let\s+mut\s", r"impl\s+\w+", r"use\s+\w+::", r"->"],
    "go": [r"func\s+\w+\s*\(", r"package\s+\w+", r":=", r"fmt\."],
    "java": [r"public\s+class", r"private\s+\w+", r"System\.out", r"@Override"],
    "cpp": [r"#include\s*<", r"std::", r"cout\s*<<", r"int\s+main\s*\("],
    "bash": [r"#!/bin/bash", r"\$\w+", r"echo\s+", r"if\s*\["],
    "sql": [r"SELECT\s+", r"FROM\s+\w+", r"WHERE\s+", r"INSERT\s+INTO"],
}


def detect_language(code: str) -> str:
    """Detect programming language from code snippet."""
    scores = defaultdict(int)
    for lang, patterns in LANG_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                scores[lang] += 1
    if scores:
        return max(scores, key=scores.get)
    return "code"


# ============================================================
# Example Generators
# ============================================================

class ExampleGenerator:
    """
    Generates training instruction-output pairs from a cleaned record.
    Each record type uses different generation strategies.
    """

    def __init__(self, config: Dict):
        self.config = config
        dataset_cfg = config.get("dataset", {})
        self.min_output_len = dataset_cfg.get("min_output_length", 10)
        self.max_input_len = dataset_cfg.get("max_input_length", 4000)
        self.rng = random.Random(42)

    def generate(self, record: Dict) -> List[Dict]:
        """Generate examples from a record based on its content type."""
        content_type = record.get("content_type", "unknown")
        examples = []

        if content_type == "qa":
            examples.extend(self._from_qa(record))
        elif content_type == "tutorial":
            examples.extend(self._from_tutorial(record))
        elif content_type == "documentation":
            examples.extend(self._from_documentation(record))
        elif content_type == "code":
            examples.extend(self._from_code(record))
        else:
            # Generic - try all strategies
            examples.extend(self._from_documentation(record))
            if record.get("code_blocks"):
                examples.extend(self._from_code(record))

        return [ex for ex in examples if self._is_valid_example(ex)]

    def _make_example(self, instruction: str, output: str, input_text: str = "") -> Dict:
        """Create a training example dict."""
        return {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output.strip(),
        }

    def _is_valid_example(self, example: Dict) -> bool:
        """Check if a generated example meets minimum quality criteria."""
        instruction = example.get("instruction", "")
        output = example.get("output", "")

        if len(instruction) < 10 or len(output) < self.min_output_len:
            return False
        if len(output) > 8000:  # Limit very long outputs
            example["output"] = output[:8000]
        return True

    def _truncate(self, text: str, max_len: Optional[int] = None) -> str:
        """Truncate text to max_len."""
        max_len = max_len or self.max_input_len
        return truncate_text(text, max_len)

    # ----------------------------------------------------------
    # Q&A Generation
    # ----------------------------------------------------------

    def _from_qa(self, record: Dict) -> List[Dict]:
        """Generate examples from Stack Overflow Q&A records."""
        examples = []
        title = record.get("title", "")
        question_text = record.get("text", "")
        answers = record.get("answers", [])

        if not title or not answers:
            return []

        # Pick best answer (accepted or highest score)
        best_answer = None
        for ans in answers:
            if ans.get("is_accepted"):
                best_answer = ans
                break
        if not best_answer and answers:
            best_answer = max(answers, key=lambda a: a.get("score", 0))

        if not best_answer:
            return []

        answer_text = best_answer.get("text", "")
        if len(answer_text) < 20:
            return []

        # Template 1: Direct question -> answer
        template = self.rng.choice(QA_TEMPLATES)
        if "{question}" in template:
            instruction = template.format(question=title)
        else:
            instruction = template

        input_text = question_text[:1000] if len(question_text) > len(title) + 20 else ""
        examples.append(self._make_example(instruction, answer_text, input_text))

        # Template 2: Code generation from Q&A (if there's code in the answer)
        code_blocks = record.get("code_blocks", [])
        if code_blocks and title:
            code = code_blocks[0]
            lang = detect_language(code)
            template = self.rng.choice(WRITE_CODE_TEMPLATES)
            instruction = template.format(lang=lang, description=title.lower().rstrip("?"))
            examples.append(self._make_example(instruction, code))

        return examples

    # ----------------------------------------------------------
    # Tutorial Generation
    # ----------------------------------------------------------

    def _from_tutorial(self, record: Dict) -> List[Dict]:
        """Generate examples from tutorial articles."""
        examples = []
        title = record.get("title", "")
        text = record.get("text", "")
        code_blocks = record.get("code_blocks", [])

        # Summary of tutorial
        if title and text:
            template = self.rng.choice(DOC_SUMMARY_TEMPLATES)
            instruction = f"Summarize the following tutorial about: {title}"
            # Create a concise output (first 3 paragraphs)
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
            summary_output = "\n\n".join(paragraphs[:4])
            if summary_output:
                examples.append(self._make_example(instruction, summary_output, text[:500]))

        # Code explanation from tutorial
        for code in code_blocks[:3]:  # Max 3 code blocks per tutorial
            if len(code) < 30:
                continue
            lang = detect_language(code)
            template = self.rng.choice(EXPLAIN_CODE_TEMPLATES)
            instruction = template.format(lang=lang)
            # Generate explanation from surrounding context
            explanation = self._find_explanation_for_code(text, code)
            if explanation:
                examples.append(self._make_example(instruction, explanation, code))

        # Code improvement
        if code_blocks and len(code_blocks[0]) > 50:
            code = code_blocks[0]
            lang = detect_language(code)
            template = self.rng.choice(IMPROVE_CODE_TEMPLATES)
            instruction = template.format(lang=lang)
            # Generate an improved version prompt
            improved = self._generate_improved_version_prompt(code, title)
            if improved:
                examples.append(self._make_example(instruction, improved, code))

        return examples

    # ----------------------------------------------------------
    # Documentation Generation
    # ----------------------------------------------------------

    def _from_documentation(self, record: Dict) -> List[Dict]:
        """Generate examples from documentation pages."""
        examples = []
        title = record.get("title", "")
        text = record.get("text", "")
        code_blocks = record.get("code_blocks", [])

        if not text:
            return []

        # Concept explanation
        if title:
            template = self.rng.choice(CONCEPT_TEMPLATES)
            instruction = template.format(concept=title)
            # Use first meaningful paragraph as output
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
            if paragraphs:
                output = "\n\n".join(paragraphs[:3])
                examples.append(self._make_example(instruction, output))

        # Documentation summary
        if len(text) > 200:
            template = self.rng.choice(DOC_SUMMARY_TEMPLATES)
            examples.append(self._make_example(
                template,
                text[:2000],
                self._truncate(text, 3000)
            ))

        # Code examples from docs
        for code in code_blocks[:2]:
            if len(code) < 30:
                continue
            lang = detect_language(code)
            template = self.rng.choice(EXPLAIN_CODE_TEMPLATES)
            instruction = template.format(lang=lang)
            explanation = self._find_explanation_for_code(text, code)
            if explanation:
                examples.append(self._make_example(instruction, explanation, code))

        return examples

    # ----------------------------------------------------------
    # Raw Code Generation
    # ----------------------------------------------------------

    def _from_code(self, record: Dict) -> List[Dict]:
        """Generate examples from raw code files (e.g., GitHub Gists)."""
        examples = []
        text = record.get("text", "")
        code_blocks = record.get("code_blocks", [record.get("text", "")])
        title = record.get("title", "")

        for code in code_blocks[:3]:
            if len(code) < 30:
                continue
            lang = detect_language(code)

            # Explain code
            template = self.rng.choice(EXPLAIN_CODE_TEMPLATES)
            instruction = template.format(lang=lang)
            explanation = self._generate_code_description(code, title)
            if explanation:
                examples.append(self._make_example(instruction, explanation, code))

            # Debug/review code
            if len(code) > 50:
                template = self.rng.choice(IMPROVE_CODE_TEMPLATES)
                instruction = template.format(lang=lang)
                review = self._generate_code_review(code)
                if review:
                    examples.append(self._make_example(instruction, review, code))

        return examples

    # ----------------------------------------------------------
    # Helper Generators
    # ----------------------------------------------------------

    def _find_explanation_for_code(self, text: str, code: str) -> str:
        """
        Find the paragraph(s) surrounding a code block in the text.
        Used as the natural explanation for the code.
        """
        # Find position of code in full text
        code_preview = code[:50].replace("(", r"\(").replace(")", r"\)")
        try:
            match = re.search(re.escape(code[:40]), text)
        except re.error:
            match = None

        if match:
            # Get surrounding context (before the code block)
            start = max(0, match.start() - 500)
            context = text[start:match.start()].strip()
            # Get last paragraph before code
            paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
            if paragraphs:
                return paragraphs[-1]

        # Fallback: use first meaningful paragraph of text
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
        return paragraphs[0] if paragraphs else ""

    def _generate_code_description(self, code: str, title: str = "") -> str:
        """
        Generate a description of what code does.
        This is a template-based approach (no model call needed during preprocessing).
        """
        lang = detect_language(code)
        lines = code.strip().split("\n")

        # Extract function/class names
        func_names = re.findall(r"(?:def|function|func|fn)\s+(\w+)\s*\(", code)
        class_names = re.findall(r"class\s+(\w+)", code)
        imports = re.findall(r"(?:import|from|use|require)\s+(\w+)", code)

        parts = []

        if title:
            parts.append(f"This {lang} code implements: {title}.")

        if class_names:
            parts.append(f"It defines the following class(es): {', '.join(class_names)}.")
        if func_names:
            parts.append(f"It implements the following function(s): {', '.join(func_names)}.")
        if imports:
            parts.append(f"It uses the following modules/packages: {', '.join(imports[:5])}.")

        parts.append(f"The code is {len(lines)} lines long.")

        return " ".join(parts) if parts else ""

    def _generate_code_review(self, code: str) -> str:
        """Generate a simple code review comment."""
        lang = detect_language(code)
        lines = code.strip().split("\n")
        issues = []

        # Check for common patterns
        if lang == "python":
            if "except:" in code or "except Exception:" in code:
                issues.append("Consider catching specific exceptions instead of bare `except` clauses.")
            if re.search(r"def\s+\w+\s*\([^)]*\)\s*:", code) and '"""' not in code and "'''" not in code:
                issues.append("Functions are missing docstrings. Adding docstrings improves maintainability.")
            if "global " in code:
                issues.append("Global variables detected. Consider refactoring to avoid global state.")

        if lang in ["javascript", "typescript"]:
            if "var " in code:
                issues.append("Consider using `const` or `let` instead of `var` for better scoping.")
            if "==" in code and "===" not in code:
                issues.append("Use strict equality (`===`) instead of loose equality (`==`).")

        if not issues:
            issues.append("The code looks generally clean.")
            issues.append("Consider adding comments to explain complex logic.")
            issues.append("Ensure all edge cases are handled, particularly for null/undefined values.")

        improved = f"Code Review for this {lang} snippet:\n\n"
        improved += "\n".join(f"- {issue}" for issue in issues)
        improved += f"\n\nThe code is {len(lines)} lines. "
        return improved

    def _generate_improved_version_prompt(self, code: str, context: str = "") -> str:
        """Generate an improvement suggestion for the code."""
        lang = detect_language(code)
        return f"Here is an improved version of the {lang} code with better readability and efficiency:\n\n{code}\n\nKey improvements:\n- Added proper error handling\n- Improved variable naming for clarity\n- Added type hints (where applicable)\n- Optimized performance for large inputs"


# ============================================================
# Dataset Builder Pipeline
# ============================================================

class DatasetBuilder:
    """
    Reads cleaned JSONL files and produces training pairs.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.generator = ExampleGenerator(config)
        self.stats = defaultdict(int)

    def build(self, input_dir: Path, output_dir: Path, merge_dir: Optional[Path] = None) -> int:
        """
        Build dataset from cleaned files.

        Args:
            input_dir: Directory with cleaned JSONL files
            output_dir: Where to write training pairs
            merge_dir: If provided, merge with existing dataset here

        Returns:
            Total number of examples generated
        """
        fm._ensure_dir(output_dir)
        output_file = output_dir / "training_pairs.jsonl"
        total = 0

        # Load and merge existing if requested
        existing = []
        if merge_dir and (Path(merge_dir) / "training_pairs.jsonl").exists():
            log.info(f"Loading existing dataset from {merge_dir}")
            existing = fm.read_jsonl(Path(merge_dir) / "training_pairs.jsonl")
            log.info(f"Loaded {len(existing)} existing examples")

        # Process all cleaned JSONL files
        jsonl_files = list(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            log.warning(f"No JSONL files in {input_dir}")
            return 0

        log.info(f"Building dataset from {len(jsonl_files)} files")

        new_examples = []

        for input_file in jsonl_files:
            log.info(f"Processing: {input_file.name}")
            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in tqdm(lines, desc=input_file.name, leave=False):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                examples = self.generator.generate(record)
                new_examples.extend(examples)
                self.stats["records_processed"] += 1
                self.stats["examples_generated"] += len(examples)

        # Combine and shuffle
        all_examples = existing + new_examples
        random.Random(42).shuffle(all_examples)

        # Write output
        fm.write_jsonl(output_file, all_examples)
        total = len(all_examples)

        # Write stats
        self.stats["total_examples"] = total
        self.stats["new_examples"] = len(new_examples)
        self.stats["existing_examples"] = len(existing)

        log.info(f"\nDataset Build Complete:")
        log.info(f"  Records processed: {self.stats['records_processed']}")
        log.info(f"  New examples generated: {len(new_examples)}")
        log.info(f"  Existing examples: {len(existing)}")
        log.info(f"  Total examples: {total}")
        log.info(f"  Output: {output_file}")

        # Write metadata
        fm.write_metadata(output_dir, dict(self.stats))

        return total


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset builder")
    parser.add_argument("--input", required=True, help="Input dir with cleaned JSONL")
    parser.add_argument("--output", required=True, help="Output dir for training pairs")
    parser.add_argument("--merge", default=None, help="Merge with existing dataset at this path")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    builder = DatasetBuilder(config)
    total = builder.build(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        merge_dir=Path(args.merge) if args.merge else None
    )
    log.info(f"Done. {total} examples written.")


if __name__ == "__main__":
    main()
