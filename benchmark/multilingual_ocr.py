"""Offline CPU Docling OCR screen on fixed multilingual image and scanned-PDF pages.

Synthetic typography checks do not certify arbitrary scans, handwriting or all languages.
The source fixtures predate this OCR experiment. No model/LLM endpoint is called.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import subprocess
import time
import unicodedata
from pathlib import Path

from benchmark.common import file_sha256, local_model_runtime
from memory_service.adapters.parsers import docling_parser

_FONT_ROOT = Path("/usr/share/fonts/truetype/noto")
_SCRIPT_FONTS = {"hi": "Devanagari", "ar": "Arabic", "th": "Thai"}


def characters(text: str) -> str:
    """Case-folded letters, marks and numbers; ignore layout/punctuation explicitly."""
    return "".join(
        char
        for char in unicodedata.normalize("NFC", text).casefold()
        if unicodedata.category(char)[0] in {"L", "M", "N"}
    )


def character_errors(expected: str, actual: str) -> dict:
    left, right = characters(expected), characters(actual)
    if max(len(left), len(right)) > 12000:
        raise ValueError("OCR scoring exceeds its explicit character budget")
    denominator = len(left)
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, target in enumerate(right, start=1):
        current = [row]
        for column, source in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (source != target),
                )
            )
        previous = current
    return {
        "edits": previous[-1],
        "expected_characters": denominator,
        "character_error_rate": previous[-1] / max(1, denominator),
        "all_numbers_preserved": set(re.findall(r"\d+", expected))
        <= set(re.findall(r"\d+", actual)),
    }


def page(case: dict):
    from PIL import Image, ImageDraw, ImageFont, features

    language = case["language"]
    if language in {"zh", "ja", "ko"}:
        font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        index = {"ja": 0, "ko": 1, "zh": 2}[language]
    else:
        suffix = _SCRIPT_FONTS.get(language, "")
        font_path = _FONT_ROOT / f"NotoSans{suffix}-Regular.ttf"
        index = 0
    if language in {"ar", "hi", "th"} and not features.check("raqm"):
        raise RuntimeError("The fixture renderer needs RAQM for script shaping")
    font = ImageFont.truetype(str(font_path), size=40, index=index)
    # Enough ordinary text for script detection; repeated material is disclosed, not
    # counted as independent examples. One sentence per line avoids invented word breaks.
    lines = case["sentences"] * 4
    canvas = Image.new("RGB", (2400, 100 + len(lines) * 78), color="white")
    draw = ImageDraw.Draw(canvas)
    for number, text in enumerate(lines):
        direction = "rtl" if language == "ar" else "ltr"
        x = 2340 if direction == "rtl" else 60
        anchor = "ra" if direction == "rtl" else "la"
        if draw.textlength(text, font=font, direction=direction) > 2280:
            raise ValueError("The fixture would crop source text")
        draw.text(
            (x, 50 + number * 78), text, font=font, fill="black", direction=direction, anchor=anchor
        )
    return (
        canvas,
        "\n".join(lines),
        {"path": str(font_path), "sha256": file_sha256(font_path), "index": index},
    )


async def run(args) -> None:
    source = json.loads(args.data.read_text())
    parser = docling_parser.DoclingParser()
    version, languages = await asyncio.gather(
        asyncio.to_thread(subprocess.check_output, ["tesseract", "--version"], text=True),
        asyncio.to_thread(subprocess.check_output, ["tesseract", "--list-langs"], text=True),
    )
    model_files = await asyncio.to_thread(
        lambda: {
            str(path.relative_to(parser._artifacts)): file_sha256(path)
            for path in sorted(parser._artifacts.rglob("*"))
            if path.is_file() and path.suffix in {".json", ".onnx", ".safetensors", ".bin", ".pt"}
        }
    )
    output = {
        "complete": False,
        "fixture_sha256": file_sha256(args.data),
        "parser_sha256": file_sha256(Path(docling_parser.__file__)),
        "fixture_provenance": source["provenance"],
        "runtime": local_model_runtime(),
        "image_digest": args.image_digest,
        "docling_version": docling_parser._docling_version(),
        "model_files_sha256": model_files,
        "tesseract_version": version.splitlines()[0],
        "installed_ocr_languages": languages.splitlines()[1:],
        "llm_calls": 0,
        "records": [],
        "limitations": [
            "Synthetic clean pages, not an independent scan/handwriting benchmark.",
            "Each language uses three source sentences repeated four times to support script detection.",
            "Character error rate ignores case, whitespace and punctuation; numbers are checked separately.",
            "Parser/component latency, not ingestion queue or retrieval endpoint latency.",
        ],
    }
    try:
        for case in source["records"]:
            if case["category"] != "antecedent":
                continue
            canvas, expected, font = page(case)
            for format_name, mime in (("PNG", "image/png"), ("PDF", "application/pdf")):
                blob = io.BytesIO()
                canvas.save(blob, format=format_name, resolution=216)
                row = {
                    "language": case["language"],
                    "format": format_name,
                    "expected": expected,
                    "font": font,
                }
                started = time.perf_counter()
                try:
                    parsed = await parser.parse(
                        document_id="ocr-screen",
                        tenant_id="ocr-screen",
                        filename=f"fixture.{format_name.lower()}",
                        media_type=mime,
                        data=blob.getvalue(),
                    )
                    actual = "\n".join(node.text for node in parsed.nodes if node.text)
                    row.update(actual=actual, metrics=character_errors(expected, actual))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                row["latency_ms"] = (time.perf_counter() - started) * 1000
                output["records"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
                print(
                    case["language"], format_name, row.get("metrics", row.get("error")), flush=True
                )
    finally:
        parser._runner.close()
    output["complete"] = True
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("tests/eval/golden/multilingual_narrative.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
