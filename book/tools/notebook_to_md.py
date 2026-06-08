#!/usr/bin/env python3
"""노트북(.ipynb)을 WikiDocs 연동용 마크다운 페이지로 변환합니다.

노트북이 단일 출처입니다. 이 스크립트는 마크다운/코드 셀을 추출해
WikiDocs `pages/` 에 들어갈 .md 한 장을 만듭니다. 의존성(nbconvert 등) 없이
표준 라이브러리만 사용합니다.

변환 규칙:
- 마크다운 셀: 그대로 통과. 단 Colab 배지 줄은 제거(WikiDocs에선 의미 없음).
- 코드 셀: ```python 펜스로 감쌈.
- 출력 셀: 텍스트 출력은 ``` 블록, 이미지(png)는 assets/ 로 추출 후 링크.

사용:
    python3 book/tools/notebook_to_md.py 01_tfidf \
        --out wikidocs/pages/01-tfidf.md --assets wikidocs/assets
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Colab 배지/열기 버튼 줄 — WikiDocs에서는 불필요하므로 제거
COLAB_BADGE_RE = re.compile(r"^\s*\[!\[.*?Colab.*?\]\(.*?\)\]\(.*?\)\s*$", re.IGNORECASE)


def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _strip_colab_badge(md: str) -> str:
    lines = [ln for ln in md.splitlines() if not COLAB_BADGE_RE.match(ln)]
    return "\n".join(lines).strip("\n")


def _render_outputs(cell: dict, assets_dir: Path, stem: str, counter: list[int]) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = "".join(out.get("text", []))
            if text.strip():
                chunks.append("```\n" + text.rstrip("\n") + "\n```")
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "image/png" in data:
                counter[0] += 1
                img_name = f"{stem}-out{counter[0]}.png"
                if assets_dir is not None:
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    raw = data["image/png"]
                    raw = raw if isinstance(raw, str) else "".join(raw)
                    (assets_dir / img_name).write_bytes(base64.b64decode(raw))
                chunks.append(f"![output](../assets/{img_name})")
            elif "text/plain" in data:
                text = "".join(data["text/plain"])
                if text.strip():
                    chunks.append("```\n" + text.rstrip("\n") + "\n```")
    return "\n\n".join(chunks)


def convert(nb_path: Path, out_path: Path, assets_dir: Path | None) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    stem = out_path.stem
    img_counter = [0]
    parts: list[str] = []

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            md = _strip_colab_badge(_cell_text(cell))
            if md.strip():
                parts.append(md)
        elif ctype == "code":
            code = _cell_text(cell).rstrip("\n")
            if code.strip():
                parts.append("```python\n" + code + "\n```")
            outs = _render_outputs(cell, assets_dir, stem, img_counter)
            if outs:
                parts.append(outs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out_path}  ({len(parts)} blocks, {img_counter[0]} images)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", help="챕터 폴더명 (예: 01_tfidf)")
    ap.add_argument("--out", required=True, help="출력 .md 경로")
    ap.add_argument("--assets", default=None, help="이미지 추출 디렉터리")
    args = ap.parse_args()

    folder = ROOT / args.chapter
    nb = folder / f"{args.chapter}.ipynb"
    if not nb.exists():
        raise SystemExit(f"노트북을 찾을 수 없습니다: {nb}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    assets_dir = None
    if args.assets:
        assets_dir = Path(args.assets)
        if not assets_dir.is_absolute():
            assets_dir = ROOT / assets_dir

    convert(nb, out_path, assets_dir)


if __name__ == "__main__":
    main()
