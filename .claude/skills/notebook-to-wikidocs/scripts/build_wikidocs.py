#!/usr/bin/env python3
"""노트북(.ipynb)을 WikiDocs 연동용 "장→절" 다중 페이지 마크다운으로 변환합니다.

`book/tools/notebook_to_md.py`의 장→절 분할 규칙을 이어받되, 핵심 차이는
**코드 실행 결과(표·로그·그림)를 함께 싣는다**는 점입니다. 노트북은 보통 출력이
비어 있으므로(Colab용 clean 상태), 다음 우선순위로 "실제 결과"를 확보합니다.

출력 원천 우선순위:
  1) --executed-notebook PATH : 미리 실행해 outputs를 담은 노트북을 출력 원천으로 사용
     (Colab/GPU에서 끝까지 돌린 뒤 저장한 .ipynb. GPU 챕터의 진짜 결과 확보용)
  2) --execute               : 이 자리에서 nbclient로 직접 실행(주로 CPU 챕터 1–6/8/19)
  3) (둘 다 없음)            : 노트북에 이미 들어있는 outputs만 사용. 없으면 코드만 출력하고
                               셀별로 "<!-- 실행 결과 없음 -->" 주석을 남겨 누락을 드러냄
                               (가짜 출력을 지어내지 않음 — 이것이 "파싱만" 금지 요구의 핵심).

출력 렌더링은 book/tools/notebook_to_tex.py의 검증된 로직을 마크다운에 맞게 포팅:
  - stream / text/plain  → 펜스 코드블록(ANSI 제거, pip 노이즈 필터, 길이 truncation)
  - text/html <table>    → 마크다운 표
  - image/png            → assets/에 저장 후 ![](...) 참조 (실제 실행 그림)
  - error                → 트레이스백 펜스 블록

사용:
    python3 build_wikidocs.py 01_tfidf \
        --num 1 --slug tfidf --title "텍스트 벡터화 (TF-IDF)" \
        --pages-dir pages --assets assets --toc TOC.md --execute
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

# 레포 루트: 이 스크립트는 .claude/skills/notebook-to-wikidocs/scripts/ 아래 있음 → parents[4]
ROOT = Path(__file__).resolve().parents[4]

COLAB_BADGE_RE = re.compile(r"^\s*\[!\[.*?Colab.*?\]\(.*?\)\]\(.*?\)\s*$", re.IGNORECASE)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
EMOJI_RE = re.compile(r"^[\s←-⇿⌀-➿⬀-⯿️\U0001F000-\U0001FAFF]+")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# pip/tqdm 등 학습과 무관한 노이즈 라인 제거 (tex 도구와 동일 기준)
SKIP_PATTERNS = (
    "TqdmWarning:",
    "IProgress not found",
    "Requirement already satisfied:",
    "WARNING: Running pip",
    "[notice] A new release of pip",
    "notice] A new release of pip",
    "To update, run:",
)

MAX_OUTPUT_LINES = 40
MAX_OUTPUT_CHARS = 2000

# 절 그룹 분류: (헤더 키워드, 그룹키) — 위에서부터 first-match.
SECTION_RULES: list[tuple[str, str]] = [
    ("삽질", "wrapup"),
    ("라이브러리", "wrapup"),
    ("체크포인트", "wrapup"),
    ("FAQ", "wrapup"),
    ("다음 챕터", "wrapup"),
    ("다음 장", "wrapup"),
    ("예고", "wrapup"),
    ("실습", "practice"),
    ("해부", "anatomy"),
    ("변형", "variation"),
]

SUBPAGES = [
    ("practice", "practice", "실습"),
    ("anatomy", "anatomy", "해부"),
    ("variation", "variation", "변형"),
    ("wrapup", "wrapup", "정리와 FAQ"),
]


# --------------------------------------------------------------------------- #
# 텍스트 유틸
# --------------------------------------------------------------------------- #
def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text).strip()


def _first_header(md: str) -> tuple[int, str] | None:
    for line in md.splitlines():
        m = HEADER_RE.match(line)
        if m:
            return len(m.group(1)), m.group(2).strip()
    return None


def _classify(header_text: str) -> str:
    for kw, group in SECTION_RULES:
        if kw in header_text:
            return group
    return "overview"


def _strip_colab_badge(md: str) -> str:
    return "\n".join(ln for ln in md.splitlines() if not COLAB_BADGE_RE.match(ln)).strip("\n")


def _demote_first_header(md: str) -> str:
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            del lines[i]
            break
    return "\n".join(lines).strip("\n")


def _strip_header_emoji(md: str) -> str:
    out = []
    for line in md.splitlines():
        m = HEADER_RE.match(line)
        if m:
            out.append(f"{m.group(1)} {_strip_emoji(m.group(2))}")
        else:
            out.append(line)
    return "\n".join(out)


def _clean_text_output(text: str) -> str:
    """ANSI 제거 → 캐리지리턴 진행바 마지막 상태만 → 노이즈 라인 제거 → 길이 제한."""
    text = ANSI_RE.sub("", text)
    lines = [seg.split("\r")[-1] for seg in text.split("\n")]
    lines = [
        ln
        for ln in lines
        if not any(p in ln for p in SKIP_PATTERNS)
        and not ln.strip().startswith("from .autonotebook import tqdm")
    ]
    # 앞뒤 빈 줄 정리
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[: MAX_OUTPUT_LINES - 1] + [
            f"... (출력 {len(lines) - MAX_OUTPUT_LINES + 1}줄 생략) ..."
        ]
    text = "\n".join(lines)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[: MAX_OUTPUT_CHARS - 4].rstrip() + "\n..."
    return text


# --------------------------------------------------------------------------- #
# HTML 표 → 마크다운 표
# --------------------------------------------------------------------------- #
class _PandasTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, list]] = []
        self.in_table = self.in_row = self.in_cell = False
        self.cell_is_header = False
        self.current_cell: list[str] = []
        self.current_row: list[tuple[bool, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
            self.tables.append({"headers": [], "rows": []})
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_table and self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.cell_is_header = tag == "th"
            self.current_cell = []
        elif self.in_cell and tag == "br":
            self.current_cell.append(" ")

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.in_cell:
            text = unescape("".join(self.current_cell))
            text = re.sub(r"\s+", " ", text).strip()
            self.current_row.append((self.cell_is_header, text))
            self.in_cell = False
            self.current_cell = []
        elif tag == "tr" and self.in_row:
            if self.current_row and self.tables:
                values = [v for _, v in self.current_row]
                header_count = sum(1 for is_h, _ in self.current_row if is_h)
                data_count = len(self.current_row) - header_count
                table = self.tables[-1]
                if header_count >= data_count:
                    table["headers"] = values
                else:
                    table["rows"].append(values)
            self.in_row = False
            self.current_row = []
        elif tag == "table":
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def _md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _html_tables_to_markdown(html: str) -> list[str]:
    parser = _PandasTableParser()
    parser.feed(html)
    out: list[str] = []
    for table in parser.tables:
        headers, rows = table["headers"], table["rows"]
        if not rows:
            continue
        width = max([len(headers)] + [len(r) for r in rows])
        if not headers:
            headers = [""] * width
        headers = (headers + [""] * width)[:width]
        rows = [(r + [""] * width)[:width] for r in rows[:30]]
        lines = [
            "| " + " | ".join(_md_escape_cell(c) for c in headers) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for r in rows:
            lines.append("| " + " | ".join(_md_escape_cell(c) for c in r) + " |")
        if len(table["rows"]) > 30:
            lines.append("| " + " | ".join(["..."] * width) + " |")
        out.append("\n".join(lines))
    return out


# --------------------------------------------------------------------------- #
# 셀 출력 렌더링
# --------------------------------------------------------------------------- #
def _render_outputs(cell: dict, assets_dir: Path | None, stem: str, counter: list[int]) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = _clean_text_output("".join(out.get("text", [])))
            if text.strip():
                chunks.append("```\n" + text + "\n```")
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            # 1) 이미지 우선 (실제 matplotlib 출력)
            if "image/png" in data:
                counter[0] += 1
                img_name = f"{stem}-out{counter[0]}.png"
                if assets_dir is not None:
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    raw = data["image/png"]
                    raw = raw if isinstance(raw, str) else "".join(raw)
                    (assets_dir / img_name).write_bytes(base64.b64decode(raw))
                chunks.append(f"![output](../assets/{img_name})")
                continue
            # 2) HTML 표 → 마크다운 표
            html = data.get("text/html")
            if isinstance(html, list):
                html = "".join(html)
            if isinstance(html, str) and "<table" in html:
                tables = _html_tables_to_markdown(html)
                if tables:
                    chunks.extend(tables)
                    continue
            # 3) text/plain
            text = data.get("text/plain")
            if text:
                text = _clean_text_output("".join(text) if isinstance(text, list) else str(text))
                if text.strip():
                    chunks.append("```\n" + text + "\n```")
        elif otype == "error":
            tb = out.get("traceback", [])
            if tb:
                text = _clean_text_output("\n".join(str(l) for l in tb[-8:]))
            else:
                text = f"{out.get('ename', 'Error')}: {out.get('evalue', '')}"
            if text.strip():
                chunks.append("```\n" + text + "\n```")
    if not chunks:
        return ""
    return "**실행 결과**\n\n" + "\n\n".join(chunks)


# --------------------------------------------------------------------------- #
# 노트북 실행 (선택)
# --------------------------------------------------------------------------- #
def execute_notebook(path: Path, timeout: int = 1800) -> dict:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    return nb


def _has_any_outputs(nb: dict) -> bool:
    return any(
        c.get("cell_type") == "code" and c.get("outputs")
        for c in nb.get("cells", [])
    )


# --------------------------------------------------------------------------- #
# 변환
# --------------------------------------------------------------------------- #
def convert(nb: dict, num: int, slug: str, title: str,
            pages_dir: Path, assets_dir: Path | None) -> tuple[list[tuple[str, str]], dict]:
    stem = f"{num:02d}-{slug}"
    img_counter = [0]
    stats = {"code_cells": 0, "code_with_output": 0, "images": 0}

    groups: dict[str, list[str]] = {
        "overview": [], "practice": [], "anatomy": [], "variation": [], "wrapup": []
    }
    sub_titles: dict[str, str] = {}
    overview_intro: list[str] = []
    setup_code: list[str] = []

    current = "overview"
    seen_h1 = False

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            md = _strip_colab_badge(_cell_text(cell))
            if not md.strip():
                continue
            hdr = _first_header(md)
            if hdr and hdr[0] == 1 and not seen_h1:
                seen_h1 = True
                body = "\n".join(md.splitlines()[1:]).strip("\n")
                if body.strip():
                    overview_intro.append(body)
                continue
            if hdr and hdr[0] == 2:
                current = _classify(hdr[1])
                if current in ("practice", "anatomy", "variation"):
                    sub_titles[current] = _strip_emoji(hdr[1])
            groups[current].append(_strip_header_emoji(md))
        elif ctype == "code":
            code = _cell_text(cell).rstrip("\n")
            if not code.strip():
                continue
            stats["code_cells"] += 1
            block = "```python\n" + code + "\n```"
            outs = _render_outputs(cell, assets_dir, stem, img_counter)
            if outs:
                stats["code_with_output"] += 1
            else:
                # 가짜 출력을 지어내지 않음. 누락을 드러내는 주석만 남김.
                outs = "<!-- 실행 결과 없음: --execute 또는 --executed-notebook 로 결과를 채우세요 -->"
            piece = block + "\n\n" + outs
            if current == "overview":
                setup_code.append(piece)
            else:
                groups[current].append(piece)

    stats["images"] = img_counter[0]
    pages_dir.mkdir(parents=True, exist_ok=True)
    toc_entries: list[tuple[str, str]] = []

    # --- 개요 페이지 ---
    ov: list[str] = []
    ov.extend(overview_intro)
    ov.extend(groups["overview"])
    present_subs = [(g, sl, sub_titles.get(g, dt)) for g, sl, dt in SUBPAGES
                    if groups[g] or (g == "practice" and setup_code)]
    roadmap = ["## 이 장의 구성"]
    for idx, (g, sl, t) in enumerate(present_subs, 1):
        roadmap.append(f"- [{num:02d}-{idx}. {t}]({stem}-{sl}.md)")
    ov.append("\n".join(roadmap))
    (pages_dir / f"{stem}.md").write_text("\n\n".join(ov).strip() + "\n", encoding="utf-8")
    toc_entries.append((f"{num:02d}. {title}", f"pages/{stem}.md"))

    # --- 절 페이지 ---
    for idx, (g, sl, dt) in enumerate(present_subs, 1):
        parts: list[str] = []
        body_blocks = list(groups[g])
        if g == "practice" and setup_code:
            parts.append("## 환경 준비\n\n" + "\n\n".join(setup_code))
        if g in ("practice", "anatomy", "variation") and body_blocks:
            body_blocks[0] = _demote_first_header(body_blocks[0])
        parts.extend(body_blocks)
        (pages_dir / f"{stem}-{sl}.md").write_text(
            "\n\n".join(p for p in parts if p).strip() + "\n", encoding="utf-8")
        t = sub_titles.get(g, dt)
        toc_entries.append((f"{num:02d}-{idx}. {t}", f"pages/{stem}-{sl}.md"))

    return toc_entries, stats


def upsert_toc(toc_path: Path, book_title: str, entries: list[tuple[str, str]]) -> None:
    """TOC.md에서 이 장(NN. / NN-N.) 블록만 교체하거나 추가. 다른 장은 보존."""
    num = entries[0][0].split(".")[0]  # "01"
    new_lines = []
    for title, path in entries:
        indent = "" if re.match(r"^\d+\.\s", title) else "  "
        new_lines.append(f"{indent}* [{title}]({path})")

    if not toc_path.exists():
        toc_path.write_text(f"# {book_title}\n\n" + "\n".join(new_lines) + "\n", encoding="utf-8")
        return

    lines = toc_path.read_text(encoding="utf-8").splitlines()
    # 이 장에 속한 기존 라인 범위 찾기: "* [NN." 또는 "* [NN-" 로 시작
    chapter_re = re.compile(rf"^\s*\*\s*\[{num}[.\-]")
    start = end = None
    for i, ln in enumerate(lines):
        if chapter_re.match(ln):
            if start is None:
                start = i
            end = i
    if start is None:
        # 없으면 끝에 추가
        out = lines + ([""] if lines and lines[-1].strip() else []) + new_lines
    else:
        out = lines[:start] + new_lines + lines[end + 1:]
    toc_path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chapter", help="챕터 폴더명 (예: 01_tfidf)")
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--pages-dir", default="pages")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--toc", default="TOC.md")
    ap.add_argument("--book-title", default="neuqes-101 — Hugging Face 입문 커리큘럼")
    ap.add_argument("--execute", action="store_true",
                    help="nbclient로 노트북을 실행해 실제 출력을 채움 (CPU 챕터용)")
    ap.add_argument("--executed-notebook", default=None,
                    help="출력이 담긴 실행본 .ipynb 경로 (Colab/GPU 결과 원천)")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else ROOT / pp

    folder = ROOT / args.chapter
    nb_path = folder / f"{args.chapter}.ipynb"
    if not nb_path.exists():
        raise SystemExit(f"노트북을 찾을 수 없습니다: {nb_path}")

    # 출력 원천 결정
    if args.executed_notebook:
        nb = json.loads(_abs(args.executed_notebook).read_text(encoding="utf-8"))
        source = f"executed-notebook ({args.executed_notebook})"
    elif args.execute:
        nb = execute_notebook(nb_path, timeout=args.timeout)
        source = "live --execute"
    else:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        source = "clean notebook (출력 없음 가능)"

    pages_dir = _abs(args.pages_dir)
    assets_dir = _abs(args.assets) if args.assets else None
    entries, stats = convert(nb, args.num, args.slug, args.title, pages_dir, assets_dir)
    upsert_toc(_abs(args.toc), args.book_title, entries)

    print(f"출력 원천: {source}")
    print(f"코드 셀 {stats['code_cells']}개 중 출력 있는 셀 {stats['code_with_output']}개, "
          f"이미지 {stats['images']}개")
    if not args.executed_notebook and not args.execute and not _has_any_outputs(nb):
        print("⚠️  실행 결과가 비어 있습니다. --execute(CPU) 또는 --executed-notebook(GPU)으로 다시 생성하세요.")
    print(f"생성: {len(entries)} 페이지")
    for t, p in entries:
        print(f"  - {t}  →  {p}")


if __name__ == "__main__":
    main()
