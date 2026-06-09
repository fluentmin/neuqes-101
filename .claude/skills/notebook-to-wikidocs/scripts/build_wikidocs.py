#!/usr/bin/env python3
"""노트북(.ipynb)을 WikiDocs 연동용 "장→절" 다중 페이지 마크다운으로 변환합니다.

`book/tools/notebook_to_md.py`의 장→절 분할 규칙을 이어받되, 핵심 차이는
**코드 실행 결과(표·로그·그림)를 함께 싣는다**는 점입니다. 노트북은 보통 출력이
비어 있으므로(Colab용 clean 상태), 다음 우선순위로 "실제 결과"를 확보합니다.

출력 원천 우선순위 (챕터별 자동):
  1) --executed-notebook PATH    : (단일 챕터) 미리 실행해 outputs를 담은 노트북
  2) executed/<폴더>.ipynb 존재   : 자동으로 출력 원천으로 사용
                                    (Colab/GPU에서 끝까지 돌린 뒤 저장·커밋한 실행본)
  3) --execute                   : 이 자리에서 nbclient로 직접 실행(주로 CPU 챕터).
                                    --save-executed 면 결과를 executed/<폴더>.ipynb 로 저장.
  4) (없음)                      : 노트북에 든 outputs만 사용. 없으면 코드만 출력하고
                                    "<!-- 실행 결과 없음 -->" 주석을 남겨 누락을 드러냄
                                    (가짜 출력을 지어내지 않음 — "파싱만" 금지 요구의 핵심).

실행본 보관 규약: GPU 챕터의 진짜 결과는 Colab T4에서 끝까지 돌린 뒤
"파일 > .ipynb 다운로드"(출력 포함)한 노트북을 `executed/<폴더>.ipynb` 로 커밋해 둔다.
챕터 폴더에는 clean 노트북만 남긴다(Colab 버튼 대상). 자세한 건 executed/README.md.

챕터 지정 (동적):
  - 위치 인자로 챕터를 받음: 폴더명(`07_bert_pipeline`), 번호(`7`/`07`) 모두 허용. 여러 개 가능.
  - 아무 챕터도 안 주고 `--all`도 없으면 에러 — 호출자가 의도(전체/일부)를 명시하게 함.
  - `--all` 이면 레포 루트의 `NN_slug/NN_slug.ipynb` 를 전부 자동 발견해 변환.
  - 챕터 메타(제목): book/tools/notebook_to_tex.py 의 CHAPTERS 레지스트리 → 노트북 첫 H1
    ("Chapter N." 접두 제거) → 슬러그 순으로 해석. 레지스트리에 없는 새 챕터도 동작.

사용:
  # 전체 (호출자가 사용자 확인 후)
  python3 build_wikidocs.py --all --execute
  # 일부
  python3 build_wikidocs.py 1 7 15 --execute
  python3 build_wikidocs.py 07_bert_pipeline --executed-notebook 07_bert_pipeline/07_bert_pipeline.executed.ipynb
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import traceback
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

# 레포 루트: 이 스크립트는 .claude/skills/notebook-to-wikidocs/scripts/ 아래 있음 → parents[4]
ROOT = Path(__file__).resolve().parents[4]

COLAB_BADGE_RE = re.compile(r"^\s*\[!\[.*?Colab.*?\]\(.*?\)\]\(.*?\)\s*$", re.IGNORECASE)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
EMOJI_RE = re.compile(r"^[\s←-⇿⌀-➿⬀-⯿️\U0001F000-\U0001FAFF]+")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
CHAPTER_FOLDER_RE = re.compile(r"^(\d{2})_(.+)$")
H1_CHAPTER_PREFIX_RE = re.compile(r"^\s*Chapter\s+\d+\s*[.．]\s*")

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

DEFAULT_BOOK_TITLE = "neuqes-101 — Hugging Face 입문 커리큘럼"


# --------------------------------------------------------------------------- #
# 텍스트 유틸
# --------------------------------------------------------------------------- #
def _cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text).strip()


def _clean_heading_text(text: str) -> str:
    """헤더 텍스트 정리: 선두 'N.'/'N)' 순번 제거 → 선두 이모지 제거.
    절 제목 중복("07-1. 1. 🚀 실습")과 이모지 잔존을 막는다. (예: '1. 🚀 실습: …' → '실습: …')
    """
    text = re.sub(r"^\s*\d+[.)]\s*", "", text.strip())
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
            out.append(f"{m.group(1)} {_clean_heading_text(m.group(2))}")
        else:
            out.append(line)
    return "\n".join(out)


def _clean_text_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    lines = [seg.split("\r")[-1] for seg in text.split("\n")]
    lines = [
        ln for ln in lines
        if not any(p in ln for p in SKIP_PATTERNS)
        and not ln.strip().startswith("from .autonotebook import tqdm")
    ]
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


def latex_title_to_plain(title: str) -> str:
    """레지스트리 제목의 LaTeX 이스케이프 해제: '\\&' → '&', '\\_' → '_' 등."""
    return (
        title.replace("\\&", "&").replace("\\_", "_")
        .replace("\\%", "%").replace("\\#", "#").replace("\\$", "$")
    )


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


def _html_tables_to_text(html: str) -> list[str]:
    """HTML 표 → 공백 정렬된 모노스페이스 텍스트(코드펜스에 넣어 블록인용 안전).
    text/plain이 없을 때의 폴백."""
    parser = _PandasTableParser()
    parser.feed(html)
    out: list[str] = []
    for table in parser.tables:
        headers, rows = table["headers"], table["rows"]
        if not rows:
            continue
        width = max([len(headers)] + [len(r) for r in rows])
        headers = (headers + [""] * width)[:width] if headers else [""] * width
        shown = [(r + [""] * width)[:width] for r in rows[:30]]
        grid = [headers] + shown
        colw = [max(len(str(row[c])) for row in grid) for c in range(width)]
        def fmt(row): return "  ".join(str(row[c]).ljust(colw[c]) for c in range(width)).rstrip()
        lines = [fmt(headers)] + [fmt(r) for r in shown]
        if len(rows) > 30:
            lines.append("...")
        out.append("\n".join(lines))
    return out


# --------------------------------------------------------------------------- #
# 셀 출력 렌더링
# --------------------------------------------------------------------------- #
def _blockquote(text: str) -> str:
    """모든 줄 앞에 '> '를 붙여 블록인용으로 감싼다(빈 줄은 '>')."""
    return "\n".join(("> " + ln) if ln.strip() else ">" for ln in text.split("\n"))


def _render_outputs(cell: dict, assets_dir: Path | None, stem: str, counter: list[int]) -> str:
    """실행 결과를 코드와 구분되는 **블록인용 카드**로 렌더링.

    WikiDocs에서 코드 블록과 출력이 같은 회색 박스로 보여 헷갈리는 문제를 해결하기 위해,
    출력 전체를 '> '로 감싼다(왼쪽 세로바 카드). 블록인용 안에 마크다운 표를 넣으면
    렌더링이 깨질 수 있어, 표 형태 출력도 text/plain(콘솔 표현)을 코드펜스로 통일한다.
    """
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = _clean_text_output("".join(out.get("text", [])))
            if text.strip():
                chunks.append("```\n" + text + "\n```")
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
                continue
            # 블록인용 중첩 안전: 표도 text/plain(콘솔 표현)을 코드펜스로.
            text = data.get("text/plain")
            if text:
                text = _clean_text_output("".join(text) if isinstance(text, list) else str(text))
                if text.strip():
                    chunks.append("```\n" + text + "\n```")
                continue
            # text/plain이 없을 때만 HTML 표 → 텍스트 표(코드펜스)로 폴백.
            html = data.get("text/html")
            if isinstance(html, list):
                html = "".join(html)
            if isinstance(html, str) and "<table" in html:
                texts = _html_tables_to_text(html)
                for t in texts:
                    chunks.append("```\n" + t + "\n```")
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
    body = "**▶ 실행 결과**\n\n" + "\n\n".join(chunks)
    return _blockquote(body)


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
    return any(c.get("cell_type") == "code" and c.get("outputs") for c in nb.get("cells", []))


def chapter_h1_title(nb: dict) -> str:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        m = _first_header(_cell_text(cell))
        if m and m[0] == 1:
            return H1_CHAPTER_PREFIX_RE.sub("", m[1]).strip()
    return ""


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
                    sub_titles[current] = _clean_heading_text(hdr[1])
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
                outs = "<!-- 실행 결과 없음: --execute 또는 --executed-notebook 로 결과를 채우세요 -->"
            piece = block + "\n\n" + outs
            if current == "overview":
                setup_code.append(piece)
            else:
                groups[current].append(piece)

    stats["images"] = img_counter[0]
    pages_dir.mkdir(parents=True, exist_ok=True)
    toc_entries: list[tuple[str, str]] = []

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


# --------------------------------------------------------------------------- #
# TOC
# --------------------------------------------------------------------------- #
def upsert_toc(toc_path: Path, book_title: str, num: int, entries: list[tuple[str, str]]) -> None:
    """TOC.md에서 이 장(NN. / NN-N.) 블록만 교체하거나 추가. 다른 장은 보존."""
    nn = f"{num:02d}"
    new_lines = []
    for title, path in entries:
        indent = "" if re.match(r"^\d+\.\s", title) else "  "
        new_lines.append(f"{indent}* [{title}]({path})")

    if not toc_path.exists():
        toc_path.write_text(f"# {book_title}\n\n" + "\n".join(new_lines) + "\n", encoding="utf-8")
        return

    lines = toc_path.read_text(encoding="utf-8").splitlines()
    chapter_re = re.compile(rf"^\s*\*\s*\[{nn}[.\-]")
    start = end = None
    for i, ln in enumerate(lines):
        if chapter_re.match(ln):
            if start is None:
                start = i
            end = i
    if start is None:
        # 번호 오름차순 유지: 다음으로 큰 장 앞에 삽입, 없으면 끝에 추가
        insert_at = len(lines)
        any_chapter = re.compile(r"^\s*\*\s*\[(\d{2})[.\-]")
        for i, ln in enumerate(lines):
            m = any_chapter.match(ln)
            if m and int(m.group(1)) > num:
                insert_at = i
                break
        out = lines[:insert_at] + new_lines + lines[insert_at:]
    else:
        out = lines[:start] + new_lines + lines[end + 1:]
    toc_path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 챕터 발견 / 선택 / 메타
# --------------------------------------------------------------------------- #
def discover_chapters() -> dict[int, tuple[str, str, Path]]:
    """{num: (folder, slug, nb_path)} — 레포 루트의 NN_slug/NN_slug.ipynb 자동 발견."""
    found: dict[int, tuple[str, str, Path]] = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = CHAPTER_FOLDER_RE.match(d.name)
        if not m:
            continue
        nb = d / f"{d.name}.ipynb"
        if nb.exists():
            found[int(m.group(1))] = (d.name, m.group(2), nb)
    return found


def load_registry_titles() -> dict[int, str]:
    """book/tools/notebook_to_tex.py 의 CHAPTERS 에서 {num: plain_title}."""
    try:
        sys.path.insert(0, str(ROOT / "book" / "tools"))
        import notebook_to_tex as t  # noqa: E402
        return {c.number: latex_title_to_plain(c.title) for c in t.CHAPTERS}
    except Exception:
        return {}


def resolve_title(num: int, slug: str, nb: dict, registry: dict[int, str]) -> str:
    if num in registry and registry[num].strip():
        return registry[num]
    h1 = chapter_h1_title(nb)
    if h1:
        return h1
    return slug.replace("_", " ")


def parse_chapter_args(tokens: list[str], available: dict[int, tuple]) -> list[int]:
    """'7' / '07' / '07_bert_pipeline' → 정렬된 챕터 번호 리스트."""
    nums: list[int] = []
    for tok in tokens:
        m = CHAPTER_FOLDER_RE.match(tok)
        if m:
            n = int(m.group(1))
        elif tok.isdigit():
            n = int(tok)
        else:
            raise SystemExit(f"챕터 인자를 해석할 수 없습니다: {tok!r} (예: 7, 07, 07_bert_pipeline)")
        if n not in available:
            raise SystemExit(f"챕터 {n:02d} 를 찾을 수 없습니다 (NN_slug/NN_slug.ipynb 없음)")
        if n not in nums:
            nums.append(n)
    return sorted(nums)


def pick_source_notebook(folder: str, slug: str, nb_path: Path,
                         executed_dir: Path, args) -> tuple[dict, str]:
    """출력 원천 우선순위에 따라 (노트북 dict, 원천설명) 반환.

    --execute 로 새로 실행했고 --save-executed 면 executed/<폴더>.ipynb 로 저장한다.
    """
    if args.executed_notebook:
        p = Path(args.executed_notebook)
        p = p if p.is_absolute() else ROOT / p
        return json.loads(p.read_text(encoding="utf-8")), f"executed-notebook({p.name})"
    archived = executed_dir / f"{folder}.ipynb"
    if archived.exists():
        return json.loads(archived.read_text(encoding="utf-8")), f"executed/{archived.name}"
    if args.execute:
        nb = execute_notebook(nb_path, timeout=args.timeout)
        if args.save_executed:
            import nbformat
            executed_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, str(executed_dir / f"{folder}.ipynb"))
        return nb, "live --execute" + (" (executed/ 저장됨)" if args.save_executed else "")
    return json.loads(nb_path.read_text(encoding="utf-8")), "clean(출력없음 가능)"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapters", nargs="*",
                    help="변환할 챕터(폴더명/번호). 비우고 --all 로 전체 지정.")
    ap.add_argument("--all", action="store_true", help="발견된 모든 챕터를 변환")
    ap.add_argument("--pages-dir", default="pages")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--toc", default="TOC.md")
    ap.add_argument("--book-title", default=DEFAULT_BOOK_TITLE)
    ap.add_argument("--execute", action="store_true",
                    help="nbclient로 실행해 실제 출력을 채움 (CPU 챕터용; GPU 챕터엔 비권장)")
    ap.add_argument("--executed-notebook", default=None,
                    help="(단일 챕터) 출력이 담긴 실행본 .ipynb 경로")
    ap.add_argument("--executed-dir", default="executed",
                    help="실행본 보관 폴더 (executed/<폴더>.ipynb 를 출력 원천으로 자동 사용)")
    ap.add_argument("--save-executed", action="store_true",
                    help="--execute 결과를 executed/<폴더>.ipynb 로 저장")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    available = discover_chapters()
    if not available:
        raise SystemExit("변환할 챕터를 찾지 못했습니다 (NN_slug/NN_slug.ipynb 없음)")

    if args.chapters:
        selected = parse_chapter_args(args.chapters, available)
    elif args.all:
        selected = sorted(available)
    else:
        raise SystemExit(
            "변환할 챕터를 지정하거나 --all 을 주세요.\n"
            f"  발견된 챕터: {', '.join(f'{n:02d}' for n in sorted(available))}"
        )

    if args.executed_notebook and len(selected) != 1:
        raise SystemExit("--executed-notebook 은 챕터 1개만 지정했을 때 씁니다.")

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else ROOT / pp

    pages_dir = _abs(args.pages_dir)
    assets_dir = _abs(args.assets) if args.assets else None
    toc_path = _abs(args.toc)
    executed_dir = _abs(args.executed_dir)
    registry = load_registry_titles()

    print(f"변환 대상 {len(selected)}개 챕터: {', '.join(f'{n:02d}' for n in selected)}\n")
    ok, failed, empty = [], [], []
    for num in selected:
        folder, slug, nb_path = available[num]
        try:
            nb, source = pick_source_notebook(folder, slug, nb_path, executed_dir, args)
            title = resolve_title(num, slug, nb, registry)
            entries, stats = convert(nb, num, slug, title, pages_dir, assets_dir)
            upsert_toc(toc_path, args.book_title, num, entries)
            no_out = stats["code_cells"] - stats["code_with_output"]
            flag = ""
            if not _has_any_outputs(nb):
                empty.append(num)
                flag = "  ⚠️ 실행 결과 비어 있음"
            print(f"[{num:02d}] {title}")
            print(f"     원천={source}  코드셀 {stats['code_cells']}개 "
                  f"(출력 {stats['code_with_output']} / 없음 {no_out}) 이미지 {stats['images']}{flag}")
            ok.append(num)
        except Exception as e:  # 챕터별 실패 격리
            failed.append((num, e))
            print(f"[{num:02d}] 실패: {e}")
            traceback.print_exc(limit=2)

    print(f"\n완료: 성공 {len(ok)} / 실패 {len(failed)}")
    if empty:
        print(f"⚠️ 실행 결과가 빈 챕터: {', '.join(f'{n:02d}' for n in empty)} "
              f"→ --execute(CPU) 또는 NN_slug/NN_slug.executed.ipynb(GPU) 로 다시 생성하세요.")
    if failed:
        print("실패 챕터: " + ", ".join(f"{n:02d}" for n, _ in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
