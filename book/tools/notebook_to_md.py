#!/usr/bin/env python3
"""노트북(.ipynb)을 WikiDocs 연동용 "장→절" 다중 페이지로 변환합니다.

노트북이 단일 출처입니다. 위키독스 인기 책(점프 투 시리즈, 공식 연동 데모)의
관례에 맞춰, 한 노트북(=장)을 개요 페이지 1개 + 절 페이지 여러 개로 쪼갭니다.
의존성(nbconvert 등) 없이 표준 라이브러리만 사용합니다.

분할 규칙 (노트북 표준 구조의 H2 헤더 키워드 기반):
- 개요(overview): 제목/도입 + 학습 흐름 + 추적표 + 변경점 + Loss + 토크나이저 노트
  → 프레이밍 섹션. 끝에 "이 장의 구성"(절 링크) 자동 추가.
- 실습 / 해부 / 변형: 각각 절 페이지 1개.
- 정리(wrapup): 라이브러리 + 체크포인트 + FAQ + 삽질 + 다음 챕터 예고.

특이 처리:
- 노트북 첫 H1 제목은 제거(위키독스는 TOC.md 제목을 페이지 제목으로 씀).
- 프레이밍 구간에 있던 환경 셋업 코드(pip install/import)는 실습 페이지 맨 앞
  "환경 준비"로 이동.
- 절 페이지(실습/해부/변형)는 자기 H2 헤더를 제거하고 본문부터 시작
  (TOC 제목과 중복 방지). 헤더의 이모지는 모두 제거.

사용:
    python3 book/tools/notebook_to_md.py 01_tfidf \
        --num 1 --slug tfidf --title "텍스트 벡터화 (TF-IDF)" \
        --pages-dir pages --assets assets --toc TOC.md
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COLAB_BADGE_RE = re.compile(r"^\s*\[!\[.*?Colab.*?\]\(.*?\)\]\(.*?\)\s*$", re.IGNORECASE)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 헤더 앞머리의 이모지/기호 제거용 (대략적인 이모지 영역 + 변형 셀렉터)
EMOJI_RE = re.compile(
    r"^[\s←-⇿⌀-➿⬀-⯿️\U0001F000-\U0001FAFF]+"
)

# 절 그룹 분류: (헤더에 포함된 키워드, 그룹키) — 위에서부터 first-match.
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
# 나머지(학습 흐름/추적표/변경점/Loss/토크나이저 노트 등)는 전부 개요로.

# 절 페이지 슬러그/순서/기본 제목
SUBPAGES = [
    ("practice", "practice", "실습"),
    ("anatomy", "anatomy", "해부"),
    ("variation", "variation", "변형"),
    ("wrapup", "wrapup", "정리와 FAQ"),
]


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
    return "\n".join(
        ln for ln in md.splitlines() if not COLAB_BADGE_RE.match(ln)
    ).strip("\n")


def _demote_first_header(md: str) -> str:
    """절 페이지의 맨 앞 H2 헤더 한 줄을 제거하고 본문부터 시작."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            del lines[i]
            break
    return "\n".join(lines).strip("\n")


def _strip_header_emoji(md: str) -> str:
    """본문 내 모든 헤더 줄의 이모지를 제거."""
    out = []
    for line in md.splitlines():
        m = HEADER_RE.match(line)
        if m:
            out.append(f"{m.group(1)} {_strip_emoji(m.group(2))}")
        else:
            out.append(line)
    return "\n".join(out)


def _render_outputs(cell: dict, assets_dir: Path | None, stem: str, counter: list[int]) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        otype = out.get("output_type")
        if otype == "stream":
            text = _clean_stream("".join(out.get("text", [])))
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


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean_stream(text: str, max_lines: int = 40) -> str:
    """학습 로그 위생: ANSI 제거, \\r 진행바는 마지막 상태만, 길면 truncate."""
    text = ANSI_RE.sub("", text)
    # 캐리지리턴 진행바: 각 줄에서 마지막 \r 이후만 남김
    lines = [seg.split("\r")[-1] for seg in text.split("\n")]
    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        lines = head + [f"... (출력 {len(lines) - max_lines + 1}줄 생략) ..."]
    return "\n".join(lines)


def convert(nb_path: Path, num: int, slug: str, title: str,
            pages_dir: Path, assets_dir: Path | None) -> list[tuple[str, str]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    stem = f"{num:02d}-{slug}"
    img_counter = [0]

    # 그룹별 블록 수집
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
                # H1 제목 제거, 뒤따르는 도입부 문단만 개요 인트로로
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
            block = ("```python\n" + code + "\n```") if code.strip() else ""
            outs = _render_outputs(cell, assets_dir, stem, img_counter)
            piece = "\n\n".join(p for p in (block, outs) if p)
            if not piece:
                continue
            if current == "overview":
                setup_code.append(piece)  # 환경 셋업 → 실습으로 이동
            else:
                groups[current].append(piece)

    pages_dir.mkdir(parents=True, exist_ok=True)
    toc_entries: list[tuple[str, str]] = []  # (title, relpath)

    # --- 개요 페이지 ---
    ov: list[str] = []
    if overview_intro:
        ov.extend(overview_intro)
    ov.extend(groups["overview"])
    # 이 장의 구성 (절 링크)
    roadmap = ["## 이 장의 구성"]
    present_subs = [(g, sl, sub_titles.get(g, dt)) for g, sl, dt in SUBPAGES
                    if groups[g] or (g == "practice" and setup_code)]
    for idx, (g, sl, t) in enumerate(present_subs, 1):
        roadmap.append(f"- [{num:02d}-{idx}. {t}]({stem}-{sl}.md)")
    ov.append("\n".join(roadmap))
    overview_path = pages_dir / f"{stem}.md"
    overview_path.write_text("\n\n".join(ov).strip() + "\n", encoding="utf-8")
    toc_entries.append((f"{num:02d}. {title}", f"pages/{stem}.md"))

    # --- 절 페이지 ---
    for idx, (g, sl, dt) in enumerate(present_subs, 1):
        parts: list[str] = []
        body_blocks = list(groups[g])
        if g == "practice" and setup_code:
            parts.append("## 환경 준비\n\n" + "\n\n".join(setup_code))
        if g in ("practice", "anatomy", "variation") and body_blocks:
            # 첫 블록(섹션 헤더 포함)의 헤더 제거
            body_blocks[0] = _demote_first_header(body_blocks[0])
        parts.extend(body_blocks)
        sub_path = pages_dir / f"{stem}-{sl}.md"
        sub_path.write_text("\n\n".join(p for p in parts if p).strip() + "\n",
                            encoding="utf-8")
        t = sub_titles.get(g, dt)
        toc_entries.append((f"{num:02d}-{idx}. {t}", f"pages/{stem}-{sl}.md"))

    print(f"생성: {len(toc_entries)} 페이지 (이미지 {img_counter[0]})")
    for t, p in toc_entries:
        print(f"  - {t}  →  {p}")
    return toc_entries


def write_toc(toc_path: Path, book_title: str, entries: list[tuple[str, str]]) -> None:
    lines = [f"# {book_title}", ""]
    for title, path in entries:
        # 개요(NN.)는 최상위, 절(NN-N.)은 2칸 들여쓰기
        indent = "" if re.match(r"^\d+\.\s", title) else "  "
        lines.append(f"{indent}* [{title}]({path})")
    toc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"TOC: {toc_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", help="챕터 폴더명 (예: 01_tfidf)")
    ap.add_argument("--num", type=int, required=True, help="장 번호")
    ap.add_argument("--slug", required=True, help="장 슬러그 (예: tfidf)")
    ap.add_argument("--title", required=True, help="장 제목")
    ap.add_argument("--pages-dir", default="pages")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--toc", default=None, help="TOC.md 경로(지정 시 이 장만으로 생성)")
    ap.add_argument("--book-title", default="neuqes-101 — Hugging Face 입문 커리큘럼")
    args = ap.parse_args()

    folder = ROOT / args.chapter
    nb = folder / f"{args.chapter}.ipynb"
    if not nb.exists():
        raise SystemExit(f"노트북을 찾을 수 없습니다: {nb}")

    def _abs(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else ROOT / pp

    pages_dir = _abs(args.pages_dir)
    assets_dir = _abs(args.assets) if args.assets else None
    entries = convert(nb, args.num, args.slug, args.title, pages_dir, assets_dir)

    if args.toc:
        write_toc(_abs(args.toc), args.book_title, entries)


if __name__ == "__main__":
    main()
