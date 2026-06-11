#!/usr/bin/env python3
"""executed/run_on_colab.ipynb 생성기.

Colab T4에서 여는 '실행 결과 러너' 노트북을 만든다. 러너는 각 챕터 clean 노트북을
nbclient로 끝까지 실행해 출력이 포함된 executed/<폴더>.ipynb 를 만들고, 포크 master 로
직접 커밋·푸시한다(고민 8, 결정 #8=A). 멱등: clean 노트북 해시가 그대로면 건너뛴다.

사용:
  python3 .claude/skills/notebook-to-wikidocs/scripts/make_colab_runner.py
  # → executed/run_on_colab.ipynb 재생성
"""
from __future__ import annotations

import json
from pathlib import Path

# 레포 루트 = 이 파일 기준 ../../../../ (.claude/skills/notebook-to-wikidocs/scripts)
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "executed" / "run_on_colab.ipynb"


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


MD_INTRO = """\
# 📥 Colab 실행 결과 러너 (`executed/` 생산)

이 노트북을 **Colab T4** 에서 열어, 각 챕터 노트북을 끝까지 실행하고
**출력이 포함된 `executed/<폴더>.ipynb`** 를 포크 `master` 로 커밋·푸시합니다.

- 변환기(`build_wikidocs.py`)가 `executed/<폴더>.ipynb` 가 있으면 **자동으로** 실제 출력 원천
  (`▶ 실행 결과`)으로 씁니다. 없으면 합성(`▶ 출력 형태`)으로 폴백합니다.
- **멱등·재개**: clean 노트북이 안 바뀌었으면(해시 동일) 건너뜁니다. 없거나 바뀐 챕터만 실행.
  Colab 세션 한계(아이들 끊김·최대 ~12h) 때문에 **여러 번 나눠 돌려도 이어집니다.**
- tex 의 결괏값은 신뢰하지 않습니다 — **executed/ 만 canonical** 입니다(고민 2·8).

**사용 순서**: ① 아래 *설정* 셀에서 대상 챕터·토큰 지정 → ② 위에서부터 전부 실행(`런타임 > 모두 실행`).
런타임 유형이 **T4 GPU** 인지 먼저 확인하세요(`런타임 > 런타임 유형 변경`).
"""

CODE_SETUP = """\
# 1) 의존성 설치 + 포크 클론
import os, subprocess

REPO   = "fluentmin/neuqes-101"   # 포크 (executed/ 를 여기 master 로 푸시)
BRANCH = "master"

get_ipython().system('pip -q install nbclient nbformat >/dev/null')

WORK = "/content/neuqes-101"
if not os.path.isdir(WORK):
    get_ipython().system('git clone -q https://github.com/{REPO}.git {WORK}'.format(REPO=REPO))
get_ipython().run_line_magic('cd', WORK)
get_ipython().system('git checkout -q {BRANCH} && git pull -q'.format(BRANCH=BRANCH))

print("GPU 확인:")
get_ipython().system('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️ GPU 없음 → 런타임 유형을 T4 로 변경하세요"')
"""

CODE_CONFIG = """\
# 2) 설정 — 대상 챕터 + GitHub 토큰
#   TARGET: "stale"  → 없거나 바뀐 챕터만 (기본, 권장)
#           "all"    → 전 챕터
#           "gpu"    → 07 이상(GPU) 전부
#           리스트    → 예: [1, 7, "24_gpt_tinystories"]  (번호 또는 폴더명)
TARGET = "stale"
FORCE  = False             # True 면 해시가 같아도 재실행
PER_CELL_TIMEOUT = 60 * 60 # 셀당 최대 실행 시간(초). GPU 학습 챕터 여유 있게.

from getpass import getpass
# contents:write 권한의 fine-grained PAT 권장. 입력값은 저장/출력되지 않습니다.
GH_TOKEN = getpass("GitHub PAT (fluentmin/neuqes-101, contents:write): ").strip()
"""

CODE_HELPERS = """\
# 3) 챕터 탐색 + 해시(멱등 판단)
import hashlib, datetime
from pathlib import Path
import nbformat

ROOT = Path(WORK)
EXEC = ROOT / "executed"; EXEC.mkdir(exist_ok=True)

def chapters():
    out = []
    for d in sorted(ROOT.glob("[0-9][0-9]_*")):
        nb = d / (d.name + ".ipynb")
        if nb.exists():
            out.append((d.name, nb))
    return out

def source_hash(nb_path):
    \"\"\"clean 노트북의 셀 소스(출력 제외) 해시 — 내용이 바뀌면 달라진다.\"\"\"
    nb = nbformat.read(nb_path, as_version=4)
    h = hashlib.sha256()
    for c in nb.cells:
        h.update(c.cell_type.encode()); h.update(b"\\0")
        h.update((c.source or "").encode()); h.update(b"\\0")
    return h.hexdigest()

def executed_hash(folder):
    p = EXEC / (folder + ".ipynb")
    if not p.exists():
        return None
    try:
        nb = nbformat.read(p, as_version=4)
        return nb.metadata.get("executed_from", {}).get("source_sha256")
    except Exception:
        return None

def is_stale(folder, nb_path):
    return source_hash(nb_path) != executed_hash(folder)

ALL = chapters()
print("발견한 챕터:", len(ALL))
"""

CODE_SELECT = """\
# 4) 대상 결정 + 현황표
def base_set(t):
    if t == "all":
        return ALL
    if t == "gpu":
        return [(f, p) for f, p in ALL if int(f[:2]) >= 7]
    if t == "stale":
        return ALL                      # staleness 필터는 아래에서
    if isinstance(t, list):
        keys = {str(x).zfill(2) if str(x).isdigit() else str(x) for x in t}
        return [(f, p) for f, p in ALL if f in keys or f[:2] in keys]
    return []

base = base_set(TARGET)
sel  = base if FORCE else [(f, p) for f, p in base if is_stale(f, p)]
sel_keys = {f for f, _ in sel}

print(f"{'':2}{'챕터':<28}{'executed':<10}상태")
for f, p in ALL:
    has = (EXEC / (f + '.ipynb')).exists()
    state = '최신' if (has and not is_stale(f, p)) else ('낡음' if has else '없음')
    mark = '▶' if f in sel_keys else ' '
    print(f"{mark} {f:<28}{'있음' if has else '-':<10}{state}")
print(f"\\n실행 대상: {len(sel)}개  (TARGET={TARGET!r}, FORCE={FORCE})")
"""

CODE_EXECUTE = """\
# 5) 실행 → executed/<폴더>.ipynb 저장 (실패해도 다음 챕터 계속)
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

manifest = []
for f, p in sel:
    print(f"\\n=== 실행: {f} ===", flush=True)
    nb = nbformat.read(p, as_version=4)
    client = NotebookClient(
        nb, timeout=PER_CELL_TIMEOUT, kernel_name="python3",
        resources={"metadata": {"path": str(p.parent)}},  # 챕터 폴더에서 실행
        allow_errors=False,
    )
    status = "ok"
    try:
        client.execute()
    except CellExecutionError as e:
        status = "error: " + str(e).splitlines()[-1][:120]
        print("  ⚠️", status)
    except Exception as e:  # 커널 타임아웃 등
        status = "fail: " + str(e)[:120]
        print("  ⚠️", status)
    nb.metadata["executed_from"] = {
        "source_sha256": source_hash(p),
        "executed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "runtime": "colab-t4",
        "status": status,
    }
    out = EXEC / (f + ".ipynb")
    nbformat.write(nb, out)
    manifest.append((f, status))
    print(f"  → 저장 executed/{f}.ipynb  [{status}]")

print("\\n=== 요약 ===")
for f, s in manifest:
    print(f"  {f}: {s}")
"""

CODE_PUSH = """\
# 6) executed/ 만 커밋·푸시 (성공한 것만 올리고 싶으면 manifest 보고 위에서 거른 뒤 재실행)
import subprocess

subprocess.run(["git", "config", "user.name",  "ChangMin Yoo"], cwd=WORK, check=True)
subprocess.run(["git", "config", "user.email", "fluentmin@gmail.com"], cwd=WORK, check=True)
subprocess.run(["git", "add", "executed/"], cwd=WORK, check=True)

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        cwd=WORK, capture_output=True, text=True).stdout.strip()
if not staged:
    print("커밋할 executed/ 변경이 없습니다.")
else:
    print("커밋 대상:\\n" + staged)
    names = ", ".join(f for f, _ in manifest) or "executed"
    msg = f"executed: Colab 실행본 갱신 ({names})"
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=WORK, check=True)
    push_url = f"https://{GH_TOKEN}@github.com/{REPO}.git"
    r = subprocess.run(["git", "push", "-q", push_url, f"HEAD:{BRANCH}"],
                       cwd=WORK, capture_output=True, text=True)
    print("push:", "✅ 성공" if r.returncode == 0 else "❌ 실패\\n" + r.stderr)
"""


def build() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            _md(MD_INTRO),
            _code(CODE_SETUP),
            _code(CODE_CONFIG),
            _code(CODE_HELPERS),
            _code(CODE_SELECT),
            _code(CODE_EXECUTE),
            _code(CODE_PUSH),
        ],
    }


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
