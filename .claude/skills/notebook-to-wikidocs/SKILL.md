---
name: notebook-to-wikidocs
description: 챕터 노트북(.ipynb)/tex을 WikiDocs용 장→절 마크다운으로 변환한다. 코드뿐 아니라 실제 실행 결과(표·로그·그림)까지 담는다. "위키독스 페이지 만들어줘", "ch07 wikidocs로 변환", "노트북을 md로" 같은 요청에 사용.
---

# notebook → WikiDocs 변환

챕터 노트북을 WikiDocs 연동용 `pages/NN-slug*.md`(장 1 + 절 여러 개)로 바꾼다.
**핵심 원칙: 코드를 실으면 그 코드의 실행 결과도 함께 싣는다. 가짜 출력을 지어내지 않는다.**

배경·설계 결정·미결 질문은 같은 폴더의 `DESIGN_NOTES.md` 참조. 새 고민이 생기면 거기에 누적한다.

## 언제 쓰나

- 특정 챕터(또는 여러 챕터)를 WikiDocs에 올릴 `.md`로 변환할 때.
- 노트북을 고친 뒤 WikiDocs 페이지를 다시 생성할 때.

## 변환 절차

챕터 지정·제목 해석은 변환기가 자동으로 한다(별도 `--num/--slug/--title` 불필요).
- 챕터 인자: 번호(`7`/`07`) 또는 폴더명(`07_bert_pipeline`), 여러 개 가능. 전체는 `--all`.
- 제목: `book/tools/notebook_to_tex.py`의 `CHAPTERS` → 노트북 첫 H1("Chapter N." 제거) → 슬러그 순.

### 1. 변환할 챕터 정하기
- 사용자가 챕터를 지정했으면 그대로 쓴다.
- **지정 안 했으면 전체(`--all`)를 돌리기 전에 반드시 사용자에게 확인**받는다.
  (32챕터 + 향후 추가까지 동적으로 변환되므로 의도치 않은 대량 실행을 막는다.)

### 2. 출력의 원천 — 실제 결과가 있으면 쓰고, 없으면 합성 골격
노트북은 보통 출력이 비어 있다(Colab용 clean). **실제 실행은 비용이 커서 필수가 아니다.**
변환기가 셀마다 다음 우선순위로 출력을 채운다.

1. **`--executed-notebook PATH`** (단일 챕터): 출력 포함 실행본 명시 → 실제 결과(`▶ 실행 결과`).
2. **`executed/<폴더>.ipynb`**: 있으면 자동 사용 → 실제 결과. (Colab에서 돌린 실행본을 둘 때만)
3. **`--execute`**: `nbclient`로 직접 실행(CPU 챕터에만). `--save-executed`면 `executed/`에 저장.
4. **합성(기본 폴백)**: 위가 없으면 `book/tools/notebook_to_tex.py`의 `synthetic_output_text()`를
   **그대로 재사용**해 `print(`가 있는 셀에 "출력은 이런 모양" 골격(값은 `...`)을 채운다
   → `▶ 출력 형태` 라벨. 함수 정의·import 등은 코드만.

즉 **기본값은 합성**이고, 실행본/`--execute`가 있을 때만 실제 결과로 승급한다. 실제 결과와
합성은 라벨로 구분한다(`▶ 실행 결과` vs `▶ 출력 형태`). 합성 로직은 tex와 단일 출처를 공유한다
(재구현 금지 — import해서 사용).

### 3. 변환 실행
```bash
# 일부 챕터 (CPU: 실행 + 실행본 저장)
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py 1 --execute --save-executed
# GPU 챕터 (executed/ 실행본을 자동 사용 — 미리 커밋돼 있어야 함)
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py 7 24
# 전체 (사용자 확인 후)
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py --all
```
- 출력: `pages/NN-slug.md`(개요) + `pages/NN-slug-{practice,anatomy,variation,wrapup}.md`(절).
- 그림: 실행 PNG를 `assets/NN-slug-outK.png`로 저장하고 `![](../assets/...)`로 참조.
- `TOC.md`: 해당 장 블록만 교체/추가하고 다른 장은 번호 순서를 지켜 보존한다.
- 챕터별 실패는 격리되어 배치 전체를 멈추지 않는다. 끝에 빈 출력/실패 챕터를 요약한다.

### 4. 품질 검토 (스킬이 자체 확인)
- [ ] 코드 셀 대부분에 `**실행 결과**` 블록이 붙었는가? `<!-- 실행 결과 없음 -->`이 남았다면
      그 셀이 출력이 없는 게 맞는지(예: 함수 정의) 확인. 학습 결과/표/그림 셀인데 비었으면
      2번으로 돌아가 실행 원천을 채운다.
- [ ] `assets/`에 PNG가 실제로 생성됐고 페이지의 상대경로(`../assets/...`)가 맞는가?
- [ ] 장→절 분할이 표준 구조(개요/실습/해부/변형/정리)와 맞는가?
- [ ] 첫 H1 제목이 제거됐는가(WikiDocs는 TOC.md 제목을 페이지 제목으로 씀)?
- [ ] `TOC.md`에서 다른 장이 안 깨졌는가?

### 5. 커밋
챕터 단위로 의미 있게 커밋. WikiDocs 출판 워크플로는 메모리의 `wikidocs-publishing` 참조.

## 주의

- `book/chapters/*.tex`는 인쇄책용이고 변환기가 건드리지 않는다(회귀 방지).
- tex는 그림을 큐레이션 대표 그림으로 치환하지만, WikiDocs md는 **실제 실행 그림**을 쓴다(정직·재현).
- 한 번에 한 챕터씩. 검증 안 된 챕터를 두고 다음으로 넘어가지 않는다(CLAUDE.md 워크플로).
