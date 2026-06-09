# executed/ — 실행본 노트북 보관소

WikiDocs 변환 시 **코드의 실제 실행 결과**(표·로그·그림)를 싣기 위한 출력 원천입니다.

## 왜 필요한가

챕터 폴더의 `NN_slug/NN_slug.ipynb`는 **출력이 없는 clean 상태**입니다(Colab에서 학습자가
직접 실행하라고 비워둠). 그래서 그대로 마크다운으로 바꾸면 결과가 비어 보입니다.
특히 GPU 챕터(7–31)는 로컬에서 돌릴 수 없어, **실제 결과의 출처가 이 폴더뿐**입니다.

## 규약

- 파일명: `executed/<폴더명>.ipynb` (예: `executed/24_gpt_tinystories.ipynb`).
- 내용: 해당 챕터를 **Colab T4에서 끝까지 실행**한 뒤 출력이 포함된 노트북.
- 챕터 폴더에는 **clean 노트북만** 둡니다(README의 Colab 버튼 대상). 실행본은 여기로 분리.

## 만드는 법

1. README.md의 Colab 버튼으로 해당 챕터를 열어 **끝까지 실행**(T4, 30분 이내).
2. Colab 메뉴 **파일 > .ipynb 다운로드** (출력이 함께 저장됩니다).
3. 받은 파일을 `executed/<폴더명>.ipynb`로 저장하고 커밋.

CPU 챕터(1–6 등)는 로컬에서 한 번에 만들 수 있습니다:
```bash
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py 1 --execute --save-executed
# → executed/01_tfidf.ipynb 저장 + pages/01-tfidf-*.md 생성
```

## 사용

변환기가 `executed/<폴더명>.ipynb`가 있으면 **자동으로** 출력 원천으로 집어 씁니다:
```bash
python3 .claude/skills/notebook-to-wikidocs/scripts/build_wikidocs.py 24
# → executed/24_gpt_tinystories.ipynb 의 출력으로 pages/24-*.md 생성
```

## 진행 현황

| 챕터 | 실행본 | 비고 |
|---|---|---|
| 01_tfidf | ✅ | CPU, `--execute --save-executed` 로 생성 |
| 02–06 | — | CPU, 로컬 생성 가능 |
| 07–32 | — | 대부분 GPU → Colab 실행본 필요 |
