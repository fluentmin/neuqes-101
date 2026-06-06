"""Build 29_benchmark_eval/appendix_eval_landscape.ipynb.

Ch 29 부록 — 생성형 LLM 평가 항해 가이드 (실무자용).

메인 챕터 (Ch 29) 는 *평가의 원리* 를 다룹니다 — task format 3가지 (MC/생성+추출/judge),
MC log-likelihood 직접 구현, few-shot. 이 부록은 그 원리를 *실무에서 어떻게 쓰는가* 입니다.

컨셉: "매주 새 모델·새 벤치마크가 쏟아지는 상황에서 힘들어하는 실무자를 위한 평가 항해 가이드."
실무자가 "새 모델 나왔는데 어떻게 평가하지? 어떤 벤치마크 믿지? 새 벤치마크 어떻게 따라가지?"
의 답을 얻는, 정보·지도 중심의 마크다운 위주 부록. 무거운 모델 추론은 없음 (T4 부담 0).

셀 구조 (마크다운 비중 높게):
  1. 제목 + Colab 배지 + 부록 안내
  2. 한 줄 질문
  §1 벤치마크 생태계 지도 (능력별 분류 표)
  §2 벤치마크의 진화 (왜 새 벤치마크가 계속 나오나 - saturation, Goodhart)
  §3 리더보드 활용 (HF Open LLM / LMSYS Arena / Open Ko-LLM / AlpacaEval)
  §4 평가 도구 비교 (lm-eval-harness / lighteval / HELM / OpenCompass) + 가벼운 코드
  §5 LLM-as-judge (장단점 + judge 프롬프트 예시)
  §6 평가의 함정 (contamination / overfitting / 단일 맹신 / 언어 편향 / 형식 민감성)
  §7 실무자를 위한 항해 전략 (결론)
  체크포인트 + FAQ
  다음 (메인/Ch 30 복귀)

빌더 패턴은 메인 _build_ch29.py 와 동일 (cells / _cid / md / code / NOTEBOOK json dump).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "29_benchmark_eval"
OUT_NB = OUT_DIR / "appendix_eval_landscape.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({
        "cell_type": "markdown",
        "id": _cid(),
        "metadata": {},
        "source": text,
    })


def code(text: str):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# ----- 1. 제목 + Colab 배지 + 부록 안내 -----
md(r"""# Chapter 29 부록 — 생성형 LLM 평가 항해 가이드 (실무자용)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/29_benchmark_eval/appendix_eval_landscape.ipynb)

> **부록 한 줄 질문** — *"매주 새 모델·새 벤치마크가 쏟아지는데, 내 모델을 대체 어떻게 평가해야 하나?"*

메인 챕터 ([`29_benchmark_eval.ipynb`](./29_benchmark_eval.ipynb)) 는 *평가의 원리* 를 다뤘습니다 — task format 3가지 (MC / 생성+추출 / LLM-judge), MC log-likelihood 직접 구현, few-shot prompting. 원리를 손에 쥐었으니, 이제 현장의 진짜 고민이 남습니다.

> *"GPT·Claude·Gemini·Qwen·EXAONE ... 매주 새 모델이 나오고, MMLU·MMLU-Pro·GPQA·KMMLU·LogicKor ... 벤치마크도 끝없이 쏟아진다. 나는 무엇을 믿고, 무엇을 추적하고, 내 서비스 모델을 어떻게 평가해야 하나?"*

이 부록은 그 질문에 답하는 **실무자를 위한 항해 지도** 입니다. 무거운 모델 추론은 거의 없습니다 — *정보·지도 중심* 이라 GPU 없이도 끝까지 읽고 돌릴 수 있습니다 (가벼운 코드 몇 개만 선택 실행). *북마크해 두고 새 모델·새 벤치마크가 나올 때마다 펴 보는 레퍼런스* 로 쓰면 좋습니다.

**환경**: GPU 불필요 (마크다운 위주, 가벼운 코드뿐). 약 5-10분 (대부분 읽기).

---

## 부록 지도

1. 🗺️ **§1 벤치마크 생태계 지도** — 능력별 분류 (지식·추론·수학·코드·진실성·한국어·대화)
2. 🧬 **§2 벤치마크의 진화** — 왜 새 벤치마크가 끝없이 나오나 (포화·오염·Goodhart)
3. 🏆 **§3 리더보드 활용** — HF Open LLM / Chatbot Arena / Open Ko-LLM / AlpacaEval, 언제 믿나
4. 🧰 **§4 평가 도구 비교** — `lm-evaluation-harness` / `lighteval` / HELM / OpenCompass
5. ⚖️ **§5 LLM-as-judge** — 강한 LLM 으로 채점, 장단점과 judge 프롬프트
6. ⚠️ **§6 평가의 함정** — 오염·과적합·단일 맹신·언어 편향·형식 민감성
7. 🧭 **§7 실무자를 위한 항해 전략** — 이 부록의 결론
8. 🎯 체크포인트 / ❓ FAQ / 다음""")

# ----- 2. 한 줄 질문 -----
md(r"""## 🧭 시작하기 전에 — 이 부록이 답하는 세 가지 질문

실무에서 생성형 LLM 평가를 마주하면 결국 세 질문으로 수렴합니다.

| 질문 | 어디서 답하나 |
|---|---|
| **"새 모델이 나왔는데 어떻게 평가하지?"** | §4 (평가 도구) + §7 (항해 전략) |
| **"어떤 벤치마크·리더보드를 믿지?"** | §1 (생태계 지도) + §3 (리더보드) + §6 (함정) |
| **"새 벤치마크가 계속 나오는데 어떻게 따라가지?"** | §2 (진화) + §7 (대표 1-2개만 추적) |

핵심 결론을 먼저 한 줄로 던지면 — **공개 벤치마크는 *참고* 일 뿐, 진짜 평가는 *내 use-case 맞춤 평가셋* 입니다.** 이 부록은 그 결론에 이르는 지도입니다.""")

# ----- §1 벤치마크 생태계 지도 -----
md(r"""## 1. 🗺️ §1 벤치마크 생태계 지도 — 능력별 분류

현대 LLM 평가는 *하나의 점수* 가 아니라 *능력별로 나뉜 수십 개 벤치마크* 의 모음입니다. 모델은 *지식은 높은데 수학은 약하거나*, *영어는 잘하는데 한국어는 약한* 식으로 능력이 들쭉날쭉하기 때문입니다. 아래 지도는 *무슨 능력을 재는가* 를 축으로 대표 벤치마크를 정리한 것입니다.

| 측정 능력 | 대표 벤치마크 | 무엇을 재나 | format |
|---|---|---|---|
| **지식** (전문 분야) | MMLU · MMLU-Pro · KMMLU(한) | 57개 분야 객관식 지식 (역사·법·의학 ...) | ① MC |
| **추론** | GPQA · BBH · HellaSwag · KoBEST(한) | 다단계 논리·상식 추론 | ① MC |
| **수학** | GSM8K · MATH | 초등 산술 → 경시대회 수학 | ② 생성+추출 |
| **코드** | HumanEval · MBPP | 함수 작성 → 실제 실행해 통과 여부 | ② 생성+실행 |
| **진실성·안전성** | TruthfulQA | 흔한 오개념·거짓에 안 속는가 | ① MC (+생성) |
| **한국어 종합** | HAERAE · LogicKor · KoMT-Bench | 한국어 지식·추론·문화·작문 | ① MC / ③ judge |
| **멀티턴·대화** | MT-Bench · Chatbot Arena | 여러 턴 대화 품질, 사람 선호 | ③ judge / 사람 |

> **핵심**: 한 모델을 *제대로* 평가하려면 이 표를 *가로질러* 봐야 합니다. **MMLU 하나만 높은 모델이 실제 대화·코드·한국어는 형편없을 수 있습니다.** 능력별로 *대표 벤치마크 1-2개씩* 을 보는 것이 현실적 출발점입니다 (구체 전략은 §7).

### 영어 vs 한국어 — 번역본의 한계
MMLU 를 한국어로 *번역* 한 벤치마크도 있지만, 번역은 *어색한 한국어 · 한국 문화 맥락 누락* 문제가 있습니다. 그래서 **KMMLU · HAERAE · LogicKor 처럼 *원어(한국어)로 직접 만든* 벤치마크** 가 한국어 능력을 더 정확히 잽니다. 한국어 모델을 평가한다면 *영어 벤치마크 번역본* 보다 *한국어 원어 벤치마크* 를 우선하세요 (§6 의 언어 편향 참고).""")

# ----- §2 벤치마크의 진화 -----
md(r"""## 2. 🧬 §2 벤치마크의 진화 — 왜 새 벤치마크가 끝없이 나오나

"왜 MMLU 만으로 안 되고 MMLU-Pro·GPQA 가 또 나오지?" — 답은 **모델이 벤치마크를 *풀어 버리기(saturate)* 때문** 입니다.

### 시간순 흐름 — 더 어렵게, 또 더 어렵게

| 세대 | 벤치마크 | 왜 다음이 필요했나 |
|---|---|---|
| 1세대 | **MMLU** (2020) | 초기엔 변별력 충분 |
| ↓ | *포화·오염* | 최신 모델이 90%+ 달성, 학습 데이터에 유출 의심 |
| 2세대 | **MMLU-Pro** · **GPQA** | 더 어렵게 (10지선다, 대학원·박사급 문제), 검색해도 안 풀리게 |
| ↓ | *또 포화* | 강한 모델이 다시 따라잡음 |
| 3세대 | **Humanity's Last Exam** 등 | *전문가도 어려운* 극한 난이도로 변별력 확보 |

### 두 가지 동력 — saturation 과 Goodhart

- **포화 (saturation)**: 모델이 벤치마크 점수를 *천장(예: 95%)* 까지 올리면, 그 벤치마크는 *모델 간 변별력* 을 잃습니다. 다 100점이면 누가 더 나은지 모릅니다. → 더 어려운 벤치마크가 필요.
- **Goodhart's law**: *"측정값이 목표가 되면, 그것은 더 이상 좋은 측정값이 아니다."* 벤치마크 점수가 *마케팅 목표* 가 되면, 모델 개발이 *그 벤치마크에 과적합* 됩니다 (벤치마크 문제 스타일만 잘 푸는 모델). 그러면 점수는 올라도 *실제 능력* 은 그만큼 안 오릅니다. → 새 벤치마크로 갈아타야 신뢰 회복.

> **실무 함의**: *"이 벤치마크는 언제 만들어졌나, 포화됐나, 오염 가능성은?"* 을 항상 물으세요. **오래된·포화된 벤치마크의 높은 점수는 변별력이 약합니다.** 새 모델을 평가할 땐 *최신·미오염* 벤치마크를 우선 봐야 합니다 (§6).""")

# ----- §3 리더보드 활용 -----
md(r"""## 3. 🏆 §3 리더보드 활용 — 무엇을 측정하고 언제 믿나

벤치마크를 직접 돌리기 전, *남이 이미 돌려 놓은 점수* 를 모은 곳이 **리더보드** 입니다. 다만 리더보드마다 *측정 방식* 이 달라, *무엇을 재는지* 를 알고 봐야 합니다.

| 리더보드 | 측정 방식 | 강점 | 주의 |
|---|---|---|---|
| **HF Open LLM Leaderboard** | 자동 벤치마크 모음 (MMLU-Pro·GPQA·MATH·BBH ...) | 표준화·재현 가능, 같은 조건 비교 | *벤치마크 능력* 만 — 실제 대화 품질 아님 |
| **LMSYS Chatbot Arena** | *사람* 이 두 모델 답변을 블라인드 비교 투표 → Elo | *실제 사용 선호* 를 직접 측정 | 투표 편향 (길고 친절한 답 선호), 느림 |
| **Open Ko-LLM Leaderboard** | 한국어 벤치마크 모음 (Ko-MMLU·HAERAE 등) | *한국어* 능력 표준 비교 | 한국어판도 오염·포화 주의 |
| **AlpacaEval** | LLM judge 가 기준 모델 대비 *승률* 채점 | 빠르고 싸게 *지시 따르기* 측정 | judge·length bias (§5) |

### 자동 벤치마크 vs 사람 투표 — 둘은 다른 걸 잰다
- **자동 벤치마크** (HF Open LLM): *지식·추론 능력* 을 객관식으로 잼. 재현 가능하지만 *실제 대화가 좋은지* 는 모름.
- **사람 arena** (Chatbot Arena): *실제 사용자 선호* 를 잼. "이 답이 더 도움된다" 는 인간 판단. 능력 점수와 *순위가 다를 수 있음* — 능력은 높은데 말투가 별로면 arena 순위가 낮습니다.

> **리더보드 gaming 주의**: 일부 모델은 *리더보드 점수를 띄우려* 그 벤치마크 스타일에 맞춰 튜닝합니다 (Goodhart, §2). **단일 리더보드 1위 = 내 task 에서도 1위, 가 아닙니다.** 리더보드는 *후보 좁히기* 용으로 쓰고, 최종 선택은 *내 task 평가셋* 으로 (§7).""")

# ----- §4 평가 도구 비교 -----
md(r"""## 4. 🧰 §4 평가 도구 비교 — 언제 무엇을 쓰나

리더보드 점수를 *직접 재현* 하거나 *내 모델을 같은 방식으로 평가* 하려면 평가 도구가 필요합니다. 메인 챕터 §5 에서 본 `lm-evaluation-harness` 가 사실상 표준이지만, 용도별로 대안이 있습니다.

| 도구 | 만든 곳 | 강점 | 언제 |
|---|---|---|---|
| **`lm-evaluation-harness`** | EleutherAI | 수백 개 task 내장, *사실상 표준* (HF 리더보드도 사용) | 일반적인 자동 벤치마크 — *기본 선택* |
| **`lighteval`** | Hugging Face | HF 생태계 통합, 가볍고 빠름 | HF 모델·데이터 파이프라인 안에서 |
| **HELM** | Stanford | *다면 평가* (정확도+공정성+효율+편향 ...) | *책임성·다차원* 평가가 중요할 때 |
| **OpenCompass** | OpenComplab | 중국어·다국어 task 풍부 | 다국어(특히 중국어) 평가 |

> **추천 출발점**: 특별한 이유가 없으면 **`lm-evaluation-harness`** 로 시작하세요 — 가장 많은 벤치마크가 표준화돼 있고, 논문·리더보드 점수와 *직접 비교* 가 됩니다. 한국어는 KoBEST·KMMLU task 가 harness 에 들어 있습니다.""")

md(r"""### 가벼운 코드 — `lm-eval` 에 어떤 task 가 있나 (선택 실행)

설치돼 있으면 *사용 가능한 task 이름* 을 들여다봅니다. 무거우면 건너뛰어도 됩니다 (이 부록은 *지도* 가 핵심). `lm-eval` 은 버전마다 task 이름이 달라, 실제로 무엇이 있는지 *직접 확인* 하는 습관이 중요합니다.""")

code(r"""# lm-eval task 목록 들여다보기 - 미설치면 건너뜀 (이 부록은 정보 위주, 실행 선택)
try:
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    all_tasks = sorted(tm.all_tasks)
    print(f"lm-eval available tasks : {len(all_tasks)}")

    # 관심 키워드별로 몇 개씩만 미리보기 (한국어·대표 벤치마크 위주)
    for kw in ["mmlu", "hellaswag", "gsm8k", "kobest", "kmmlu", "truthfulqa", "humaneval"]:
        hits = [t for t in all_tasks if kw in t.lower()][:6]
        print(f"  [{kw:10s}] {hits}")
except ImportError:
    print("lm-eval not installed - skipping (this appendix is map-first, not run-first).")
    print("install: pip install lm-eval   then re-run this cell to browse tasks.")
except Exception as e:
    print(f"lm-eval present but task listing failed: {type(e).__name__}: {e}")
    print("version differences are common - check lm_eval docs for your version.")""")

# ----- §5 LLM-as-judge -----
md(r"""## 5. ⚖️ §5 LLM-as-judge — 강한 LLM 으로 채점

객관식(MC)·정규식으로는 *열린 질문* (에세이, 조언, 대화) 을 채점할 수 없습니다 — 정답이 하나가 아니니까요 (메인 챕터 §1 의 어려움). 그래서 등장한 것이 **LLM-as-judge** 입니다. *강한 LLM (예: GPT-4, Claude)* 이 사람 대신 답변 품질을 1-10점 또는 *A vs B 승부* 로 채점합니다. MT-Bench, AlpacaEval, 한국어 LogicKor 가 이 방식입니다.

### 장점과 단점

| | 내용 |
|---|---|
| **장점** | *주관적 품질* (유창함·도움됨·안전성) 을 잴 수 있음. 사람 평가보다 *싸고 빠름*. 열린 질문 평가의 거의 유일한 자동 수단 |
| **단점** | *position bias* (먼저 보여준 답 선호), *length bias* (긴 답 선호), *self-preference* (judge 가 자기 계열 모델 답 선호), *비용* (judge API 호출), judge 모델의 *편향 상속* |

> **실무 권장**: judge 결과는 *절대 점수* 보다 *상대 비교* (A vs B 승률) 로, 그리고 *MC·생성 평가와 병행* 해서 보세요. position bias 는 *순서를 바꿔 두 번 채점* 해 평균내면 줄어듭니다.""")

md(r"""### judge 프롬프트 예시 (실제 API 호출은 선택)

LLM judge 의 핵심은 *채점 기준을 명확히 준 프롬프트* 입니다. 아래는 *A vs B 승부* 판정 프롬프트의 뼈대입니다 (실제 API 호출은 비용·키 문제로 주석 처리 — 형식만 익히세요).""")

code(r'''# LLM-as-judge 프롬프트 뼈대 - 실제 호출은 주석 (형식 학습용)
JUDGE_PROMPT = """You are an impartial judge. Compare two AI answers to the same question.
Judge by: helpfulness, correctness, and clarity. Ignore answer length and order.

[Question]
{question}

[Answer A]
{answer_a}

[Answer B]
{answer_b}

Output ONLY one of: "A", "B", or "tie". Then a one-line reason.
"""

example = JUDGE_PROMPT.format(
    question="건강한 식습관 3가지를 알려줘.",
    answer_a="규칙적인 식사, 충분한 채소 섭취, 물을 자주 마시기입니다.",
    answer_b="음 글쎄요, 그냥 적당히 드세요.",
)
print(example)
print("=" * 60)
print("position bias 줄이기: A/B 순서를 바꿔 한 번 더 채점하고 결과를 평균합니다.")
print("실제 채점: 아래처럼 강한 judge 모델에 이 프롬프트를 보내면 됩니다 (키·비용 필요).")
print()
print("# from openai import OpenAI            # 또는 anthropic")
print("# client = OpenAI()")
print("# verdict = client.chat.completions.create(")
print("#     model='gpt-4o', messages=[{'role':'user','content': example}])")''')

# ----- §6 평가의 함정 -----
md(r"""## 6. ⚠️ §6 평가의 함정 — 실무에서 꼭 피해야 할 것

벤치마크 점수를 그대로 믿으면 위험합니다. 실무에서 자주 발목 잡는 다섯 가지 함정입니다.

| # | 함정 | 무엇이 문제 | 어떻게 대응 |
|---|---|---|---|
| 1 | **오염 (contamination)** | 벤치마크 문항이 *학습 데이터에 유출* → 암기로 맞힘 | 최신·비공개 테스트셋 선호, n-gram 중복 검사 |
| 2 | **벤치마크 과적합** | 점수만 띄우려 *그 벤치마크 스타일* 에 튜닝 (Goodhart) | 여러 벤치마크 + 내 task 로 교차 확인 |
| 3 | **단일 벤치마크 맹신** | MMLU 1위 = 만능, 이 아님 | 능력별로 *가로질러* 봄 (§1) |
| 4 | **언어·문화 편향** | 영어 벤치마크 *번역* 은 한국어 능력 과대/과소평가 | KMMLU 같은 *원어* 벤치마크 (§1) |
| 5 | **형식 민감성** | 프롬프트 한 줄·예시 순서에 점수 출렁 | 동일 포맷·동일 few-shot 으로 *공정 비교* |

### 가장 흔한 실수 — "점수는 높은데 실제론 별로"
이 세 함정의 조합이 *"벤치마크 점수는 높은데 실제 서비스에선 별로"* 의 정체입니다.

- 오염 (1) 으로 점수가 *부풀고*,
- 과적합 (2) 으로 *벤치마크 스타일만* 잘하고,
- 단일 맹신 (3) 으로 *약점을 못 보고* 골랐기 때문입니다.

> **방어선은 결국 §7 의 결론으로 이어집니다** — *내 use-case 평가셋* 으로 최종 검증하면, 위 다섯 함정 대부분이 한 번에 걸러집니다. 공개 벤치마크는 *후보 좁히기*, 내 평가셋이 *최종 판정* 입니다.""")

# ----- §7 항해 전략 (결론) -----
md(r"""## 7. 🧭 §7 실무자를 위한 항해 전략 — 이 부록의 결론

지도를 다 봤으니, *실제로 어떻게 움직일 것인가* 로 마무리합니다. 다섯 가지 원칙입니다.

### ① 내 use-case 맞춤 평가셋이 *가장 중요* 하다
공개 벤치마크는 *참고* 일 뿐입니다. **진짜 평가는 *내 서비스가 실제로 받는 입력* 으로 만든 평가셋** 입니다 — 우리 챗봇이 받는 질문 50-200개, 우리 도메인 문서 요약 샘플, 우리 고객이 쓰는 말투. 이 평가셋에서의 점수가 *공개 벤치마크 1위보다* 의미 있습니다. (§6 의 함정 대부분을 한 번에 방어)

### ② 새 *모델* 따라가기 — 채널을 좁혀라
- HF Hub *trending* (주간 인기 모델), Papers with Code, 주요 리더보드 *watch*
- 매주 다 보지 말고 *능력 도약이 보고된* 모델만 (예: 같은 크기에서 벤치마크가 크게 오른 경우)

### ③ 새 *벤치마크* 따라가기 — 대표 1-2개만
- §1 의 *능력별로* 대표 벤치마크 **1-2개씩만** 추적. 전부 좇으면 지칩니다.
- 기존 벤치마크가 *포화* 됐을 때만 (§2) 후속 벤치마크로 갈아타기

### ④ 최소 평가 세트 — 이렇게 시작하라
| 무엇 | 왜 |
|---|---|
| 능력별 대표 벤치마크 (지식·추론·수학·코드 1개씩) | *기본 능력* 빠른 스크리닝 |
| **내 task 평가셋** (50-200 샘플) | *실제 성능* 의 진짜 신호 |
| LLM judge (열린 질문 일부) | MC 로 못 재는 *생성 품질* |

### ⑤ 평가는 *한 번이 아니라 지속* 이다
모델·프롬프트·데이터를 바꿀 때마다 *회귀 평가(regression test)* 를 돌리세요. 한 번 좋았다고 영원히 좋은 게 아닙니다 — 작은 변경이 *조용히* 품질을 떨어뜨립니다. 평가셋을 *CI 처럼* 자동으로 돌리는 것이 성숙한 실무입니다.

> **한 줄 결론** — *공개 벤치마크로 후보를 좁히고, 내 평가셋으로 최종 판정하고, 그 평가를 지속하라.* 매주 새 모델·새 벤치마크가 쏟아져도, 이 원칙만 지키면 흔들리지 않습니다.

---""")

# ----- 체크포인트 + FAQ -----
md(r"""## 🎯 체크포인트 질문

1. **포화와 Goodhart** — 잘 만든 벤치마크가 시간이 지나면 *변별력* 을 잃는 두 가지 이유 (saturation, Goodhart's law) 를 각각 설명해 보세요.
2. **자동 벤치마크 vs 사람 arena** — HF Open LLM Leaderboard 와 LMSYS Chatbot Arena 가 *서로 다른 능력* 을 잰다고 했습니다. 한 모델이 전자에선 높고 후자에선 낮을 수 있는 이유는?
3. **항해 전략의 핵심** — "공개 벤치마크 1위" 보다 "내 use-case 평가셋 점수" 를 우선해야 하는 이유를, §6 의 함정 중 두 가지를 근거로 들어 보세요.""")

md(r"""## ❓ FAQ

**Q1. 영어 벤치마크를 한국어로 *번역* 해서 쓰면 안 되나요?**
부분적으로는 쓸 수 있지만 한계가 큽니다. 번역본은 (1) *어색한 한국어* 로 모델이 질문 자체를 오해할 수 있고, (2) *한국 문화·법·역사 맥락* 이 빠져 한국어 능력을 정확히 못 잽니다. 예를 들어 미국 법 문제를 번역해도 한국 법 지식은 안 측정됩니다. 그래서 **KMMLU · HAERAE · LogicKor 처럼 *원어(한국어)로 직접 출제* 한 벤치마크** 가 한국어 능력의 더 정확한 신호입니다. 번역본은 *없는 것보다 낫다* 정도로, 원어 벤치마크가 있으면 그쪽을 우선하세요.

**Q2. LLM-as-judge 점수를 믿어도 되나요?**
*상대 비교* 로는 꽤 쓸 만하지만, *절대 점수* 는 조심하세요. judge 는 *긴 답·자기 계열 모델·먼저 보여준 답* 을 선호하는 편향이 있습니다 (length / self-preference / position bias). 그래서 (1) A/B 순서를 바꿔 두 번 채점해 평균내고, (2) MC·생성 평가와 *병행* 하고, (3) 가끔 *사람 평가* 로 judge 자체를 검증하는 것이 안전합니다. judge 는 *값싼 1차 필터* 로 쓰고, 중요한 결정은 사람·내 평가셋으로 확정하세요.

**Q3. 벤치마크 점수는 높은데 실제 서비스에선 별로인 이유가 뭔가요?**
세 가지가 겹쳤을 가능성이 큽니다. (1) *오염* — 벤치마크 문항이 학습 데이터에 유출돼 점수가 부풀었거나, (2) *과적합* — 벤치마크 스타일에만 튜닝돼 일반 입력엔 약하거나 (Goodhart), (3) *단일 맹신* — 한 벤치마크만 보고 다른 약점을 못 봤거나. 핵심 해법은 §7 — *내 use-case 평가셋* 으로 최종 검증하면 이 셋이 한 번에 걸러집니다. 공개 점수는 후보 좁히기용일 뿐입니다.

**Q4. 새 벤치마크가 매주 나오는데, 현실적으로 어떻게 따라가나요?**
*전부 좇지 마세요.* §1 의 능력별로 **대표 벤치마크 1-2개씩만** 고정해 추적하고, 기존 것이 *포화*(최신 모델들이 천장 점수) 됐을 때만 후속으로 갈아타세요 (예: MMLU → MMLU-Pro/GPQA). 새 벤치마크는 HF Hub trending·Papers with Code·주요 리더보드 공지로 충분히 따라잡힙니다. *모든 벤치마크를 다 돌리는 것* 보다 *내 task 평가셋을 잘 유지* 하는 데 시간을 쓰는 편이 훨씬 효율적입니다.

**Q5. 작은 회사라 평가 인프라가 없는데, 어떻게 시작하나요?**
*가볍게 시작* 하면 됩니다. (1) **내 task 평가셋 50-200개** 를 손으로 만드세요 — 실제 사용자 질문·기대 답변 쌍. 이게 가장 가치 있습니다. (2) 자동 채점이 가능한 것 (분류·숫자·키워드 포함) 은 *간단한 스크립트* 로, 열린 질문은 *LLM judge* (GPT/Claude API) 로 채점. (3) 능력 스크리닝이 필요하면 `lm-evaluation-harness` 로 대표 벤치마크 몇 개만. 인프라보다 *내 평가셋을 꾸준히 모으고 회귀 평가로 돌리는 습관* 이 핵심입니다.
```python
# 가장 작은 시작 - 내 task 평가셋 한 줄씩 (질문, 기대 키워드)
my_evalset = [
    {"q": "환불 정책 알려줘", "must_include": ["7일", "영수증"]},
    {"q": "배송 얼마나 걸려?",  "must_include": ["영업일"]},
]
# 모델 답변에 must_include 키워드가 들어있는지로 1차 자동 채점 → 회귀 평가 CI 로
```""")

# ----- 다음 -----
md(r"""## 다음 — 메인 챕터 / Ch 30 으로

이 부록에서 *매주 쏟아지는 모델·벤치마크 속에서 어떻게 평가를 항해하는가* 를, 능력별 생태계 지도 → 벤치마크의 진화 → 리더보드·평가 도구 → LLM judge → 함정 → 실무 전략으로 정리했습니다. **핵심 한 줄 — 공개 벤치마크로 후보를 좁히고, 내 use-case 평가셋으로 최종 판정하고, 그 평가를 지속하라.**

- **메인 챕터로 돌아가기**: [`29_benchmark_eval.ipynb`](./29_benchmark_eval.ipynb) — 평가의 *원리* (MC log-likelihood 직접 구현, 생성+추출, few-shot).
- **다음 챕터 예고**: **Chapter 30 — DPO (alignment)**. 평가(Ch 29)로 *능력을 측정* 했으니, 이제 *alignment* (Ch 30-31) 로 *선호·안전성을 정렬* 합니다. *얼마나 잘하는가* 에서 *얼마나 사람이 원하는 방식으로 하는가* 로 — 좋은 평가가 있어야 정렬이 *나아졌는지* 도 측정할 수 있습니다.

> 부록의 메시지 한 줄 — *평가는 한 번의 점수가 아니라 지속되는 항해다.* 새 모델·새 벤치마크가 나올 때마다 이 지도를 펴 보세요. 무엇을 믿고 무엇을 추적할지의 기준은 변하지 않습니다 — *내 use-case 가 진짜 평가다.*""")


# ----- 노트북 저장 -----
NOTEBOOK = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT_NB.relative_to(REPO)}  ({len(cells)} cells)")
