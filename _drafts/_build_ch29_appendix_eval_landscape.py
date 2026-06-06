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
8. 📑 **§8 실전 — 최신 LLM tech report 큐레이션** — EXAONE 4.0 / Gemma 3 / Qwen3 / GLM-4.5 / DeepSeek-R1 의 평가 보고 방식 비교
9. 🎯 체크포인트 / ❓ FAQ / 다음""")

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
md(r"""## 5. ⚖️ §5 LLM-as-judge 와 human evaluation — 열린 질문은 누가 채점하나

객관식(MC)·정규식으로는 *열린 질문* (에세이, 조언, 대화) 을 채점할 수 없습니다 — 정답이 하나가 아니니까요 (메인 챕터 §1 의 어려움). 그래서 *주관적 품질* 을 채점하는 두 방식이 있습니다: **LLM-as-judge** (강한 LLM 이 채점) 와 **human evaluation** (사람이 직접 채점). 둘은 *경쟁이자 보완* 관계입니다.

### LLM-as-judge — 개념

*강한 LLM (예: GPT-4o, Claude)* 이 사람 대신 답변 품질을 채점합니다. 채점 방식은 세 가지:

| 방식 | 설명 | 예 |
|---|---|---|
| **Pointwise** (단일 점수) | 답변 하나에 1-10점 | MT-Bench single-answer |
| **Pairwise** (A vs B) | 두 답을 비교해 승/패/무 | Chatbot Arena auto, AlpacaEval |
| **Reference-based** | *모범 답안* 을 참고해 채점 | rubric + gold answer |

### 장점과 단점 (충실 비교)

| 축 | LLM-as-judge |
|---|---|
| **장점** | *싸고 빠름* (사람 대비 수십-수백 배), *확장성* (수천 건 자동), *재현성* (같은 judge·temperature=0 이면 같은 결과), *세밀한 rubric* 적용 가능, 열린 질문 평가의 거의 유일한 *자동* 수단 |
| **단점 — bias** | *position bias* (먼저 보여준 답 선호), *length/verbosity bias* (긴 답 선호), *self-preference* (judge 가 자기 계열 모델 답 선호), *sycophancy* (자신감 있는 톤 선호) |
| **단점 — 근본 한계** | *judge ceiling* — judge 보다 *똑똑한 답* 은 제대로 평가 못함 (채점자가 학생보다 약하면 곤란). judge 모델의 *편향·오류 상속*. *adversarial 취약* (judge 를 속이는 답) |
| **비용** | judge API 호출 비용 (사람보다는 싸지만 0 은 아님) |

> **bias 완화법**: position bias → *A/B 순서를 바꿔 두 번 채점* 후 평균 (또는 둘 다 일관될 때만 인정). length bias → 프롬프트에 *"길이를 무시하라"* 명시 + length-controlled 승률. self-preference → *서로 다른 계열의 judge 여러 개* 사용.

> **실무 권장**: judge 결과는 *절대 점수* 보다 *상대 비교* (A vs B 승률) 로, *MC·생성 평가와 병행*, 그리고 *소량의 human eval 로 judge 를 calibration* (사람 판단과의 상관관계 측정).""")

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

md(r"""### 🧑 사람이 직접 평가 (human evaluation) — *진짜 gold standard 인가?*

LLM-as-judge 가 등장하기 전부터, 그리고 지금도 *최종 진실* 로 여겨지는 건 **사람이 직접 채점** 하는 방식입니다 — Likert 점수(1-5), A/B 선호, 순위 매기기. 형태도 둘로 나뉩니다:

- **대중 평가** (crowdsourcing) — 일반 사용자 다수의 투표. 예: **LMSYS Chatbot Arena** (익명 두 모델 대결에 사람이 투표 → Elo)
- **전문가 평가** — 도메인 전문가·훈련된 annotator 가 가이드라인대로 채점

그런데 *"human eval 이 정말 gold standard 인가"* 를 두고 **찬반이 팽팽** 합니다.

| 입장 | 주장 |
|---|---|
| **찬성 (human 이 진실)** | 결국 *사람이 쓸* 답이니 사람 판단이 궁극 기준. *뉘앙스·맥락·문화·안전성* 의 미묘함을 사람만 잡아냄. 모델의 *예상 못한 실패·새 능력* 을 발견. 자동 metric 이 못 보는 *실제 유용성* 을 측정 |
| **반대 (human 도 못 믿음)** | *비싸고 느림* → 확장 불가, 반복 측정 어려움. *평가자 간 불일치* (inter-annotator agreement 가 낮음) — 같은 답에 사람마다 다른 점수. *평가자 편향* (길이·자신감 있는 톤·친숙한 형식 선호 — LLM judge 와 똑같은 편향!), *피로·집중력 저하*, *전문성 부족* (어려운 답의 정확성을 일반 평가자가 판별 못함). *재현 불가* (같은 평가를 다시 못 함) |

**논쟁의 핵심 — 대중 평가 vs 전문가 평가**:
- *Chatbot Arena 같은 대중 투표가 gold standard?* 회의론: 대중은 *정확성보다 그럴듯함·유창함·길이* 를 선호하는 경향 (실제로 *더 길고 자신감 있는 오답* 이 짧은 정답을 이기기도). 즉 *인기 ≠ 정확성*.
- *그럼 전문가 평가가 답?* 일관성은 높지만 *비싸고 도메인 한정* 이라 확장이 어렵고, 전문가 *개인 편향* 도 존재.
- 결국 **human eval 도 완벽한 진실이 아니라 *또 하나의 노이즈 있는 측정*** 이라는 인식이 실무에 자리잡는 중입니다.

> **실무 합의 (현재)**: *어느 하나만 믿지 않는다.* 자동 벤치마크(능력) + LLM-judge(열린 질문 대량) + 소량 human eval(judge calibration·최종 sanity check) 을 **삼각측량(triangulation)** 합니다. LLM-judge 는 *human 과의 상관관계* 로 신뢰도를 검증하고, human eval 은 *명확한 가이드라인 + 다수 평가자 + agreement 측정* 으로 노이즈를 줄입니다. **"한 가지 평가로 모델을 판단하지 말라"** 가 §6·§7 로 이어지는 핵심 교훈입니다.""")

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

# ----- §8 실전 — 최신 LLM tech report 큐레이션 -----
md(r"""## 📑 실전 — 최신 LLM tech report 는 평가를 어떻게 보고하는가

지금까지는 *지도와 원리* 였습니다. 메인 챕터에서는 작은 모델 (Qwen2.5-0.5B) 로 평가 *방법론* 을 직접 돌려 봤지만, 작은 모델은 점수가 random 근처라 *실전 감각* 이 부족합니다. 그래서 이 섹션에서는 **실제 최신 SOTA 모델들의 tech report 가 평가를 어떻게 보고하는지** 를 발췌해 읽어 봅니다. *코드 실행이 아니라, 진짜 리포트를 읽는 법* 을 익히는 큐레이션입니다.

대상은 최근 공개된 다섯 개 모델 패밀리의 tech report 입니다 — **EXAONE 4.0** (LG AI Research), **Gemma 3** (Google DeepMind), **Qwen3** (Alibaba), **GLM-4.5** (Zhipu AI / Z.ai), **DeepSeek-R1** (DeepSeek). 각 리포트가 *어떤 능력을 강조하고, 어떤 벤치마크로, 어떻게 보고하는지* 를 §1-§7 의 틀로 다시 봅니다.

> **읽는 관점 네 가지** — 각 리포트를 (1) *어떤 능력군을 평가하나*, (2) *대표 벤치마크는*, (3) *보고 방식의 특징* (점수표 / Arena Elo / LLM-judge / 자체 eval, few-shot, base vs instruct), (4) *그 회사만의 강조점* 으로 읽습니다. 회사마다 *자기 모델이 강한 쪽을 부각* 하므로, 리포트는 *기술 문서이자 마케팅 문서* 라는 점을 기억하세요.""")

md(r"""### 모델별 평가 큐레이션 — 한눈 비교표

아래 표는 다섯 리포트의 *평가 방식* 을 §1 능력군 틀로 정리한 것입니다. 수치는 *리포트에 명시된 것만* 옮겼고, 확인되지 않은 값은 적지 않았습니다 (방법론·강조점 위주로 일반화).

| 모델 (출처) | 강조 능력군 | 대표 벤치마크 (리포트에 등장) | 보고 방식의 특징 | 그 회사만의 강조점 |
|---|---|---|---|---|
| **EXAONE 4.0**<br>(LG AI Research) | 지식·수학·코드·instruction·long-context·**agentic**·**한국어** | MMLU-Redux/-Pro, GPQA-Diamond, AIME 2025, LiveCodeBench v5/v6, IFEval, BFCL-v3, Tau-Bench, **KMMLU-Pro·KMMLU-Redux·Ko-LongBench** | reasoning / non-reasoning **모드별 표 분리**, 자체 한국어 벤치마크 다수 | **한국어** 전문지식 (KMMLU 계열) + agentic tool use |
| **Gemma 3**<br>(Google DeepMind) | 수학·코드·**다국어**·long-context·vision·**안전성** | MMLU-Pro, MATH, GPQA Diamond, LiveCodeBench, MBPP, Global MMLU-Lite, FACTS Grounding, MMMU | 자동 벤치마크 + **LMSYS Arena Elo** (27B-IT 1338) 동시 보고, 별도 **책임성/안전성 평가** 섹션 | **안전성·책임성** (암기·아동안전·CBRN) + 경량 다국어 |
| **Qwen3**<br>(Alibaba) | 지식·추론·수학·코드·**다국어(119개 언어)** | MMLU, MMLU-Pro, MMLU-Redux, BBH, SuperGPQA, GPQA, GSM8K, MATH, EvalPlus, MultiPL-E, MGSM, MMMLU, INCLUDE | **few-shot 설정 명시** (MMLU 5-shot, GSM8K 4-shot 등), base 모델 표 분리, thinking/non-thinking 모드 | **다국어** 폭 (29→119 언어) + 코드(MultiPL-E) |
| **GLM-4.5**<br>(Zhipu AI / Z.ai) | **agentic**·**추론**·**코드** (ARC) | MMLU-Pro, AIME 24, MATH-500, GPQA, LiveCodeBench, SWE-bench Verified, Terminal-Bench, TAU-Bench, BFCL V3, BrowseComp | agentic·coding 벤치마크 비중 큼, 경쟁 모델 대비 **순위(rank)** 강조 | **agent / tool-use / 코드** (SWE-bench, TAU-Bench) |
| **DeepSeek-R1**<br>(DeepSeek) | **추론**·수학·코드 | AIME 2024, MATH-500, GPQA Diamond, LiveCodeBench, Codeforces, MMLU(-Pro/-Redux), IFEval, AlpacaEval 2.0, Arena-Hard | **pass@1 + cons@64**(다수결), 열린질문은 **LLM-judge**(GPT-4-Turbo) length-controlled 승률 | **추론(reasoning)** — 순수 RL 로 추론 능력 유도""")

md(r"""### 리포트별 평가 철학 — 한 단락씩

각 tech report 의 *평가 철학* 을 출처와 함께 짚습니다. (수치는 리포트에 명시된 것만, 나머지는 *강조 방향* 수준으로 일반화했습니다.)

**EXAONE 4.0 — 한국어 + agentic 을 전면에** ([arxiv 2507.11407](https://arxiv.org/abs/2507.11407))
non-reasoning 모드와 reasoning 모드를 *별도 표로 분리* 해 보고하는 것이 특징입니다. 영어 표준 벤치마크 (MMLU-Redux/-Pro, GPQA-Diamond, AIME 2025, LiveCodeBench) 위에, **한국어 전문지식을 직접 출제한 KMMLU-Pro·KMMLU-Redux** 와 자체 long-context 벤치마크 Ko-LongBench 를 더해 한국어 실무 적용성을 강조합니다. agentic tool use (BFCL-v3, Tau-Bench) 도 핵심 평가 축입니다 — *한국어 + 도구 사용* 이 LG 의 차별점입니다.

**Gemma 3 — 능력 + 안전성을 함께** ([arxiv 2503.19786](https://arxiv.org/abs/2503.19786))
자동 벤치마크 (MMLU-Pro, MATH, GPQA Diamond, LiveCodeBench, Global MMLU-Lite, MMMU) 점수와 함께 **LMSYS Chatbot Arena Elo** (27B-IT 가 1338) 를 같이 보고해, *자동 점수* 와 *사람 선호* 를 둘 다 제시합니다 (§3 의 두 축을 한 리포트에서). 특히 **책임성/안전성 평가** 섹션이 두꺼워 — 암기(memorization) 감사, 아동 안전, CBRN(화생방핵) 위험 능력 평가 — *경량·다국어·안전* 을 전면에 둡니다.

**Qwen3 — 다국어 폭과 few-shot 투명성** ([arxiv 2505.09388](https://arxiv.org/abs/2505.09388))
지식·추론·수학·코드 전 영역을 다루되, **few-shot 설정을 벤치마크마다 명시** (MMLU 5-shot, GSM8K 4-shot, BBH 3-shot CoT 등) 해 *공정 비교* 를 돕습니다. 무엇보다 **다국어 폭** 이 강조점입니다 — 사전학습 언어를 29개에서 **119개 언어·방언** 으로 확장하고 MGSM·MMMLU·INCLUDE 같은 다국어 벤치마크로 검증합니다. base 모델 표를 별도로 두어 *base vs post-trained* 구분도 분명합니다.

**GLM-4.5 — Agentic·Reasoning·Coding(ARC) 에 올인** ([arxiv 2508.06471](https://arxiv.org/abs/2508.06471))
리포트 제목부터 *"Agentic, Reasoning, and Coding (ARC) Foundation Models"* 로, 평가도 그 세 축에 집중합니다. **SWE-bench Verified, Terminal-Bench, TAU-Bench, BFCL V3, BrowseComp** 같은 *agent/tool-use·실전 코딩* 벤치마크 비중이 크고, 경쟁 모델 대비 *전체 순위* 를 강조해 *적은 파라미터로 상위권* 임을 부각합니다 — 전형적인 *우리가 강한 축을 전면에* 전략입니다.

**DeepSeek-R1 — 추론(reasoning) 단일 축** ([arxiv 2501.12948](https://arxiv.org/abs/2501.12948))
순수 강화학습(RL) 으로 추론 능력을 유도했다는 주장답게, 평가가 **추론·수학·코드** 에 집중됩니다 (AIME 2024, MATH-500, GPQA Diamond, LiveCodeBench, Codeforces). 보고 방식이 특징적입니다 — **pass@1** 에 더해 **cons@64**(64회 샘플링 다수결) 를 함께 보고해 *추론의 안정성* 을 보여 주고, 열린 질문(AlpacaEval 2.0, Arena-Hard)은 **LLM-as-judge** (GPT-4-Turbo) 의 length-controlled 승률로 채점합니다 (§5 의 실제 사례).""")

md(r"""### 종합 인사이트 — 실무자가 가져갈 다섯 교훈

다섯 리포트를 가로질러 읽으면 *공통 패턴* 과 *실무 교훈* 이 보입니다.

1. **회사마다 강조 능력·벤치마크가 다릅니다** — DeepSeek 은 추론, Qwen 은 다국어·코드, EXAONE 은 한국어, Gemma 는 안전성·경량, GLM 은 agent/coding. 리포트는 *자기 모델이 강한 축을 전면에* 둡니다. 즉 **tech report 는 기술 문서이자 마케팅 문서** 입니다. 강조된 벤치마크만 보면 *그 회사 프레임* 에 갇힙니다.

2. **자체 벤치마크가 점점 늘어납니다** — EXAONE 의 KMMLU 계열·Ko-LongBench, Gemma 의 HiddenMath·FACTS Grounding 처럼 *직접 만든 평가셋* 이 늘고 있습니다. 공개 벤치마크의 *포화·오염* (§2, §6) 때문입니다. 자체 벤치마크는 신선하지만 *제3자 재현이 어렵다* 는 단점도 함께 봐야 합니다.

3. **base vs instruct, few-shot 설정, 프롬프트를 봐야 공정 비교입니다** — Qwen3 처럼 *few-shot 수를 명시* 한 리포트가 있는가 하면, 모드(thinking/non-thinking)·기준 모델이 제각각입니다. **같은 벤치마크라도 설정이 다르면 점수를 직접 비교하면 안 됩니다** (§6 의 형식 민감성).

4. **한국어 실무자는 영어 벤치마크만 보고 모델을 고르면 안 됩니다** — 대부분 리포트의 메인 표는 *영어 벤치마크* 입니다. EXAONE 만이 KMMLU 계열을 전면에 둡니다. 한국어 서비스라면 **KMMLU·HAERAE·LogicKor (또는 자체 한국어 셋)** 로 *반드시 따로 검증* 해야 합니다 (§1, §6 의 언어 편향).

5. **리포트 점수 ≠ 내 use-case 성능입니다** — 리포트의 SOTA 점수는 *그 벤치마크에서의* 성능일 뿐입니다. cherry-picking·자체 벤치마크·설정 차이를 걷어내고 나면, 결국 **§7 의 결론** 으로 돌아옵니다 — *공개 리포트로 후보를 좁히고, 내 task 평가셋으로 최종 판정하라.* 리포트는 *지도*, 내 평가셋이 *나침반* 입니다.

> **읽는 순서 팁** — 새 모델 리포트를 받으면 (1) *abstract 에서 강조 축* 을 먼저 보고, (2) *평가 표의 설정(few-shot·모드·기준 모델)* 을 확인하고, (3) *내 능력군(예: 한국어·코드)에 해당하는 행만* 발췌해, (4) *내 평가셋으로 교차 검증* 하세요. 표 전체를 외울 필요는 없습니다.""")

md(r"""### (선택) 가벼운 코드 — HF Hub 에서 이 모델들 메타 조회

리포트를 읽은 뒤, 실제 모델 카드·다운로드 수를 HF Hub API 로 들여다볼 수 있습니다. *어떤 모델이 실무에서 실제로 많이 쓰이나* 의 보조 신호입니다 (리포트 점수와는 또 다른 축). 네트워크가 없으면 건너뛰어도 됩니다 — 이 섹션의 핵심은 *리포트 읽는 법* 이지 다운로드 수가 아닙니다.""")

code(r"""# HF Hub 에서 다섯 모델 패밀리의 대표 모델 메타 조회 - 네트워크 없으면 건너뜀
# (리포트 점수와는 별개의 "실사용" 신호: 다운로드 수, like 수)
CANDIDATES = [
    "LGAI-EXAONE/EXAONE-4.0-32B",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-32B",
    "zai-org/GLM-4.5",
    "deepseek-ai/DeepSeek-R1",
]

try:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"{'model':40s} {'downloads':>12s} {'likes':>8s}")
    print("-" * 64)
    for repo_id in CANDIDATES:
        try:
            info = api.model_info(repo_id)
            dl = info.downloads if info.downloads is not None else 0
            likes = info.likes if info.likes is not None else 0
            print(f"{repo_id:40s} {dl:>12,d} {likes:>8,d}")
        except Exception as e:
            # repo 이름이 바뀌었거나 비공개일 수 있음 - 행 단위로만 실패 처리
            print(f"{repo_id:40s} {'(lookup failed: ' + type(e).__name__ + ')':>20s}")
    print()
    print("note: download count is a usage signal, NOT a quality score.")
    print("always cross-check report scores with your own use-case evalset (see strategy section).")
except ImportError:
    print("huggingface_hub not installed - skipping (this section is curation-first).")
    print("install: pip install huggingface_hub   then re-run to browse model metadata.")
except Exception as e:
    print(f"HF Hub lookup unavailable (offline?): {type(e).__name__}: {e}")
    print("that's fine - the point of this section is how to READ reports, not the numbers.")""")

# ----- 체크포인트 + FAQ -----
md(r"""## 🎯 체크포인트 질문

1. **포화와 Goodhart** — 잘 만든 벤치마크가 시간이 지나면 *변별력* 을 잃는 두 가지 이유 (saturation, Goodhart's law) 를 각각 설명해 보세요.
2. **자동 벤치마크 vs 사람 arena** — HF Open LLM Leaderboard 와 LMSYS Chatbot Arena 가 *서로 다른 능력* 을 잰다고 했습니다. 한 모델이 전자에선 높고 후자에선 낮을 수 있는 이유는?
3. **항해 전략의 핵심** — "공개 벤치마크 1위" 보다 "내 use-case 평가셋 점수" 를 우선해야 하는 이유를, §6 의 함정 중 두 가지를 근거로 들어 보세요.
4. **리포트 읽기** — 다섯 tech report (EXAONE 4.0 / Gemma 3 / Qwen3 / GLM-4.5 / DeepSeek-R1) 는 *각자 강조하는 능력군* 이 달랐습니다 (§8). 각 회사가 *어떤 축* 을 전면에 뒀는지 하나씩 떠올려 보고, 왜 *"리포트는 마케팅 문서이기도 하다"* 고 했는지 설명해 보세요.""")

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
```

**Q6. tech report 의 SOTA 점수를 그대로 믿어도 되나요?**
조심해야 합니다 (§8). 리포트 점수는 *세 가지 이유* 로 부풀거나 한쪽으로 치우칠 수 있습니다. (1) **cherry-picking** — 회사는 *자기 모델이 강한 벤치마크* 를 골라 전면에 둡니다 (DeepSeek 은 추론, EXAONE 은 한국어 식). (2) **자체 벤치마크** — 직접 만든 평가셋은 신선하지만 *제3자 재현이 어렵고* 자기 모델에 유리하게 설계됐을 수 있습니다. (3) **설정 차이** — 같은 벤치마크라도 few-shot 수·모드(thinking/non-thinking)·기준 모델이 다르면 *직접 비교가 무의미* 합니다. 그래서 리포트는 *후보 좁히기* 로만 쓰고, **내 능력군에 해당하는 행만 발췌해 내 평가셋으로 교차 검증** 하세요. 특히 한국어 서비스라면 영어 메인 표만 보지 말고 KMMLU·HAERAE·LogicKor 로 따로 확인해야 합니다.""")

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
