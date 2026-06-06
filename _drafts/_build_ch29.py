"""Build 29_benchmark_eval/29_benchmark_eval.ipynb — Phase 4, 평가 챕터.

Ch 28 (KoGPT2 SFT) 로 *생성형 모델* 을 만들었으니, *이제 어떻게 평가하는가?*
Ch 1-23 의 분류 모델은 accuracy/F1 하나로 끝났지만, 생성형 LLM 은 *task 마다*
평가 방식이 다릅니다. 이 챕터는 *학습 없이 추론·평가만* 합니다.

핵심 교육 메시지:
  1. 분류 평가 vs 생성 평가 대비
  2. 벤치마크 task format 3가지 (Multiple-choice / Generation+추출 / LLM-as-judge)
  3. few-shot prompting (in-context learning) — zero/few-shot 차이

핵심 셀 (§2): MC log-likelihood 를 lm-eval-harness 없이 *원리 그대로* 직접 구현.
  - 각 선택지 토큰의 평균 log-prob 를 model.logits 로 직접 계산 → argmax
  - KoBEST HellaSwag (4지선다) + BoolQ (2지선다) subset

모델: Qwen/Qwen2.5-0.5B-Instruct (작아서 T4 OK, 한·영 지원) 메인.
      작은 모델은 벤치마크에서 거의 random — 그 자체가 교훈.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "29_benchmark_eval"
OUT_NB = OUT_DIR / "29_benchmark_eval.ipynb"
OUT_README = OUT_DIR / "README.md"

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


# ----- 1. 제목 -----
md(r"""# Chapter 29. 분야별 벤치마크 평가 — 생성형 LLM 은 어떻게 평가하는가

**목표**: Phase 4 의 *평가* 챕터. Ch 28 까지 우리는 *학습* 만 했습니다 — 사전학습, continual pretraining, SFT. *그렇게 만든 생성형 모델을 이제 어떻게 측정할까요?* Ch 1-23 의 분류 모델은 `accuracy` / `F1` 하나로 끝났지만, **생성형 LLM 은 task 마다 평가 방식이 다릅니다**. 이 챕터는 *왜 LLM 평가가 특별한가* + *벤치마크 task format 의 다양성* + *MC log-likelihood 평가를 직접 구현해 원리 이해* + *`lm-evaluation-harness` 소개* 를 다룹니다.

**이 챕터는 학습 (`Trainer`) 이 없습니다 — 추론·평가만 합니다.** 그래서 *학습 축* 도 없습니다.

**환경**: Google Colab **T4 GPU 권장** (학습이 없어 CPU 에서도 느리지만 동작).

**예상 소요 시간**: 약 12-18분 (`lm-eval`·`datasets` 설치 약 2분 + Qwen2.5-0.5B 로드 약 1분 + KoBEST subset 로드 약 1분 + MC log-likelihood 평가 약 4-6분 + 생성 평가 약 2-3분 + zero/few-shot 비교 약 2분)

---

## 학습 흐름

1. 📊 **누적 추적표** (Ch 27/28 + **29 강조** + Ch 30 예고). *평가 챕터라 학습 축 없음*
2. 🔄 **변경점**: 이전 챕터들은 *학습*, 이 챕터는 *평가만*. 분류 평가 (accuracy) → 생성 평가 (task별)
3. 🎯 **분류 평가 vs 생성 평가 대비 표** + **벤치마크 task format 3가지 표** (이 챕터의 뼈대)
4. 🛠️ **환경 셋업** + §1 평가 대상 모델 로드 (Qwen2.5-0.5B-Instruct)
5. 🔬 **§2 Multiple-choice 평가 직접 구현** (핵심) — 각 선택지의 log-likelihood 계산 → argmax. *생성 안 함*
6. 🚀 **§3 Generation 평가** — few-shot prompt → 생성 → 정답 추출 (정규식)
7. 📈 **§4 zero-shot vs few-shot** — in-context learning 효과
8. 🧰 **§5 `lm-evaluation-harness` 소개** — 직접 구현과 *같은 원리, 표준화된 도구*
9. 🗺️ **§6 분야별 벤치마크 지도** — *측정 능력별* 분류
10. 🧭 **§7 해석** — 작은 모델의 한계 + scaling + 벤치마크 오염 주의
11. 🎯 **체크포인트** / ❓ **FAQ** (답변 포함) / 다음 챕터 예고

---

> 📒 **사전 학습 자료**: Ch 28 (SFT 모델 — 평가 대상), Ch 24-27 (GPT 사전학습·continual pretraining), Ch 23 (분류 평가 — accuracy/F1 의 마지막). 본 챕터는 *분류 평가 (라벨 비교) 에서 생성 평가 (task별 다른 방식) 로* 의 전환점입니다. **MC 벤치마크가 왜 생성을 안 하고 log-likelihood 만 보는지** 를 직접 코드로 확인하는 것이 핵심입니다.""")

# ----- 2. 누적 추적표 -----
md(r"""## 📊 누적 추적표

| Ch | 모델 | 단계 | 데이터 | 학습 축 | 평가 |
|---|---|---|---|---|---|
| 27 | KoGPT2 `skt/kogpt2-base-v2` (125M) | continual pretraining | 한국어 TinyStories 30K | next-token CE | perplexity / 생성 샘플 |
| 28 | KoGPT2 (125M, SFT) | SFT (instruction tuning) | KoAlpaca instruction-response | next-token CE (response-only) | instruction following (정성) |
| **29 ← 여기** | **Qwen2.5-0.5B-Instruct (+KoGPT2 SFT 대조)** | **(학습 없음 — 평가만)** | **KoBEST / 산술 subset** | **— (평가 챕터)** | **task format 별: MC log-likelihood / 생성+추출 / LLM-judge** |
| 30 (다음) | Ch 28 SFT 모델 + ref | alignment (DPO) | preference (chosen/rejected) | DPO loss (response-only) | preference 정렬 |

**평가 챕터라 학습 축이 없습니다.** Ch 24-28 이 *능력을 만드는* 단계였다면, 본 챕터는 *그 능력을 측정* 하는 단계입니다. Ch 30 부터는 다시 학습 (alignment) 으로 돌아갑니다.

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

---

## 🌏 GPT 시대 학습 4단계 — 본 챕터는 *측정* 단계

| 단계 | 용어 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|
| 1 | Pretraining | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | Continual pretraining | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | SFT (Instruction tuning) | Ch 28 | |
| — | **평가 (benchmark)** | **Ch 29 ← 여기** | ✅ (학습 아님) |
| 4 | Alignment (DPO / GRPO) | Ch 30·31 | |

> 능력을 *만든 뒤* (단계 1-3), *얼마나 잘하는지* 를 정량화하는 것이 벤치마크 평가입니다. *정렬 (단계 4)* 로 넘어가기 전, *지금 무엇을 할 수 있는지* 를 먼저 측정합니다.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 24-28)

| 축 | Ch 24-28 (학습 챕터들) | Ch 29 (본 챕터, 평가) |
|---|---|---|
| 무엇을 하는가 | *학습* (`Trainer` / `SFTTrainer`) | **평가만** (추론, `Trainer` 없음) |
| 측정 방식 | loss / perplexity | **task format 별 다른 평가** |
| 평가 대상 | 학습 중인 본체 | **이미 학습된 instruct 모델** |
| 출력 형태 | (학습은 출력 없음) | MC 는 *생성 안 함* (log-likelihood), 생성 task 는 토큰 시퀀스 |

분류 시대 (Ch 1-23) 의 평가는 단순했습니다 — 모델이 고정 클래스 중 하나를 고르면 정답 라벨과 비교해 `accuracy` / `F1` 를 냈습니다. **생성형 LLM 은 자유 토큰 시퀀스를 출력** 하므로, *무엇을 정답으로 보고 어떻게 비교할지* 가 task 마다 다릅니다. 그래서 *하나의 metric* 이 아니라 *여러 평가 방식의 혼합* 이 필요합니다.

> 핵심 전환: **"정답 라벨과 비교" (분류) → "task format 에 맞는 방식으로 측정" (생성)**. 이 챕터의 §4 표가 그 *3가지 format* 입니다.""")

# ----- 4. 핵심 대비 표 + task format 3가지 -----
md(r"""## 🎯 분류 평가 vs 생성 평가 — 무엇이 달라지나

|  | 분류 (Ch 1-23) | 생성 LLM (Ch 24-) |
|---|---|---|
| metric | `accuracy` / `F1` (정답 라벨 비교) | **task 마다 다름** |
| 출력 | 고정 클래스 (예: 5개 별점) | 자유 토큰 시퀀스 |
| 평가 | 한 가지 방식 | **여러 방식 혼합** |
| 예시 | "이 리뷰는 긍정/부정?" → 라벨 일치 | "이 수학 문제를 풀어라" → 생성 후 답 추출 |

분류 모델은 *출력 공간이 닫혀 있어* (클래스가 정해져 있어) 비교가 자명합니다. 생성 모델은 *출력 공간이 열려 있어* (어떤 토큰 시퀀스든 나올 수 있어) — 같은 정답도 표현이 다양하고, 어떤 task 는 *답이 맞았는지* 를 코드로 채점하기조차 어렵습니다.

---

## 🧱 벤치마크 task format 3가지 — 이 챕터의 뼈대

생성형 LLM 벤치마크는 *평가 방식* 에 따라 크게 3가지로 나뉩니다. 이 챕터는 앞 두 개를 직접 구현하고, 세 번째는 개념으로 다룹니다.

| format | 평가 방식 | 생성? | 벤치마크 예 | 본 챕터 |
|---|---|---|---|---|
| **① Multiple-choice** | 각 선택지의 *log-likelihood* 를 비교해 argmax | **생성 안 함** | MMLU, KMMLU, HellaSwag, ARC, KoBEST | **§2 직접 구현** |
| **② Generation + 정답 추출** | 생성한 뒤 정답을 파싱 / 실행해 채점 | 생성함 | GSM8K (수학), HumanEval (코드) | **§3 시연** |
| **③ LLM-as-judge** | 다른 (강한) LLM 이 답변의 품질을 채점 | 생성함 | LogicKor, MT-Bench | §6 개념 |

> **핵심 직관**: *객관식 (MC)* 은 모델에게 답을 *쓰게 하지 않습니다*. 대신 *각 선택지를 모델이 얼마나 "그럴듯하다" 고 보는가* (log-likelihood) 를 측정해 가장 높은 것을 고릅니다. 이게 §2 에서 코드로 확인할 *MC 평가의 본질* 입니다. MMLU·KMMLU·HellaSwag 같은 대표 벤치마크 대부분이 이 방식입니다.""")

# ----- 4-b. 왜 생성형은 정량 평가가 어려운가 - 예제로 -----
md(r"""## 🧩 왜 생성형은 정량 평가가 어려운가 — 쉬운 예제로

위 표가 *무엇이 다른가* 를 한눈에 보여 줬다면, 여기서는 *왜 어려운가* 를 **구체적인 예제** 로 손에 잡히게 풀어 봅니다. 핵심 한 문장은 이렇습니다 — **분류는 정답이 하나라 채점이 자명하지만, 생성은 정답이 무수히 많아 채점이 주관적입니다.**

### ① 분류 (Ch 1-23): 정답이 하나, 채점이 자명

> 입력: `"이 영화 최고예요"`  →  출력: **긍정 / 부정** 둘 중 하나

정답 라벨이 *딱 하나* (`긍정`) 입니다. 모델이 `긍정` 이라 하면 맞음, `부정` 이라 하면 틀림. `accuracy` 가 *맞췄나 틀렸나* 로 명확히 정의됩니다. 출력 공간이 *닫혀 있어* (클래스가 정해져 있어) 비교가 기계적입니다.

### ② 생성 (Ch 24-): 정답이 무수히 많음, 채점이 주관적

> 입력: `"건강한 식습관 3가지 알려줘"`  →  출력: **무한히 많은 타당한 답**

`"규칙적인 식사, 채소 섭취, 물 충분히"` 도 정답이고, `"아침 거르지 않기, 가공식품 줄이기, 천천히 먹기"` 도 정답입니다. *어떤 답이 더 좋은가?* 는 *주관적* 입니다. 정답 라벨 하나와 비교하는 방식이 *통째로 무너집니다*.

### 같은 질문, 여러 타당한 답 — 능력별로 보면

| 능력 | 정답이 하나인가? | 그래서 어떻게 채점? |
|---|---|---|
| **수학** (GSM8K) | 최종 *숫자* 는 하나지만 *풀이 과정* 은 다양 | 생성 후 **최종 숫자만 추출** 해 비교 (§3) |
| **번역·요약** | *여러 표현* 이 다 정답 (`"고양이가 잤다"` = `"고양이는 잠들었다"`) | BLEU·ROUGE 같은 *n-gram 매칭* — 그러나 표현이 다르면 정답도 점수가 깎이는 **한계** |
| **열린 질문** | *정답 자체가 없음* (식습관 조언, 에세이 ...) | LLM judge 또는 사람 평가 (③) |

### 부분 정답 문제 — 이진 채점이 안 됨

분류는 *맞음 / 틀림* 의 이진입니다. 생성은 그 사이가 넓습니다.

- *절반만 맞은 답*: "건강한 식습관 3가지" 를 물었는데 2가지만 맞게 답함 → 0점? 0.67점?
- *형식은 틀렸지만 내용은 맞은 답*: 정답 숫자 `24` 를 `"이십사"` 라고 답함 → 단순 문자열 비교로는 *틀림*
- *맞지만 군더더기가 많은 답*: 정답에 불필요한 설명이 섞임 → 어디까지 정답?

이런 *부분 정답* 은 분류의 라벨 비교로는 표현할 수 없습니다. 그래서 단순 매칭 대신 *task 마다 다른 채점 방식* 이 필요합니다.""")

md(r"""### 짧은 코드 — 왜 exact match 가 부족한가

같은 질문에 *형식만 다른 두 정답* 을 단순 string match (exact match) 로 채점하면, **둘 다 "틀림" 으로 나옵니다**. 내용은 맞는데도 말이죠. 아래에서 직접 시연합니다 (모델 없이 문자열만 비교 — 즉시 실행).""")

code(r"""# 같은 수학 문제의 "정답" 과, 형식만 다른 두 모델 답변
gold = "24"                       # 채점 기준 정답 (최종 숫자)
answer_a = "정답은 24입니다."        # 내용 O, 형식 다름 (설명이 붙음)
answer_b = "이십사"                # 내용 O, 형식 다름 (한글 표기)

# 방식 1) exact match - 문자열이 완전히 같아야 정답
def exact_match(pred, gold):
    return pred.strip() == gold.strip()

# 방식 2) 숫자 추출 후 비교 - 생성 평가가 실제로 쓰는 방식 (§3)
import re
def extract_int_match(pred, gold):
    nums = re.findall(r"-?\d+", pred)
    return bool(nums) and nums[0] == gold

print("question        : 6 곱하기 4는?")
print(f"gold answer     : {gold!r}\n")
for name, ans in [("answer_a", answer_a), ("answer_b", answer_b)]:
    em = exact_match(ans, gold)
    nm = extract_int_match(ans, gold)
    print(f"{name} = {ans!r}")
    print(f"   exact match        : {em}   (둘 다 내용은 맞는데 exact 는 False)")
    print(f"   extract-int match  : {nm}\n")

print("=> exact match 는 형식이 다르면 내용이 맞아도 '틀림'.")
print("   생성 평가는 이래서 task 마다 정교한 채점(숫자 추출/n-gram/LLM judge)이 필요합니다.")""")

md(r"""> 위에서 `answer_a` 는 *숫자 추출* 로는 맞다고 잡히지만, `answer_b` (`"이십사"`) 는 *숫자 추출조차* 실패합니다 — 내용은 정답인데도요. **이것이 생성형 정량 평가의 본질적 어려움** 입니다. 정답을 *인식하는 것 자체* 가 task 마다 다른 규칙을 요구합니다. §2 의 MC 평가가 *생성을 피해* log-likelihood 만 보는 것도, 이 형식 변동 문제를 우회하려는 설계입니다.""")

# ----- 4-c. 평가의 중요성·방대함 + 부록 예고 -----
md(r"""## 🌐 평가는 어쩌다 *하나의 거대한 분야* 가 되었나

여기서 한 걸음 물러나 큰 그림을 봅니다. **모델 능력이 올라갈수록 평가는 더 어렵고 더 중요해집니다.**

- 쉬운 문제는 최신 모델이 *다 풀어 버립니다*. 초창기 벤치마크 (예: 단순 분류·문법 판정) 는 이미 포화 상태입니다.
- 그래서 *어려운 능력* (전문 지식, 다단계 추론, 코드, 안전성) 을 구별하려면 *점점 더 정교한 벤치마크* 가 필요합니다. 평가가 모델을 *뒤쫓아* 함께 어려워집니다.

그 결과 평가는 *metric 하나* 가 아니라 **하나의 거대한 분야** 로 자랐습니다.

- 능력별 *수십 개* 벤치마크 (지식·추론·수학·코드·진실성·한국어 ...)
- 자동 *리더보드* 와 사람 투표 *arena*
- *LLM-as-judge* 로 열린 질문까지 채점
- *안전성·편향·환각* 평가 트랙

매주 새 모델과 새 벤치마크가 쏟아지는 지금, *"이 방대한 평가 생태계를 어떻게 항해하는가"* 자체가 실무 역량이 되었습니다.

> 📚 **상세 평가 부록 안내** — 이 방대한 생태계를 *실무자 관점* 에서 항해하는 지도를 별도 부록으로 두었습니다: [**`appendix_eval_landscape.ipynb`**](./appendix_eval_landscape.ipynb). *벤치마크 생태계 지도 · 벤치마크의 진화 · 리더보드 활용 · 평가 도구 비교 · LLM-as-judge · 평가의 함정 · 실무자를 위한 항해 전략* 을 다룹니다. 본 챕터(§2-§5)가 *평가의 원리* 라면, 부록은 *그 원리를 실무에서 어떻게 쓰는가* 입니다.""")

# ----- 5. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업

평가 챕터라 학습 라이브러리는 가볍게, 대신 표준 평가 도구 `lm-eval` (lm-evaluation-harness) 를 설치합니다. §2-§4 의 직접 구현은 `transformers` + `datasets` 만으로 동작하고, `lm-eval` 은 §5 (표준 도구 소개) 에서만 씁니다.

> `lm-eval` 은 의존성이 많아 설치에 1-2분 걸립니다. §5 를 건너뛰려면 `lm-eval` 설치 라인을 빼도 §1-§4 는 그대로 동작합니다.""")

code(r"""%pip install -q -U datasets transformers accelerate
# lm-eval 은 §5 (표준 도구 소개) 에서만 사용. 설치가 무거우면 이 줄만 주석 처리하세요.
%pip install -q lm-eval""")

code(r"""import re
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"device : cuda  ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device : cpu  (evaluation will be slow - Colab T4 recommended)")

# 추론에서도 fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
DTYPE = torch.float16 if USE_FP16 else torch.float32

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"torch  : {torch.__version__}")
print(f"fp16   : {USE_FP16}")""")

# ----- 6. §1 모델 로드 -----
md(r"""## 1. 평가 대상 모델 로드

**`Qwen/Qwen2.5-0.5B-Instruct`** — 약 0.5B (494M) 파라미터의 *작은 instruct 모델* 입니다. T4 에서 가볍게 돌고, 한국어·영어를 모두 지원해 한국어 벤치마크에서도 *random 보다 나은* 점수를 냅니다. 작은 모델이라 점수 자체는 낮지만, *평가 파이프라인을 끝까지 돌려보기* 에 적합합니다.

> 비교용으로 Ch 28 에서 만든 KoGPT2 SFT 모델을 함께 평가할 수도 있습니다. KoGPT2 (125M) 는 너무 작아 대부분의 벤치마크에서 *거의 random* 입니다 — **그 자체가 §7 의 교훈**: 작은 모델의 벤치마크 한계. 본 노트북은 Qwen 을 메인으로 진행합니다.""")

code(r"""MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"model     : {MODEL_NAME}")
print(f"params    : {n_params/1e6:.1f}M")
print(f"vocab     : {tokenizer.vocab_size}")
print(f"eos token : {tokenizer.eos_token!r}")""")

# ----- 7. §2 MC 평가 직접 구현 (핵심) -----
md(r"""## 2. Multiple-choice 평가 직접 구현 (핵심)

객관식 벤치마크 (MMLU, KMMLU, HellaSwag, KoBEST ...) 는 **모델에게 답을 생성하게 하지 않습니다**. 대신 각 선택지를 *문맥에 이어붙였을 때 모델이 얼마나 그럴듯하게 보는가* — 즉 **log-likelihood** 를 계산해 가장 높은 선택지를 정답으로 예측합니다.

### 왜 생성이 아니라 log-likelihood 인가
- 생성하면 *형식 변동* (모델이 "정답은 3번" vs "세 번째" vs 그냥 본문을 이어 씀) 때문에 채점이 불안정합니다.
- log-likelihood 는 *각 선택지에 대한 모델의 확신* 을 직접 수치로 비교 — 형식에 흔들리지 않고 *재현 가능* 합니다.

### 계산 원리
선택지 $c$ 의 토큰을 $(t_1, ..., t_k)$ 라 하면, 문맥 (prompt) 뒤에 이어질 확률의 로그는

$$\log P(c \mid \text{prompt}) = \sum_{i=1}^{k} \log P(t_i \mid \text{prompt}, t_{<i})$$

모델의 `logits` 에 `log_softmax` 를 씌워 *각 선택지 토큰 위치의 정답 토큰 log-prob* 를 뽑아 더하면 됩니다. **`lm-eval-harness` 도 내부에서 이 계산을 합니다** — 우리는 그 원리를 그대로 구현합니다.""")

md(r"""### log-prob 합 (sum) vs 평균 (mean) — 길이 정규화

선택지마다 토큰 수가 다르면 *단순 합* 은 *짧은 선택지* 에 유리합니다 (log-prob 는 음수라, 토큰이 적을수록 합이 덜 깎임). 그래서 **토큰 수로 나눈 평균 log-prob** (length-normalized) 를 쓰기도 합니다.

| 방식 | 식 | 성질 |
|---|---|---|
| sum (`acc`) | $\sum_i \log P(t_i)$ | 길이가 비슷한 선택지에 적합 |
| mean (`acc_norm`) | $\frac{1}{k}\sum_i \log P(t_i)$ | 길이 편향 완화 (HellaSwag 처럼 선택지 길이가 다를 때) |

`lm-eval-harness` 가 `acc` 와 `acc_norm` 두 점수를 함께 내는 이유가 이것입니다. 아래에서 둘 다 계산해 비교합니다.""")

code(r"""@torch.no_grad()
def continuation_logprob(prompt: str, continuation: str):
    '''prompt 뒤에 continuation 이 이어질 때, continuation 토큰들의 log-prob 를
    (sum, mean) 으로 반환. teacher forcing - 생성하지 않고 한 번의 forward 로 계산.'''
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids
    # continuation 이 차지하는 토큰 수 (경계는 tokenizer 가 정함)
    cont_len = max(1, full_ids.shape[1] - prompt_ids.shape[1])

    full_ids = full_ids.to(device)
    logits = model(full_ids).logits[0]                 # (T, V)
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # (T, V)

    # 위치 i 의 토큰은 위치 i-1 의 logits 가 예측 -> 한 칸 시프트
    target = full_ids[0, 1:]                            # (T-1,)
    pred_lp = log_probs[:-1]                            # (T-1, V)
    token_lp = pred_lp[torch.arange(target.shape[0]), target]  # (T-1,)

    cont_lp = token_lp[-cont_len:]                     # 마지막 cont_len 개 = continuation
    return cont_lp.sum().item(), cont_lp.mean().item()


def mc_predict(prompt: str, choices: list[str]):
    '''각 선택지의 (sum, mean) log-prob 를 구해 argmax. 두 방식의 예측을 모두 반환.'''
    sums, means = [], []
    for c in choices:
        s, m = continuation_logprob(prompt, c)
        sums.append(s)
        means.append(m)
    return int(np.argmax(sums)), int(np.argmax(means)), sums, means


# 동작 확인 - 정답이 명확한 간단한 한 문항 (4지선다)
demo_prompt = "1 더하기 1은 "
demo_choices = ["2입니다.", "3입니다.", "5입니다.", "10입니다."]
pred_sum, pred_mean, sums, means = mc_predict(demo_prompt, demo_choices)
demo_df = pd.DataFrame({
    "choice": demo_choices,
    "logprob_sum": [round(x, 2) for x in sums],
    "logprob_mean": [round(x, 3) for x in means],
})
print(demo_df.to_string(index=False))
print(f"\npredicted (sum)  : {demo_choices[pred_sum]}")
print(f"predicted (mean) : {demo_choices[pred_mean]}")""")

md(r"""### KoBEST HellaSwag — 4지선다 상식추론

**`skt/kobest_v1`** 의 `hellaswag` task 는 *문맥 (context)* 뒤에 가장 자연스러운 *다음 문장* 을 4개 후보 (`ending_1..4`) 중에서 고르는 한국어 상식추론 벤치마크입니다. 선택지 길이가 제각각이라 *길이 정규화 (mean)* 효과가 잘 드러납니다. T4 + 시간 제약상 **test split 의 앞 50문항** 만 평가합니다.""")

code(r"""from datasets import load_dataset

N_HELLASWAG = 50  # T4 30분 룰 - subset 만. 전체 500문항은 너무 오래 걸림
hellaswag = load_dataset("skt/kobest_v1", "hellaswag", split="test").select(range(N_HELLASWAG))
print(f"HellaSwag subset : {len(hellaswag)} 문항 (4지선다)")
print(f"columns          : {hellaswag.column_names}")

# 한 문항 예시
ex = hellaswag[0]
print("\n--- example ---")
print("context :", ex["context"][:60], "...")
for i in range(1, 5):
    print(f"ending_{i} :", ex[f'ending_{i}'][:40])
print("label   :", ex["label"])""")

code(r"""def eval_hellaswag(dataset):
    '''각 문항에서 context 뒤 4개 ending 의 log-prob 를 비교해 argmax.
    sum 방식 (acc) 과 mean 방식 (acc_norm) 정확도를 함께 반환.'''
    correct_sum = correct_mean = 0
    for ex in dataset:
        prompt = ex["context"] + " "
        choices = [ex[f"ending_{i}"] for i in range(1, 5)]
        pred_sum, pred_mean, _, _ = mc_predict(prompt, choices)
        correct_sum += int(pred_sum == ex["label"])
        correct_mean += int(pred_mean == ex["label"])
    n = len(dataset)
    return correct_sum / n, correct_mean / n


acc_sum, acc_mean = eval_hellaswag(hellaswag)
print(f"KoBEST HellaSwag  (n={len(hellaswag)})")
print(f"  acc      (sum  / log-prob)     : {acc_sum:.3f}")
print(f"  acc_norm (mean / length-norm)  : {acc_mean:.3f}")
print(f"  random baseline (1/4)          : 0.250")""")

md(r"""### KoBEST BoolQ — 2지선다 (예 / 아니오)

`boolq` task 는 *본문 (paragraph)* 과 *질문 (question)* 을 주고 **예 / 아니오** 를 묻습니다. 선택지가 2개라 *random baseline 은 0.5* — MC 평가를 *가장 단순한 형태* 로 보여줍니다. 여기서는 질문 뒤에 "예" / "아니오" 를 이어붙여 log-prob 를 비교합니다 (label 0 = 아니오, 1 = 예).""")

code(r"""N_BOOLQ = 50
boolq = load_dataset("skt/kobest_v1", "boolq", split="test").select(range(N_BOOLQ))
BOOLQ_CHOICES = ["아니오", "예"]  # 인덱스가 곧 label (0=아니오, 1=예)


def eval_boolq(dataset):
    correct = 0
    for ex in dataset:
        prompt = f"본문: {ex['paragraph']}\n질문: {ex['question']}\n답변: "
        _, pred_mean, _, _ = mc_predict(prompt, BOOLQ_CHOICES)
        correct += int(pred_mean == ex["label"])
    return correct / len(dataset)


acc_boolq = eval_boolq(boolq)
print(f"KoBEST BoolQ  (n={len(boolq)})")
print(f"  acc             : {acc_boolq:.3f}")
print(f"  random baseline : 0.500  (2지선다)")""")

md(r"""> **여기까지가 MC 평가의 전부입니다.** 생성은 한 번도 하지 않았습니다 — 오직 *각 선택지의 log-likelihood* 를 forward 한 번씩으로 계산해 argmax 했을 뿐입니다. 점수가 random 근처라면, 그건 *0.5B 라는 작은 모델의 한계* 입니다 (§7). 같은 코드를 더 큰 모델에 그대로 적용하면 점수가 오릅니다.""")

# ----- 8. §3 Generation 평가 -----
md(r"""## 3. Generation + 정답 추출 평가 (생성 기반)

두 번째 format 은 **모델이 실제로 답을 생성** 한 뒤, 그 텍스트에서 *정답을 파싱* 해 채점합니다. GSM8K (초등 수학), HumanEval (코드 실행) 가 대표적입니다. 여기서는 가벼운 **산술 문제 subset** 으로 *생성 → 정규식으로 숫자 추출 → 정답 비교* 흐름을 시연합니다.

### few-shot prompt
모델에게 *예시 몇 개* (few-shot) 를 먼저 보여줘 *답변 형식* 을 유도합니다. 예시 없이 (zero-shot) 던지면 모델이 형식을 못 맞춰 추출이 실패하기 쉽습니다 — §4 에서 그 차이를 정량으로 봅니다.""")

code(r"""# 가벼운 산술 subset (GSM8K 대신 - 빠르고 정답이 명확해 추출 평가에 적합)
ARITHMETIC = [
    ("6 곱하기 4는 얼마인가요?", 24),
    ("15 더하기 9는 얼마인가요?", 24),
    ("20 빼기 8은 얼마인가요?", 12),
    ("7 곱하기 7은 얼마인가요?", 49),
    ("100 빼기 37은 얼마인가요?", 63),
    ("13 더하기 28은 얼마인가요?", 41),
]

# few-shot 예시 (답변 형식을 보여줌)
FEWSHOT = (
    "Q: 사과 3개와 5개를 더하면 몇 개인가요?\nA: 정답은 8입니다.\n\n"
    "Q: 12에서 7을 빼면 얼마인가요?\nA: 정답은 5입니다.\n\n"
)


def extract_first_int(text: str):
    '''생성 텍스트에서 첫 정수를 추출 (정답 파싱). 없으면 None.'''
    m = re.findall(r"-?\d+", text)
    return int(m[0]) if m else None


@torch.no_grad()
def generate_answer(prompt: str, max_new_tokens: int = 24):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # greedy - 평가는 재현성 위해 deterministic
        pad_token_id=tokenizer.eos_token_id,
    )
    # 새로 생성된 토큰만 디코드
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def eval_generation(problems, shots: str):
    correct = 0
    rows = []
    for q, ans in problems:
        prompt = shots + f"Q: {q}\nA:"
        gen = generate_answer(prompt).strip()
        pred = extract_first_int(gen)
        ok = (pred == ans)
        correct += int(ok)
        rows.append({"question": q, "generated": gen[:30], "pred": pred, "answer": ans, "ok": ok})
    return correct / len(problems), pd.DataFrame(rows)


acc_gen, gen_df = eval_generation(ARITHMETIC, FEWSHOT)
print(gen_df.to_string(index=False))
print(f"\nfew-shot 산술 정확도 : {acc_gen:.3f}  (n={len(ARITHMETIC)})")""")

md(r"""> **생성 평가의 어려움**: 모델이 정답 숫자를 *맞게 말해도* 형식이 다르면 (`24` vs `이십사` vs 본문에 다른 숫자가 먼저 등장) 정규식 추출이 어긋날 수 있습니다. GSM8K·HumanEval 의 공식 평가가 *정교한 파싱 / 코드 실행* 을 쓰는 이유입니다. MC (§2) 가 *형식에 안 흔들리는* 것과 대조됩니다.""")

# ----- 9. §4 zero vs few-shot -----
md(r"""## 4. zero-shot vs few-shot — in-context learning

같은 산술 task 를 **예시 없이 (zero-shot)** 과 **예시 2개 (2-shot)** 으로 평가해 점수 차이를 봅니다. 모델 가중치는 *전혀 바뀌지 않는데* (학습이 아닙니다), 프롬프트에 예시를 넣는 것만으로 성능이 달라지는 현상이 **in-context learning** 입니다.

- **zero-shot**: 예시 없이 문제만. 모델이 *답변 형식* 을 스스로 정해야 함 → 추출 실패 잦음
- **few-shot**: 예시가 *형식 (정답은 N입니다)* 을 보여줘 모델이 그대로 따라 함 → 추출 성공률·정확도 상승""")

code(r"""# zero-shot (예시 없음) - 형식 유도가 없어 더 어려움
acc_zero, zero_df = eval_generation(ARITHMETIC, shots="")
# few-shot (위에서 정의한 FEWSHOT 2개)
acc_few = acc_gen  # §3 에서 이미 계산

compare = pd.DataFrame({
    "setting": ["zero-shot (0 examples)", "few-shot (2 examples)"],
    "accuracy": [round(acc_zero, 3), round(acc_few, 3)],
})
print(compare.to_string(index=False))
print(f"\nin-context learning 효과 : {acc_few - acc_zero:+.3f}  (few - zero)")
print("(작은 모델·작은 subset 이라 변동 큼 - 경향만 참고. 큰 모델일수록 효과 뚜렷)")""")

md(r"""> few-shot 이 점수를 올리는 핵심은 *지식을 새로 가르치는 게 아니라* **답변 형식을 정렬** 시키는 데 있습니다. MMLU 같은 벤치마크가 보통 *5-shot* 으로 보고되는 이유 — 모델이 *객관식 답 형식* 에 적응하도록.""")

# ----- 10. §5 lm-eval-harness -----
md(r"""## 5. `lm-evaluation-harness` 소개 — 표준 도구

§2-§4 에서 직접 구현한 것과 **정확히 같은 원리** 를, EleutherAI 의 **`lm-evaluation-harness`** (`lm-eval` 패키지) 가 *표준화된 방식* 으로 수행합니다. 수백 개 벤치마크 (MMLU, HellaSwag, GSM8K, KoBEST, KMMLU ...) 가 *task 정의로 내장* 되어, *프롬프트 포맷 · few-shot 예시 선택 · log-likelihood 계산 · 정규식 추출* 이 모두 통일됩니다. 논문·리더보드의 점수가 이 도구로 측정됩니다.

### 직접 구현 (§2) 과의 관계
| | 직접 구현 (§2) | `lm-eval-harness` |
|---|---|---|
| MC log-likelihood | `continuation_logprob` 손으로 | 내부에서 동일 계산 (`loglikelihood`) |
| 프롬프트 포맷 | 우리가 문자열 조립 | task yaml 에 정의 |
| few-shot 선택 | 우리가 고정 | seed 기반 자동 샘플링 |
| 결과 | `acc` 하나 | `acc` + `acc_norm` + stderr |

> *원리는 같고, 표준화·재현성이 다릅니다.* 직접 구현으로 *무슨 일이 일어나는지* 를 이해한 뒤, 실제 보고용 점수는 `lm-eval` 로 내는 것이 일반적인 흐름입니다.""")

md(r"""### `lm_eval.simple_evaluate` API (실행은 선택)

`lm-eval` 의 파이썬 API 핵심은 `simple_evaluate` 한 함수입니다. 아래는 *KoBEST BoolQ* 한 task 를 우리 Qwen 모델로 평가하는 코드입니다. **`lm-eval` 은 버전마다 task 이름·인자가 달라질 수 있어**, 무거우면 실행을 건너뛰고 *사용법* 만 익혀도 됩니다 (직접 구현 §2 가 메인). 셀 상단에서 설치 여부를 확인하고, 설치돼 있을 때만 실행합니다.""")

code(r"""# lm-eval 실행은 선택 - 미설치거나 무거우면 건너뜀 (직접 구현 §2 가 메인)
try:
    import lm_eval
    print(f"lm-eval version : {lm_eval.__version__}")
    HAS_LM_EVAL = True
except ImportError:
    print("lm-eval 미설치 - 이 셀은 건너뜁니다 (셋업 셀의 pip install lm-eval 참고).")
    print("§2-§4 의 직접 구현은 lm-eval 없이도 모두 동작합니다.")
    HAS_LM_EVAL = False""")

code(r"""# lm-eval 표준 도구로 한 task 평가 (설치돼 있을 때만)
# API 는 버전 변동이 큽니다 - 설치된 버전 기준으로 task 이름이 다르면 lm_eval.tasks 로 확인하세요.
if HAS_LM_EVAL:
    from lm_eval.models.huggingface import HFLM

    # 이미 로드한 model/tokenizer 를 그대로 lm-eval 에 래핑 (중복 로드 방지)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["kobest_boolq"],   # 버전에 따라 "kobest_boolq" / "kobest" 등 - 미존재 시 except 로
        num_fewshot=0,
        limit=50,                 # subset - T4 30분 룰
    )
    # 결과 표 출력
    for task, metrics in results["results"].items():
        print(f"[{task}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:14s}: {v:.3f}")
else:
    print("lm-eval 미설치 - §2 의 직접 구현 결과를 표준 점수로 참고하세요.")""")

md(r"""> 위 셀이 버전 문제로 실패하면 (`task 'kobest_boolq' not found` 등), `lm_eval.tasks.TaskManager().all_tasks` 로 *설치된 버전의 task 이름* 을 확인해 바꾸면 됩니다. **핵심은 점수 자체가 아니라, §2 의 직접 구현과 `lm-eval` 이 *같은 log-likelihood 원리* 위에 있다는 것** 입니다.""")

# ----- 11. §6 분야별 벤치마크 지도 -----
md(r"""## 6. 분야별 벤치마크 지도 — 무엇을 측정하나

벤치마크는 *측정하는 능력* 에 따라 분류됩니다. 하나의 점수가 아니라 *여러 능력* 을 *각기 다른 벤치마크* 로 재는 것이 현대 LLM 평가입니다. 아래는 영어·한국어 대표 벤치마크를 능력별로 정리한 지도입니다.

| 측정 능력 | 영어 벤치마크 | 한국어 벤치마크 | format |
|---|---|---|---|
| **지식** (전문 분야) | MMLU | KMMLU, HAERAE-Bench | ① MC |
| **상식추론** | HellaSwag, ARC | KoBEST (HellaSwag/COPA/BoolQ) | ① MC |
| **수학** | GSM8K, MATH | — (GSM8K 번역본) | ② 생성+추출 |
| **코드** | HumanEval, MBPP | — | ② 생성+실행 |
| **진실성** | TruthfulQA | — | ① MC (+ 생성) |
| **종합 대화/지시** | MT-Bench | LogicKor | ③ LLM-judge |

### format 별 다시 보기
- **① MC** (지식·상식·진실성): §2 에서 직접 구현한 *log-likelihood argmax*. 가장 흔하고 재현성 높음
- **② 생성+추출** (수학·코드): §3 의 *생성 후 파싱/실행*. 채점이 까다롭지만 *실제 풀이 능력* 측정
- **③ LLM-judge** (대화·지시): 다른 강한 LLM (예: GPT-4) 이 답변을 1-10 점으로 채점. *주관적 품질* 을 측정 — §7 의 한계 참고

> 한 모델을 *제대로 평가* 하려면 이 표 전체를 가로질러야 합니다 — *지식은 높은데 수학은 약한* 모델이 흔하기 때문입니다. **한 벤치마크 점수만으로 모델을 판단하면 안 되는** 이유입니다.""")

# ----- 12. §7 해석 -----
md(r"""## 7. 해석 — 작은 모델의 한계, scaling, 벤치마크 오염

### 작은 모델의 벤치마크 한계
§2 에서 Qwen2.5-0.5B 의 KoBEST 점수는 *random 을 살짝 웃도는* 수준이었습니다. Ch 28 의 KoGPT2 SFT (125M) 를 같은 코드로 평가하면 *거의 정확히 random* 입니다. **벤치마크는 어느 정도 규모 이상에서만 의미 있는 신호** 를 줍니다 — 작은 모델은 *지식·추론 용량 자체* 가 부족합니다.

### scaling 의 필요성
같은 `continuation_logprob` 코드를 7B / 70B 모델에 그대로 적용하면 점수가 크게 오릅니다. *코드가 아니라 모델 규모* 가 점수를 만듭니다. 이것이 LLM 에서 *scaling* 이 강조되는 이유 — 평가 방식은 그대로 두고 모델만 키워도 능력이 따라옵니다.

### 벤치마크 오염 (contamination) 과 한계
- **오염**: 벤치마크 문항이 *사전학습 데이터에 섞여* 들어가면, 모델이 *외워서* 맞히는 것일 수 있습니다 (실제 능력 과대평가). 새 모델일수록 의심해야 합니다.
- **단일 벤치마크의 위험**: MMLU 만 높은 모델이 실제 대화는 형편없을 수 있습니다. *여러 능력 (§6)* 을 가로질러 봐야 합니다.
- **format 편향**: MC 점수가 높다고 *실제 생성 품질* 이 좋은 건 아닙니다 — MC 는 *고르기만* 하면 되지 *쓰지* 는 않으니까요.

> 평가는 *모델을 만드는 것만큼 어렵습니다*. 좋은 평가 없이는 *무엇이 나아졌는지* 알 수 없고, 다음 단계 (alignment) 를 향한 방향도 잡을 수 없습니다.""")

# ----- 13. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. 분류 평가 (Ch 1-23) 와 생성 LLM 평가가 *근본적으로 다른* 이유는 무엇인가요? metric 이 하나가 아닌 까닭을 출력 공간 관점에서 설명해 보세요.
2. Multiple-choice 벤치마크는 왜 *생성하지 않고* log-likelihood 만 계산하나요? 생성 방식 대비 어떤 장점이 있나요?
3. log-prob 의 *합 (sum)* 과 *평균 (mean)* 중, 선택지 길이가 제각각인 HellaSwag 에서 어느 쪽이 길이 편향에 강한가요? 그 이유는?
4. few-shot 예시가 모델 가중치를 바꾸지 않는데도 점수를 올리는 현상의 이름은 무엇이며, 무엇을 "정렬" 시키기 때문인가요?
5. 한 모델을 *하나의 벤치마크 점수* 로만 판단하면 안 되는 이유를 §6·§7 을 근거로 두 가지 이상 들어 보세요.""")

# ----- 14. FAQ -----
md(r"""## ❓ FAQ

**Q1. MC 벤치마크는 왜 답을 생성하게 하지 않고 log-likelihood 를 보나요?**
생성하면 *형식 변동* 때문에 채점이 불안정합니다 — 모델이 "정답은 2번", "두 번째", 혹은 그냥 본문을 이어 쓸 수도 있습니다. log-likelihood 는 *각 선택지에 대한 모델의 확신* 을 직접 수치로 비교하므로 형식에 흔들리지 않고 *재현 가능* 합니다. 또한 forward 한 번이면 되어 *생성 (autoregressive decode) 보다 빠릅니다*. §2 의 `continuation_logprob` 이 그 계산의 전부입니다.

**Q2. log-prob 를 토큰 수로 나누는 (mean) 정규화는 왜 필요한가요?**
log-prob 는 음수라, *토큰이 많은 선택지일수록 합이 더 깎입니다*. 그러면 *짧은 선택지* 가 부당하게 유리해집니다. 토큰 수로 나눠 *토큰당 평균 log-prob* 를 보면 이 길이 편향이 완화됩니다. HellaSwag 처럼 선택지 길이가 크게 다른 task 에서 `acc_norm` (mean) 이 `acc` (sum) 보다 흔히 더 높게 나오는 이유입니다.
```python
score_sum  = cont_lp.sum()             # acc       - 길이 편향 있음
score_mean = cont_lp.sum() / cont_len  # acc_norm  - 길이 정규화
```

**Q3. few-shot 이 왜 점수를 올리나요? 학습도 안 하는데요.**
few-shot 은 *지식을 새로 가르치는 게 아니라* **답변 형식을 정렬** 시킵니다. 예시가 "정답은 N입니다" 같은 틀을 보여주면 모델이 그 틀을 따라 출력해, 정규식 추출 성공률과 정확도가 오릅니다. 가중치는 그대로지만 *프롬프트 안의 예시* 가 모델의 다음 토큰 분포를 그쪽으로 *조건화* 합니다 — 이것이 in-context learning 입니다. MMLU 가 보통 5-shot 으로 보고되는 이유이기도 합니다.

**Q4. LLM-as-judge (③) 의 장단점은 무엇인가요?**
- **장점**: *주관적 품질* (유창함, 도움됨, 안전성) 을 측정할 수 있습니다. MC·정규식으로는 잴 수 없는 *자유 생성* 의 품질을 사람 대신 강한 LLM 이 채점합니다 (MT-Bench, LogicKor).
- **단점**: judge 모델의 *편향* 을 물려받습니다 (긴 답변·자기 출력 선호, 위치 편향). 비용도 듭니다 (judge 호출). 그래서 *MC·생성 평가와 병행* 하고, judge 결과는 절대 점수보다 *상대 비교* 로 보는 것이 안전합니다.

**Q5. 벤치마크 오염 (contamination) 이 뭔가요? 왜 주의해야 하나요?**
벤치마크 문항이 *사전학습 데이터에 섞여* 들어가면, 모델이 *추론이 아니라 암기* 로 맞힐 수 있습니다 — 실제 능력보다 점수가 과대평가됩니다. 인터넷 크롤 데이터로 학습한 최신 모델일수록 의심해야 합니다. 그래서 *공개 직후의 새 벤치마크*, *비공개 테스트셋*, *오염 탐지 (n-gram 중복 검사)* 같은 장치가 쓰입니다. 한 모델의 *유난히 높은 한 벤치마크 점수* 는 오염을 의심할 신호일 수 있습니다.

**Q6. `lm-eval-harness` 를 쓰면 되는데 왜 직접 구현하나요?**
*무슨 일이 일어나는지* 를 이해하기 위해서입니다. `lm-eval` 의 점수가 *어떻게 나온 것인지* 모르면, 점수가 이상할 때 디버깅할 수 없습니다. §2 에서 본 것처럼 MC 평가의 본질은 *log-likelihood argmax* 한 줄이고, harness 는 그것을 *표준화·자동화* 한 도구일 뿐입니다. 원리를 이해한 뒤 보고용 점수는 harness 로 내는 것이 실무 흐름입니다.

**Q7. 작은 모델 (0.5B, KoGPT2 125M) 점수가 random 근처인데, 평가가 의미 있나요?**
점수 *자체* 는 의미가 약하지만 (작은 모델은 용량 부족), *평가 파이프라인이 올바른지* 확인하는 데는 충분합니다. 같은 코드를 큰 모델에 적용하면 점수가 오르는 것을 §7 에서 짚었습니다. 또 *작은 모델이 random 근처* 라는 사실 자체가 *벤치마크는 일정 규모 이상에서만 신호를 준다* 는 교훈입니다 — scaling 의 필요성을 보여줍니다.""")

# ----- 15. 다음 챕터 예고 -----
md(r"""## 다음 챕터 예고

**Chapter 30. DPO — 사람 선호로 정렬 (Alignment, 학습 단계 4)**

- 본 챕터 (벤치마크) 가 *능력 측정* 이었다면, Ch 30-31 의 **alignment** 는 *선호·안전성 정렬* 입니다. *얼마나 잘하는가* 에서 *얼마나 사람이 원하는 방식으로 하는가* 로.
- **DPO (Direct Preference Optimization)**: Ch 28 의 SFT 모델을 *preference 데이터 (chosen / rejected 쌍)* 로 정렬. 사람이 *더 선호하는* 답변의 확률을 올리고 덜 선호하는 답변의 확률을 내립니다.
- **`labels = -100` thread 가 DPO 에서도 이어집니다** — chosen / rejected *둘 다 response 부분만* log-prob 를 계산해 비교합니다. 본 챕터 §2 의 `continuation_logprob` 과 *정확히 같은 log-prob 계산* 이 DPO loss 의 핵심 재료입니다.

**Phase 4 GPT 시대 흐름 — 본 챕터의 위치**:

| 챕터 | 단계 | 하는 일 |
|---|---|---|
| Ch 24·26 | 1 (pretraining) | 본체를 만든다 |
| Ch 25·27 | 2 (continual pretraining) | 도메인에 적응 |
| Ch 28 | 3 (SFT) | 지시를 따르게 |
| **Ch 29 ← 여기** | **— (평가)** | **무엇을 할 수 있는지 측정** |
| Ch 30·31 | 4 (alignment) | 사람 선호로 정렬 |

> **변하는 축** (Ch 28 → Ch 29): *학습 → 평가*. 모델·loss 가 아니라 *측정 방식* 이 주제입니다. §2 의 MC log-likelihood 가 Ch 30 DPO 의 chosen/rejected log-prob 비교로 *재료 그대로* 이어집니다.""")


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


# ----- README.md 작성 -----
README = """# 29_benchmark_eval — 분야별 벤치마크 평가 (생성형 LLM 은 어떻게 평가하는가)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/29_benchmark_eval/29_benchmark_eval.ipynb)

## 한 줄 목표
Ch 28 까지 *학습* (사전학습 → continual pretraining → SFT) 으로 생성형 모델을 만들었습니다. 이 챕터는 *그 모델을 어떻게 측정하는가* 를 다룹니다. 분류 시대 (Ch 1-23) 는 `accuracy` / `F1` 하나로 끝났지만, **생성형 LLM 은 task 마다 평가 방식이 다릅니다**. *왜 LLM 평가가 특별한가* + *벤치마크 task format 3가지* + *MC log-likelihood 평가 직접 구현* + *`lm-evaluation-harness` 소개* 를 다룹니다. **학습 (`Trainer`) 없이 추론·평가만** 합니다.

## 이 챕터의 뼈대 — 벤치마크 task format 3가지

| format | 평가 방식 | 생성? | 벤치마크 예 | 본 챕터 |
|---|---|---|---|---|
| **① Multiple-choice** | 각 선택지의 *log-likelihood* 를 비교해 argmax | **생성 안 함** | MMLU, KMMLU, HellaSwag, KoBEST | §2 직접 구현 |
| **② Generation + 추출** | 생성 후 정답 파싱 / 실행 | 생성함 | GSM8K, HumanEval | §3 시연 |
| **③ LLM-as-judge** | 강한 LLM 이 답변 채점 | 생성함 | LogicKor, MT-Bench | §6 개념 |

> **핵심 직관**: 객관식 (MC) 은 모델에게 답을 *쓰게 하지 않습니다*. 각 선택지를 모델이 얼마나 그럴듯하게 보는가 (log-likelihood) 를 측정해 argmax 합니다. MMLU·KMMLU·HellaSwag 대부분이 이 방식입니다.

## 분류 평가 vs 생성 평가

|  | 분류 (Ch 1-23) | 생성 LLM (Ch 24-) |
|---|---|---|
| metric | accuracy / F1 (정답 라벨 비교) | task 마다 다름 |
| 출력 | 고정 클래스 | 자유 토큰 시퀀스 |
| 평가 | 한 가지 | 여러 방식 혼합 |

## 다루는 핵심 개념
- **MC log-likelihood 평가** (§2, 핵심) — `model.logits` → `log_softmax` → 선택지 토큰 log-prob 합/평균 → argmax. `lm-eval-harness` 없이 *원리 그대로* 구현. 생성하지 않음
- **log-prob 정규화** (sum=`acc` vs mean=`acc_norm`) — 선택지 길이 편향 완화. HellaSwag 처럼 길이가 다를 때 중요
- **Generation + 정답 추출** (§3) — few-shot prompt → greedy 생성 → 정규식 숫자 추출 → 채점. 생성 평가의 형식 변동 어려움
- **zero-shot vs few-shot** (§4) — in-context learning. 가중치 안 바꾸고 *답변 형식 정렬* 로 점수 상승
- **`lm-evaluation-harness`** (§5) — 표준 도구. `lm_eval.simple_evaluate` / `HFLM`. 직접 구현과 *같은 원리, 표준화*
- **분야별 벤치마크 지도** (§6) — 지식 (MMLU/KMMLU) / 상식 (HellaSwag/ARC/KoBEST) / 수학 (GSM8K) / 코드 (HumanEval) / 진실성 (TruthfulQA) / 대화 (LogicKor/MT-Bench)
- **작은 모델의 한계 · scaling · 벤치마크 오염** (§7) — 0.5B 는 random 근처, 규모가 점수를 만듦, contamination 주의

## 사용 모델 · 벤치마크
- 모델: **`Qwen/Qwen2.5-0.5B-Instruct`** (494M, 한·영 지원, T4 OK). 작은 모델이라 점수는 낮지만 *파이프라인을 끝까지 돌려보기* 에 적합 — 그 자체가 §7 의 교훈
- 벤치마크: **KoBEST** (`skt/kobest_v1`) 의 HellaSwag (4지선다 상식추론, 50문항) + BoolQ (2지선다, 50문항) subset + 가벼운 산술 subset (생성 평가용)

## Loss / 학습
**없음.** 이 챕터는 추론·평가만 합니다. `Trainer` / `SFTTrainer` 가 등장하지 않습니다.

## 라이브러리 주의 — `lm-eval` 버전
`lm-eval` (lm-evaluation-harness) 은 버전마다 *task 이름·인자* 가 달라집니다 (`kobest_boolq` vs `kobest` 등). §5 의 `simple_evaluate` 실행은 *선택* 으로 두고, 실패하면 `lm_eval.tasks.TaskManager().all_tasks` 로 설치된 버전의 task 이름을 확인하면 됩니다. **§2-§4 의 직접 구현은 `lm-eval` 없이도 모두 동작** 합니다 (메인 경로).

## 환경
Google Colab **T4 GPU 권장** (학습이 없어 CPU 에서도 동작하지만 느림). 약 12-18분 (`lm-eval` 설치 약 2분 + 모델·데이터 로드 약 2분 + MC 평가 약 4-6분 + 생성 평가 약 2-3분 + zero/few-shot 비교 약 2분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능.

## 변화 추적

| Ch | 모델 | 단계 | 데이터 | 평가 |
|---|---|---|---|---|
| 27 | KoGPT2 (125M) | continual pretraining | 한국어 TinyStories | perplexity / 생성 샘플 |
| 28 | KoGPT2 (125M, SFT) | SFT | KoAlpaca | instruction following (정성) |
| **29** | **Qwen2.5-0.5B-Instruct** | **(평가 — 학습 없음)** | **KoBEST / 산술 subset** | **MC log-likelihood / 생성+추출 / LLM-judge (task format 별)** |
| 30 (다음) | Ch 28 SFT + ref | alignment (DPO) | preference (chosen/rejected) | preference 정렬 |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[30_dpo](../30_dpo/) (예정) — DPO alignment. 본 챕터가 *능력 측정* 이라면 DPO 는 *사람 선호로 정렬*. §2 의 `continuation_logprob` (response log-prob 계산) 이 DPO 의 chosen/rejected log-prob 비교로 *재료 그대로* 이어집니다.
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}")
