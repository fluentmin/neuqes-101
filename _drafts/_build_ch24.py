"""Build 24_gpt_tinystories/24_gpt_tinystories.ipynb — Phase 4 첫 챕터.

GPT (decoder-only) from scratch + TinyStories 사전학습 + generation. Phase 1·2·3
(BERT, Ch 7-23) 의 *encoder + masked token 예측 + task head 부착 fine-tune* 패러다임에서
Phase 4 (Ch 24-30) 의 *decoder + next-token 예측 + LM head 그대로 + SFT* 패러다임으로
전환하는 출발점.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "24_gpt_tinystories"
OUT_NB = OUT_DIR / "24_gpt_tinystories.ipynb"
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
md(r"""# Chapter 24. GPT — TinyStories 로 작은 언어모델 사전학습

**목표**: Phase 4 의 첫 챕터. Ch 7-23 까지 다룬 **BERT (encoder, MLM, task head 부착 fine-tune)** 패러다임에서, 이번엔 **GPT (decoder-only, causal LM, LM head 그대로)** 패러다임으로 전환합니다. `GPT2LMHeadModel` 을 *random init* 으로 from scratch 띄우고, **TinyStories** subset 으로 next-token 예측 사전학습 → 같은 prompt 에 *학습 전 / 학습 후* generation 결과를 나란히 비교합니다. Ch 20·22 의 *사전·사후 [MASK] 비교* 와 같은 깊이로, *사전학습이 본체에 어떤 next-token 분포를 새겼는가* 를 직접 봅니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 25-30분 (데이터 로드 약 2분 + BPE 토크나이저 학습 약 3분 + 학습 전 generation 약 30초 + 모델 학습 약 18분 + 학습 후 generation + reference 비교 약 2분)

---

## 학습 흐름

1. 📊 **변화 추적표 + Phase 전환 도입부** — Encoder (BERT) → Decoder (GPT) 큰 그림 한 화면
2. 🔄 **변경점** — 모델 패밀리 (encoder → decoder), 학습 목표 (MLM → CausalLM), 토크나이저 (WordPiece → BPE 직접 학습)
3. 📐 **Loss** — `CrossEntropyLoss(next-token)`. MLM 의 *15% 자리* vs CausalLM 의 *거의 모든 자리* 차이
4. 🔤 **토크나이저 노트** — BPE 직접 학습 (Ch 19 의 WordPiece/WordLevel 과 비교)
5. 🚀 **실습**: TinyStories 30K stories → BPE vocab=2048 학습 → 작은 `GPT2LMHeadModel` (약 3M params) 학습
6. 🔬 **사전·사후 generation 비교** — 같은 prompt 3개, *학습 전 (random init) vs 학습 후* 나란히, 그리고 reference `gpt2` (124M, WebText) 도 함께
7. 🛠️ **변형**: `temperature / top_k / top_p` sampling 비교
8. 📦 **등장 라이브러리** / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)

---

> 📒 **사전 학습 자료**: Ch 20-23 (작은 BERT scratch MLM + 분류 fine-tune). Ch 24 는 *같은 from-scratch 사전학습* 흐름인데, 본체가 *encoder (BERT) → decoder (GPT)*, 학습 목표가 *MLM → CausalLM*, 산출물이 *fine-tune 체크포인트 → generation 모델* 로 바뀝니다.""")

# ----- 2. 변화추적표 + Phase 4 도입 -----
md(r"""## 📊 변화 추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 20 | 작은 BERT (영어, scratch) | `bert-base-uncased` (가져옴) | Wikitext-103 paragraphs | MLM head | `CrossEntropyLoss` (masked 15%) |
| 21 | Ch 20 + 분류 헤드 | (Ch 20 과 동일) | Yelp 이진 (다른 도메인) | `Linear(H, 2)` | `CrossEntropyLoss` |
| 22 | 작은 BERT (한국어, scratch) | `klue/bert-base` (가져옴) | 한국어 위키 paragraphs | MLM head | `CrossEntropyLoss` (masked 15%) |
| 23 | Ch 22 + 분류 헤드 | (Ch 22 와 동일) | NSMC 이진 (다른 도메인) | `Linear(H, 2)` | `CrossEntropyLoss` |
| **24 ← 여기** | **작은 GPT2 (직접, scratch)** | **BPE (직접 학습, vocab=2048)** | **TinyStories 30K stories** | **`Linear(H, V)` (LM head, weight tied)** | **`CrossEntropyLoss` (next-token, 거의 모든 자리)** |
| 25 (다음) | `gpt2` (124M, OpenAI WebText 사전학습) | BPE (GPT2 그대로) | TinyStories (Ch 24 와 동일) | `Linear(H, V)` (LM head 그대로) | `CrossEntropyLoss` (next-token) - **continual pretraining** |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

---

## Phase 전환 — Encoder (BERT) → Decoder (GPT) 패러다임

Ch 7-23 의 BERT 챕터들이 *encoder + masked token 예측 + task head 부착 fine-tune* 패러다임이라면, Phase 4 (Ch 24-30) 는 *decoder + next-token 예측 + LM head 그대로 + SFT(behavior alignment)* 패러다임입니다. 본 챕터가 그 출발점.

| 축 | Phase 1·2·3 (BERT, Ch 7-23) | **Phase 4 (GPT, Ch 24-30)** |
|---|---|---|
| 본체 | Encoder (양방향 attention) | **Decoder (causal / masked attention)** |
| 사전학습 task | MLM (가려진 토큰 예측) | **CausalLM (next-token 예측)** |
| 학습 신호 위치 | 선택된 약 15% 만 (`-100` 다수) | **거의 모든 토큰** (`-100` pad 만) |
| 출력 head | task 별 부착 (`Linear(H, K)`) | **LM head (`Linear(H, V)`) 그대로** |
| Downstream 적응 | head 교체 + 본체 fine-tune (*task 적응*) | **SFT (*behavior alignment*)** + alignment (DPO/GRPO) |
| "Fine-tune" 의미 | task 별 특화 | **prompt 만 바꿔도 다른 일** |

> 본 챕터는 그 *출발점* — 작은 GPT 를 처음부터 학습해 *next-token 예측이 어떻게 generation 으로 이어지는지* 를 직접 봅니다. Ch 25 (대규모 사전학습 `gpt2` 를 TinyStories 로 **continual pretraining**) / Ch 28 (SFT) / Ch 30-31 (DPO / GRPO) 가 같은 본체 위에 차곡차곡 쌓여 갑니다.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 23)

| 축 | Ch 23 (한국어 BERT 분류 fine-tune) | Ch 24 (GPT scratch + TinyStories) |
|---|---|---|
| **모델 패밀리** | Encoder (`BertForSequenceClassification`) | **Decoder (`GPT2LMHeadModel`)** ← *Phase 전환의 핵심* |
| 사전학습 task | MLM (Ch 22 산출물 본체 + 분류 head) | **CausalLM (next-token, from scratch)** |
| 토크나이저 | `klue/bert-base` WordPiece (가져옴, vocab 32K) | **BPE 직접 학습** (vocab 2,048) |
| 데이터 | NSMC 한국어 영화 리뷰 (이진 라벨) | **TinyStories 영어 short stories** (라벨 없음) |
| Output head | `Linear(H, 2)` (새로 부착) | **`Linear(H, V)` LM head** (모델이 내장 + weight tied) |
| Loss | `CrossEntropyLoss` (분류, K=2) | **`CrossEntropyLoss` (next-token, K=V=2048)** |
| 산출물 | 분류 정확도 | **generation 텍스트** (`model.generate()`) |

> **변경점이 한꺼번에 많은 이유** — Phase 가 바뀌는 *전환 챕터* 라 *축 자체* 가 새로 정의됩니다. Ch 25 부터는 다시 *한 가지 축* 만 바뀝니다 (Ch 25: 본체 출발점 = scratch → 사전학습 모델, 같은 *continual pretraining* task / Ch 26: 언어 (한국어 scratch) / Ch 27: 한국어 continual pretraining / Ch 28: 학습 단계 = pretraining → SFT, `labels[prompt] = -100`).""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss — `CrossEntropyLoss` (next-token)

수식은 MLM 의 CE 와 *완전히 같음*. 다만 *어느 자리에서 loss 가 계산되는가* 가 다릅니다.

### 수식

입력 토큰 시퀀스 $x = (x_1, \dots, x_n)$ 에 대해, 각 위치 $i$ 에서 *그 다음 토큰* $x_{i+1}$ 을 예측:

$$L_{\text{CLM}} = -\frac{1}{n-1} \sum_{i=1}^{n-1} \log P(x_{i+1} \mid x_1, \dots, x_i)$$

- $P(x_{i+1} \mid x_{\leq i})$: 모델이 *지금까지 본 토큰만으로* 다음 토큰을 예측할 확률 (vocab 2,048 차원 softmax)
- 평균 분모 $n-1$: pad 가 아닌 *거의 모든* 자리에서 loss 계산 (MLM 의 15% 와 대비)

### 숫자로 감 잡기 (vocab=2048)

| 모델 상태 | 정답 토큰 확률 | $-\log p$ |
|---|---|---|
| 균등 추측 (random init 직후) | $1/2048 \approx 4.88 \times 10^{-4}$ | **7.62** ← random baseline |
| 약하게 학습 (정답 확률 0.02) | $0.02$ | 3.91 |
| 잘 학습된 작은 GPT (정답 확률 0.05-0.15) | $0.05$ - $0.15$ | **1.9 - 3.0** ← 이번 챕터 목표 영역 |
| 큰 사전학습 GPT (정답 확률 0.3+) | $0.3$ | 1.20 |
| 완벽 (정답 확률 1.0) | $1.0$ | 0.00 |

**관전 포인트**:
- 학습 첫 step loss 가 약 7.6 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab + 모델 정상.
- 목표는 *vocab 후보를 좁히는* 단계 (약 2-3). TinyStories 의 단순한 어휘·문법 덕분에 3M 짜리 작은 모델로도 도달 가능.

### Perplexity (PPL)

$\text{PPL} = e^{L}$ — *다음 토큰을 평균 몇 후보 중에서 고민하는가*:

| CLM loss | PPL | 해석 |
|---|---|---|
| 7.62 | 2,048 | 균등 (전체 vocab) |
| 4.0 | 55 | 약 50 개 후보 |
| 2.5 | 12 | 약 12 개 후보 ← 본 챕터 목표 |
| 1.0 | 2.7 | 거의 결정적 |

> MLM 의 `ln(30522) ≈ 10.33` random baseline 과 같은 직관. *vocab 차원* 만 작아진 것 (2,048).""")

# ----- 5. labels=-100 thread 환기 -----
md(r"""## 💡 `labels = -100` thread 환기 — MLM 의 *15% 만* vs CausalLM 의 *거의 모든 자리*

Ch 20·22 의 MLM 에서 봤던 `labels = -100` ignore_index 트릭이 GPT CausalLM 사전학습에서도 등장하지만, **적용 자리가 정반대** 입니다.

| 단계 | 챕터 | `labels` 구성 | loss 계산 자리 |
|---|---|---|---|
| MLM 사전학습 | Ch 20 (영어), Ch 22 (한국어) | 선택된 약 15% 만 원본 token id, 나머지 = `-100` | *가려진 자리만* |
| **GPT CausalLM 사전학습** | **Ch 24 (영어, 본 챕터), Ch 26 (한국어)** | **`input_ids.clone()` - pad 만 `-100`** | **거의 *전 자리*** |
| SFT / Instruction Tuning | Ch 28 (한국어 KoGPT2 SFT) | **prompt 부분 = `-100`**, *답변 토큰만* 원본 id | *답변 부분만* |

> 같은 `-100` 트릭, *적용 자리만 정반대*. MLM 은 *대부분을 가리고 일부만 학습*, GPT 사전학습은 *거의 가리지 않음*, SFT 는 *prompt 만 가림*. 한 step 에 학습되는 토큰 수만 봐도 *GPT 사전학습은 MLM 대비 약 5-6배 효율* (15% vs 거의 100%).

본 챕터에서는 `DataCollatorForLanguageModeling(mlm=False)` 이 자동으로 `labels = input_ids.clone()` 을 만들어 줍니다 — 뒤에 collator 출력 셀에서 직접 확인하겠습니다. Ch 28 의 *왜 모델이 instruction 을 따라가게 되는가* 는 *한 줄 코드 `labels[prompt_mask] = -100`* 로 정확히 설명되는데, 그 코드를 이해할 토대가 *이 챕터의 collator 출력* 입니다.""")

# ----- 6. 파인튜닝 의미 변화 thread -----
md(r"""## ⚠️ GPT 시대의 학습은 *네 단계* — 용어가 BERT 와 다릅니다

Ch 21·23 에서 본 *fine-tune* 은 **BERT 시대 의미** — *사전학습된 본체 + 새 task-specific head (`Linear(H, K)`)*. 분류·회귀·QA 마다 다른 head, *task 별 특화*. 한 모델 = 한 task.

Phase 4 GPT 시대는 *fine-tune* 한 단어가 *여러 의미* 로 섞여 쓰입니다. 학술적으로는 **네 단계** 로 분리됩니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 | 본 커리큘럼 |
|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | 일반 코퍼스 위에 random init 본체부터 학습 | 모든 토큰 (`labels = input_ids`) | **Ch 24** (영어 scratch, TinyStories), **Ch 26** (한국어 scratch) |
| 2 | **Continual pretraining** (계속 사전학습 / continual learning) | *사전학습된 본체* 를 *새 데이터* 로 *같은 CausalLM task* 더 학습. **head 그대로, task 그대로, 데이터만 새로** | 모든 토큰 (pretraining 과 동일) | **Ch 25** (`gpt2` + TinyStories) |
| 3 | **SFT** (Supervised Fine-Tuning / Instruction tuning) | instruction-response 쌍으로 *행동 정렬*. `labels[prompt] = -100` 으로 답변 부분만 학습 | **답변 토큰만** | **Ch 28** (KoGPT2 + KoAlpaca) |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference 또는 verifier reward 로 *선호 정렬* | preference log-likelihood ratio / RL advantage | **Ch 30** (DPO), **Ch 31** (GRPO) |

**세 가지 공통점** (모두 GPT 시대):
- **모델 클래스 그대로** — `AutoModelForCausalLM` (BERT 처럼 task head 부착 안 함)
- **출력 형식 그대로** — 토큰 시퀀스
- **학습 신호 종류 그대로** — next-token CE (alignment 만 예외)

**다른 점은 *데이터 형식* 과 *어느 토큰에 학습 신호를 주는가*** — `labels = -100` 자리만 변함.

| 단계 | 데이터 | `labels = -100` 자리 |
|---|---|---|
| Pretraining (Ch 24·26) | 일반 텍스트 | pad 만 |
| **Continual pretraining (Ch 25)** | **새 도메인 텍스트** | **pad 만 (Pretraining 과 동일)** |
| SFT (Ch 28) | instruction + response | **prompt 부분** |
| Alignment (Ch 30·31) | preference 쌍 / verifier reward | (RL 내부) |

> **BERT 의 "fine-tune" 은 *task 적응* 한 가지였지만, GPT 의 "fine-tune" 은 *continual pretraining / SFT / alignment* 셋이 섞인 통칭**. 정확히 말하려면 단계별 용어를 구분합니다. 본 챕터는 단계 1 (사전학습) 그 자체. Ch 25 에서 단계 2 (continual pretraining) 가, Ch 28 에서 단계 3 (SFT — 진짜 *행동 정렬*) 이, Ch 30-31 에서 단계 4 (alignment) 가 본격 등장합니다.

> *왜 GPT 모델 하나가 모든 task 를 해내는가* 의 답은 단계 3 (SFT) 부터 — head 가 task 별로 분기하지 않으니 *입력 프롬프트* 만 바꾸면 같은 모델이 다른 일을 합니다.""")

# ----- 7. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — BPE 직접 학습 (vocab=2048)

GPT-2 와 같은 종류 (byte-level BPE) 의 작은 vocab 을 직접 학습합니다. Ch 19 에서 봤던 WordPiece / WordLevel 토크나이저 학습 절차의 *BPE 판*.

| 토크나이저 | 학습 방식 | 등장 챕터 |
|---|---|---|
| WordLevel | 공백 + 빈도 - 가장 단순 | Ch 19 (직접 학습) |
| WordPiece | 빈도 기반 subword (BERT) | Ch 7-23 (BERT 챕터들 - 가져옴), Ch 19 (직접 학습) |
| **BPE (byte-level)** | **빈도 높은 byte 쌍 반복 병합 (GPT-2)** | **Ch 24 (본 챕터, 직접 학습), Ch 25-26 (GPT 챕터들)** |

### WordPiece vs BPE 의 결합 방식 차이

같은 입력 `"unhappiness"` 에 대해:

- **WordPiece** (BERT): `["un", "##happiness"]` 또는 `["un", "##happy", "##ness"]` - 단어 *중간* subword 에 `##` 접두사. 단어 경계를 명시.
- **BPE** (GPT-2): `["un", "happiness"]` 또는 `["un", "h", "app", "iness"]` - 접두사 없이 *byte 시퀀스 그대로*. 단어 경계는 *공백 자체가 한 byte* 로 처리.

byte-level BPE 의 핵심 장점: *어떤 유니코드 문자열이든* (이모지, 한글, 특수 기호) UNK 없이 표현 가능 - 가장 작은 단위가 *byte (256개)* 라 vocab 에 모든 byte 를 포함하면 *완전 가역*.

### 특수 토큰 컨벤션

GPT-2 는 특수 토큰을 *최소화* 합니다 - `<|endoftext|>` 하나만 사용 (bos = eos = pad 겸용). BERT 의 `[CLS] [SEP] [MASK] [PAD] [UNK]` 5종과 대비.

> Ch 19 의 "토크나이저는 모델과 운명공동체" 원칙이 본 챕터에서도 유효 - vocab 2,048 의 BPE 를 직접 학습한 뒤, *같은 vocab 으로 GPT 본체를 random init* 합니다. Ch 25 에서는 *반대로* - `gpt2` (124M) 의 vocab 50,257 BPE 를 그대로 가져와 같은 TinyStories 데이터로 **continual pretraining**. 토크나이저 + 모델이 *함께* 변하는 게 Ch 24-25 의 핵심 비교.""")

# ----- 8. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업""")

code(r"""%pip install -q -U transformers tokenizers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import math
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device     : cuda  ({device_name})")
    print(f"VRAM total : {vram_gib:.2f} GiB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device     : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device     : cpu  (training will be very slow - Colab T4 recommended)")

print(f"torch      : {torch.__version__}")

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
print(f"use fp16   : {USE_FP16}")""")

# ----- 9. 데이터 -----
md(r"""## 1. TinyStories 데이터 로드

`roneneldan/TinyStories` 는 GPT-3.5 / GPT-4 가 *4세 어린이가 이해할 단어만* 으로 생성한 짧은 영어 동화 약 2.1M 편 (Eldan & Li 2023, arXiv:2305.07759). 어휘·문법이 단순해 **3-5M 파라미터** 짜리 작은 모델로도 grammatical 한 생성이 가능합니다.

학습 split 의 처음 **30,000 stories** 만 사용 (T4 30분 룰 안).""")

code(r"""from datasets import load_dataset

N_TRAIN = 30_000      # 더 길게 돌리려면 키우세요 (full 은 약 2.1M stories)
N_VAL   = 500

raw_train = load_dataset("roneneldan/TinyStories", split=f"train[:{N_TRAIN}]")
raw_val   = load_dataset("roneneldan/TinyStories", split=f"validation[:{N_VAL}]")
print("train:", raw_train)
print("val  :", raw_val)
print("\n=== sample story ===")
print(raw_train[0]["text"][:400])""")

# ----- 10. 토크나이저 -----
md(r"""## 2. BPE 토크나이저 직접 학습

`tokenizers.BPE` + ByteLevel pre-tokenizer 로 vocab 2,048 의 BPE 를 코퍼스에서 직접 학습합니다. Ch 19 의 토크나이저 학습 절차와 같은 패턴 - 다른 점은 *알고리즘* 만 (WordPiece → BPE).""")

code(r"""from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

VOCAB_SIZE = 2048
EOS = "<|endoftext|>"

bpe = Tokenizer(BPE(unk_token=None))
bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
bpe.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[EOS],
    initial_alphabet=ByteLevel.alphabet(),
    show_progress=True,
)

t0 = time.time()
bpe.train_from_iterator((ex["text"] for ex in raw_train), trainer, length=len(raw_train))
print(f"BPE training done: {time.time()-t0:.1f}s, vocab={bpe.get_vocab_size()}")

# HF 표준 인터페이스로 wrap - bos = eos = pad 모두 <|endoftext|> 로 (GPT-2 컨벤션)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=bpe,
    bos_token=EOS,
    eos_token=EOS,
    pad_token=EOS,
)

print("\n=== encode/decode demo ===")
sample = "Once upon a time, a little rabbit went to the forest."
enc = tokenizer(sample)
print(f"input      : {sample}")
print(f"ids        : {enc['input_ids']}")
print(f"tokens     : {tokenizer.convert_ids_to_tokens(enc['input_ids'])}")
print(f"decode     : {tokenizer.decode(enc['input_ids'])}")
print(f"vocab_size : {tokenizer.vocab_size}")
print(f"eos_token  : {tokenizer.eos_token}  id={tokenizer.eos_token_id}")""")

md(r"""**관전 포인트** — `Once upon a time` 같이 TinyStories 에 *자주 등장* 하는 표현은 *적은 수의 토큰* 으로 압축, `rabbit` 처럼 덜 등장한 단어는 *여러 byte 조각* 으로 분할되는 경향. vocab 2,048 은 작은 모델에 맞춘 *최소한의 크기* 입니다.""")

# ----- 11. 토큰화 + group_texts -----
md(r"""## 3. 토큰화 + `group_texts` (HF 표준 CLM 전처리)

HuggingFace 의 causal LM 학습 표준 패턴 (`run_clm.py`) 그대로:

1. 전체 코퍼스를 토큰화 (배치 단위)
2. 각 story 끝에 `<|endoftext|>` 부착 (story 경계 표시)
3. 모든 토큰을 이어붙여 1D 스트림으로 만든 뒤 `block_size=128` 단위로 잘라 chunk 화
4. 각 chunk 가 한 학습 sample - `DataCollatorForLanguageModeling(mlm=False)` 가 `labels = input_ids` 를 자동으로 채워 next-token prediction loss 가 됨

Ch 20·22 의 `group_texts` 와 *완전히 같은 패턴*. MLM 챕터들에선 *마스킹* 만 추가됐다면, CausalLM 챕터에선 *labels = input_ids* 그대로.""")

code(r"""BLOCK_SIZE = 128

def tokenize_fn(batch):
    return tokenizer(batch["text"])

# 토큰화 (text 컬럼 제거)
tok_train = raw_train.map(tokenize_fn, batched=True, remove_columns=["text"], desc="tokenize train")
tok_val   = raw_val.map(tokenize_fn,   batched=True, remove_columns=["text"], desc="tokenize val")

# 각 story 끝에 EOS 부착 (story 경계 표시)
def add_eos(batch):
    new_ids, new_mask = [], []
    for ids in batch["input_ids"]:
        ids = ids + [tokenizer.eos_token_id]
        new_ids.append(ids)
        new_mask.append([1] * len(ids))
    return {"input_ids": new_ids, "attention_mask": new_mask}

tok_train = tok_train.map(add_eos, batched=True, desc="add eos train")
tok_val   = tok_val.map(add_eos,   batched=True, desc="add eos val")

# group_texts - 모든 토큰을 이어붙여 BLOCK_SIZE 단위로 자름
def group_texts(batch):
    concatenated = {k: sum(batch[k], []) for k in batch.keys()}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // BLOCK_SIZE) * BLOCK_SIZE
    return {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }

lm_train = tok_train.map(group_texts, batched=True, desc="group train")
lm_val   = tok_val.map(group_texts,   batched=True, desc="group val")

print(f"\ntrain chunks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"val   chunks: {len(lm_val):,}")
print(f"approx. train tokens: {len(lm_train) * BLOCK_SIZE / 1e6:.2f} M")
print("\nfirst chunk decode (first 200 chars):")
print(tokenizer.decode(lm_train[0]["input_ids"])[:200])""")

# ----- 12. collator 의 labels 확인 -----
md(r"""### 🔬 Collator 가 만드는 `labels` 확인 - *거의 모든 자리* 가 학습 신호

`DataCollatorForLanguageModeling(mlm=False)` 가 *내부적으로* `labels = input_ids.clone()` 을 만들어 `-100` 자리는 *없거나 pad 토큰 자리만* 임을 직접 확인합니다. Ch 20·22 의 MLM collator 가 약 85% 를 `-100` 으로 채웠던 것과 *정확히 반대*.""")

code(r"""from transformers import DataCollatorForLanguageModeling

collator_demo = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
demo_batch = collator_demo([lm_train[0], lm_train[1]])

input_ids = demo_batch["input_ids"]
labels = demo_batch["labels"]

print(f"input_ids shape: {tuple(input_ids.shape)}")
print(f"labels shape   : {tuple(labels.shape)}")

# -100 자리 vs 학습 신호 자리 비율
total = labels.numel()
n_ignored = (labels == -100).sum().item()
n_train_signal = total - n_ignored
print(f"\n=== 'labels = -100' thread - CausalLM vs MLM comparison ===")
print(f"total positions      : {total}")
print(f"  ignored (-100)     : {n_ignored:>5d}  ({100 * n_ignored / total:5.2f}%)")
print(f"  train signal       : {n_train_signal:>5d}  ({100 * n_train_signal / total:5.2f}%)")
print(f"\n[MLM (Ch 20/22)]     approx. 85% = -100, 15% = train signal")
print(f"[CausalLM (this ch)] {100 * n_ignored / total:5.2f}% = -100, {100 * n_train_signal / total:5.2f}% = train signal  <- almost every position")
print(f"\n=> a single step's token-learning efficiency: GPT pretrain is approx. 5-6x higher than MLM")

# input_ids 와 labels 의 동일성 검증 (pad 가 아닌 자리)
identical = (input_ids == labels).sum().item()
print(f"\n(input_ids == labels) positions: {identical}/{total}  - clone as-is")""")

md(r"""> **`-100` thread 환기** - MLM 은 *마스킹된 자리만* 학습, CausalLM 은 *거의 모든 자리* 학습. 같은 PyTorch `CrossEntropyLoss(ignore_index=-100)` 트릭이 *적용 자리만 정반대*. Ch 28 (SFT) 에서는 *prompt 자리만 -100* - 같은 트릭의 세 번째 적용. 그 한 줄 코드가 *모델이 instruction 을 따라가게 만드는 핵심* 입니다.

본 챕터의 collator 셋업이 *그 토대* - `labels = input_ids.clone()` 의 직관을 손에 익혀 두면 Ch 28 의 `labels[:prompt_len] = -100` 가 단번에 이해됩니다.""")

# ----- 13. 모델 -----
md(r"""## 4. `GPT2LMHeadModel` from scratch

`GPT2Config` 의 핵심 필드만 작게 잡고 *random init* (사전학습 X) 시작.

- `n_layer=4, n_head=4, n_embd=256` → 약 3M params, BERT 챕터들의 small DistilBERT 와 비슷한 스케일
- `n_positions = BLOCK_SIZE = 128` - 학습한 만큼만 context 사용
- bos / eos / pad token id 를 토크나이저와 동기화
- `tie_word_embeddings=True` (기본) - LM head 와 input embedding 의 weight 를 공유 → 파라미터 절약

### BERT 와의 차이가 코드로 드러나는 곳

- `BertForMaskedLM` 이 아니라 `GPT2LMHeadModel` - 클래스 자체가 *causal attention 내장*
- `from_pretrained(...)` 없이 `GPT2LMHeadModel(config)` - 무작위 초기화 from scratch (Ch 20·22 의 `BertForMaskedLM(config)` 와 같은 패턴, *모델 패밀리만* 다름)""")

code(r"""from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=BLOCK_SIZE,
    n_embd=256,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    activation_function="gelu_new",
    resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
)

model = GPT2LMHeadModel(config).to(device)   # 학습 전 generation 시연용으로 미리 GPU 로
n_params = model.num_parameters()
print(f"#params           : {n_params/1e6:.2f} M")
print(f"weight tying      : {config.tie_word_embeddings}  (lm_head <-> wte shared)")
print(f"fp32 weight size  : {n_params * 4 / 1024**2:.2f} MiB")
print(f"\nmodel: {type(model).__name__}")
print(f"  - body : {type(model.transformer).__name__}  (Decoder, causal attention)")
print(f"  - head : {type(model.lm_head).__name__}(in={model.lm_head.in_features}, out={model.lm_head.out_features})")""")

# ----- 14. 학습 전 generation -----
md(r"""## 5. 학습 *전* generation - 비교 기준선 (random init baseline)

Ch 20·22 의 *사전학습 전 [MASK] top-5 후보* 와 같은 역할. random init 모델은 통계적으로 *어느 토큰이든 거의 균등한 확률* 로 뽑으니, 생성 텍스트가 *영어와 거리가 먼 byte 조각 / 의미 없는 짧은 단어 나열* 이 나옵니다.

같은 prompt 와 sampling 설정을 학습 *전 / 후* 모두에서 호출 → loss 곡선 없이도 *학습이 본체에 무엇을 새겼는가* 가 한 화면에 드러납니다.""")

code(r"""PROMPTS = [
    "Once upon a time,",
    "The little girl",
    "A big dog",
]
GEN_KWARGS = dict(max_new_tokens=60, do_sample=True, temperature=0.8, top_k=50)


@torch.no_grad()
def generate_text(active_model, prompt: str, gen_tokenizer=None, **kwargs):
    tok = gen_tokenizer if gen_tokenizer is not None else tokenizer
    enc = tok(prompt, return_tensors="pt").to(active_model.device)
    out = active_model.generate(
        **enc,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        **kwargs,
    )
    return tok.decode(out[0], skip_special_tokens=True)


# 재현성을 위해 학습 전·후 동일 seed
torch.manual_seed(SEED)
model.eval()
before_outputs = []
print("=" * 70)
print("UNTRAINED model - generation from random initial weights")
print("=" * 70)
for p in PROMPTS:
    text = generate_text(model, p, **GEN_KWARGS)
    before_outputs.append(text)
    print(f"\n[prompt] {p}")
    print(text)""")

md(r"""**관전 포인트** - 학습 전 출력은 *무작위 토큰 나열* (반복되는 짧은 byte 조각, 의미 없는 단어들). Ch 20·22 의 *학습 전 [MASK] top-5* 가 *the / a / of / , / .* 같은 통계적 빈도 토큰이었던 것과 같은 현상의 *generation 판* 입니다. 학습 후 출력과 *나란히 비교* 하면 사전학습이 본체에 *next-token 분포* 를 새긴 증거를 직접 보게 됩니다.""")

# ----- 15. 학습 -----
md(r"""## 6. `Trainer` 로 사전학습

BERT 챕터들 (Ch 20·22) 과 *완전히 같은* Trainer 패턴 - 바뀌는 건 모델 클래스와 collator 의 `mlm=False` 두 곳.

- `DataCollatorForLanguageModeling(mlm=False)` → `labels = input_ids` (next-token prediction)
- `max_steps=1500`, `batch_size=32`, `fp16=True` - T4 약 15-18분
- `eval_steps=150` 으로 train / val loss 추이 관찰""")

code(r"""from transformers import (DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, TrainerCallback)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="./out_gpt_tinystories",
    max_steps=1500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_beta1=0.9, adam_beta2=0.95,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                       # T4 는 bf16 불가
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=150,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    seed=SEED,
)


class VRAMCallback(TrainerCallback):
    '''step 별 peak VRAM 기록 (로깅 윈도우 단위로 reset). CUDA 에서만 유효.'''

    def __init__(self):
        self.steps, self.peak_MiB = [], []

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            self.steps.append(state.global_step)
            self.peak_MiB.append(peak)
            torch.cuda.reset_peak_memory_stats()


vram_cb = VRAMCallback()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=lm_train,
    eval_dataset=lm_val,
    data_collator=collator,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== training summary ===")
print(f"elapsed       : {elapsed/60:.2f} min")
print(f"global_step   : {train_out.global_step}")
print(f"train_loss    : {train_out.training_loss:.4f}")
print(f"random baseline (ln vocab): {math.log(tokenizer.vocab_size):.4f}")
if torch.cuda.is_available():
    print(f"final peak    : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")""")

code(r"""# loss curve + VRAM trace
log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]
eval_pts  = [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# loss
ax1.plot([s for s, _ in train_pts], [l for _, l in train_pts], "-",
         color="tab:blue", alpha=0.6, label="train")
if eval_pts:
    ax1.plot([s for s, _ in eval_pts], [l for _, l in eval_pts], "s-",
             color="tab:red", label="eval")
ax1.axhline(math.log(tokenizer.vocab_size), ls=":", color="gray",
            label=f"uniform baseline = ln({tokenizer.vocab_size}) approx. {math.log(tokenizer.vocab_size):.2f}")
ax1.set_xlabel("step"); ax1.set_ylabel("cross-entropy loss")
ax1.set_title("TinyGPT-2 on TinyStories - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

# VRAM (CUDA 만)
if vram_cb.steps:
    ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
             label="peak VRAM (per log window)")
    ax2.set_title(f"VRAM trace  (bs=32, fp16, n_pos={BLOCK_SIZE})")
else:
    ax2.text(0.5, 0.5, "VRAM trace available on CUDA only",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("VRAM trace - CUDA only")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()""")

md(r"""**관전 포인트** - 학습 첫 step loss 가 약 7.6 (random baseline `ln(2048)`) 부근에서 시작해 *수백 step 안에 약 4-5* 로 빠르게 떨어지고, 1500 step 끝에 *약 2.5-3.0* 부근에서 안정화되면 정상. perplexity 로 환산하면 vocab 2,048 중 *약 12-20 개 후보* 로 좁힌 상태 - 다음 토큰을 *어느 정도 결정적* 으로 뽑는 수준.""")

# ----- 16. 학습 후 generation + before/after 비교 -----
md(r"""## 7. 학습 *후* generation + before/after 비교

같은 `PROMPTS / GEN_KWARGS` 로 학습 후 모델에서 다시 생성하고, §5 의 학습 전 결과와 나란히 비교합니다. **이 챕터의 합격 기준**: 학습 후 텍스트가 *전* 보다 명확히 *영어 문장* 에 가까워졌는가 - Ch 20·22 의 *사전·사후 [MASK] top-5* 비교의 *generation 판*.""")

code(r"""torch.manual_seed(SEED)
model.eval()
after_outputs = []
print("=" * 70)
print("TRAINED model - generation after Trainer.train()")
print("=" * 70)
for p in PROMPTS:
    text = generate_text(model, p, **GEN_KWARGS)
    after_outputs.append(text)
    print(f"\n[prompt] {p}")
    print(text)""")

code(r"""# before / after 나란히 - 사전학습이 본체에 새긴 next-token 분포의 직접적 증거
print("=" * 78)
print("BEFORE (random init) vs AFTER (trained on TinyStories 30K)")
print("=" * 78)
for p, before, after in zip(PROMPTS, before_outputs, after_outputs):
    print(f"\nPROMPT  : {p}")
    print("-" * 78)
    print(f"BEFORE  : {before[len(p):].strip()[:280]}")
    print(f"AFTER   : {after[len(p):].strip()[:280]}")""")

md(r"""**해석 가이드 - 사전학습이 만든 차이**

- **BEFORE (random init)**: *영어와 거리가 먼 byte 조각 / 의미 없는 짧은 단어 반복*. logits 가 random 초기값이라 sampling 이 통계적 빈도 토큰들 사이에서만 흔들림.
- **AFTER (TinyStories 30K × 1500 steps)**: *말이 되는 영어 문장* - 짧지만 *주어 + 동사 + 목적어* 구조, *동화 풍 어휘* (rabbit, forest, friend, mom, happy, ...). 완벽하진 않아도 *학습이 본체에 next-token 분포를 새긴 증거* 가 한 줄에서 명확.

> Ch 20·22 의 *사전·사후 [MASK] top-5* 비교에서 `[MASK]` 자리에 *the / a / of* 같은 빈도 토큰만 뽑히던 random init 모델이, 학습 후엔 *문맥에 맞는 정답 토큰* 을 top-5 에 담아내던 그 변화의 *generation 판* 입니다.""")

# ----- 17. reference gpt2 비교 -----
md(r"""### 🔬 Reference 비교 - `gpt2` (124M, OpenAI WebText) 의 같은 prompt generation

같은 prompt 3개를 *학습이 충분히 잘 된* 표준 `gpt2` (124M params, WebText 약 40GB 사전학습) 에 넣어 *우리 작은 GPT (약 3M, TinyStories 30K)* 와 격차를 직접 비교. Ch 20 의 *3-way [MASK] top-5 비교* (before / ours / `bert-base-uncased`) 와 같은 패턴.

T4 에서 약 1분 추가. 데이터·파라미터 격차가 generation 품질의 격차로 어떻게 드러나는지 한 화면에.""")

code(r"""from transformers import AutoTokenizer, AutoModelForCausalLM

print("loading reference gpt2 (124M, OpenAI WebText pretraining)...")
ref_tok = AutoTokenizer.from_pretrained("gpt2")
ref_tok.pad_token = ref_tok.eos_token
ref_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
print(f"  vocab_size : {ref_tok.vocab_size:,}")
print(f"  #params    : {ref_model.num_parameters()/1e6:.1f} M")

torch.manual_seed(SEED)
ref_outputs = []
print("\n" + "=" * 70)
print("REFERENCE gpt2 (124M, WebText) - generation on same prompts")
print("=" * 70)
for p in PROMPTS:
    text = generate_text(ref_model, p, gen_tokenizer=ref_tok, **GEN_KWARGS)
    ref_outputs.append(text)
    print(f"\n[prompt] {p}")
    print(text)

# 메모리 정리
del ref_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()""")

code(r"""# 3-way 비교 - BEFORE (random) / OURS (3M, TinyStories) / REF (gpt2 124M, WebText)
print("=" * 78)
print("3-way comparison: BEFORE (random) / OURS (3M, TinyStories 30K) / REF (gpt2 124M, WebText)")
print("=" * 78)
for p, before, after, ref in zip(PROMPTS, before_outputs, after_outputs, ref_outputs):
    print(f"\nPROMPT : {p}")
    print("-" * 78)
    print(f"BEFORE : {before[len(p):].strip()[:240]}")
    print(f"OURS   : {after[len(p):].strip()[:240]}")
    print(f"REF    : {ref[len(p):].strip()[:240]}")""")

md(r"""**해석 가이드 - 데이터·파라미터 규모가 만든 격차**

- **BEFORE (random)**: 영어와 거리 먼 byte 조각.
- **OURS (3M, TinyStories 30K × 1500 steps)**: *동화 풍 단순 영어* - 어휘는 동화 도메인에 강하지만 (rabbit, forest, mom, friend, ...) *복잡한 문장 구조 / 추상적 어휘* 는 약함.
- **REF (gpt2 124M, WebText 약 40GB)**: *다양한 도메인 어휘 + 자연스러운 문장 흐름* - 같은 prompt 에 대해 *동화풍이 아닌 일반 산문 / 뉴스 / 대화* 등 다양한 톤. 학습 데이터 분포 (WebText) 의 다양성이 generation 다양성으로 직결.

> **세 모델의 격차가 정확히 *모델 크기 + 데이터 크기 + 데이터 다양성* 의 격차** - 우리 작은 GPT (3M, TinyStories 30K stories) → reference `gpt2` (124M, WebText 약 40GB) 사이에 *파라미터 약 40배, 데이터 규모 약 수천 배, 도메인 다양성 격차*. 그게 generation 의 *질적 차이* 로 정확히 드러납니다.

> Ch 25 가 이 격차를 *데이터 축을 통제하고* 좁히는 챕터입니다 - `gpt2` (124M) 의 사전학습 *위에* 같은 TinyStories 30K 로 **continual pretraining**. *대규모 일반 사전학습 모델을 작은 도메인 데이터로 적응* 시킬 때의 generation 품질이, 우리 from-scratch 작은 GPT 와 어떻게 다른지 직접 비교.""")

# ----- 18. 변형: sampling -----
md(r"""## 🛠️ 변형 - sampling hyperparam 비교

같은 prompt 에 `temperature / top_k / top_p` 만 바꿔 generation 스타일 변화 관찰. *학습된 본체는 그대로* - 변하는 건 *sampling 분포* 뿐.""")

code(r"""prompt = "Once upon a time, a little rabbit"
configs = [
    {"label": "T=0.3, top_k=20  (conservative)", "temperature": 0.3, "top_k": 20,  "top_p": None},
    {"label": "T=0.8, top_k=50  (balanced)",    "temperature": 0.8, "top_k": 50,  "top_p": None},
    {"label": "T=1.0, top_p=0.9 (nucleus)",     "temperature": 1.0, "top_k": 0,   "top_p": 0.9},
    {"label": "T=1.2, top_k=100 (diverse)",     "temperature": 1.2, "top_k": 100, "top_p": None},
]
for c in configs:
    torch.manual_seed(SEED)
    print("=" * 70)
    print(f"[{c['label']}]")
    print(generate_text(model, prompt, max_new_tokens=60, do_sample=True,
                        temperature=c["temperature"], top_k=c["top_k"], top_p=c["top_p"]))
    print()""")

md(r"""**관전 포인트**

- `temperature` ↑ → logits 분포 *평탄화* → 다양성 ↑, 일관성 ↓
- `top_k=20` → 매 step 후보를 *상위 20 개* 로만 한정 → 안전하지만 반복적
- `top_p=0.9` (nucleus) → 누적 확률 90% 이내 후보 → *모델이 확신 있을 땐 좁게, 애매할 땐 넓게* 자동 조정
- `T=1.2, top_k=100` → 가장 다양하지만 *말이 안 되는 토큰* 도 종종 섞임""")

# ----- 19. 등장한 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.GPT2Config` | GPT-2 구조 hyperparam (n_layer, n_head, n_embd, n_positions, ...) | Ch 25 - `gpt2` 사전학습 config 그대로 로드 |
| `transformers.GPT2LMHeadModel` | decoder + LM head 내장, causal attention 자동 처리 | Ch 25 - `AutoModelForCausalLM.from_pretrained("gpt2")` |
| `transformers.GPT2LMHeadModel(config)` (random init) | from scratch 사전학습 모델 생성 | Ch 26 - 한국어 TinyStories scratch |
| `tokenizers.BPE + ByteLevel` | byte-level BPE 토크나이저 직접 학습 | Ch 26 - 한국어 BBPE |
| `DataCollatorForLanguageModeling(mlm=False)` | CausalLM collator - `labels = input_ids.clone()` 자동 | Ch 25-26 동일 / Ch 28 SFT 는 *`-100` 자리만 다름* |
| `group_texts` 패턴 (HF run_clm.py 표준) | 가변 길이 텍스트 → 고정 length 토큰 블록 스트림 | Ch 25-26 동일 |
| `model.generate(do_sample=True, ...)` | sampling-based text generation (temperature / top_k / top_p) | Ch 25-30 전반에서 활용 |
| `<\|endoftext\|>` 단일 special token | GPT-2 컨벤션 (bos = eos = pad 겸용) | Ch 25 - 같은 컨벤션 |""")

# ----- 20. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. GPT CausalLM 사전학습은 *거의 모든 자리* 가 학습 신호인데, BERT MLM 은 *15% 자리* 만 학습 신호입니다. 왜 BERT 는 *전 자리* 를 학습하지 못할까요? (causal attention vs bidirectional attention 의 정보 누출 관점)
2. `tie_word_embeddings=True` (weight tying) 가 `Linear(H, V)` 의 파라미터를 어떻게 절약하나요? vocab 2,048 / hidden 256 일 때 절약되는 파라미터 수를 직접 계산해 보세요.
3. 학습 첫 step loss 가 `ln(2048) ≈ 7.62` 가 아니라 *5.0* 이라면 무엇을 의심해야 하나요? *15.0* 이라면?
4. Ch 28 (SFT) 에서는 `labels[:prompt_len] = -100` 한 줄로 *prompt 부분만 학습 신호에서 제외* 합니다. 이번 챕터의 collator 출력 (거의 모든 자리가 학습 신호) 과 비교해 *왜 이 한 줄이 모델이 instruction 을 따라가게 만드는지* 설명해 보세요.""")

# ----- 21. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) 왜 GPT 는 *거의 모든 토큰* 을 학습 신호로 쓰고 BERT 는 15% 만 쓰나요?

**causal attention vs bidirectional attention 의 정보 누출 차이** 때문입니다.

BERT 의 *bidirectional* attention 은 토큰 $i$ 의 hidden 이 좌·우 모든 토큰을 다 봅니다. 만약 *모든 자리* 의 정답 토큰을 *예측 task* 로 두면, 모델은 *자기 자신을 그대로 복사* 하는 trivial 해를 학습합니다 (input 이 hidden 에 그대로 들어 있으니까). 그래서 BERT 는 일부 토큰을 `[MASK]` 로 *가려야만* 의미 있는 학습 신호가 생깁니다 - *주변 문맥* 으로 *가려진 자리* 를 복원.

GPT 의 *causal* attention 은 토큰 $i$ 의 hidden 이 *과거 (j ≤ i)* 만 봅니다. 미래 토큰을 못 보니 *next-token 예측* 이 trivial 하지 않습니다 - 모든 자리에서 *다음 토큰* 을 예측해도 cheating 이 안 됩니다. 그래서 *전 자리* 가 학습 신호.

코드 한 줄로 갈리는 차이:

```python
# BERT MLM
DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
# - 약 15% 자리만 학습 신호 (labels = original token id)
# - 나머지 85% 는 labels = -100 (loss 계산 제외)

# GPT CausalLM
DataCollatorForLanguageModeling(tokenizer, mlm=False)
# - 거의 모든 자리가 학습 신호 (labels = input_ids.clone())
# - pad 토큰 자리만 -100
```

한 step 의 *토큰 학습 효율* 은 GPT 가 약 5-6배 높습니다 (15% vs 거의 100%). 그래서 같은 step 수라도 GPT 가 더 많은 토큰을 학습.

### Q2. (이론) TinyStories 는 *일반 도메인* 인가요 *task corpus* 인가요? 왜 일반 위키 (Wikitext-103) 가 아닌가요?

**TinyStories 는 *합성된 simple 스토리* 라 *generation 시연 가치* 가 우선** 인 데이터입니다. *진정한 일반 도메인 사전학습* 의 의미에서는 Wikitext-103 보다 약하지만, 본 챕터의 목적은 *작은 모델로 generation 이 어떻게 동작하는지를 직접 보는 것* - 일반 위키 (Wikitext-103) 로 같은 셋업을 돌리면 3M 모델이 *문장 구조를 학습하기 전에 학습이 끝남*. TinyStories 의 단순한 어휘·문법 덕분에 *작은 모델로도 grammatical 한 생성이 가능* 합니다.

Ch 25 가 그 *trade-off 의 반대편* 을 다룹니다 - *큰 모델 (gpt2 124M) + 대규모 일반 코퍼스 (WebText)* 의 사전학습된 본체를 TinyStories 로 **continual pretraining**. *작은 + 합성 도메인 from-scratch* vs *큰 + 일반 도메인 사전학습 후 continual pretraining* 의 generation 품질 격차가 핵심 비교.

### Q3. (이론) BPE 토크나이저는 Ch 19 의 WordPiece / WordLevel 과 어떻게 다른가요?

세 방식 모두 *vocab 학습 알고리즘* 이지만, *어떻게 subword 를 만드는가* 가 다릅니다.

- **WordLevel**: 공백 + 빈도 - 단어 통째로 vocab 등록. UNK 가 많음.
- **WordPiece (BERT)**: 빈도 + likelihood 기반 subword 병합. 단어 *중간* subword 에 `##` 접두사 (`unhappiness` → `["un", "##happiness"]`).
- **BPE (GPT-2)**: *byte 쌍 빈도* 기반 반복 병합. 접두사 없이 byte 시퀀스 그대로 (`unhappiness` → `["un", "happiness"]`).

본 챕터는 *byte-level BPE* - 가장 작은 단위가 *byte (256개)* 라 *어떤 유니코드 문자열* (이모지, 한글, 특수 기호) 도 UNK 없이 표현 가능. Ch 19 의 직접 학습 챕터에서 본 세 알고리즘 중 *GPT 계열이 BPE 를 선호* 하는 이유.

```python
# 본 챕터의 BPE 학습 - vocab 2,048
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

bpe = Tokenizer(BPE(unk_token=None))
bpe.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(vocab_size=2048, initial_alphabet=ByteLevel.alphabet())
bpe.train_from_iterator(text_iter, trainer)
```

### Q4. (이론) `tie_word_embeddings=True` (weight tying) 가 정확히 무엇을 공유하나요?

**Input embedding (`wte`) 의 weight 와 LM head (`Linear(H, V)`) 의 weight 가 *같은 텐서를 공유*** 합니다.

```python
# 개념적으로
model.lm_head.weight = model.transformer.wte.weight   # 같은 텐서, 같은 메모리
```

직관: input embedding 은 *vocab token → hidden* 변환, LM head 는 *hidden → vocab logit* 변환. 둘이 *transpose 관계* 라 같은 weight 를 공유해도 의미가 통합니다. 효과:

- **파라미터 절약**: `vocab_size × hidden_size` 만큼 (본 챕터: 2,048 × 256 = 524,288 = 약 0.5M params). 전체 3M 모델의 약 17% - 작은 모델에선 비중이 큼.
- **학습 안정**: input 과 output 이 *같은 임베딩 공간* 을 공유 → 일관성 ↑.

GPT-2 의 기본값. 우리 모델도 자동으로 적용됩니다 (`config.tie_word_embeddings=True`).

### Q5. (실무) temperature / top_k / top_p sampling 의 의미는?

`model.generate(do_sample=True, ...)` 의 세 핵심 hyperparam:

- **`temperature`** (T): logits 를 *나눠* softmax → T<1 은 분포를 뾰족하게 (안전), T>1 은 평탄화 (다양). $p_i = \text{softmax}(\text{logits}_i / T)$.
- **`top_k`**: 매 step *상위 k 개* 후보로만 한정. 작으면 안전하지만 반복적.
- **`top_p`** (nucleus): *누적 확률 p* 이내 후보로 한정. 모델이 확신 있을 땐 좁게 (top-1 이 0.9), 애매할 땐 넓게 (top-100 이 0.9) 자동 조정.

```python
# 셋이 같이 쓰일 때의 적용 순서
# 1. logits / T  → softmax
# 2. top_k 로 후보 잘라냄
# 3. top_p 로 후보 더 잘라냄
# 4. 남은 후보에서 multinomial sampling
model.generate(do_sample=True, temperature=0.8, top_k=50, top_p=0.9, max_new_tokens=60)
```

일반적 추천: *대화* 에는 `T=0.7-0.9, top_p=0.9`, *코드 / 수식* 에는 `T=0.2-0.4, top_k=20`, *창의적 생성* 에는 `T=1.0-1.2, top_p=0.95`.

### Q6. (실무) `labels = -100` 트릭이 CausalLM 사전학습에서 거의 안 쓰이는데, 그럼 언제 쓰이나요?

본 챕터의 collator 출력에서 봤듯 CausalLM 사전학습은 *pad 토큰 자리만* `-100` (그것도 `group_texts` 로 chunk 길이가 모두 같으면 pad 도 없음). 거의 안 쓰임.

하지만 같은 트릭이 **Ch 28 (SFT, Instruction Tuning)** 에서 *결정적 한 줄* 로 부활합니다:

```python
# Ch 28 의 SFT 데이터 - "instruction + response" 형식
# 모델이 *response 부분만* 학습하길 원함 (instruction 은 외우면 안 됨)
prompt = "### 질문: 한국의 수도는?\n### 답변: "
response = "서울입니다."

input_ids = tokenizer(prompt + response)["input_ids"]
labels = input_ids.copy()
prompt_len = len(tokenizer(prompt)["input_ids"])
labels[:prompt_len] = [-100] * prompt_len   # <- 이 한 줄이 SFT 의 핵심
```

이 한 줄로 모델은 *prompt 토큰을 외우지 않고 response 만 학습* - 같은 instruction 에 대한 *다양한 response* 가 학습 가능하고, 추론 시에는 *주어진 instruction 에 대해 response 를 생성* 하게 됩니다. *모델이 instruction 을 따라간다* 는 게 이 한 줄의 효과.

본 챕터의 collator 출력 (거의 모든 자리 = 학습 신호) 을 손에 익혀 두면 Ch 28 의 `labels[:prompt_len] = -100` 가 단번에 이해됩니다.

### Q7. (실무) 다음 챕터 (Ch 25 — continual pretraining) 와의 비교는 어떻게 되나요?

Ch 25 = *OpenAI 가 사전학습한 `gpt2` (124M params, WebText 약 40GB) 를 TinyStories 로* **continual pretraining** (같은 CausalLM task, 같은 LM head — *task adaptation 의미의 fine-tune 이 아님*). 본 챕터의 *작은 from-scratch 모델* 과 *완전 반대 출발점*:

| 축 | Ch 24 (본 챕터) | Ch 25 (다음) |
|---|---|---|
| 모델 크기 | 약 3M params | **약 124M (40배)** |
| 사전학습 | from scratch (random init) | **OpenAI WebText 약 40GB 사전학습** |
| TinyStories 학습 | 사전학습 그 자체 (1500 steps) | **continual pretraining** (수백 steps 면 충분) |
| 토크나이저 | 직접 학습 BPE vocab 2,048 | **gpt2 BPE vocab 50,257 (그대로)** |
| Generation 품질 | grammatical 한 동화 풍 영어 | **자연스러운 동화 + 일반 도메인 폭** |
| 학습 시간 | 약 18분 (사전학습) | **약 5-8분** (continual pretraining 만) |

**핵심 메시지**: *대규모 일반 사전학습된 본체* + *작은 도메인 continual pretraining* 이 *작은 from-scratch 모델* 보다 *빠르게, 그리고 좋게* 도달합니다. *왜 실무는 보통 from-scratch 가 아니라 사전학습 모델을 가져와 새 데이터로 계속 학습하거나 SFT 하는가* 의 답. (단계 3 SFT 는 Ch 28 에서 본격.)""")

# ----- 22. 다음 챕터 -----
md(r"""## 다음 챕터 예고

**Chapter 25. GPT2 (124M) Continual Pretraining 으로 TinyStories 에 적응 — *대규모 사전학습 모델의 도메인 계속 학습***

- `AutoModelForCausalLM.from_pretrained("gpt2")` - OpenAI WebText 약 40GB 로 사전학습된 124M params 모델 로드
- **같은 TinyStories 30K** 데이터 (본 챕터와 동일) 로 **continual pretraining** (계속 사전학습 / continual learning — *같은 CausalLM task, 새 데이터, head 그대로*. *task adaptation 의미의 fine-tune 이 아니라 단계 2*)
- **핵심 비교**: 본 챕터 (3M, from scratch, 18분) vs Ch 25 (124M, continual pretraining, 5-8분) 의 generation 품질·학습 곡선 격차
- *trainer 자체는 Ch 24 와 동일* (`transformers.Trainer` + `DataCollatorForLanguageModeling(mlm=False)`) — *변하는 건 모델 로드 한 줄 + lr (scratch 5e-4 → continual pretraining 2e-5)*
- 작은 데이터 + 큰 사전학습 모델 = *왜 실무가 from-scratch 가 아니라 사전학습 모델 위에 계속 학습 패턴인가* 의 정량 답변
- *진짜 task adaptation 의미의 fine-tune (instruction tuning)* 은 Ch 28 SFT 에서 본격 등장

> **변하는 축**: *모델 크기 + 사전학습 여부* (3M / scratch → 124M / pretrained). 데이터·토크나이저 규약·loss·trainer 는 같음. Phase 4 의 *학습 단계 2 (continual pretraining)* 가 본격적으로 자리 잡는 챕터.""")


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
README = """# 24_gpt_tinystories — GPT (TinyStories) from-scratch 사전학습 (Phase 4 첫 챕터)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/24_gpt_tinystories/24_gpt_tinystories.ipynb)

## 한 줄 목표
Phase 4 의 첫 챕터. Ch 7-23 의 *BERT (encoder, MLM, task head 부착 fine-tune)* 패러다임에서, *GPT (decoder-only, causal LM, LM head 그대로)* 패러다임으로 전환합니다. `GPT2LMHeadModel` 을 random init 으로 from scratch 띄우고, **TinyStories 30,000 stories** 로 next-token 예측 사전학습 → 같은 prompt 에 *학습 전 / 학습 후* generation 결과를 나란히 비교 (+ reference `gpt2` 124M 까지 3-way). Ch 20·22 의 *사전·사후 [MASK] top-5 비교* 와 같은 깊이로, *사전학습이 본체에 어떤 next-token 분포를 새겼는가* 를 직접 확인합니다.

## Phase 4 도입

| 축 | Phase 1·2·3 (BERT, Ch 7-23) | **Phase 4 (GPT, Ch 24-30)** |
|---|---|---|
| 본체 | Encoder (양방향 attention) | **Decoder (causal attention)** |
| 사전학습 task | MLM (가려진 토큰 예측) | **CausalLM (next-token 예측)** |
| 학습 신호 위치 | 선택된 약 15% 만 (`-100` 다수) | **거의 모든 토큰** (`-100` pad 만) |
| Output head | task 별 부착 (`Linear(H, K)`) | **LM head (`Linear(H, V)`) 그대로** |
| Downstream 적응 | head 교체 + fine-tune (*task 적응*) | **SFT (*behavior alignment*)** + alignment (DPO / GRPO) |
| "Fine-tune" 의미 | task 별 특화 | **prompt 만 바꿔도 다른 일** |

> 본 챕터는 그 *출발점* - 작은 GPT 를 처음부터 학습해 *next-token 예측이 어떻게 generation 으로 이어지는지* 를 직접 봅니다. Ch 25 (대규모 사전학습 `gpt2` **continual pretraining**) / Ch 28 (SFT) / Ch 30-31 (DPO / GRPO) 가 같은 본체 위에 쌓여 갑니다.

## 다루는 핵심 개념
- **GPT2LMHeadModel(config)** from scratch - `from_pretrained` 없이 random init (Ch 20·22 의 `BertForMaskedLM(config)` 와 같은 패턴, 모델 패밀리만 다름)
- **`GPT2Config` 핵심 필드** - `n_layer / n_head / n_embd / n_positions`, `tie_word_embeddings` (기본 True, 약 0.5M params 절약)
- **causal attention** - encoder (BERT) 와 본질적 차이. 모델 클래스가 내장 처리
- **`DataCollatorForLanguageModeling(mlm=False)`** - labels = input_ids.clone() 자동. *거의 모든 자리* 가 학습 신호 (MLM 의 15% 와 정반대)
- **`group_texts` 패턴** (HF run_clm.py 표준) - 가변 길이 텍스트 → 고정 길이 `block_size=128` 블록 스트림
- **byte-level BPE 토크나이저 직접 학습** - `tokenizers.BPE + ByteLevel`, vocab 2,048. WordPiece 와의 결합 방식 차이
- **GPT-2 special token 컨벤션** - `<|endoftext|>` 하나가 bos / eos / pad 겸용
- **`model.generate(do_sample=True, ...)`** - temperature / top_k / top_p sampling 비교
- **`-100` thread 환기** - MLM (15% 자리) vs CausalLM (거의 모든 자리) vs SFT (response 부분만, Ch 28) - 같은 트릭, 정반대 자리
- **파인튜닝 의미 변화 thread 환기** - BERT 시대 (task head 부착) vs GPT 시대 (head 그대로, 행동 정렬)
- Random baseline loss `ln(2048) ≈ 7.62`, TinyStories 3M 모델은 보통 *약 2.5-3.0* 까지 도달
- **Reference 비교** - `gpt2` (124M, WebText 약 40GB) 의 같은 prompt generation 으로 *모델 크기 + 데이터 격차* 의 generation 품질 차이 직접 확인

## Loss
`CrossEntropyLoss` (next-token, `mlm=False`) - BERT MLM 의 CE 와 수식적으로 동일. 마스킹 위치만 다름 (BERT: 무작위 15% / GPT: 모든 토큰의 다음 위치, 거의 모든 자리). 모델 forward 안에서 `logits` 와 `labels` 가 한 칸 shift 되어 처리.

수식: $L_{\\text{CLM}} = -\\frac{1}{n-1} \\sum_{i=1}^{n-1} \\log P(x_{i+1} \\mid x_{\\leq i})$

## 데이터
`roneneldan/TinyStories` (Eldan & Li 2023, arXiv:2305.07759) - GPT-3.5 / GPT-4 가 *4세 어린이 어휘* 로 생성한 짧은 영어 동화 약 2.1M 편. 본 챕터는 *학습 split 의 처음 30,000 stories* 만 사용 (약 4-6M 토큰).

`block_size=128` 로 `group_texts` 후 train 약 30,000-50,000 chunks / eval 약 500 chunks.

## 모델
**`GPT2LMHeadModel`** with `n_layer=4, n_head=4, n_embd=256, n_positions=128`. 약 **3M params** (weight tying 자동 적용). BERT 챕터들 (Ch 20·22 의 작은 BERT 약 10M, Ch 9-18 의 DistilBERT 약 66M) 과 다르게 *완전 random init* 에서 시작.

## Hyperparams
- `max_steps=1500`, `per_device_train_batch_size=32`, `learning_rate=3e-4`
- `lr_scheduler_type="cosine"`, `warmup_steps=100`
- AdamW `betas=(0.9, 0.95)`, `weight_decay=0.1`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가)
- `eval_strategy="steps"`, `eval_steps=150`

## 환경
Google Colab **T4 GPU 필수**. 약 25-30분 (데이터 로드 약 2분 + BPE 학습 약 3분 + 학습 전 generation 약 30초 + 모델 학습 약 18분 + 학습 후 generation + reference `gpt2` 비교 약 2분).

device 자동 감지 (CUDA / MPS / CPU) - 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 22-23 | 작은 BERT (한국어, scratch) | klue/bert-base (가져옴) | 한국어 위키 → NSMC | MLM head / Linear(H, 2) | CE (masked / class) |
| **24** | **작은 GPT2 (직접, scratch)** | **BPE (직접 학습, vocab 2,048)** | **TinyStories 30K stories** | **Linear(H, V) (LM head, weight tied)** | **CE (next-token, 거의 모든 자리)** |
| 25 (다음) | gpt2 (124M, OpenAI WebText 사전학습) | BPE (GPT2 그대로, vocab 50,257) | TinyStories (Ch 24 와 동일) | Linear(H, V) (LM head 그대로) | CE (next-token) - **continual pretraining** |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[25_gpt2_continual_pretrain](../25_gpt2_continual_pretrain/) - OpenAI `gpt2` (124M, WebText 약 40GB 사전학습) 을 *같은 TinyStories 30K* 로 **continual pretraining** (계속 사전학습 / continual learning — 같은 CausalLM task, head 그대로). *데이터를 통제하고 본체 출발점만 다름*. 본 챕터 (3M, from scratch, 약 18분) vs Ch 25 (124M, continual pretraining, 약 5-8분) 의 generation 품질·학습 곡선 격차가 *왜 실무는 from-scratch 가 아니라 대규모 사전학습 모델을 활용하는가* 의 정량 답변. *진짜 task adaptation 의미의 fine-tune (instruction tuning)* 은 Ch 28 SFT 에서 본격 등장.
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}")
