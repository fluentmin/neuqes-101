"""Build 20_en_bert_pretrain/20_en_bert_pretrain.ipynb — Phase 3, scratch MLM.

작은 BertConfig 로 BERT 를 random init 하고 MLM 으로 사전학습.
토크나이저는 `bert-base-uncased` 의 WordPiece 를 그대로 가져옴.
데이터는 *일반 도메인* Wikitext-103 paragraphs — 원본 BERT 의 Wikipedia + BookCorpus 정신을 따라
*task 도메인이 아닌* 일반 위키 본문으로 MLM. Ch 21 의 분류 fine-tune (Yelp 영화 리뷰) 은
*완전히 다른 도메인* — 일반 표상 학습 → 다른 task 로 transfer 메시지가 정직해짐.
산출물은 ./ch20_small_bert_mlm 체크포인트 (Ch 21 에서 fine-tune).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "20_en_bert_pretrain"
OUT_NB = OUT_DIR / "20_en_bert_pretrain.ipynb"
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


# ----- 1. Title -----
md(r"""# Chapter 20. 작은 BERT 직접 사전학습 — 영어 MLM (scratch)

**목표**: Phase 3 의 두 번째 챕터. Ch 19 에서 *토크나이저를 직접 학습* 해 봤다면, 이번엔 **모델 본체를 직접 random init 해 사전학습** 합니다. 표준 BERT 보다 *훨씬 작은* (약 10M params) BERT 를 짜서 **일반 도메인 Wikitext-103** paragraphs 로 **Masked Language Modeling (MLM)** 사전학습. 원본 BERT 의 Wikipedia + BookCorpus 정신을 따라 *task 도메인이 아닌* 일반 위키 본문 사용 — Ch 21 의 분류 fine-tune (Yelp 영화 리뷰) 은 *완전히 다른 도메인* 으로 *일반 표상 → 다른 task* transfer 메시지가 정직해집니다. 토크나이저는 학습 안정성을 위해 표준 `bert-base-uncased` 를 그대로 가져옵니다.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 20-25분 (`bert-base-uncased` 토크나이저 로드 + Wikitext-103 다운로드 + 5K paragraphs 필터링·토큰화 약 3분 + MLM 2 epoch 약 15-20분 + 평가/저장)

---

## 학습 흐름

1. 🔤 **토크나이저**: `bert-base-uncased` WordPiece (vocab 30,522) 그대로 로드
2. 📥 **데이터**: `Salesforce/wikitext`, config `wikitext-103-raw-v1` paragraphs 5,000 (일반 도메인, 라벨 없음)
3. 🚀 **토큰화 + `group_texts`**: HF `run_mlm.py` 표준 — 모든 텍스트를 이어붙여 토큰 스트림으로 만든 뒤 `block_size=128` 단위로 자름
4. 🏗️ **모델 구성**: `BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)` + `BertForMaskedLM(config)` random init
5. 🚀 **학습**: `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` + Trainer, fp16, 2 epoch
6. 🔬 **평가**: MLM loss 학습 곡선, perplexity, masked token 예측 시연 ([MASK] top-5 후보 — 위키 도메인 + Yelp 도메인 혼합)
7. 💾 **저장**: `model.save_pretrained("./ch20_small_bert_mlm")` — Ch 21 에서 `from_pretrained` 로 재사용

---

> 📒 **사전 학습 자료**: Ch 19 (토크나이저 직접 학습) — 토크나이저가 "어떻게 만들어지는지" 를 본 뒤, 이번 챕터는 *모델이 어떻게 사전학습되는지* 를 봅니다. 둘이 합쳐져 "사전학습된 BERT 를 가져다 쓰는" 흐름 (Ch 7-18) 의 *안쪽* 이 드러납니다.""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 17 | klue/bert-base | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | sigmoid (각각) | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | sigmoid + 태스크별 | `BCEWithLogitsLoss + λ·L_aux` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — | — |
| **20 ← 여기** | **작은 BERT (직접, scratch)** | **`bert-base-uncased` 토크나이저 (가져옴)** | **Wikitext-103 paragraphs (일반 도메인)** | **MLM head** | softmax (MLM) | **`CrossEntropyLoss` (masked token)** |
| 21 (다음) | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 (다른 도메인 transfer) | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 안에서의 위치** — Ch 19 (토크나이저 학습) → Ch 20 (모델 사전학습) → Ch 21 (분류 fine-tune). 사전학습 모델을 *받아서 fine-tune* 하던 Phase 1·2 의 흐름을 이번엔 *직접 만들어* 봅니다.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 19)

| 축 | Ch 19 (토크나이저 학습 전용) | Ch 20 (작은 BERT scratch MLM) |
|---|---|---|
| **이 챕터의 task** | 토크나이저 학습 (모델 없음) | **모델 사전학습 (MLM)** ← *유일한 변화* |
| 모델 | 없음 | **작은 `BertForMaskedLM` (random init, 약 10M params)** |
| 토크나이저 | WordPiece + WordLevel *직접 학습* | **`bert-base-uncased` *가져옴*** (vocab 30,522) |
| 데이터 | Yelp text + NSMC text (vocab 학습용) | **Wikitext-103 paragraphs (일반 도메인 MLM 학습용)** |
| Loss | 없음 (vocab + merge rules 가 산출물) | **`CrossEntropyLoss` (masked token 위치만)** |
| 산출물 | tokenizer json 파일 4 종 | **모델 체크포인트** (`./ch20_small_bert_mlm`) — Ch 21 재사용 |

> **변경점 한 가지 원칙** — Phase 3 안에서 *모델 축* 이 변합니다 (없음 → 작은 BERT scratch). 토크나이저는 *직접 학습이 가능함을 본 뒤* 표준으로 돌아옵니다 — `bert-base-uncased` 가 *공개되어 검증된* vocab 이라 사전학습 안정성이 높음. 토크나이저 *학습 절차* 와 *모델 학습 절차* 가 두 챕터로 분리되어 각각의 메커니즘이 또렷이 보이게 합니다.

### 왜 토크나이저는 가져오고 모델만 직접 학습하나

(1) **vocab 신뢰성** — Ch 19 의 vocab 8K 토크나이저는 5K 문장 학습이라 어휘 커버리지가 좁음. `bert-base-uncased` 의 30,522 vocab 은 Wikipedia + BookCorpus 로 학습되어 *영어 일반 분포* 를 잘 표현. 모델 학습이 vocab 노이즈에 영향받지 않음. (2) **다음 챕터 호환** — Ch 21 에서 같은 토크나이저로 분류 fine-tune 하면, *문체가 다른* downstream 입력에도 안정. (3) **표준 패턴** — 실무에서도 보통 *모델은 직접 사전학습하지만 vocab 은 검증된 것* 을 쓰는 패턴 (예: HF `roberta-base` 도 GPT-2 의 BPE 그대로 가져옴).

### 왜 task corpus (Yelp) 가 아니라 일반 위키인가 — 원본 BERT 의 정신

원본 BERT (Devlin et al., 2018) 는 *영어 Wikipedia + BookCorpus* 라는 **일반 도메인** 코퍼스로 사전학습한 뒤, *완전히 다른 downstream task* (GLUE, SQuAD 등) 로 fine-tune 했습니다. 본 챕터도 같은 패턴 — Wikitext-103 *일반 위키 paragraphs* 로 MLM 사전학습 → Ch 21 에서 *영화 리뷰 (Yelp)* 라는 *다른 도메인* 으로 transfer.

만약 Yelp text 로 MLM 사전학습한 뒤 Yelp 분류로 fine-tune 하면 *domain-adaptive pretraining* (DAPT) 에 가까워져 *일반 표상 학습 → 다른 task transfer* 의 본질이 흐려집니다. *일반 도메인 → 다른 도메인 transfer* 가 *진짜 사전학습-fine-tune 패러다임*. Ch 22 (한국어 위키 → NSMC 분류) 가 같은 패턴.""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss 함수의 변화 — Masked Language Modeling (MLM)

이전 분류 챕터들 (Ch 11-18) 의 loss 는 *문장 한 개에 라벨 하나*. 이번 챕터는 *문장 안의 가려진 토큰들* 을 맞춰야 합니다 — 토큰 위치 하나하나가 *분류 task* 가 됩니다.

### 수식

입력 토큰 시퀀스 $x = (x_1, \dots, x_n)$ 의 일부를 무작위로 `[MASK]` 로 가린 뒤, 모델이 *원래 토큰* 을 예측:

$$L_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid x_{\setminus M})$$

- $M$: 가려진 위치 집합 (전체 토큰의 15%)
- $P(x_i \mid x_{\setminus M})$: 모델이 $i$ 번 위치에 *원래 토큰* 을 예측할 확률 (vocab 30,522 차원 softmax)
- $|M|$: 가려진 토큰 수로 평균

각 가려진 위치에서 *vocab 전체에 대한 `CrossEntropyLoss`*. 분류 헤드의 K (이전 챕터들의 2, 5, 7) 가 이번엔 **V = 30,522** 로 폭증.

### 숫자로 감 잡기 (vocab=30,522)

| 모델 상태 | 정답 토큰 확률 | $-\log p$ |
|---|---|---|
| 균등 추측 (random init 초기) | $1/30522 \approx 3.28 \times 10^{-5}$ | **10.33** ← random baseline |
| 약하게 학습 (정답 확률 0.01) | $0.01$ | 4.61 |
| 잘 학습된 작은 BERT (정답 확률 0.05-0.1) | $0.05$ - $0.1$ | **2.3 - 3.0** ← 이번 챕터 목표 영역 |
| 큰 사전학습 BERT (정답 확률 0.3+) | $0.3$ | 1.20 |
| 완벽 (정답 확률 1.0) | $1.0$ | 0.00 |

**관전 포인트**:
- 학습 첫 step 의 loss 가 약 10 부근이면 random init 직후 *균등 추측* 상태. 첫 100 step 안에 빠르게 떨어지면 vocab 정상.
- 목표는 *vocab 의 일부 후보를 추려내는* 단계 (약 2.5-4.0). 작은 모델 + 5K paragraphs + 2 epoch 으로 *완벽* 은 불가능 — 그러나 Ch 21 의 fine-tune 출발점으로는 충분.

### Perplexity (PPL)

언어 모델 표준 metric. $\text{PPL} = e^{L}$ — *모델이 다음 토큰을 평균 몇 후보 중에서 고민하는가* 의 직관:

| MLM loss | PPL | 해석 |
|---|---|---|
| 10.33 | 30,522 | 균등 (전체 vocab) |
| 5.0 | 148 | vocab 의 일부로 좁혀짐 |
| 3.0 | 20 | 20 개 후보 중에서 결정 |
| 1.0 | 2.7 | 거의 결정적 |

> 이전 분류 챕터의 `random baseline = log K` (K=2 → 0.69, K=7 → 1.95) 와 같은 직관을 *vocab 차원에 확장* 한 게 MLM. `ln(30522) ≈ 10.33` 이 그 random baseline.""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

이번 챕터부터 토크나이저는 *표준 사전학습 모델 것* 을 가져옵니다.

- `AutoTokenizer.from_pretrained("bert-base-uncased")` — 영어 BERT 의 표준 WordPiece.
- vocab_size = 30,522 (Ch 19 의 8K 와 비교해 약 4 배).
- 학습 코퍼스 = Wikipedia (영어) + BookCorpus → 영어 일반 분포가 잘 반영.
- 특수 토큰: `[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102`, `[MASK]=103`.

### 같은 문장의 토큰화 — Ch 19 직접 학습 vs Ch 20 가져옴

`"The capital of France is Paris, located on the Seine river."` 같은 *일반 위키풍* 문장이:

- **Ch 19 의 8K WordPiece (Yelp 5K 학습)**: 학습 코퍼스 (Yelp) 분포에 *최적화* 되어 있어 *위키 도메인 단어* (`Seine`, `capital`, `located` 등) 가 작은 조각으로 쪼개짐 — Yelp 에 적게 등장하는 어휘일수록 더 잘게 분할.
- **Ch 20 의 `bert-base-uncased` 30K WordPiece (Wiki + BookCorpus 학습)**: 위키 어휘를 *덜 쪼개짐*. 30K vocab + 일반 영어 학습이라 일반 도메인 + Yelp 같은 task 도메인 *둘 다* 폭넓게 커버.

본 챕터의 사전학습 데이터 (Wikitext-103) 와 토크나이저 (`bert-base-uncased`) 가 *둘 다 일반 위키 분포* 라 *vocab 미스매치* 가 작습니다. Ch 21 에서 Yelp 분류로 fine-tune 할 때도 같은 토크나이저로 *문체 다른* 도메인을 처리 — 일반 영어 어휘는 거의 덜 쪼개지고, *영화 리뷰 특유 표현* 만 약간 더 분할되는 정도.

### "토크나이저는 모델과 운명공동체"

Ch 19 §5-4 의 cross-language 실험에서 봤듯, *학습 언어가 다른* 토크나이저를 모델에 끼우면 거의 100% UNK 가 됩니다. 모델 weight 와 vocab 은 *함께 학습되어* 그 vocab 의 토큰 임베딩 공간에서 의미를 형성합니다.

이번 챕터에서 *토크나이저는 vocab 만 빌려오고 모델은 random init* 입니다 — 즉 *vocab 구조* 와 *토큰 임베딩 의미* 가 분리됩니다. 학습 초기에는 임베딩이 random 이라 vocab 구조의 가치가 안 보이지만, MLM 으로 학습이 진행되면 임베딩이 vocab 구조에 *맞춰 정렬* 됩니다 — *이 챕터의 본질이 바로 그 정렬 과정*.

> Ch 21 부터는 *이 챕터의 모델 + 토크나이저 쌍* 을 통째로 가져가 fine-tune. 둘은 *함께 가야* 의미가 유지됩니다.""")

# ----- 6. install + import -----
md(r"""## 🛠️ 환경 셋업""")

code(r"""%pip install -q -U transformers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

plt.rcParams["axes.unicode_minus"] = False

# device 자동감지 — Colab(T4) 은 CUDA, 로컬 Mac 은 MPS, 그 외 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:         {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
elif DEVICE == "cpu":
    print("Warning: CPU runtime — MLM training will be very slow. Switch to T4 recommended.")""")

# ----- 7. nvidia-smi -----
md(r"""**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):""")
code(r"""!nvidia-smi""")

# ----- 8. 토크나이저 -----
md(r"""## 1. 🔤 토크나이저 — `bert-base-uncased` 그대로 로드

vocab 30,522 의 영어 WordPiece. *모델은 random init* 이지만 토크나이저는 *완성품* 을 가져옵니다.""")

code(r"""TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")
print(f"special tokens:")
for name in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token"):
    tok = getattr(tokenizer, name)
    tid = tokenizer.convert_tokens_to_ids(tok) if tok is not None else None
    print(f"  {name:>11}: {tok!r:>10}  (id={tid})")

# 간단 시연 — 일반 위키풍 문장
SAMPLE = "The capital of France is Paris, located on the Seine river."
enc = tokenizer(SAMPLE, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nsample: {SAMPLE!r}")
print(f"tokens ({len(tokens)}): {tokens}")
print(f"ids:    {enc['input_ids'][0].tolist()}")""")

# ----- 9. 데이터 -----
md(r"""## 2. 📥 데이터 — Wikitext-103 paragraphs (일반 도메인 사전학습 코퍼스)

원본 BERT 가 영어 Wikipedia + BookCorpus 라는 *일반 도메인* 코퍼스로 사전학습한 정신을 따라, 본 챕터도 **Wikitext-103** 본문으로 MLM 사전학습합니다 — *task 도메인 (Yelp 영화 리뷰) 으로 사전학습하면 domain-adaptive pretraining 에 가까워져 사전학습의 진짜 메시지 (일반 표상 학습 → 다른 task 로 transfer) 가 흐려지기 때문*.

**원본**: `Salesforce/wikitext`, config `wikitext-103-raw-v1` (CC-BY-SA, 정제된 영문 위키 paragraphs). HF Hub 정제본 — line 단위로 이미 정리되어 있어 빈 줄 / 너무 짧은 줄 / 너무 긴 줄만 제외하면 바로 사용 가능. Ch 21 의 분류 fine-tune (Yelp 이진) 은 *완전히 다른 도메인* — 사전학습 → fine-tune transfer 메시지가 정직해집니다. Ch 22 의 한국어 (한국어 Wikipedia paragraphs) 와 *대칭* 패턴.""")

code(r"""SEED = 42
N_TRAIN_TEXT = 5000
N_EVAL_TEXT  = 500

print("downloading Wikitext-103 (Salesforce/wikitext, wikitext-103-raw-v1)...")
raw_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
raw_eval  = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
print(f"  raw train lines: {len(raw_train):,}")
print(f"  raw eval  lines: {len(raw_eval):,}")

# 빈 줄 / 너무 짧은 줄 (제목·메타) / 너무 긴 줄 (목록·인용) 제외
def is_good(ex, min_len=50, max_len=2000):
    t = ex["text"].strip()
    return min_len <= len(t) <= max_len

train_filtered = raw_train.filter(is_good).shuffle(seed=SEED).select(range(N_TRAIN_TEXT))
eval_filtered  = raw_eval.filter(is_good).shuffle(seed=SEED).select(range(N_EVAL_TEXT))

# text 컬럼만 유지
ds_train_raw = train_filtered.remove_columns([c for c in train_filtered.column_names if c != "text"])
ds_eval_raw  = eval_filtered.remove_columns([c for c in eval_filtered.column_names if c != "text"])

print(f"\nsampled train: {len(ds_train_raw):,} paragraphs")
print(f"sampled eval:  {len(ds_eval_raw):,} paragraphs")
print()
print(f"sample text length stats (chars):")
lens = [len(t) for t in ds_train_raw["text"]]
print(f"  mean: {np.mean(lens):.1f}, median: {np.median(lens):.0f}, max: {max(lens)}")
print()
print(f"first sample previews:")
for i in range(3):
    t = ds_train_raw[i]["text"]
    print(f"  Sample {i}: {t[:120]}")""")

# ----- 10. 토큰화 + group_texts -----
md(r"""## 3. 🚀 토큰화 + `group_texts` — HF `run_mlm.py` 표준 패턴

MLM 사전학습의 표준 입력 포맷은 *고정 길이 블록*. 변동 길이 문장에 그대로 padding 하면 *손실*: (a) 짧은 문장이 많으면 PAD 비율이 높아 GPU 시간 낭비, (b) 긴 문장은 truncation 으로 정보 손실.

**해결책**: 모든 문서를 *이어 붙여 토큰 스트림* 으로 만든 뒤, `block_size=128` 단위로 자름. 문장 경계가 사라지는 trade-off 가 있지만, BERT 사전학습은 *임의 위치의 토큰 예측* 이라 문장 경계가 중요하지 않음.

Wikitext-103 paragraphs 는 *제한 50-2000자 필터링* 으로 평균 문장 길이가 일정 (수백 자 위주). 5,000 paragraphs 가 `block_size=128` 로 잘리면 약 1,000-2,000 블록 정도로 정리됩니다. 코드는 HF `examples/pytorch/language-modeling/run_mlm.py` 의 `group_texts` 함수를 그대로 따른 표준 패턴.""")

code(r"""BLOCK_SIZE = 128

def tokenize_function(examples):
    # 특수 토큰 부착 안 함 — 블록 단위로 자를 거라 [CLS]/[SEP] 가 의미 없음
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

tokenized_train = ds_train_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
tokenized_eval = ds_eval_raw.map(
    tokenize_function, batched=True, remove_columns=["text"],
)
print(f"tokenized_train: {tokenized_train}")
print(f"first 30 input_ids of sample 0: {tokenized_train[0]['input_ids'][:30]}")""")

code(r"""def group_texts(examples):
    '''HF 표준 group_texts — 모든 토큰 스트림을 이어 붙인 뒤 block_size 로 자름.'''
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    # block_size 배수로 잘라내기 (마지막 토막은 버림)
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    # labels = input_ids 사본 (collator 가 mask 위치만 골라냄)
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result


lm_train = tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_eval  = tokenized_eval.map(group_texts, batched=True, batch_size=1000)

print(f"lm_train: {lm_train}")
print(f"lm_eval:  {lm_eval}")
print(f"\nblock_size:           {BLOCK_SIZE}")
print(f"train blocks: {len(lm_train):,}  (approx. {len(lm_train) * BLOCK_SIZE:,} tokens)")
print(f"eval blocks:  {len(lm_eval):,}   (approx. {len(lm_eval) * BLOCK_SIZE:,} tokens)")
print(f"\nsample block 0 first 20 ids: {lm_train[0]['input_ids'][:20]}")
print(f"sample block 0 first 20 tok: {tokenizer.convert_ids_to_tokens(lm_train[0]['input_ids'][:20])}")""")

# ----- 11. 모델 -----
md(r"""## 4. 🏗️ 작은 `BertConfig` + `BertForMaskedLM` — random init

표준 `bert-base-uncased` 는 hidden=768, layer=12, head=12, intermediate=3072 = **110M params** — T4 에서 scratch 학습은 *수일* 필요.

이번 챕터는 *입문용 작은 BERT* 로 축소:

| hyperparam | 표준 `bert-base-uncased` | 이번 챕터 (작은 BERT) |
|---|---|---|
| `hidden_size` | 768 | **256** |
| `num_hidden_layers` | 12 | **4** |
| `num_attention_heads` | 12 | **4** |
| `intermediate_size` | 3072 | **1024** |
| `max_position_embeddings` | 512 | **128** (BLOCK_SIZE 와 같음) |
| 총 파라미터 | 약 110M | **약 10M** (toy 규모) |

크기는 1/10 이지만 *MLM 학습이 진행되는지* 보기에는 충분. Ch 21 에서 분류 fine-tune 할 때 성능 비교가 진짜 결과.""")

code(r"""HIDDEN_SIZE         = 256
NUM_HIDDEN_LAYERS   = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE   = 1024
MAX_POS_EMBED       = 128  # = BLOCK_SIZE

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
)

model = BertForMaskedLM(config)  # random init — pretrained weight 없음!

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
emb = sum(p.numel() for n, p in model.named_parameters() if "embeddings" in n)
encoder = sum(p.numel() for n, p in model.named_parameters() if "encoder" in n)
head = sum(p.numel() for n, p in model.named_parameters() if "cls" in n)

print(f"Config: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, "
      f"head={NUM_ATTENTION_HEADS}, intermediate={INTERMEDIATE_SIZE}")
print(f"max_position_embeddings: {MAX_POS_EMBED}")
print()
print(f"Total parameters:    {total:>13,}  ({total/1e6:.2f} M)")
print(f"Trainable:           {trainable:>13,}")
print(f"  embeddings:        {emb:>13,}  ({emb/total:.1%})  ← vocab 30522 x hidden 256")
print(f"  encoder (4 layer): {encoder:>13,}  ({encoder/total:.1%})")
print(f"  MLM head:          {head:>13,}  ({head/total:.1%})  ← tied with embeddings")""")

md(r"""**관찰** — 작은 BERT 의 파라미터는 *임베딩 테이블이 절반 이상* 차지합니다 (vocab 30522 × hidden 256 ≈ 7.8M). encoder body 자체는 약 2M. 이게 *vocab 큰데 모델이 작은* 셋업의 특징 — 표준 BERT (vocab 30K × hidden 768 ≈ 23M / 110M = 21%) 와 비율이 매우 다릅니다.

> MLM head 의 weight 는 입력 임베딩과 *tied* (공유) — `BertForMaskedLM` 기본 동작. vocab 차원 출력 layer 가 임베딩 테이블과 같아 파라미터 절약.""")

# ----- 12. Collator + Trainer -----
md(r"""## 5. 🚀 `DataCollatorForLanguageModeling` + Trainer 학습

collator 가 매 batch 마다 *무작위로 15% 토큰을 [MASK]* 로 바꾸고, 그 위치의 정답 토큰을 `labels` 로 표시 (나머지 위치는 -100 → CrossEntropyLoss 무시).

**MLM masking 규칙** (BERT 원논문):
- 선택된 15% 중 80%: 실제로 `[MASK]` 로 교체
- 10%: 무작위 다른 토큰으로 교체
- 10%: 원래 토큰 유지

이 비율은 *모델이 [MASK] 토큰 자체에 과도하게 의존하지 않게* 하는 트릭. `DataCollatorForLanguageModeling` 이 자동 처리.""")

code(r"""data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)""")

md(r"""### 🔍 [MASK] 가 들어가는 원리 — 한 눈에 보는 80/10/10

`DataCollatorForLanguageModeling` 은 매 step 마다 *입력 토큰의 약 15%* 를 *무작위로* 선택하고, 선택된 위치마다 세 가지 중 하나를 적용합니다.

| 선택된 토큰 운명 | 비율 | 의도 |
| --- | --- | --- |
| `[MASK]` 로 교체 | **80%** | 표준 마스킹 — 모델이 *주변 문맥만으로* 원래 토큰을 맞추도록 |
| **다른 random 토큰** 으로 교체 | 10% | inference 때는 `[MASK]` 가 없으니, 모델이 *항상* 자기 입력을 *의심* 하게 만듦 |
| **원본 그대로** 유지 | 10% | 동일 — 입력과 정답이 일치하는 케이스도 학습 데이터에 포함 |

**나머지 85%** 의 토큰은 `labels = -100` 으로 두어 *loss 계산에서 제외* 됩니다 (PyTorch CE 의 `ignore_index` 기본값). 즉 한 step 의 MLM loss 는 *선택된 15% 자리만* 모아 평균한 값.

> 이 `labels = -100` 트릭은 BERT-만의 것이 아닙니다 — Phase 4 GPT 사전학습은 *거의 모든 토큰* 을 학습 (`labels = input_ids`), SFT (Ch 27) 는 *prompt 만 -100, 답변만 학습*. 같은 트릭, 정반대 자리. Ch 21 에서 더 자세히.""")

code(r"""# 짧은 예시 문장 하나에 collator 한 번 돌려서 어떤 자리가 어떻게 바뀌는지 직접 봅니다.
import pandas as pd

DEMO_SENT = "Pretraining a language model on Wikipedia teaches it general English structure."
demo_enc = tokenizer(DEMO_SENT, return_tensors=None)
demo_ids = demo_enc["input_ids"]

torch.manual_seed(0)  # 재현성: 같은 seed 면 같은 마스킹
demo_batch = [{"input_ids": demo_ids, "attention_mask": [1] * len(demo_ids)}]
demo_out = data_collator(demo_batch)

masked_ids = demo_out["input_ids"][0].tolist()
labels     = demo_out["labels"][0].tolist()
mask_id    = tokenizer.mask_token_id

orig_tokens   = tokenizer.convert_ids_to_tokens(demo_ids)
masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)

rows = []
for orig_id, new_id, lab, orig_tok, new_tok in zip(demo_ids, masked_ids, labels, orig_tokens, masked_tokens):
    if lab == -100:
        kind = "—"
    elif new_id == mask_id:
        kind = "[MASK] (80%)"
    elif new_id == orig_id:
        kind = "kept (10%)"
    else:
        kind = "random (10%)"
    rows.append({
        "pos": len(rows),
        "original": orig_tok,
        "after_collator": new_tok,
        "label_id": lab,
        "what_happened": kind,
    })

demo_df = pd.DataFrame(rows)
print(demo_df.to_string(index=False))""")

code(r"""# 큰 batch 통계 — 80/10/10 비율이 실제로 맞는지 확인
torch.manual_seed(0)
N_DEMO = 64
big_batch = [
    {"input_ids": lm_train[i]["input_ids"], "attention_mask": [1] * BLOCK_SIZE}
    for i in range(N_DEMO)
]
big_out = data_collator(big_batch)

in_ids = big_out["input_ids"]
lab    = big_out["labels"]

selected = (lab != -100)
n_total    = lab.numel()
n_selected = selected.sum().item()
n_mask     = ((in_ids == mask_id) & selected).sum().item()
n_kept     = ((in_ids == lab) & selected).sum().item()
n_random   = n_selected - n_mask - n_kept

print(f"Total tokens:                      {n_total:>7,}")
print(f"Selected for loss (target 15%):    {n_selected:>7,}  ({100 * n_selected / n_total:5.2f}%)")
print(f"  └─ replaced with [MASK]:         {n_mask:>7,}  ({100 * n_mask / n_selected:5.2f}% of selected)")
print(f"  └─ replaced with random:         {n_random:>7,}  ({100 * n_random / n_selected:5.2f}% of selected)")
print(f"  └─ kept as original:             {n_kept:>7,}  ({100 * n_kept / n_selected:5.2f}% of selected)")
print()
print("Target: 선택 15% / 그 중 80-10-10 으로 [MASK]-random-kept. 표본 크면 비율 안정.")""")

md(r"""**관전 포인트**

- `what_happened` 가 `—` 인 자리 (약 85%) 는 *입력과 정답이 그대로* — loss 에 기여하지 않음. 모델은 *문맥을 만들어 주는* 역할만.
- `[MASK]` 자리 (약 12%) 가 본 task 의 *진짜 학습 신호*. 주변 토큰들의 attention 결과로 *가려진 자리* 의 vocab 분포를 예측.
- `random` (약 1.5%) 와 `kept` (약 1.5%) 는 *inference 분포 일치* 를 위한 정규화. 추론 시에는 `[MASK]` 가 없으므로 *입력을 절대 신뢰하면 안 된다* 는 신호를 학습에 섞어 줌.
- 매 epoch · 매 batch 마다 마스킹은 *새로 무작위* — 같은 문장이 epoch 마다 다른 자리에서 가려져 학습됨 (data augmentation 효과).""")

code(r"""USE_FP16 = (DEVICE == "cuda")   # T4 는 fp16, MPS/CPU 는 fp32
NUM_EPOCHS = 2

training_args = TrainingArguments(
    output_dir="./ch20_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,            # scratch 학습이라 fine-tune (2e-5) 보다 크게
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",            # 마지막에 직접 save_pretrained
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print(f"epochs:        {NUM_EPOCHS}")
print(f"batch size:    {training_args.per_device_train_batch_size}")
print(f"learning rate: {training_args.learning_rate}")
print(f"fp16:          {USE_FP16}")
print(f"train blocks:  {len(lm_train):,}")
print(f"steps / epoch: {len(lm_train) // training_args.per_device_train_batch_size}")""")

md(r"""### 🔬 학습 직전 baseline — 사전학습 전·후 비교 준비

`trainer.train()` 을 호출하기 *전* 의 모델 상태 (`BertForMaskedLM(config)` random init) 로 두 가지를 측정해 둡니다 — *학습 후와 나란히* 보면 *사전학습이 본체에 무엇을 새겼는지* 가 한 화면에 드러납니다.

1. **`eval_loss` / `perplexity`** — random init 이므로 vocab 30,522 균등 분포 (`ln V` ≈ 10.33) 근처가 기대치.
2. **같은 문장의 `[MASK]` top-5** — random init 의 logits 는 거의 균등이라 *문맥과 무관한 토큰* (자주 등장하는 관사·전치사·특수문자 등) 이 뽑힙니다.

학습이 끝난 뒤 6-1 셀에서 *완전히 같은 문장* 으로 다시 측정해 *직접 비교* 합니다.""")

code(r"""# predict_mask 함수 정의 — 학습 전·후 두 번 호출하므로 먼저 정의
def predict_mask(text, top_k=5):
    '''text 안의 [MASK] 자리 top-k 토큰과 확률 반환.'''
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return None
    results = []
    for pos in mask_positions:
        probs = torch.softmax(logits[pos], dim=-1)
        top_p, top_i = probs.topk(top_k)
        candidates = [(tokenizer.convert_ids_to_tokens(int(i)), float(p))
                       for p, i in zip(top_p, top_i)]
        results.append((int(pos), candidates))
    return results


# 검증용 문장 — 학습 전·후 동일하게 사용
# 위키 일반 도메인 (사전학습 직접 본 분포) + Yelp 도메인 (Ch 21 downstream, 다른 도메인 transfer)
test_sentences = [
    # 위키 도메인 — 사전학습 직접 본 분포, 향상 명확히 기대
    f"The capital of France is {tokenizer.mask_token}.",
    f"Water freezes at {tokenizer.mask_token} degrees Celsius.",
    # Yelp 도메인 (Ch 21 fine-tune 대상) — 다른 도메인 transfer 한계 확인
    f"The food at this restaurant was absolutely {tokenizer.mask_token}.",
    f"I would {tokenizer.mask_token} recommend this place.",
]

# ---- 사전학습 전 eval_loss / perplexity ----
pre_eval = trainer.evaluate()
pre_eval_loss = pre_eval["eval_loss"]
pre_eval_ppl  = math.exp(pre_eval_loss)
random_baseline_loss = math.log(tokenizer.vocab_size)

print("=" * 78)
print("BEFORE pretraining  (random init body)")
print("=" * 78)
print(f"  eval_loss       : {pre_eval_loss:.4f}   (random baseline ln V = {random_baseline_loss:.4f})")
print(f"  eval_perplexity : {pre_eval_ppl:,.0f}     (random baseline V    = {tokenizer.vocab_size:,})")
print()

# ---- 사전학습 전 [MASK] top-5 ----
pre_top5_records = []
for sent in test_sentences:
    results = predict_mask(sent, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    pre_top5_records.append({"sentence": sent, "top5_before": top5_tokens})
    print(f"input: {sent}")
    print(f"  top-5 before pretraining: {top5_tokens}")
    print()""")

code(r"""t0 = time.time()
train_result = trainer.train()
elapsed = time.time() - t0
print(f"\nMLM pretraining done in {elapsed/60:.1f} min")
print(f"mean train loss: {train_result.training_loss:.4f}")
print(f"random baseline loss (uniform over vocab): {math.log(tokenizer.vocab_size):.4f}")""")

code(r"""!nvidia-smi""")

# ----- 13. 평가 -----
md(r"""## 6. 🔬 평가 — MLM loss 곡선 + perplexity + masked token 예측

학습이 *실제로 진행* 됐는지 세 각도로 확인:
1. step-by-step train loss 곡선 — 빠르게 10.33 (random baseline) → 5 이하로 떨어졌는지
2. eval set 의 perplexity — 외부 텍스트에서도 일관된 수준인지
3. 임의 문장에 `[MASK]` 를 끼워 top-5 후보 출력 — *어떤 단어를 예측하는지* 정성 평가""")

code(r"""# 학습 로그에서 train loss 추출
log_history = trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(tokenizer.vocab_size)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="train MLM loss")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":", label=f"random baseline (ln V = {random_baseline:.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("MLM loss (CrossEntropy)")
    ax.set_title("MLM training loss — small BERT scratch on Wikitext-103")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")""")

code(r"""eval_metrics = trainer.evaluate()
eval_loss = eval_metrics["eval_loss"]
eval_ppl = math.exp(eval_loss)
print("=== eval (held-out Wikitext-103 paragraphs) ===")
for k, v in eval_metrics.items():
    if isinstance(v, float):
        print(f"  {k:>22}: {v:.4f}")
print()
print(f"  MLM loss:               {eval_loss:.4f}")
print(f"  perplexity (exp loss):  {eval_ppl:.2f}")
print(f"  random baseline PPL:    {tokenizer.vocab_size:,}  (uniform over vocab)")
print(f"  -> model narrowed vocab to approx. {eval_ppl:.0f} candidates per masked position")""")

md(r"""### 6-1. 🔬 사전학습 전·후 비교 — random init 본체 vs 2 epoch 학습 후

학습 직전 (5번 마지막 셀에서 측정해 둔 `pre_eval_loss` / `pre_top5_records`) 와 *완전히 같은 문장·같은 평가 셋* 에 학습 후 모델을 적용해 두 결과를 나란히 봅니다. *사전학습이 본체에 무엇을 새겼는가* 의 가장 직접적인 증거.""")

code(r"""# ---- 사전학습 후 eval_loss / perplexity ----
post_eval = trainer.evaluate()
post_eval_loss = post_eval["eval_loss"]
post_eval_ppl  = math.exp(post_eval_loss)

print("=" * 78)
print("AFTER pretraining  (2 epoch MLM on Wikitext-103)")
print("=" * 78)
print(f"  eval_loss       : {post_eval_loss:.4f}   (before: {pre_eval_loss:.4f})")
print(f"  eval_perplexity : {post_eval_ppl:,.2f}        (before: {pre_eval_ppl:,.0f})")
print(f"  -> narrowed vocab to approx. {post_eval_ppl:.0f} candidates per masked position")
print()

# ---- 사전학습 후 [MASK] top-5 ----
post_top5_records = []
for sent in test_sentences:
    results = predict_mask(sent, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    post_top5_records.append({"sentence": sent, "top5_after": top5_tokens})
    print(f"input: {sent}")
    print(f"  top-5 after pretraining: {top5_tokens}")
    print()""")

md(r"""### 6-2. eval_loss / perplexity — 수치 비교

두 측정치를 한 표·한 막대 그래프로.""")

code(r"""# 사전·사후 수치 비교 표
metric_compare = pd.DataFrame({
    "metric":           ["eval_loss", "eval_perplexity"],
    "before (random)":  [pre_eval_loss,  pre_eval_ppl],
    "after (2 epoch)":  [post_eval_loss, post_eval_ppl],
    "random baseline":  [random_baseline_loss, float(tokenizer.vocab_size)],
})
print("Before vs After — eval metrics")
print(metric_compare.round(4).to_string(index=False))""")

code(r"""# 막대 그래프 두 장 (eval_loss / perplexity)
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

loss_values = [pre_eval_loss, post_eval_loss]
loss_labels = ["before (random)", "after (2 epoch)"]
axes[0].bar(loss_labels, loss_values, color=["#999999", "#4878D0"])
axes[0].axhline(random_baseline_loss, color="black", lw=1.0, ls=":",
                label=f"random baseline ln V = {random_baseline_loss:.2f}")
axes[0].set_ylabel("eval_loss")
axes[0].set_title("MLM eval_loss")
axes[0].legend(loc="upper right", fontsize=10)

ppl_values = [pre_eval_ppl, post_eval_ppl]
axes[1].bar(loss_labels, ppl_values, color=["#999999", "#4878D0"])
axes[1].set_yscale("log")
axes[1].axhline(tokenizer.vocab_size, color="black", lw=1.0, ls=":",
                label=f"random baseline V = {tokenizer.vocab_size:,}")
axes[1].set_ylabel("perplexity (log scale)")
axes[1].set_title("MLM perplexity")
axes[1].legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()""")

md(r"""### 6-3. 🏆 학습이 *충분히 잘 된 경우* 의 기준점 — 표준 `bert-base-uncased` 비교

우리 작은 BERT (10M, 5K paragraphs × 2 epoch) 의 top-5 가 *방향성은 맞지만 정답이 잘 안 보이는* 이유는 단순합니다 — **학습 데이터·모델 크기·학습 시간 모두 부족**. *그럼 학습이 충분히 잘 되면 어떤 결과가 나오나?* 의 답을 같은 문장에 표준 `bert-base-uncased` (110M, 위키+BookCorpus 약 33억 토큰) 를 적용해 직접 봅니다.

같은 토크나이저 (`bert-base-uncased`) 를 쓰고 있으므로 *모델만 바꿔* 두 결과를 나란히.""")

code(r"""# 표준 bert-base-uncased 로드 — 학습이 충분히 잘 된 경우의 기준점
from transformers import AutoModelForMaskedLM

ref_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
ref_model.to(model.device)
ref_model.eval()

ref_param_count = sum(p.numel() for p in ref_model.parameters())
our_param_count = sum(p.numel() for p in model.parameters())
print(f"Our small BERT params: {our_param_count/1e6:.1f}M")
print(f"Reference BERT params: {ref_param_count/1e6:.1f}M  ({ref_param_count/our_param_count:.0f}x larger)")""")

code(r"""# Reference 모델로 같은 문장의 top-5 측정
def predict_mask_with(text, ref, top_k=5):
    '''임의의 MLM 모델로 [MASK] 자리 top-k 예측.'''
    ref.eval()
    inputs = tokenizer(text, return_tensors="pt").to(ref.device)
    with torch.no_grad():
        outputs = ref(**inputs)
    logits = outputs.logits[0]
    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return None
    results = []
    for pos in mask_positions:
        probs = torch.softmax(logits[pos], dim=-1)
        top_p, top_i = probs.topk(top_k)
        candidates = [(tokenizer.convert_ids_to_tokens(int(i)), float(p))
                       for p, i in zip(top_p, top_i)]
        results.append((int(pos), candidates))
    return results


ref_top5_records = []
for sent in test_sentences:
    results = predict_mask_with(sent, ref_model, top_k=5)
    top5_tokens = [tok for tok, _ in results[0][1]] if results else []
    ref_top5_records.append({"sentence": sent, "top5_ref": top5_tokens})

# 참조 모델 메모리 해제 (분류 fine-tune 챕터 21 가 같은 노트북이 아니므로 안전)
del ref_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()""")

md(r"""### 6-4. [MASK] top-5 — 3-way 비교 (before / ours / reference BERT)

같은 문장 4개의 [MASK] 자리 top-5 후보를 *사전학습 전 → 우리 작은 BERT 학습 후 → 표준 bert-base-uncased* 셋으로 나란히.""")

code(r"""# 3-way top-5 비교 표
rows = []
for pre, post, ref in zip(pre_top5_records, post_top5_records, ref_top5_records):
    rows.append({
        "sentence":          pre["sentence"],
        "top5_before":       ", ".join(pre["top5_before"]),
        "top5_ours":         ", ".join(post["top5_after"]),
        "top5_ref_bert":     ", ".join(ref["top5_ref"]),
    })

top5_compare = pd.DataFrame(rows)
print("Before (random) vs Ours (small BERT, 5K paragraphs) vs Reference (bert-base-uncased, approx. 3.3B tokens)")
print("=" * 100)
for _, row in top5_compare.iterrows():
    print(f"input: {row['sentence']}")
    print(f"  before (random)        : {row['top5_before']}")
    print(f"  ours  (small, 5K para) : {row['top5_ours']}")
    print(f"  ref   (bert-base)      : {row['top5_ref_bert']}")
    print()""")

md(r"""**해석 가이드 — 사전학습이 만든 차이**

- **`eval_loss`**: random baseline `ln V ≈ 10.33` 에서 약 5-7 부근까지 떨어졌으면 본체가 *언어 구조 일부* 를 학습. *완전한* BERT 수준은 아니어도 표준 BERT 가 학습한 것의 *방향* 은 맞춤.
- **`perplexity`**: 30,522 (vocab 전체) 에서 수십-수백 부근으로. *마스크 자리마다 후보를 약 50-500 개로 좁힌 상태* 라는 직관적 해석.
- **top-5 토큰** (3-way 비교):
  - *before (random)*: 자주 등장하는 *관사·전치사·특수문자* (`the`, `a`, `,`, `.`, `of`) — random init 이지만 logits 가 미세하게 흔들려 *통계적 빈도* 높은 토큰만 뽑힘.
  - *ours (small BERT, 5K paragraphs × 2 epoch)*: 위키 도메인은 *방향성이 보이기 시작* — 일반 부사·형용사, 위키 어휘 일부. 다만 정답 토큰 (`paris`, `0` 등) 이 top-5 안에 *안정적으로* 들어오지는 못함. **데이터·모델 크기 부족의 한계**.
  - *ref (bert-base-uncased, 약 33억 토큰 × 40 epoch)*: 위키 도메인은 *정답이 top-1* — `paris`, `zero` 같은 자연스러운 답. Yelp 도메인 (다른 도메인) 도 *감성 형용사* (`amazing`, `delicious`, `highly`) 가 자연스럽게 top-5 에 들어옴. **이게 사전학습이 충분히 잘 됐을 때의 모습**.

> **세 모델의 격차가 정확히 *데이터 규모 + 모델 크기 + 학습 시간* 의 격차** — 우리 작은 BERT (10M, 5K paragraphs, 2 epoch) → reference (110M, 3.3B tokens, 40 epoch) 사이에 *데이터 약 5,000배, 파라미터 11배, epoch 20배*. 그 격차가 top-5 의 *질적 차이* 로 정확히 드러납니다.

이번 챕터의 작은 BERT 는 *Wikitext-103 5K paragraphs × 2 epoch* 로 학습한 *일반 도메인 mini BERT*. 위키 도메인은 직접 본 분포라 향상이 빠르지만, Yelp 영화 리뷰 영역은 *다른 도메인* 이라 fine-tune 단계에서 적응이 필요합니다 — 이게 *진짜 사전학습 → fine-tune 패러다임* 의 핵심. Ch 21 에서 Yelp 이진 분류로 fine-tune 할 때 진짜 transfer 비교 — *우리가 직접 만든 작은 영어 BERT (일반 위키 5K, 약 10M)* vs *Ch 10 의 DistilBERT (대규모 Wikipedia+BookCorpus, 약 66M)* vs *random init baseline*.""")

# ----- 14. 저장 -----
md(r"""## 7. 💾 모델 저장 — Ch 21 에서 재사용

`model.save_pretrained()` 와 `tokenizer.save_pretrained()` 를 *같은 폴더* 에 저장. Ch 21 에서는 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 한 줄로 *이 BERT body* 를 가져와 분류 헤드를 새로 얹습니다.""")

code(r"""SAVE_DIR = "./ch20_small_bert_mlm"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import os
print(f"Saved to: {SAVE_DIR}")
print(f"Files:")
for f in sorted(os.listdir(SAVE_DIR)):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    if size > 1024 * 1024:
        size_str = f"{size / 1024 / 1024:.1f} MB"
    else:
        size_str = f"{size / 1024:.1f} KB"
    print(f"  {f:>30s}  {size_str}")""")

md(r"""**저장된 파일 구조** — `from_pretrained` 가 인식하는 HF 표준 레이아웃:

| 파일 | 역할 |
|---|---|
| `config.json` | `BertConfig` 직렬화 (hidden, layer, head, vocab 등) |
| `model.safetensors` (또는 `pytorch_model.bin`) | 모델 weight |
| `tokenizer.json` / `vocab.txt` | 토크나이저 (Ch 21 fine-tune 에서 같은 토크나이저 사용) |
| `special_tokens_map.json`, `tokenizer_config.json` | 특수 토큰 메타 |

> Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 호출 시, `BertForMaskedLM` 의 *MLM head 는 버려지고* encoder body 만 가져옴. 그 위에 새 `Linear(256, 2)` 분류 헤드를 random init 으로 부착 — Ch 7-18 의 fine-tune 셋업과 *동일한 구조*. 이번 챕터의 사전학습이 *얼마나 도움 됐는지* 가 Ch 21 의 학습 곡선에서 직접 비교됩니다.""")

# ----- 15. 변형 -----
md(r"""## 🛠️ 변형 — 학습 step 더 늘리거나 block_size 변경

작은 BERT 의 성능은 *학습량* 에 민감합니다. 두 가지 변형을 시뮬레이션 (실제 실행은 시간 관계상 한 셋업만):

| 변형 축 | 이번 챕터 (기본) | 변형 예 | 예상 효과 |
|---|---|---|---|
| `num_train_epochs` | 2 | 5 | eval loss 하락, perplexity 약 2-3 감소 |
| `BLOCK_SIZE` | 128 | 64 | 블록 수 약 2배, 한 블록 짧아져 *문맥* 줄음 → loss 약간 상승 |
| `BLOCK_SIZE` | 128 | 256 | 블록 수 절반, 한 블록 길어 *문맥* 풍부 → loss 하락 가능, VRAM 약 4배 (attention $O(n^2)$) |
| `N_TRAIN_TEXT` | 5,000 paragraphs | 30,000 paragraphs | loss 큰 폭 하락, 시간 약 6배 증가 (T4 30분 룰 안에선 1 epoch 만 가능) |
| `mlm_probability` | 0.15 | 0.30 | 더 어려운 task → loss 상승, 학습 신호 증가 (논문 BERT 는 15% 가 sweet spot) |

> **T4 30분 룰 안에서 가능한 가장 큰 개선** — Wikitext-103 paragraphs 를 5K → 20K (약 4배) 로 늘리고 batch 32 유지하면 한 epoch 약 20-25분, 1 epoch 로 마무리. 이번 챕터의 *짧고 빠른* 실험 이후 직접 변형해 보세요.""")

# ----- 16. 등장한 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.BertConfig` | BERT 구조 hyperparam 컨테이너 (hidden, layer, head 등) | Ch 22 에서 한국어 작은 BERT 설계 |
| `transformers.BertForMaskedLM` | encoder + MLM head, MLM 사전학습 전용 모델 클래스 | Ch 22 에서 한국어 MLM |
| `transformers.DataCollatorForLanguageModeling` | 매 batch 마다 자동 masking (15% rule) | Ch 22 같음 |
| `BertForMaskedLM(config)` (random init) | pretrained weight 없이 모델 생성 | Ch 22 같음 |
| `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")` | Wikitext-103 HF 정제본 로드 (line 단위) | Ch 22 한국어 Wikipedia (`wikimedia/wikipedia`) 와 같은 패턴 |
| `group_texts` 패턴 (HF run_mlm.py 표준) | 가변 길이 텍스트를 고정 길이 블록 스트림으로 | Ch 22 같음 |
| `model.save_pretrained()` / `from_pretrained()` | HF 표준 체크포인트 인터페이스 | Ch 21 에서 분류 fine-tune 로드 |
| `math.log(vocab_size)` | MLM 의 random baseline loss | 사전학습 챕터 공통 진단 도구 |""")

# ----- 17. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. MLM 사전학습의 random baseline loss 가 `ln(30522) ≈ 10.33` 인 이유는 무엇이고, 학습 첫 step 의 loss 가 이 값과 *크게* 다르다면 무엇을 의심해야 하나요?
2. `DataCollatorForLanguageModeling` 이 매 batch 마다 *다른* 위치를 mask 합니다. 같은 위치를 *고정해서* mask 하면 어떤 문제가 생길까요?
3. `BertForMaskedLM` 의 MLM head 가 *입력 임베딩과 tied* 됩니다. 왜 이렇게 묶으면 파라미터 절약 + 학습 안정에 모두 도움이 되나요?
4. Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 를 호출하면, *이번 챕터 모델의 어떤 부분* 이 이어지고 *어떤 부분* 이 버려지나요?""")

# ----- 18. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) 왜 Yelp text 가 아니라 일반 위키 (Wikitext-103) 로 사전학습하나요? Ch 21 의 분류가 Yelp 인데 같은 도메인으로 학습하는 게 더 유리하지 않나요?

**일반 도메인 → 다른 도메인 transfer** 가 *진짜 사전학습-fine-tune 패러다임* 이기 때문입니다. Yelp text 로 MLM 사전학습 → Yelp 분류 fine-tune 의 흐름은 사실 *domain-adaptive pretraining* (DAPT) 에 더 가깝습니다 — 사전학습이 *이미 task 도메인의 표현* 을 학습한 상태에서 분류만 얹는 셈.

원본 BERT (Devlin et al., 2018) 가 *Wikipedia + BookCorpus* 라는 일반 도메인 코퍼스로 사전학습한 뒤 *완전히 다른* GLUE/SQuAD task 로 fine-tune 한 게 *transfer 의 본질* — 일반 표상이 다른 도메인에도 적용 가능한가의 시험.

```python
# 이번 챕터 (Ch 20 → Ch 21) 흐름 — 원본 BERT 와 같은 정신
# 1. 일반 위키 (Wikitext-103) 로 MLM 사전학습
# 2. 영화 리뷰 (Yelp) 로 분류 fine-tune  ← 다른 도메인 transfer

# 만약 Yelp 로 사전학습 → Yelp 분류 fine-tune 이었다면
# domain-adaptive pretraining 에 가까워져 transfer 메시지가 약해짐
```

Ch 22-23 의 한국어 흐름 (한국어 Wikipedia → NSMC 영화 리뷰) 도 *대칭* 패턴. 일반 위키 학습이 task 도메인 fine-tune 에 *얼마나 효율적으로 transfer 되는가* 가 본 챕터의 진짜 측정 대상.

> **참고** — Ch 10 (DistilBERT) 도 Wikipedia + BookCorpus 사전학습 → Yelp 분류 fine-tune 의 *같은 패턴*. 본 챕터의 작은 BERT 와 Ch 10 의 DistilBERT 가 *같은 일반 도메인 사전학습 → 같은 task fine-tune* 이라 *공정한 비교* 가 됩니다. Yelp 로 사전학습했다면 Ch 10 과 비교 자체가 unfair.

### Q2. (실무) `bert-base-uncased` 의 모델 weight 도 같이 가져오면 더 빠르지 않나요?

맞습니다 — 그게 *fine-tuning* 흐름 (Ch 7-18). 이번 챕터는 그 흐름을 *뒤집어*, "사전학습이 어떻게 이뤄지는지" 를 보는 게 목적입니다.

`BertForMaskedLM.from_pretrained("bert-base-uncased")` 라고 쓰면 110M 사전학습 모델이 로드되어 *이미 잘 작동* 함. 반면 이번 챕터는:

```python
config = BertConfig(hidden_size=256, num_hidden_layers=4, ...)
model = BertForMaskedLM(config)   # random init, weight 없음
```

`from_pretrained` 가 아니라 `BertForMaskedLM(config)` 라는 생성자 호출로 *비어 있는* 모델을 만듭니다. 학습이 *0 에서 시작* 하는 것 자체가 핵심.

### Q3. (이론) `mlm_probability=0.15` 의 15% 는 어디서 나온 숫자인가요?

BERT 원논문 (Devlin et al., 2018, arXiv:1810.04805) 의 sweet spot:

- 너무 작으면 (5% 정도): 한 batch 의 *학습 신호* (loss 가 계산되는 위치) 가 너무 적어 학습 효율 ↓
- 너무 크면 (40%+): 모델이 *문맥* 으로 볼 수 있는 토큰이 너무 적어 mask 추측이 *불가능에 가까워짐*. loss 가 크지만 학습 가치는 작음
- 15% 가 *학습 신호 양 + 추측 가능성* 의 균형점

후속 연구 (RoBERTa, ELECTRA) 는 *동적 masking* (매 batch 다른 위치, 이미 본 챕터에서 collator 가 자동 처리) 또는 *교체-기반 학습* (ELECTRA) 같은 변형을 시도했지만, *15% mask 비율* 자체는 거의 표준으로 정착.

### Q4. (실무) MLM 학습 중 loss 가 갑자기 *발산* 하면 어떻게 해야 하나요?

작은 BERT scratch 학습은 fine-tune 보다 *학습률에 민감* 합니다. 발산 (loss → NaN 또는 100+) 의 흔한 원인:

```python
# 1. 학습률 낮추기 (5e-4 → 1e-4 → 5e-5 순서로)
training_args = TrainingArguments(learning_rate=1e-4, ...)

# 2. warmup_ratio 늘리기 (0.06 → 0.1)
training_args = TrainingArguments(warmup_ratio=0.1, ...)

# 3. gradient clipping (Trainer 기본 1.0, 더 빡빡하게)
training_args = TrainingArguments(max_grad_norm=0.5, ...)

# 4. fp16 끄고 fp32 로 시도 (loss scale overflow 가능성)
training_args = TrainingArguments(fp16=False, ...)
```

이번 챕터의 `lr=5e-4, warmup=0.06, fp16=True` 셋업은 *작은 BERT + 5K 데이터* 에 맞춰 보수적으로 잡았습니다. 모델 키우거나 데이터 늘리면 위 옵션을 조정.

### Q5. (실무) 사전학습이 *얼마나* 도움 되는지 어떻게 확인하나요?

Ch 21 에서 두 모델을 *같은 분류 task* 로 fine-tune 해 비교하는 게 가장 직접적입니다:

```python
# A. 이번 챕터 사전학습 모델 (Ch 20 산출물)
model_pretrained = AutoModelForSequenceClassification.from_pretrained(
    "./ch20_small_bert_mlm", num_labels=2
)

# B. 같은 구조 + random init (사전학습 안 한 baseline)
config = BertConfig(hidden_size=256, num_hidden_layers=4, ...)
model_scratch = BertForSequenceClassification(config)
```

두 모델을 *같은 Yelp 이진 학습 데이터* 로 fine-tune → eval accuracy 비교. 사전학습이 도움 됐다면 (A) 가 (B) 보다 *빨리* 그리고 *높이* 도달. Ch 21 의 핵심 실험.

> **참고**: 이번 챕터의 *작은 사전학습* (5K 문장, 1-2 epoch) 은 *큰 효과* 를 기대하기 어렵습니다. 그러나 *방향성* (random 보다 시작점이 낫다) 은 분명히 나옵니다. 큰 효과를 보려면 데이터 100K+, epoch 5+, BERT 표준 크기 — 이건 T4 30분 룰 밖.

### Q6. (이론) `group_texts` 가 문장 경계를 무시하는데, BERT 가 잘 학습되나요?

원논문의 BERT 는 NSP (Next Sentence Prediction) 같은 *문장 쌍* task 도 같이 학습했지만, 후속 연구 (RoBERTa, 2019) 가 *NSP 를 빼고 그냥 토큰 스트림으로 학습* 해도 성능이 *더 좋다* 는 걸 보였습니다. 이번 챕터는 그 단순화된 흐름 (MLM only).

토큰 스트림이 *문장 경계 정보를 잃지만* 얻는 게 더 큽니다:
- 짧은 문장이 PAD 로 가득 차지 않음 → GPU 활용도 ↑
- 긴 문장이 잘리지 않음 → 정보 손실 ↓
- 학습 신호 (mask 위치) 가 *균등하게 분포*

문장 경계는 분류·NLI 같은 downstream 에서 다시 명시적으로 입력됩니다 ([CLS] 토큰).

### Q7. (실무) 저장된 체크포인트가 너무 무거우면 어떻게 가볍게 하나요?

이 챕터의 작은 BERT 는 약 40MB 정도라 무겁지 않지만, 큰 모델의 경우:

```python
# 1. safetensors 형식 강제 (bin 보다 약간 작음 + 안전)
model.save_pretrained("./ch20_small_bert_mlm", safe_serialization=True)

# 2. fp16 으로 저장 (weight 자체를 half 로)
model.half().save_pretrained("./ch20_small_bert_mlm")

# 3. 양자화 (advanced — bitsandbytes 8-bit/4-bit)
# from transformers import BitsAndBytesConfig
# config = BitsAndBytesConfig(load_in_8bit=True)
```

이번 챕터는 *학습용 체크포인트* 이므로 fp32 그대로 저장 (Ch 21 fine-tune 시 정밀도 유지). 배포용이면 inference 단계에서 quantize 고려.

### Q8. (이론) 큰 BERT (110M) 와 비교해 이번 작은 BERT (10M) 의 *근본 한계* 는?

| 차원 | 작은 BERT (이번 챕터) | bert-base-uncased | 차이의 영향 |
|---|---|---|---|
| hidden_size | 256 | 768 | 표현 공간 차원이 1/3 → 미세한 의미 구분 어려움 |
| num_layers | 4 | 12 | *깊은* 추론 (구문 → 의미 → 문맥) 단계 부족 |
| 학습 데이터 | 5K paragraphs (약 700K-1M 토큰, Wikitext-103) | 33억 토큰 (BERT-base) | 어휘 다양성·문맥 풍부함 격차 약 5000배 |
| 학습 시간 | 20분 | 4 일 (TPU v3-256) | 압축한 *정보량* 자체가 다름 |

**결론**: 이번 챕터의 산출물은 *fine-tune 출발점으로는 random 보다 나음* 정도. *zero-shot* 또는 *복잡한 downstream* 에선 표준 BERT 와 비교 불가. *작은 모델 + 작은 데이터로도 일반 도메인 사전학습이 가능하다는 메커니즘* 을 *경험* 하는 게 이 챕터의 목적이고, *실용 모델* 은 표준 사전학습품을 가져다 쓰는 게 정답.""")

# ----- 19. 다음 챕터 -----
md(r"""## 다음 챕터 예고

**Chapter 21. 영어 BERT 분류 (Ch 20 사전학습 모델 fine-tune) — *일반 도메인 → 다른 도메인 transfer***

- 이번 챕터의 `./ch20_small_bert_mlm` 체크포인트를 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` 로 로드 → MLM head 떼고 분류 헤드 부착
- **Yelp 이진 분류 fine-tune** (Ch 10·11 과 같은 데이터·셋업) — *완전히 다른 도메인 transfer*. 일반 위키로 사전학습한 본체가 *영화 리뷰 도메인* 에 얼마나 잘 적응하는가 측정
- **핵심 비교**: 이번 작은 사전학습 BERT (약 10M params, Wikitext-103 5K paragraphs MLM) vs Ch 10 의 DistilBERT (약 66M params, 대규모 Wikipedia + BookCorpus 사전학습). 둘 다 *일반 도메인 → Yelp transfer* 라 비교가 *fair* — *사전학습 규모* 차이만 측정됨
- 작은 모델 + 작은 데이터 사전학습이 *얼마나 도움 되는가* 의 정량 측정 — fine-tune 학습 곡선·최종 accuracy·confusion matrix 모두 나란히
- 일부러 *random init* baseline (사전학습 없이 분류 직접) 도 함께 학습해 *사전학습의 순 효과* 분리

> **변하는 축**: Phase 3 안에서 *task 가 사전학습 (MLM) → 분류 (fine-tune)* 로 전환. 모델 구조·토크나이저는 그대로, 데이터 도메인이 *위키 → Yelp* 로 바뀌고 loss·평가 metric 이 분류 표준으로 돌아옴.""")

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
README = """# 20_en_bert_pretrain — 작은 BERT 직접 사전학습 (영어 MLM scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/20_en_bert_pretrain/20_en_bert_pretrain.ipynb)

## 한 줄 목표
Phase 3 의 두 번째 챕터. Ch 19 에서 *토크나이저* 를 직접 학습해 봤다면, 이번엔 **모델 본체를 random init 해 일반 도메인 MLM 사전학습** 합니다. 표준 BERT (110M) 의 1/10 크기 작은 BERT (약 10M, hidden=256/layer=4) 를 `BertConfig` 로 직접 설계, `bert-base-uncased` 의 WordPiece 토크나이저는 그대로 가져와 **Wikitext-103 paragraphs 5,000** (일반 도메인) 으로 MLM 사전학습 → 체크포인트 저장 → Ch 21 에서 *완전히 다른 도메인 (Yelp 영화 리뷰)* 이진 분류 fine-tune.

## 다루는 핵심 개념
- **MLM (Masked Language Modeling)** — 입력 토큰의 15% 를 `[MASK]` 로 가리고 원래 토큰을 맞추는 self-supervised task
- **일반 도메인 사전학습** — 원본 BERT 의 Wikipedia + BookCorpus 정신을 따라 Wikitext-103 본문 사용. task 도메인 (Yelp) 으로 학습하지 않아 *진정한 transfer* 측정 가능
- `transformers.BertConfig` 로 작은 BERT 직접 설계 (`hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024`)
- `BertForMaskedLM(config)` 로 *random init* (pretrained weight 없이) — `from_pretrained` 와 반대 흐름
- `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` — 매 batch 마다 동적 masking (80% [MASK] / 10% random / 10% keep)
- `group_texts` 패턴 (HF `run_mlm.py` 표준) — 가변 길이 텍스트를 고정 길이 `block_size=128` 블록 스트림으로
- MLM head 가 입력 임베딩과 *tied* — vocab 차원 출력이라 파라미터 절약
- random baseline loss `ln(vocab_size) ≈ 10.33`, perplexity 로 변환 가능 (`exp(loss)`)
- `[MASK]` top-5 예시 — 위키 도메인 (사전학습이 본 분포) + Yelp 도메인 (다른 도메인 transfer) 혼합 시연
- `model.save_pretrained()` / `tokenizer.save_pretrained()` 로 HF 표준 체크포인트 저장 — Ch 21 에서 `from_pretrained` 로 로드

## Loss
`CrossEntropyLoss` — 가려진 위치들의 *원래 토큰* 을 vocab 30,522 차원 softmax 로 예측. 라벨이 -100 인 위치는 자동 무시 (collator 가 처리).

수식: $L_{\\text{MLM}} = -\\frac{1}{|M|} \\sum_{i \\in M} \\log P(x_i \\mid x_{\\setminus M})$ — 가려진 토큰 위치 $M$ 에서의 평균 음의 로그 우도.

## 데이터
Wikitext-103 (일반 도메인) — `Salesforce/wikitext` config `wikitext-103-raw-v1` (CC-BY-SA, HF Hub 정제본). line 단위 정제 후 빈 줄 / 너무 짧은 줄 (50자 미만) / 너무 긴 줄 (2000자 초과) 제외. train 5,000 / eval 500 paragraphs, seed 42.

`block_size=128` 로 `group_texts` 후 train 약 1,000-2,000 블록 / eval 약 100-200 블록. Ch 22 (한국어 Wikipedia) 와 같은 *일반 위키 패턴*.

## 환경
Google Colab T4 GPU (fp16). 약 20-25분 (`bert-base-uncased` 토크나이저 로드 + Wikitext-103 다운로드 + 5K paragraphs 필터링·토큰화 약 3분 + MLM 2 epoch 약 15-20분 + 평가/저장).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 17 | klue/bert-base | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | `BCEWithLogitsLoss + λ·L_aux` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — |
| **20** | **작은 BERT (직접, scratch)** | **`bert-base-uncased` 토크나이저 (가져옴)** | **Wikitext-103 paragraphs (일반 도메인)** | **MLM head** | **`CrossEntropyLoss` (masked)** |
| 21 (다음) | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 (다른 도메인 transfer) | `Linear(H, 2)` | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 산출물
`./ch20_small_bert_mlm/` 폴더에 `config.json + model.safetensors + tokenizer.json + vocab.txt + ...` 저장. Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 한 줄로 *encoder body* 를 가져와 새 분류 헤드를 부착해 fine-tune.

## 다음 챕터
[21_en_bert_classify](../21_en_bert_classify/) — 이번 챕터 사전학습 모델을 *완전히 다른 도메인 (Yelp 영화 리뷰)* 이진 분류로 fine-tune. **Ch 10 (DistilBERT 대규모 Wikipedia + BookCorpus 사전학습 모델 fine-tune) 과 직접 비교** — 둘 다 *일반 도메인 → Yelp transfer* 라 비교가 fair, 작은 사전학습 BERT (약 10M, Wikitext-103 5K paragraphs MLM) vs 표준 사전학습 BERT (약 66M, 대규모 corpus) 의 정량 격차가 *사전학습 규모 차이만* 측정. random init baseline 도 함께 학습해 *사전학습의 순 효과* 분리.
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}  ({len(README.splitlines())} lines)")
