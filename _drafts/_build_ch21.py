"""Build 21_en_bert_classify/21_en_bert_classify.ipynb — Phase 3, scratch MLM → 분류.

Ch 20 의 작은 BERT scratch MLM 사전학습 패턴을 *압축 재현* 한 뒤, 같은 본체 위에
분류 헤드를 얹어 Yelp 이진 분류로 fine-tune. Ch 10 (DistilBERT 대규모 사전학습)
과 직접 비교해 *사전학습 규모* 가 분류 정확도에 얼마나 영향 주는지 보입니다.
self-contained — Ch 20 체크포인트에 의존하지 않고 노트북 안에서 MLM 부터 시작.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "21_en_bert_classify"
OUT_NB = OUT_DIR / "21_en_bert_classify.ipynb"
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
md(r"""# Chapter 21. 작은 BERT 분류 — 영어 Yelp 이진 (scratch 사전학습 + fine-tune)

**목표**: Phase 3 의 세 번째 챕터. Ch 20 에서 *작은 BERT 를 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 fine-tune** 합니다. Ch 10 (DistilBERT, ~66M params, 수십억 토큰 사전학습) 과 같은 Yelp 이진 분류 셋업에 *우리가 만든 작은 BERT* (~10M params, Yelp 5K 문장 MLM) 를 붙여 두 결과를 나란히 비교 — *사전학습 규모* 가 downstream 정확도에 얼마나 차이를 만드는지 정량으로 확인.

self-contained 노트북: Ch 20 의 MLM 학습을 1 epoch 짧게 재현 → 같은 본체로 분류 fine-tune → Ch 10 결과와 비교. *MLM 없이 random init 으로 바로 분류* 하는 baseline 도 변형 셀에서 함께 학습해 *사전학습 자체의 순 효과* 도 분리.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 25분 (MLM 1 epoch 약 10-12분 + 분류 fine-tune 2 epoch 약 8-10분 + 평가 약 2분)

---

## 학습 흐름

1. 🚀 **데이터**: `fancyzhx/yelp_polarity` 이진 분류 (Ch 10 과 같은 5K/1K split, seed 42)
2. 🔤 **토크나이저**: `bert-base-uncased` (Ch 20 과 동일)
3. 🏗️ **MLM 사전학습 재현 (Ch 20 압축본)**: 같은 작은 BertConfig 로 1 epoch 만 짧게
4. 🔀 **헤드 교체**: `BertForMaskedLM` → `BertForSequenceClassification(num_labels=2)`. 본체는 그대로, MLM head 떼고 분류 head 부착
5. 🚀 **분류 fine-tune**: Trainer fp16, 2 epoch
6. 🔬 **평가**: accuracy / precision / recall / F1 / AUC (Ch 10 과 같은 5종)
7. 🆚 **Ch 10 vs Ch 21 비교 표**: 정확도, 모델 크기, 사전학습 토큰량
8. 🛠️ **변형 — random init baseline**: MLM 건너뛰고 바로 분류 fine-tune. 사전학습의 순 효과 정량

---

> 📒 **사전 학습 자료**: Ch 20 (작은 BERT scratch MLM), Ch 10 (DistilBERT 사전학습 + Yelp 이진 분류). Ch 21 은 두 챕터를 *합쳐서* — Ch 20 의 사전학습 흐름 그대로 + Ch 10 의 fine-tune 평가 그대로.""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 10 | DistilBERT 파인튜닝 (~66M) | `bert-base-uncased` WordPiece | Yelp 이진 (4-5 → 1, 1-2 → 0) | `Linear(H, 1)` | sigmoid | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | sigmoid + 태스크별 | `BCEWithLogitsLoss + λ·L_aux` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Yelp text (라벨 무시) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |
| **21 ← 여기** | **Ch 20 사전학습 BERT + 분류 헤드 (~10M)** | (Ch 20과 동일) | **Yelp 이진화** | **`Linear(H, 2)`** | **softmax** | **`CrossEntropyLoss`** |
| 22 (다음) | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | NSMC text (라벨 무시) | MLM head | softmax (MLM) | `CrossEntropyLoss` (masked token) |

전체 챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.

**Phase 3 안에서의 위치** — Ch 19 (토크나이저 학습) → Ch 20 (모델 사전학습) → **Ch 21 (분류 fine-tune)** → Ch 22 (한국어 사전학습) → Ch 23 (한국어 분류). Ch 21 이 Phase 3 의 *영어 절반 마무리* — Ch 10 비교가 클라이맥스.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 20)

| 축 | Ch 20 (작은 BERT scratch MLM) | Ch 21 (작은 BERT 분류 fine-tune) |
|---|---|---|
| **이 챕터의 task** | MLM 사전학습 (masked token 예측) | **이진 분류 (긍정/부정)** ← *유일한 변화* |
| 모델 클래스 | `BertForMaskedLM` | **`BertForSequenceClassification(num_labels=2)`** |
| 본체 (embedding + encoder) | random init → MLM 학습 | **Ch 20 사전학습 본체 그대로 이어받음** |
| 출력 헤드 | MLM head (vocab 30,522 차원) | **분류 head (`Linear(256, 2)`)** ← 새 random init |
| 토크나이저 | `bert-base-uncased` (vocab 30,522) | (그대로) |
| 데이터 | Yelp text (라벨 무시) | **Yelp 이진** (라벨 사용) |
| Loss | `CrossEntropyLoss` (vocab 30,522 logits) | **`CrossEntropyLoss` (2 logits)** ← K 만 큰 변화 |
| 학습률 | 5e-4 (scratch 사전학습) | **2e-5** (fine-tune, 표준) |

> **변경점 한 가지 원칙** — Phase 3 안에서 *task 축* 이 변합니다 (MLM → 분류). 모델 본체·토크나이저·데이터(텍스트 자체) 는 그대로, 헤드와 라벨 형식만 바뀝니다. 이게 *사전학습-fine-tune 패러다임* 의 핵심: 본체는 한 번 학습한 표상을 재사용, downstream 마다 *작은 헤드 + 작은 학습률* 로 적응.

### Ch 10 (DistilBERT) 과의 비교가 본 챕터의 메인 메시지

| 차원 | Ch 10 (DistilBERT) | Ch 21 (이 챕터) | 비고 |
|---|---|---|---|
| 본체 파라미터 | ~66M | **~10M** | Ch 21 은 1/6 작음 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (~33억 토큰) | **Yelp 5K 문장 (~70만 토큰)** | 약 5000배 격차 |
| 사전학습 시간 | TPU 수일 (대규모 인프라) | **T4 약 10분** | |
| 분류 fine-tune 셋업 | Ch 10 = 이번 챕터 동일 (같은 데이터, 같은 hyperparams) | | 변하는 건 *본체 출발점* 뿐 |
| 기대 accuracy | ~92-95% | **~75-85% 예상** | 비교는 실측치로 확인 |

이 격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다. 동시에 *작은 사전학습도 random init 보다는 낫다* 는 것을 변형 셀에서 추가 확인.""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss 함수의 변화 — MLM CE (vocab=30,522) → 분류 CE (K=2)

Ch 20 의 MLM 도 본질은 *vocab 위에서의 다중 분류* 였습니다. 다만 K = vocab_size = 30,522 라 어려운 task. 이번 챕터는 K = 2 의 *훨씬 쉬운* 분류 task.

### 수식

분류 task 의 CE 는 Ch 11 과 같습니다 (K=2):

$$L_{\text{cls}} = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i, y_i}$$

- $\hat p_{i, k} = \mathrm{softmax}(z_i)_k$ — K=2 차원 softmax
- $y_i \in \{0, 1\}$ — 정수 라벨

### 두 CE 비교 (random baseline)

| task | K | random baseline loss $\log K$ | 학습 어려움 |
|---|---|---|---|
| MLM (Ch 20) | 30,522 | **10.33** | 매우 어려움 — 가려진 토큰 자리에 *vocab 전체 후보* 중 정답을 |
| 분류 (Ch 21) | 2 | **0.693** | 상대적으로 쉬움 — 긍정/부정 둘 중 하나 |

학습 첫 step 의 loss 가 ~0.693 부근이면 모델이 *균등 추측* 단계. fine-tune 첫 step 에서 분류 헤드만 새로 init 됐으므로 *이 정도* 가 정상.

### 사전학습 효과가 *loss 곡선* 에 어떻게 드러나나

| 셋업 | 학습 첫 step loss | 학습 종료 loss (epoch 2) | 메모 |
|---|---|---|---|
| random init + 분류 (변형 셀) | ~0.693 | ~0.5-0.6 | 본체도 분류 헤드도 random — 학습이 *느림* |
| Ch 20 MLM 사전학습 본체 + 분류 (메인) | ~0.693 | **~0.3-0.5** | 본체에 *언어 구조* 가 들어 있어 헤드만 빠르게 적응 |
| Ch 10 DistilBERT 사전학습 본체 + 분류 | ~0.693 | **~0.15-0.25** | 대규모 사전학습이 만든 표상의 위력 |

random baseline 은 *세 셋업 모두 같음* — 사전학습이 *학습 속도* 와 *수렴점* 에 영향. 학습 첫 step loss 가 같다고 사전학습이 의미 없는 게 아닙니다.

> **숫자로 감 잡기** (K=2, 정답 = 클래스 1):
> | logits $(z_0, z_1)$ | softmax → $\hat p_1$ | 손실 |
> |---|---|---|
> | (0, 0) | 0.5 | **0.693** ← random |
> | (-1, +1) | 0.881 | 0.127 |
> | (-2, +2) | 0.982 | 0.018 |
> | (+2, -2) | 0.018 | **4.018** ← 자신 있게 틀림 |""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

Ch 20 과 *완전히 동일* — `AutoTokenizer.from_pretrained("bert-base-uncased")`, vocab 30,522 영어 WordPiece. 사전학습-fine-tune 패러다임의 핵심: **토크나이저는 사전학습 ~ 분류 전 구간에서 동일** 해야 함. 그래야 본체가 학습한 토큰 임베딩이 그대로 의미를 유지.

### 분류 task 에서 [CLS] 토큰의 의미

MLM 사전학습 (Ch 20) 에서는 `group_texts` 패턴으로 *특수 토큰 없이* 토큰 스트림을 잘랐습니다. 분류 fine-tune 에서는 *문장 단위* 입력이라 표준 BERT 포맷:

```
[CLS] the food was excellent and the service was great [SEP]
```

- `[CLS]` 의 최종 hidden state $h_{[CLS]} \in \mathbb{R}^{256}$ 가 *문장 표상*. 분류 헤드 `Linear(256, 2)` 가 이 위에 얹힘.
- MLM 학습 중에는 `[CLS]` 의 hidden 이 *암묵적* 으로만 학습됨 (옆 토큰들과 attention 공유). 분류 fine-tune 단계에서 *이 자리* 가 본격 활용.

### 헤드 교체 시 어떤 파라미터가 어떻게 이어지나

| 모델 부분 | Ch 20 학습 끝 → Ch 21 시작 | 운명 |
|---|---|---|
| 임베딩 (vocab 30,522 × hidden 256) | 사전학습으로 *언어 구조* 학습 | **그대로 이어받음** |
| Encoder 4 layer (attention + FFN) | MLM 으로 *문맥 의존 표상* 학습 | **그대로 이어받음** |
| MLM head (`cls.predictions`) | vocab 위 분류 헤드 | **버려짐** |
| 분류 head (`classifier`, `Linear(256, 2)`) | (없었음) | **새로 random init** ← fine-tune 으로 학습 |

> Ch 10 의 DistilBERT 가 같은 흐름 (MLM 사전학습 → 분류 fine-tune) 을 *큰 규모* 로 거친 결과. 우리도 같은 흐름을 *작은 규모* 로 직접 거칩니다.""")

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
    BertForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score, confusion_matrix,
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
    print("Warning: CPU runtime — both MLM and classification will be very slow. Switch to T4 recommended.")""")

md(r"""**baseline VRAM** (CUDA 환경에서만 의미 있는 출력 — Colab T4 기준):""")
code(r"""!nvidia-smi""")

# ----- 7. 데이터 로드 -----
md(r"""## 1. 📥 Yelp 이진 분류 데이터 로드 — Ch 10 과 같은 split

`fancyzhx/yelp_polarity` 는 *이미 이진화된* (긍정/부정) 5점 척도 yelp 리뷰 데이터셋. Ch 10 의 `Yelp/yelp_review_full` + 별점 이진화 와 *완전히 같은 형태* 의 결과가 나오도록 같은 seed·같은 sample 수를 적용. **5,000 train / 1,000 eval, seed 42**.""")

code(r"""SEED = 42
N_TRAIN = 5000
N_EVAL = 1000

ds_raw = load_dataset("fancyzhx/yelp_polarity")
print(f"splits: {list(ds_raw.keys())}")
print(f"train size: {len(ds_raw['train']):,}")
print(f"test size:  {len(ds_raw['test']):,}")
print(f"label names: {ds_raw['train'].features['label'].names}")

# Ch 10 과 동일한 seed·크기로 sample
ds_train_full = ds_raw["train"].shuffle(seed=SEED).select(range(N_TRAIN))
ds_eval_full  = ds_raw["test"].shuffle(seed=SEED).select(range(N_EVAL))

# 클래스 분포
train_labels = np.array(ds_train_full["label"])
eval_labels  = np.array(ds_eval_full["label"])
print(f"\nsampled train: {len(ds_train_full):,}")
print(f"  positive rate: {train_labels.mean():.1%}  (label 1)")
print(f"sampled eval:  {len(ds_eval_full):,}")
print(f"  positive rate: {eval_labels.mean():.1%}  (label 1)")

print(f"\nfirst train sample:")
print(f"  label: {ds_train_full[0]['label']} ({ds_raw['train'].features['label'].names[ds_train_full[0]['label']]})")
print(f"  text:  {ds_train_full[0]['text'][:200]}...")""")

# ----- 8. 토크나이저 로드 -----
md(r"""## 2. 🔤 토크나이저 — `bert-base-uncased` (Ch 20 과 동일)

vocab 30,522 의 영어 WordPiece. MLM 사전학습과 분류 fine-tune 전 구간에서 *같은 토크나이저* 를 써야 본체가 학습한 임베딩의 의미가 유지됩니다.""")

code(r"""TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

print(f"tokenizer:        {TOKENIZER_NAME}")
print(f"vocab_size:       {tokenizer.vocab_size:,}")
print(f"model_max_length: {tokenizer.model_max_length}")

# 분류 입력 예시
SAMPLE = "The food was unforgettable and the service was excellent."
enc = tokenizer(SAMPLE, return_tensors="pt", truncation=True, max_length=128)
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
print(f"\nsample: {SAMPLE!r}")
print(f"tokens ({len(tokens)}): {tokens}")""")

# ----- 9. MLM 사전학습 (Ch 20 압축본) -----
md(r"""## 3. 🏗️ MLM 사전학습 — Ch 20 패턴 압축 재현 (1 epoch)

이 노트북을 *self-contained* 로 만들기 위해 Ch 20 의 MLM 사전학습을 여기서 짧게 재현합니다. Ch 20 보다 *짧은 1 epoch* (시간 단축) 라 사전학습 깊이는 얕지만, *random init 보다는 낫다* 는 차이를 만들기에는 충분합니다.

같은 작은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024) → `BertForMaskedLM(config)` random init → Yelp text 5K 문장 MLM 1 epoch.""")

code(r"""# Ch 20 과 같은 작은 BERT 설정
HIDDEN_SIZE         = 256
NUM_HIDDEN_LAYERS   = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE   = 1024
MAX_POS_EMBED       = 128
BLOCK_SIZE          = 128

mlm_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
)

mlm_model = BertForMaskedLM(mlm_config)  # random init
total = sum(p.numel() for p in mlm_model.parameters())
print(f"Small BERT config: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, head={NUM_ATTENTION_HEADS}")
print(f"Total parameters:  {total:,}  ({total/1e6:.2f} M)")""")

code(r"""# MLM 학습용 데이터셋: text 만 추출 (라벨 무시 — self-supervised)
mlm_train_raw = ds_train_full.remove_columns([c for c in ds_train_full.column_names if c != "text"])
mlm_eval_raw  = ds_eval_full.remove_columns([c for c in ds_eval_full.column_names if c != "text"])

def mlm_tokenize(examples):
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

mlm_tokenized_train = mlm_train_raw.map(mlm_tokenize, batched=True, remove_columns=["text"])
mlm_tokenized_eval  = mlm_eval_raw.map(mlm_tokenize,  batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result

lm_train = mlm_tokenized_train.map(group_texts, batched=True, batch_size=1000)
lm_eval  = mlm_tokenized_eval.map(group_texts,  batched=True, batch_size=1000)

print(f"MLM train blocks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")
print(f"MLM eval blocks:  {len(lm_eval):,}")""")

code(r"""mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

USE_FP16 = (DEVICE == "cuda")
MLM_EPOCHS = 1   # Ch 20 의 1-2 epoch 중 짧은 쪽으로 (분류 fine-tune 시간 확보)

mlm_args = TrainingArguments(
    output_dir="./ch21_mlm_output",
    num_train_epochs=MLM_EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    data_collator=mlm_collator,
    processing_class=tokenizer,
)

print(f"MLM epochs:     {MLM_EPOCHS}")
print(f"MLM batch size: {mlm_args.per_device_train_batch_size}")
print(f"MLM learning rate: {mlm_args.learning_rate}")
print(f"MLM fp16:       {USE_FP16}")
print(f"MLM steps:      {len(lm_train) // mlm_args.per_device_train_batch_size * MLM_EPOCHS}")""")

code(r"""t0 = time.time()
mlm_result = mlm_trainer.train()
mlm_elapsed = time.time() - t0
print(f"\nMLM pretraining done in {mlm_elapsed/60:.1f} min")
print(f"mean train loss: {mlm_result.training_loss:.4f}")
print(f"random baseline (ln vocab): {math.log(tokenizer.vocab_size):.4f}")""")

code(r"""mlm_eval_metrics = mlm_trainer.evaluate()
mlm_eval_loss = mlm_eval_metrics["eval_loss"]
print(f"MLM eval loss:        {mlm_eval_loss:.4f}")
print(f"MLM eval perplexity:  {math.exp(mlm_eval_loss):.2f}")
print(f"(random baseline PPL: {tokenizer.vocab_size:,})")""")

md(r"""**관전 포인트** — MLM loss 가 *random baseline 10.33* 에서 시작해 4-6 부근까지 떨어졌다면 본체가 *언어 구조의 일부* 를 학습한 상태. perplexity 로 환산하면 vocab 30,522 중 *수백 개 후보* 로 좁혀진 정도. Ch 20 의 2 epoch 보다는 약간 얕지만, 분류 fine-tune 출발점으로는 충분합니다.

> **체크포인트 저장은 생략** — 노트북 안에서 바로 본체 가중치를 분류 모델로 옮기기 때문. Ch 20 처럼 디스크에 저장하려면 `mlm_model.save_pretrained("./ch21_mlm_ckpt")` 한 줄.""")

# ----- 10. 헤드 교체 + 분류 fine-tune -----
md(r"""## 4. 🔀 헤드 교체 — MLM → 분류 + Fine-tune

이제 *방금 학습된 작은 BERT 본체* 를 분류 모델로 옮깁니다. 두 가지 흐름:

1. `BertForMaskedLM.bert` (embedding + encoder) 를 그대로 가져옴
2. 새 `BertForSequenceClassification(config)` 을 만들고, 1 의 본체를 *복사*. 분류 헤드는 새로 random init

이렇게 만든 모델을 같은 Yelp 데이터의 *라벨* 까지 사용해 분류 fine-tune. Ch 10 의 hyperparams 와 *완전히 같이* (`lr=2e-5, batch=16, epoch=2, fp16=True`) 둬서 *본체 출발점* 외 모든 조건을 통제.""")

code(r"""# 분류용 config: 같은 본체 구조 + num_labels=2 + problem_type
cls_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
    num_labels=2,
    problem_type="single_label_classification",
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)

cls_model = BertForSequenceClassification(cls_config)

# MLM 본체 (embeddings + encoder) 를 분류 모델로 *복사* — pooler 까지 같이
missing, unexpected = cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
print(f"본체 가중치 복사 완료")
print(f"  missing keys (분류 측에만 있는 부분): {len(missing)}  e.g. {missing[:3] if missing else []}")
print(f"  unexpected keys (MLM 측 잉여):       {len(unexpected)}  e.g. {unexpected[:3] if unexpected else []}")

# 파라미터 수 비교
total_cls = sum(p.numel() for p in cls_model.parameters())
total_body = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" not in n)
total_head = sum(p.numel() for n, p in cls_model.named_parameters() if "classifier" in n)
print(f"\nClassification model parameters:")
print(f"  body (embeddings + encoder + pooler): {total_body:>10,}  ({total_body/total_cls:.1%})")
print(f"  classifier head Linear(256, 2):       {total_head:>10,}  ({total_head/total_cls:.1%})")
print(f"  total:                                 {total_cls:>10,}  ({total_cls/1e6:.2f} M)")""")

md(r"""**`bert.load_state_dict` 가 한 일** — `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* (이름 `self.bert`) 을 갖습니다. 그 본체만 통째로 옮긴 셈. MLM head (`cls.predictions`) 와 분류 head (`classifier`) 는 *모델 객체의 다른 자리* 라 자동으로 분리됩니다.

> Ch 7-18 의 `AutoModelForSequenceClassification.from_pretrained(...)` 가 디스크에서 같은 일을 합니다. 우리는 *방금 학습한 본체* 를 디스크 없이 in-memory 로 옮긴 셈.""")

code(r"""# 분류용 토큰화 — 문장 단위, [CLS]/[SEP] 부착, max_length=128
def cls_tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

cls_train = ds_train_full.map(cls_tokenize, batched=True).remove_columns(
    [c for c in ds_train_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
cls_eval  = ds_eval_full.map(cls_tokenize,  batched=True).remove_columns(
    [c for c in ds_eval_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)

print(cls_train)
print(f"\nFirst sample label: {cls_train[0]['labels']}  (int 0 or 1)")""")

code(r"""def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=2)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    preds = probs_full.argmax(axis=1)
    probs_pos = probs_full[:, 1]   # 클래스 1 의 확률 = AUC 입력

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs_pos)),
    }""")

code(r"""# Ch 10 과 같은 hyperparams — 변하는 건 *본체 출발점* 뿐
cls_args = TrainingArguments(
    output_dir="./ch21_cls_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

cls_trainer = Trainer(
    model=cls_model,
    args=cls_args,
    train_dataset=cls_train,
    eval_dataset=cls_eval,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

t0 = time.time()
cls_result = cls_trainer.train()
cls_elapsed = time.time() - t0
print(f"\nClassification fine-tune done in {cls_elapsed/60:.1f} min")
print(f"mean train loss: {cls_result.training_loss:.4f}")
print(f"random baseline (ln 2): {math.log(2):.4f}")""")

code(r"""!nvidia-smi""")

# ----- 11. 평가 -----
md(r"""## 5. 🔬 평가 — Ch 10 과 같은 5종 metric + 학습 곡선

`accuracy / precision / recall / F1 / AUC` 전부 같은 정의. 마지막에 confusion matrix 와 학습 곡선을 같이 그려 *본체 출발점 변화가 학습 동역학에 어떻게 드러나는지* 시각화.""")

code(r"""cls_eval_metrics = cls_trainer.evaluate()
print("Ch 21 small BERT (scratch MLM 1 epoch + classification fine-tune) — eval:")
for k, v in cls_eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")""")

code(r"""preds_output = cls_trainer.predict(cls_eval)
cls_logits = preds_output.predictions
cls_labels = preds_output.label_ids.astype(int)

exp = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
cls_probs_full = exp / exp.sum(axis=1, keepdims=True)
cls_preds = cls_probs_full.argmax(axis=1)
cls_probs_pos = cls_probs_full[:, 1]

print(f"Logits shape: {cls_logits.shape}")
print(f"Predicted positive rate: {(cls_preds == 1).mean():.1%}")
print(f"Top-1 prob mean: correct={cls_probs_full.max(axis=1)[cls_preds == cls_labels].mean():.4f}, "
      f"wrong={cls_probs_full.max(axis=1)[cls_preds != cls_labels].mean():.4f}")
print()
print(classification_report(
    cls_labels, cls_preds,
    target_names=["negative", "positive"],
    digits=4, zero_division=0,
))""")

md(r"""### 5-1. 학습 곡선 — MLM 사전학습 효과가 보이는 자리

분류 fine-tune 의 step-by-step train loss 를 그려, *시작점* 과 *수렴점* 을 같이 확인.""")

code(r"""log_history = cls_trainer.state.log_history
train_logs = [(e["step"], e["loss"]) for e in log_history if "loss" in e and "eval_loss" not in e]

if train_logs:
    steps, losses = zip(*train_logs)
    random_baseline = math.log(2)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, "o-", color="#4878D0", label="train CE loss (small BERT)")
    ax.axhline(random_baseline, color="black", lw=1.0, ls=":",
               label=f"random baseline (ln 2 = {random_baseline:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("CE loss (binary)")
    ax.set_title("Classification fine-tune loss — small BERT (Ch 20 MLM body)")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No train loss logs found.")""")

md(r"""### 5-2. Confusion matrix""")

code(r"""sns.set_theme(style="white", context="talk")
cm = confusion_matrix(cls_labels, cls_preds, labels=[0, 1])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm, annot=cm, fmt="d",
    cmap="Blues", vmin=0, vmax=1,
    xticklabels=["negative", "positive"],
    yticklabels=["negative", "positive"],
    cbar_kws={"label": "row-normalized (recall)"}, ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Ch 21 small BERT — Confusion Matrix")
plt.tight_layout()
plt.show()""")

# ----- 12. Ch 10 비교 -----
md(r"""## 6. 🆚 Ch 10 (DistilBERT) vs Ch 21 (작은 BERT scratch) — 본 챕터의 핵심 결과

같은 데이터·같은 hyperparams 에 *본체 출발점만 다른* 두 셋업의 정확도 비교. Ch 10 의 수치는 본 챕터를 작성하는 시점에 *해당 노트북의 README/실행 결과* 를 참고해 인용 — 학습자가 노트북을 돌려 본인 수치로 갱신해 보면 더 좋습니다.

| 차원 | Ch 10 (DistilBERT pretrained) | Ch 21 (작은 BERT scratch + 1 epoch MLM) | 비고 |
|---|---|---|---|
| 본체 파라미터 | ~66M | ~10M | Ch 21 은 1/6 크기 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (~33억 토큰) | Yelp 5K 문장 (~70만 토큰) | 약 5000배 격차 |
| 사전학습 시간 | TPU 수일 | T4 약 10-12분 | |
| 분류 fine-tune 셋업 | (같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | | 본체 외 통제 |""")

code(r"""# Ch 10 reference 수치 — yelp_polarity 5K/1K + DistilBERT fine-tune 2 epoch 의 *전형적* 결과
# (실측치는 학습자가 Ch 10 노트북을 돌려 본인 값으로 갱신 권장)
CH10_REFERENCE = {
    "accuracy":  0.93,
    "precision": 0.93,
    "recall":    0.93,
    "f1":        0.93,
    "auc":       0.98,
}

ch21_metrics = {k.replace("eval_", ""): v for k, v in cls_eval_metrics.items()
                if k.startswith("eval_") and isinstance(v, float)
                and k.replace("eval_", "") in CH10_REFERENCE}

comparison = pd.DataFrame({
    "metric":              list(CH10_REFERENCE.keys()),
    "Ch10 DistilBERT (ref)": [CH10_REFERENCE[k] for k in CH10_REFERENCE.keys()],
    "Ch21 small BERT":     [ch21_metrics.get(k, float("nan")) for k in CH10_REFERENCE.keys()],
})
comparison["delta (Ch21 - Ch10)"] = comparison["Ch21 small BERT"] - comparison["Ch10 DistilBERT (ref)"]
print("Ch10 vs Ch21 — classification metrics")
print(comparison.round(4).to_string(index=False))""")

code(r"""# bar chart 로 한눈에 보기
sns.set_theme(style="whitegrid", context="talk")
plot_df = comparison.melt(
    id_vars=["metric"],
    value_vars=["Ch10 DistilBERT (ref)", "Ch21 small BERT"],
    var_name="model", value_name="score",
)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=plot_df, x="metric", y="score", hue="model",
    palette={"Ch10 DistilBERT (ref)": "#4878D0", "Ch21 small BERT": "#EE854A"},
    ax=ax,
)
ax.set_ylim(0, 1.05)
ax.set_title("Yelp binary classification — Ch10 vs Ch21")
ax.set_xlabel("metric")
ax.set_ylabel("score")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.show()""")

md(r"""**관찰 — *5000배 사전학습 격차* 가 분류 정확도에 어떻게 드러나나**

전형적으로:
- Ch 10 (DistilBERT): accuracy ~92-95%, AUC ~0.97-0.99
- Ch 21 (작은 BERT): accuracy ~75-85%, AUC ~0.85-0.92

**accuracy 10-15%p 격차** 가 나옵니다. 이게 *사전학습 규모의 가치* — Wikipedia + BookCorpus 의 *일반 영어 지식* 이 DistilBERT 본체에 압축되어 있어, Yelp 분류 같은 *처음 보는 도메인* 에도 빠르게 적응합니다.

> 한편 Ch 21 의 accuracy 가 *random (50%) 보다 훨씬 높다* 는 것도 중요한 결과입니다. 작은 사전학습 + 작은 모델로도 *기본 신호* (긍정/부정 단어들의 통계) 는 잡힙니다. 다음 변형 셀에서 *MLM 없이 random init* 으로 바로 분류했을 때 얼마나 더 떨어지는지 확인.""")

# ----- 13. 변형 -----
md(r"""## 🛠️ 변형 — MLM 사전학습 없이 random init 으로 바로 분류 fine-tune

*사전학습 자체의 순 효과* 를 측정. 위와 *완전히 같은* 분류 셋업이지만 본체를 *random init 그대로* (MLM 학습 없음) 사용. 두 결과 차이가 *Ch 20 의 MLM 사전학습이 분류 성능에 얼마나 도움 됐는지* 의 측정치.

| 셋업 | 본체 출발점 | 비교 |
|---|---|---|
| **메인** (위) | Ch 20 패턴 MLM 1 epoch 학습 → 본체 복사 | *사전학습 있음* |
| **변형** (이 셀) | random init 그대로 (`BertForSequenceClassification(config)`) | *사전학습 없음* — pure baseline |

같은 데이터·같은 hyperparams·같은 seed.""")

code(r"""# 같은 config 의 *fresh* random init 분류 모델
random_cls_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
    num_labels=2,
    problem_type="single_label_classification",
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)
random_cls_model = BertForSequenceClassification(random_cls_config)

random_args = TrainingArguments(
    output_dir="./ch21_random_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=USE_FP16,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

random_trainer = Trainer(
    model=random_cls_model,
    args=random_args,
    train_dataset=cls_train,
    eval_dataset=cls_eval,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

t0 = time.time()
random_result = random_trainer.train()
random_elapsed = time.time() - t0
print(f"\nRandom init classification done in {random_elapsed/60:.1f} min")
print(f"mean train loss: {random_result.training_loss:.4f}")""")

code(r"""random_eval_metrics = random_trainer.evaluate()
print("Random init small BERT (no MLM pretraining) — eval:")
for k, v in random_eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")""")

code(r"""# 세 결과 한꺼번에 비교
random_metrics = {k.replace("eval_", ""): v for k, v in random_eval_metrics.items()
                  if k.startswith("eval_") and isinstance(v, float)
                  and k.replace("eval_", "") in CH10_REFERENCE}

three_way = pd.DataFrame({
    "metric":              list(CH10_REFERENCE.keys()),
    "Ch10 DistilBERT (ref)": [CH10_REFERENCE[k] for k in CH10_REFERENCE.keys()],
    "Ch21 small BERT + MLM": [ch21_metrics.get(k, float("nan")) for k in CH10_REFERENCE.keys()],
    "Ch21 random init":    [random_metrics.get(k, float("nan")) for k in CH10_REFERENCE.keys()],
})
print("Three-way comparison — pretraining effect")
print(three_way.round(4).to_string(index=False))""")

code(r"""# 세 모델 bar chart
sns.set_theme(style="whitegrid", context="talk")
plot_df3 = three_way.melt(
    id_vars=["metric"],
    value_vars=["Ch10 DistilBERT (ref)", "Ch21 small BERT + MLM", "Ch21 random init"],
    var_name="model", value_name="score",
)

fig, ax = plt.subplots(figsize=(10, 5.5))
sns.barplot(
    data=plot_df3, x="metric", y="score", hue="model",
    palette={
        "Ch10 DistilBERT (ref)": "#4878D0",
        "Ch21 small BERT + MLM": "#EE854A",
        "Ch21 random init":    "#999999",
    },
    ax=ax,
)
ax.set_ylim(0, 1.05)
ax.set_title("Pretraining effect — DistilBERT vs small BERT (MLM) vs random init")
ax.set_xlabel("metric")
ax.set_ylabel("score")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()""")

md(r"""**해석 — 세 셋업의 정렬**

1. **Ch 10 DistilBERT** — 대규모 사전학습. 최고 성능.
2. **Ch 21 small BERT + MLM** — 작은 사전학습 (5K 문장, 1 epoch). 중간 성능. *random 보다 분명히 나음*.
3. **Ch 21 random init** — 사전학습 없음. 최저 성능. *작은 모델 + 작은 데이터로 처음부터 분류* 는 일반적으로 *수렴이 느림* 또는 *수렴해도 낮은 정확도*.

`(2) - (3)` 차이가 *Ch 20 의 MLM 사전학습이 만든 가치*. `(1) - (2)` 차이가 *대규모 vs 작은 사전학습의 격차*. **두 격차가 비슷한 크기로 나오면 "데이터 규모 5000배 격차" 와 "사전학습 유무" 가 비슷한 영향력** 이라는 메시지.

> **주의 — 작은 모델 + 작은 데이터의 분산** — Ch 21 의 두 셋업 (MLM vs random) 사이 격차가 *seed 에 따라 변동* 이 큽니다. 정확한 비교를 위해선 seed 3-5개 평균이 더 신뢰 가능. 이번 챕터는 *방향성* 만 보는 데 의의.""")

# ----- 14. 등장한 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `transformers.BertForSequenceClassification` | encoder + 분류 head, 분류 fine-tune 전용 | Ch 23 한국어 분류 |
| `BertForSequenceClassification(config)` (random init) | pretrained weight 없이 모델 생성 | Ch 23 같음 (random baseline) |
| `model.bert.load_state_dict(other.bert.state_dict())` | 본체만 통째로 옮기는 in-memory 헤드 교체 | Ch 23 같음 |
| `transformers.BertForMaskedLM` (재등장) | MLM 사전학습 (Ch 20 압축 재현) | Ch 22 한국어 MLM |
| `sklearn.metrics.precision_recall_fscore_support(..., average="binary")` | 이진 분류 metric 한 묶음 | Ch 23 동일 |
| `sklearn.metrics.roc_auc_score` | AUC | Ch 23 동일 |""")

# ----- 15. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. `BertForMaskedLM` 과 `BertForSequenceClassification` 둘 다 *내부에 같은 `BertModel`* 을 갖습니다. 두 모델 사이에서 *어떤 파라미터* 가 이어지고 *어떤 파라미터* 가 새로 학습되나요?
2. MLM 학습 첫 step 의 loss 가 약 10.33 인 반면, 분류 fine-tune 첫 step 의 loss 는 약 0.693 입니다. 이 *4 배 차이* 가 모델의 학습 어려움 차이를 의미하나요? (힌트: K=vocab_size vs K=2)
3. Ch 21 의 작은 BERT 가 Ch 10 의 DistilBERT 보다 *낮은 정확도* 를 보입니다. 이 격차가 (a) *모델 크기* 차이 (~10M vs ~66M), (b) *사전학습 데이터 양* 차이 (~70만 토큰 vs ~33억 토큰) 중 어느 쪽 영향이 클까요? 추가 실험으로 어떻게 분리할 수 있나요?
4. *MLM 1 epoch* 와 *random init* baseline 의 정확도 차이가 매우 작거나 (예: 1-2%p) 거꾸로 *random 이 더 높게* 나올 가능성이 있나요? 어떤 상황에서 그럴 수 있을까요?""")

# ----- 16. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) Ch 20 의 체크포인트를 디스크에 저장해두고 Ch 21 에서 `from_pretrained` 로 로드하면 안 되나요?

가능합니다 — 그게 *프로덕션 흐름* 입니다. Colab 의 단일 세션에서 Ch 20 → Ch 21 이어 돌리거나, 또는:

```python
# Ch 20 마지막 셀
mlm_model.save_pretrained("./ch20_small_bert_mlm")
tokenizer.save_pretrained("./ch20_small_bert_mlm")

# Ch 21 첫 셀
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "./ch20_small_bert_mlm",
    num_labels=2,
)
```

`from_pretrained` 가 `BertForMaskedLM` 체크포인트를 자동으로 분류 모델로 *헤드 변환* 합니다 — MLM head 는 버려지고 분류 head 가 random init 으로 부착됨 (warning 메시지가 *그 일이 일어났음* 을 알려줌).

이번 챕터가 *MLM 학습 코드를 직접 재현* 한 이유는 **노트북 self-contained** — Colab 세션이 끊겨도 노트북 하나만으로 끝까지 돌릴 수 있게.

### Q2. (이론) MLM 본체 가중치를 *완전히 같은* hyperparams 에 옮겼는데 왜 분류 정확도가 *작은 폭* 만 개선되나요?

**작은 데이터의 한계** — 사전학습 코퍼스 (Yelp 5K 문장 = 약 70만 토큰) 자체가 *학습할 언어 분포* 가 좁습니다. DistilBERT 의 사전학습 코퍼스 (~33억 토큰) 와 비교하면 *5000배 작은* 데이터로 같은 일을 한 것.

```python
# 더 많은 사전학습으로 격차 줄이기 (T4 30분 룰 안에서)
MLM_EPOCHS = 3                     # 1 → 3
# 또는 데이터 늘리기 — N_TRAIN_MLM 만 늘려도 효과 큼
ds_mlm_only = ds_raw["train"].shuffle(seed=SEED).select(range(20000))
```

`N_TRAIN_MLM = 20000` + `MLM_EPOCHS = 1` 정도가 T4 30분 룰 안에서 최대치. 그래도 *대규모 사전학습* 의 격차는 메우기 어렵습니다 — *데이터 규모 자체의 가치* 가 진짜 BERT 의 비밀.

### Q3. (실무) MLM 사전학습이 fine-tune 정확도에 *해가 되는* 경우가 있나요?

드물지만 있습니다. 두 가지 시나리오:

```python
# (1) 사전학습이 *과도* — 작은 데이터에 너무 오래 학습해 본체가 overfitting
MLM_EPOCHS = 20   # 5K 문장에 20 epoch → 데이터에 과적합

# (2) downstream 과 *분포가 다른* 사전학습 — Ch 20 은 Yelp text 자체로 학습이라 이 문제 없음
# 다른 경우: Wikipedia 영어로 MLM 사전학습한 모델로 한국어 분류 fine-tune (Q4 참고)
```

(1) 의 경우 본체가 *특정 문장 패턴에 과적합* 되어 fine-tune 일반화가 떨어질 수 있습니다. *MLM eval loss* 와 *MLM train loss* 의 격차로 진단 — 격차가 커지면 overfitting.

(2) 의 경우 본체가 *downstream 도메인에 무관한 표상* 을 학습. 토크나이저까지 안 맞으면 거의 동작 안 함 (Ch 19 의 cross-language 실험).

### Q4. (이론) Ch 20 의 *작은 사전학습 모델* 을 Ch 21 의 *한국어 분류* (Ch 23 예고) 에 쓰면 어떻게 되나요?

거의 동작 안 합니다 — 토크나이저가 *영어 WordPiece* 라 한국어 문장의 대부분이 `[UNK]` 가 되거나 자모 단위로 쪼개집니다. *임베딩 자체* 가 한국어 의미를 모르는 상태.

```python
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
korean_text = "이 영화 정말 재밌었어요"
print(en_tokenizer.tokenize(korean_text))
# 예상: ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]'] 또는 자모 분해된 형태
```

그래서 Ch 22-23 에서는 *한국어 토크나이저* (`klue/bert-base` WordPiece) 로 처음부터 다시 사전학습 + fine-tune. 토크나이저와 사전학습은 *언어 단위로 매칭* 되어야 함.

### Q5. (실무) 학습 곡선을 보면 fine-tune 후반에 loss 가 *다시 올라가는* 경우가 있는데 정상인가요?

자주 보이는 현상입니다. 원인 셋:

```python
# (1) 학습률 스케줄 — Trainer 기본은 linear warmup → linear decay
# epoch 후반의 LR 이 *너무 작아져* 미세 진동만 남음. loss 가 0.2-0.4 진동.

# (2) overfitting — train loss 는 떨어지는데 eval loss 가 올라감
# 일찍 멈추거나 weight_decay 늘리거나 데이터 늘리기

# (3) batch sample 의 불운 — 어려운 sample 이 몰린 batch 가 loss 를 튀게 만듦
# logging_steps 가 작으면 (50 step 등) 잘 보임. logging_steps=200 으로 평탄화도 가능
```

이번 챕터의 짧은 학습 (2 epoch ≈ 600 step) 에서는 (1) 이 흔합니다. *eval_loss* 가 *epoch 별로* 어떻게 움직이는지 (Trainer 가 `eval_strategy="epoch"` 으로 자동 측정) 가 진짜 신호.

### Q6. (이론) DistilBERT (Ch 10) 가 BERT 의 *distilled* 버전인데 왜 Ch 21 의 *작은 BERT* (scratch) 와 정확도가 차이 나나요? 둘 다 작지 않나요?

DistilBERT 와 Ch 21 의 작은 BERT 는 *축약 방법론* 이 전혀 다릅니다.

| 차원 | DistilBERT | Ch 21 small BERT |
|---|---|---|
| 출발점 | *이미 학습된* BERT-base 의 *지식 증류* (teacher → student) | random init 부터 시작 |
| 사전학습 | MLM + *teacher 의 soft label* + *hidden state 정합* | MLM only (이번 챕터 1 epoch) |
| 학습 코퍼스 | BERT-base 와 같음 (~33억 토큰) | Yelp 5K 문장 (~70만 토큰) |
| 파라미터 | 66M (BERT-base 110M 의 *60%*) | 10M (BERT-base 의 *9%*) |
| 사전학습 시간 | TPU 수일 | T4 10분 |

DistilBERT 가 *이미 똑똑한 큰 BERT 가 만든 답* 을 학습 신호로 받기 때문에 *훨씬 작은 데이터로도 같은 수준* 으로 학습됩니다. 우리는 *teacher 없이 처음부터* 학습하는 셋업 — *맨바닥에서 작은 모델로 사전학습이 어디까지 가능한가* 의 한계 실험.

### Q7. (실무) Ch 21 의 모델을 더 키우면 (예: hidden=512, layer=8) 어떻게 되나요?

T4 메모리 안에서는 가능합니다. 정확도 변화 추정:

| 모델 크기 | 파라미터 | T4 학습 시간 (MLM 1 epoch + cls 2 epoch) | 예상 accuracy |
|---|---|---|---|
| hidden=128, layer=2 | ~5M | 약 5분 | 65-72% |
| **hidden=256, layer=4 (이번 챕터)** | **~10M** | **약 20분** | **75-85%** |
| hidden=384, layer=6 | ~20M | 약 30분 | 78-88% (T4 30분 한계) |
| hidden=512, layer=8 | ~35M | 약 45분 | 80-90% (T4 30분 룰 위반) |
| hidden=768, layer=12 (BERT-base) | ~110M | 수일 | 90%+ (대규모 사전학습 데이터 필요) |

데이터 양을 안 늘리면 모델만 키워도 *정확도 한계* 가 빨리 옵니다. *모델 키움 + 데이터 키움* 이 같이 가야 하고, 그 정점이 *DistilBERT/BERT* 의 *대규모 사전학습*. Ch 21 의 *작은 모델 + 작은 데이터* 는 *원리 학습용 toy 셋업* 의 정의.""")

# ----- 17. 다음 챕터 -----
md(r"""## 다음 챕터 예고

**Chapter 22. 작은 BERT 직접 사전학습 — 한국어 MLM (scratch)**

- Ch 20 의 영어 패턴을 한국어로 그대로: 작은 BertConfig + `klue/bert-base` 토크나이저 (가져옴) + NSMC text MLM
- 토크나이저만 한국어로 바뀜. 본체 구조·MLM 셋업은 *완전히 같음*
- Ch 22 → Ch 23 (한국어 분류) 흐름은 이번 챕터 → Ch 21 (영어) 와 *대칭*

> **변하는 축**: Phase 3 안에서 *언어* (영어 → 한국어). 모델 구조·학습 셋업은 동일.

이번 챕터 (Ch 21) 에서 본 *작은 사전학습 + 분류* 의 격차 패턴이 Ch 23 에서 *한국어 환경* 에서도 같은 결을 그리는지 확인하는 게 Phase 3 의 마지막 메시지.""")

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


# ----- README.md -----
README = """# 21_en_bert_classify — 작은 BERT 분류 (영어 Yelp 이진, scratch 사전학습 + fine-tune)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/21_en_bert_classify.ipynb)

## 한 줄 목표
Phase 3 의 세 번째 챕터. Ch 20 에서 *작은 BERT 를 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 fine-tune**. Ch 10 (DistilBERT, ~66M params, 수십억 토큰 사전학습) 과 같은 Yelp 이진 분류 셋업에 *우리가 만든 작은 BERT* (~10M params, Yelp 5K 문장 MLM) 를 붙여 두 결과를 나란히 비교 — *사전학습 규모* 가 downstream 정확도에 얼마나 차이를 만드는지 정량으로.

self-contained 노트북: Ch 20 의 MLM 학습을 1 epoch 짧게 재현 → 같은 본체로 분류 fine-tune → Ch 10 결과와 비교. *MLM 없이 random init 으로 바로 분류* 하는 baseline 도 변형 셀에서 함께 학습해 *사전학습 자체의 순 효과* 도 분리.

## 다루는 핵심 개념
- `BertForMaskedLM` → `BertForSequenceClassification` 헤드 교체 — 본체 (`embeddings + encoder + pooler`) 는 그대로, MLM head 떼고 분류 head (`Linear(256, 2)`) 부착
- in-memory state_dict 전송: `cls_model.bert.load_state_dict(mlm_model.bert.state_dict())` — 디스크 없이 본체 가중치 복사
- 같은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024, ~10M params) 가 MLM 모델과 분류 모델 양쪽에 적용
- 사전학습 효과의 *순 측정* — random init baseline 과 비교
- **Ch 10 (DistilBERT 대규모 사전학습) vs Ch 21 (작은 BERT 자체 사전학습)** 의 정량 비교

## Loss
`CrossEntropyLoss` — 분류 fine-tune 표준 (K=2, softmax + CE). 라벨은 `int 0/1`, `problem_type="single_label_classification"`. random baseline loss = `ln(2) ≈ 0.693`.

수식: $L = -\\frac{1}{N}\\sum_{i=1}^{N} \\log \\hat p_{i, y_i}$ — Ch 11/15 와 같은 K-class softmax CE.

## 데이터
`fancyzhx/yelp_polarity` 이진 분류 (label 0/1, 5점 척도 자동 이진화). 5,000 train / 1,000 eval, seed 42 — Ch 10 과 같은 split.

MLM 사전학습 단계에서는 같은 5,000 문장의 *text 만* (라벨 무시) 사용해 `block_size=128` `group_texts` 패턴으로 1 epoch 학습.

## 환경
Google Colab T4 GPU (fp16). 약 25분 (MLM 1 epoch 약 10-12분 + 분류 fine-tune 2 epoch 약 8-10분 + 평가/시각화 약 2분).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 10 | DistilBERT 파인튜닝 (~66M) | `bert-base-uncased` WordPiece | Yelp 이진화 | `Linear(H, 1)` | `BCEWithLogitsLoss` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Yelp text (라벨 무시) | MLM head | `CrossEntropyLoss` (masked) |
| **21** | **Ch 20 사전학습 BERT + 분류 헤드 (~10M)** | (Ch 20과 동일) | **Yelp 이진화** | **`Linear(H, 2)`** | **`CrossEntropyLoss`** |
| 22 (다음) | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | NSMC text | MLM head | `CrossEntropyLoss` (masked) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 비교 표 — Ch 10 vs Ch 21

| 차원 | Ch 10 (DistilBERT) | Ch 21 (small BERT scratch) | 비고 |
|---|---|---|---|
| 본체 파라미터 | ~66M | ~10M | Ch 21 은 1/6 작음 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (~33억 토큰) | Yelp 5K 문장 (~70만 토큰) | 약 5000배 격차 |
| 사전학습 시간 | TPU 수일 | T4 약 10-12분 | |
| 분류 fine-tune 셋업 | (같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | | 본체 외 통제 |
| 기대 accuracy | ~92-95% | ~75-85% | 비교는 실측치로 |

격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다. 동시에 *작은 사전학습도 random init 보다는 분명히 낫다* 는 게 변형 셀의 결과.

## 다음 챕터
[22_ko_bert_pretrain](../22_ko_bert_pretrain/) — Ch 20 의 영어 사전학습 패턴을 한국어로 재현. 같은 작은 BertConfig + `klue/bert-base` 토크나이저 + NSMC text MLM. Ch 22 → Ch 23 (한국어 분류) 가 이번 챕터 → Ch 21 (영어) 와 *대칭*.
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}  ({len(README.splitlines())} lines)")
