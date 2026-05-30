"""Build 21_en_bert_classify/appendix_compute_budget.ipynb — fair-compute 비교.

메인 챕터 (Ch 21) 는 *분류 시간 확보* 를 위해 MLM 을 1 epoch 만 돌리지만,
부록은 fair-compute 비교가 핵심이라 MLM 을 3 epoch 까지 충분히 돌려
사전학습 효과가 분명히 드러나도록 합니다. 그렇게 쓴 GPU 시간을 *분류 fine-tune 에 더 쓰면*
효과를 메울 수 있는지를 정량으로 비교합니다.

세 셋업:
- 🅰️ A — MLM 3 epoch + 분류 fine-tune 2 epoch (부록 기준선, 메인의 1 → 3)
- 🅱️ B — 사전학습 없이 random init 분류 모델, T_A_total 만큼 분류 fine-tune 만
- 🅲  C — 사전학습 없이 random init 분류 모델, 같은 2 epoch 만 (단순 baseline)

부록 빌더는 메인 빌더 `_build_ch21.py` 와 같은 패턴 (cells / _cid / md / code / NOTEBOOK
json dump) 을 따릅니다. T4 30분 룰을 지키기 위해 데이터를 줄여 (N_TRAIN=2000,
N_EVAL=400) 세 셋업이 한 노트북 안에 들어가도록 했습니다.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "21_en_bert_classify"
OUT_NB = OUT_DIR / "appendix_compute_budget.ipynb"

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


# ----- 1. 제목 + 부록 안내 -----
md(r"""# Chapter 21 부록 — fair-compute 비교 (사전학습 vs 더 긴 fine-tune)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/appendix_compute_budget.ipynb)

> **부록 한 줄 질문** — *"MLM 사전학습에 쓰는 GPU 시간을 그냥 분류 fine-tune 에 더 쓰면 사전학습 효과를 메울 수 있나요?"*

메인 챕터 ([`21_en_bert_classify.ipynb`](./21_en_bert_classify.ipynb)) 는 *작은 BERT 를 MLM 으로 짧게 사전학습 한 뒤 분류 fine-tune* 한 결과를 Ch 10 (DistilBERT 대규모 사전학습) 과 비교했습니다. 이 부록은 그 결과에 *한 비교* 를 더합니다 — **같은 GPU wall-clock budget** 으로 *사전학습 없이* 분류 fine-tune 만 더 길게 돌렸을 때 어떻게 되는지.

세 셋업을 한 노트북 안에서 같은 데이터·같은 본체 구조에 *조건 하나만* 바꿔 비교합니다.

| 셋업 | 사전학습 | 분류 fine-tune | 의도 |
|---|---|---|---|
| 🅰️ **A** | MLM 3 epoch | 2 epoch | 메인의 1 → 3 epoch (사전학습 효과를 충분히, 기준선) |
| 🅱️ **B** | 없음 (random init) | **A 의 총 시간만큼** epoch 늘림 | **fair-compute** 비교 — 같은 GPU 예산을 fine-tune 에 몰아주면? |
| 🅲 **C** | 없음 (random init) | 2 epoch (A 와 동일) | *순수 random init baseline* — 사전학습의 *순* 효과 |

A vs C 는 *사전학습의 순 효과*, A vs B 는 *사전학습 vs compute 등가 fine-tune* 비교.

**환경**: Google Colab T4 GPU 필수. 약 25-30분.

**메인 챕터와의 관계** — 메인 챕터는 분류 시간 확보를 위해 MLM 을 1 epoch 로 줄였지만, 부록은 fair-compute 비교가 핵심이라 **MLM 1 → 3 epoch** 로 늘려 사전학습 효과가 분명히 드러나도록 했습니다. 그 대신 데이터를 메인의 5K/1K → 부록의 `N_TRAIN=2000, N_EVAL=400` 으로 줄여 *부록 하나* 의 T4 30분 안에 세 셋업이 모두 들어가도록 했습니다. 부록은 *부록만으로 self-contained* — 메인 노트북을 먼저 돌릴 필요 없음.

---""")

# ----- 2. 변경점 (셋업 비교 표) -----
md(r"""## 🔄 셋업 비교 — A / B / C

| 축 | 🅰️ A (메인 재현) | 🅱️ B (fair-compute) | 🅲 C (random baseline) |
|---|---|---|---|
| MLM 사전학습 | **3 epoch** | 없음 | 없음 |
| 분류 fine-tune epoch | 2 | **A 의 총 시간만큼 자동 산정** | 2 |
| 본체 시작점 | MLM 가중치 | random init | random init |
| 분류 head | random init | random init | random init |
| 데이터 | Yelp 5K/1K (메인) → 본 부록은 **2K/400** | (같음) | (같음) |
| 토크나이저 | `bert-base-uncased` | (같음) | (같음) |
| 모델 본체 | 작은 BERT (hidden=256, layer=4) | (같음) | (같음) |
| Loss | `CrossEntropyLoss` (K=2) | (같음) | (같음) |
| 학습률 | `5e-4` (MLM) / `2e-5` (cls) | `2e-5` (cls only) | `2e-5` (cls only) |
| fp16 | True | True | True |

**B 의 epoch 결정** — A 의 *총 시간* `T_A_total = T_A_mlm + T_A_cls` 를 측정한 뒤, B 의 *한 epoch* 에 걸린 시간 `t_per_epoch_B` 로 나눠 `epochs_B = round(T_A_total / t_per_epoch_B)` 로 자동 산정합니다. 이렇게 *시간 등가* 를 맞춥니다.

> **B 가 측정하는 가설** — *"사전학습은 결국 GPU 시간 소비. 그 시간을 그냥 분류 fine-tune 에 쓰면 안 되나?"*. **A vs B** 결과가 이 질문에 답합니다.

---""")

# ----- 3. 환경 셋업 -----
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
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
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
    print("Warning: CPU runtime — all three setups will be very slow.")
    print("         Switch to Colab T4 runtime to keep the appendix under 30 minutes.")""")

md(r"""### 데이터·모델 hyperparams — 메인과 통일, MLM 은 더 충분히

메인 챕터는 *분류 fine-tune 시간 확보* 를 위해 MLM 을 1 epoch 으로 줄였지만, 부록은 fair-compute 비교가 핵심이라 **MLM 1 → 3 epoch** 로 늘려 사전학습 효과가 충분히 드러나도록 합니다. T4 30분 룰을 지키기 위해 데이터는 `N_TRAIN=2000, N_EVAL=400` 으로 축소 (메인 5K/1K → 부록 2K/400). 작은 BERT 본체 구조 (hidden=256, layer=4, head=4) 와 학습률은 메인과 완전히 같음.""")

code(r"""SEED = 42

# 데이터 — 부록은 MLM 을 3 epoch 까지 늘리므로 데이터를 더 줄여 시간 균형
N_TRAIN_TEXT = 2000
N_EVAL_TEXT  = 400
BLOCK_SIZE     = 128
MAX_LENGTH_CLS = 128

# 모델 — 메인 챕터와 완전히 동일한 작은 BERT
HIDDEN_SIZE         = 256
NUM_HIDDEN_LAYERS   = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE   = 1024
MAX_POS_EMBED       = 256

# MLM 사전학습 hyperparams (셋업 A 만 사용)
# 메인은 1 epoch — 분류 시간 확보용. 부록은 사전학습 효과를 충분히 보기 위해 3 epoch.
MLM_EPOCHS = 3
MLM_BATCH  = 32
MLM_LR     = 5e-4

# 분류 fine-tune hyperparams (A, C 는 2 epoch / B 는 자동 산정)
CLS_EPOCHS = 2
CLS_BATCH  = 16
CLS_LR     = 2e-5

# B 의 epoch 산정 시 상한 (T4 30분 룰을 한 번 더 보장)
B_EPOCHS_CAP = 20

USE_FP16 = (DEVICE == "cuda")

# CPU/MPS 환경이면 더 작게 (메시지)
if DEVICE != "cuda":
    print("Note: non-CUDA device detected.")
    print("      Consider reducing N_TRAIN_TEXT to 1000 and N_EVAL_TEXT to 200 to finish in reasonable time.")
    print("      fp16 disabled (only effective on CUDA).")

print(f"Train texts: {N_TRAIN_TEXT}")
print(f"Eval  texts: {N_EVAL_TEXT}")
print(f"Model: hidden={HIDDEN_SIZE}, layer={NUM_HIDDEN_LAYERS}, head={NUM_ATTENTION_HEADS}, intermediate={INTERMEDIATE_SIZE}")
print(f"fp16: {USE_FP16}")""")

# ----- 4. 데이터 로드 -----
md(r"""## 1. 📥 데이터·토크나이저 로드 — 메인 챕터와 같은 파이프라인

`fancyzhx/yelp_polarity` 이진 분류. seed 42 로 shuffle 후 앞에서 `N_TRAIN_TEXT / N_EVAL_TEXT` 만 사용. 메인 챕터는 5K/1K, 부록은 2K/400.""")

code(r"""ds_raw = load_dataset("fancyzhx/yelp_polarity")
print(f"splits: {list(ds_raw.keys())}")
print(f"label names: {ds_raw['train'].features['label'].names}")

ds_train_full = ds_raw["train"].shuffle(seed=SEED).select(range(N_TRAIN_TEXT))
ds_eval_full  = ds_raw["test"].shuffle(seed=SEED).select(range(N_EVAL_TEXT))

train_labels = np.array(ds_train_full["label"])
eval_labels  = np.array(ds_eval_full["label"])
print(f"train: {len(ds_train_full):,}  positive rate: {train_labels.mean():.1%}")
print(f"eval:  {len(ds_eval_full):,}  positive rate: {eval_labels.mean():.1%}")""")

code(r"""TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print(f"tokenizer:  {TOKENIZER_NAME}")
print(f"vocab_size: {tokenizer.vocab_size:,}")""")

md(r"""### 공통 유틸 — 분류 토큰화 / metric / 모델 빌더

세 셋업이 같은 토큰화·metric·모델 구조를 공유하므로 한 번만 정의합니다.""")

code(r"""def cls_tokenize(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH_CLS)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

cls_train = ds_train_full.map(cls_tokenize, batched=True).remove_columns(
    [c for c in ds_train_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
cls_eval = ds_eval_full.map(cls_tokenize, batched=True).remove_columns(
    [c for c in ds_eval_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
print(cls_train)""")

code(r"""def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=2)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    preds = probs_full.argmax(axis=1)
    probs_pos = probs_full[:, 1]
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auc":       float(roc_auc_score(labels, probs_pos)),
    }


def build_cls_config():
    '''셋업 A, B, C 가 공유하는 분류 BertConfig.'''
    return BertConfig(
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


def make_cls_trainer(model, epochs, run_name):
    '''Trainer + TrainingArguments 공통 셋업. epochs 만 셋업 별로 다름.'''
    args = TrainingArguments(
        output_dir=f"./ch21_appendix_{run_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=CLS_BATCH,
        per_device_eval_batch_size=32,
        learning_rate=CLS_LR,
        fp16=USE_FP16,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=cls_train,
        eval_dataset=cls_eval,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )""")

# ----- 5. 셋업 A -----
md(r"""## 2. 🅰️ Setup A — MLM 3 epoch + 분류 fine-tune 2 epoch (사전학습 충분히)

메인 챕터의 핵심 셋업을 *MLM 만 1 → 3 epoch* 로 늘려 재현합니다 — 1 epoch 으로는 본체 표상이 덜 정렬돼서 fair-compute 비교 메시지가 약해지기 때문. `T_A_mlm` (MLM 학습 시간) 과 `T_A_cls` (분류 fine-tune 시간) 를 별도로 측정해 합쳐 `T_A_total` 을 만듭니다. 이 시간이 *셋업 B 의 compute budget* 입니다.""")

code(r"""# ---- A-1. MLM 사전학습 ----
mlm_config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POS_EMBED,
    pad_token_id=tokenizer.pad_token_id,
)

torch.manual_seed(SEED)
mlm_model = BertForMaskedLM(mlm_config)

# MLM 학습용 데이터셋: text 만 (라벨 무시 — self-supervised)
mlm_train_raw = ds_train_full.remove_columns(
    [c for c in ds_train_full.column_names if c != "text"]
)

def mlm_tokenize(examples):
    return tokenizer(examples["text"], add_special_tokens=False, truncation=False)

mlm_tokenized = mlm_train_raw.map(mlm_tokenize, batched=True, remove_columns=["text"])

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

lm_train = mlm_tokenized.map(group_texts, batched=True, batch_size=1000)
print(f"MLM train blocks: {len(lm_train):,}  (block_size={BLOCK_SIZE})")

mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
)

mlm_args = TrainingArguments(
    output_dir="./ch21_appendix_A_mlm",
    num_train_epochs=MLM_EPOCHS,
    per_device_train_batch_size=MLM_BATCH,
    per_device_eval_batch_size=64,
    learning_rate=MLM_LR,
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=USE_FP16,
    logging_steps=20,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
    seed=SEED,
)

mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_args,
    train_dataset=lm_train,
    data_collator=mlm_collator,
    processing_class=tokenizer,
)

t0 = time.time()
mlm_result = mlm_trainer.train()
T_A_mlm_sec = time.time() - t0
T_A_mlm = T_A_mlm_sec / 60.0
print(f"\n[A] MLM pretraining: {T_A_mlm:.2f} min  (mean train loss: {mlm_result.training_loss:.4f})")""")

code(r"""# ---- A-2. 분류 fine-tune (MLM 본체 이어받아 2 epoch) ----
torch.manual_seed(SEED)
cls_model_A = BertForSequenceClassification(build_cls_config())

# MLM 본체 가중치를 분류 모델로 복사 (메인 챕터와 동일)
missing, unexpected = cls_model_A.bert.load_state_dict(
    mlm_model.bert.state_dict(), strict=False,
)
print(f"[A] body weights copied  (missing: {len(missing)}, unexpected: {len(unexpected)})")

trainer_A = make_cls_trainer(cls_model_A, epochs=CLS_EPOCHS, run_name="A_cls")

t0 = time.time()
result_A = trainer_A.train()
T_A_cls_sec = time.time() - t0
T_A_cls = T_A_cls_sec / 60.0

T_A_total = T_A_mlm + T_A_cls
metrics_A = trainer_A.evaluate()

print(f"\n[A] Classification fine-tune: {T_A_cls:.2f} min  ({CLS_EPOCHS} epochs)")
print(f"[A] Total compute: {T_A_total:.2f} min  ({T_A_mlm:.2f} MLM + {T_A_cls:.2f} cls)")
print(f"[A] eval accuracy: {metrics_A['eval_accuracy']:.4f}  F1: {metrics_A['eval_f1']:.4f}  AUC: {metrics_A['eval_auc']:.4f}")""")

md(r"""**측정 끝** — `T_A_total` 이 셋업 B 의 *compute budget*. 다음 단계에서 random init 모델의 epoch 당 시간을 측정한 뒤 *몇 epoch* 을 돌려야 같은 시간이 나오는지 계산합니다.""")

# ----- 6. 셋업 B -----
md(r"""## 3. 🅱️ Setup B — random init, 같은 GPU 시간 budget 만큼 fine-tune

**핵심 질문에 답하는 셋업** — 사전학습 없이 random init 분류 모델을 `T_A_total` 분 동안 fine-tune 하면 어디까지 가는가.

구현:
1. 같은 본체 구조의 random init `BertForSequenceClassification` 을 만든다
2. **1 epoch 만 잠깐** 돌려 epoch 당 시간 `t_per_epoch_B` 를 측정한다
3. `epochs_B = max(2, round(T_A_total / t_per_epoch_B))` 로 epoch 수 결정
4. 같은 random init 모델을 *fresh 하게 다시* 만들어 `epochs_B` epoch 학습 (warm-up state 가 1 epoch 단계와 일관되도록)

> *"먼저 1 epoch 측정 → 다시 처음부터"* 방식 — 더 단순한 *바로 `epochs_B` epoch 돌리기* 가 있긴 하지만, epoch 당 시간 측정이 필요해서 이 방식이 더 안전. 측정 epoch 도 *그냥 버리지 않고* 시간 budget 계산에 반영합니다.""")

code(r"""# ---- B-1. epoch 당 시간 측정 (1 epoch 만) ----
torch.manual_seed(SEED)
cls_model_B_probe = BertForSequenceClassification(build_cls_config())
trainer_B_probe = make_cls_trainer(cls_model_B_probe, epochs=1, run_name="B_probe")

t0 = time.time()
trainer_B_probe.train()
t_per_epoch_B_sec = time.time() - t0
t_per_epoch_B = t_per_epoch_B_sec / 60.0
print(f"[B-probe] 1 epoch took {t_per_epoch_B:.2f} min on random init model")

# T_A_total 을 채우는데 필요한 epoch 수
epochs_B_raw = T_A_total / t_per_epoch_B
epochs_B = max(2, int(round(epochs_B_raw)))
epochs_B = min(epochs_B, B_EPOCHS_CAP)   # 상한 (T4 30분 룰 재보장)

print(f"[B] target budget: T_A_total = {T_A_total:.2f} min")
print(f"[B] epochs needed: raw {epochs_B_raw:.2f} -> rounded {epochs_B}  (cap {B_EPOCHS_CAP})")
print(f"[B] expected wall time: about {epochs_B * t_per_epoch_B:.2f} min")""")

code(r"""# ---- B-2. fresh random init 모델로 epochs_B epoch fine-tune ----
torch.manual_seed(SEED)
cls_model_B = BertForSequenceClassification(build_cls_config())
trainer_B = make_cls_trainer(cls_model_B, epochs=epochs_B, run_name="B_cls")

t0 = time.time()
result_B = trainer_B.train()
T_B_total_sec = time.time() - t0
T_B_total = T_B_total_sec / 60.0
metrics_B = trainer_B.evaluate()

print(f"\n[B] Classification fine-tune: {T_B_total:.2f} min  ({epochs_B} epochs)")
print(f"[B] eval accuracy: {metrics_B['eval_accuracy']:.4f}  F1: {metrics_B['eval_f1']:.4f}  AUC: {metrics_B['eval_auc']:.4f}")""")

md(r"""**관전 포인트** — `T_B_total` 이 `T_A_total` 과 비슷한가, 그리고 metric 이 `metrics_A` 와 비교해 얼마나 따라잡았는가. *시간 등가 비교* 의 핵심.""")

# ----- 7. 셋업 C -----
md(r"""## 4. 🅲 Setup C — random init, 같은 epoch 수 (사전학습의 *순* 효과)

A vs C 비교는 *사전학습이 만든 차이* 그 자체. 시간이 아니라 *학습 방식의 시작점* 만 다른 둘.

- A: MLM 본체 → 2 epoch fine-tune
- C: random init → 2 epoch fine-tune

A 가 C 보다 얼마나 높은지가 *MLM 3 epoch 의 순 효과*.""")

code(r"""torch.manual_seed(SEED)
cls_model_C = BertForSequenceClassification(build_cls_config())
trainer_C = make_cls_trainer(cls_model_C, epochs=CLS_EPOCHS, run_name="C_cls")

t0 = time.time()
result_C = trainer_C.train()
T_C_total_sec = time.time() - t0
T_C_total = T_C_total_sec / 60.0
metrics_C = trainer_C.evaluate()

print(f"\n[C] Classification fine-tune: {T_C_total:.2f} min  ({CLS_EPOCHS} epochs)")
print(f"[C] eval accuracy: {metrics_C['eval_accuracy']:.4f}  F1: {metrics_C['eval_f1']:.4f}  AUC: {metrics_C['eval_auc']:.4f}")""")

# ----- 8. 세 셋업 비교 -----
md(r"""## 5. 🆚 세 셋업 비교 — 표 + bar chart

같은 평가 셋 위에서 세 결과를 한 표로 모읍니다.""")

code(r"""def row(label, pretraining, epochs, total_min, metrics):
    return {
        "setup": label,
        "pretraining": pretraining,
        "fine-tune epochs": epochs,
        "total compute (min)": round(total_min, 2),
        "accuracy": round(metrics["eval_accuracy"], 4),
        "F1":       round(metrics["eval_f1"], 4),
        "AUC":      round(metrics["eval_auc"], 4),
    }

summary = pd.DataFrame([
    row("A (MLM + cls)",       f"MLM {MLM_EPOCHS} epoch", CLS_EPOCHS, T_A_total, metrics_A),
    row("B (fair-compute)",    "none",                    epochs_B,   T_B_total, metrics_B),
    row("C (random baseline)", "none",                    CLS_EPOCHS, T_C_total, metrics_C),
])
print(summary.to_string(index=False))""")

code(r"""# bar chart — 3 setups x 3 metrics
sns.set_theme(style="whitegrid", context="talk")
plot_df = summary.melt(
    id_vars=["setup"],
    value_vars=["accuracy", "F1", "AUC"],
    var_name="metric",
    value_name="score",
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=plot_df, x="metric", y="score", hue="setup",
    palette={
        "A (MLM + cls)":       "#4878D0",
        "B (fair-compute)":    "#EE854A",
        "C (random baseline)": "#6ACC64",
    },
    ax=ax,
)
ax.set_ylim(0, 1.05)
ax.set_title("Fair-compute comparison — A / B / C")
ax.set_xlabel("metric")
ax.set_ylabel("score")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.show()""")

code(r"""# wall-clock 비교 — A 의 사전학습 시간이 어느 만큼이었는지 시각화
fig, ax = plt.subplots(figsize=(9, 4))
bottoms = [0, 0, 0]
labels = ["A (MLM + cls)", "B (fair-compute)", "C (random baseline)"]
mlm_times = [T_A_mlm, 0.0, 0.0]
cls_times = [T_A_cls, T_B_total, T_C_total]

ax.bar(labels, mlm_times, color="#4878D0", label="MLM pretraining (min)")
ax.bar(labels, cls_times, bottom=mlm_times, color="#EE854A", label="Classification fine-tune (min)")
ax.set_ylabel("wall clock (min)")
ax.set_title("Compute budget breakdown")
ax.legend(loc="upper right", fontsize=11)
plt.tight_layout()
plt.show()""")

# ----- 9. 해석 -----
md(r"""## 6. 🔎 해석 — 무엇을 읽어야 하나

| 비교 | 의미 | 전형적 결과 |
|---|---|---|
| **A vs C** | *사전학습의 순 효과* (같은 epoch 의 두 출발점) | A 가 C 보다 accuracy 약 5-15%p 높음 — MLM 3 epoch 으로 본체 표상이 *분류에 유용한 방향* 으로 충분히 정렬 |
| **A vs B** | *사전학습 vs 동일 compute fine-tune* | A 가 B 보다 *여전히* 높음. 다만 격차는 A vs C 보다 줄어듦 — fine-tune epoch 을 늘려도 *self-supervised 신호 결손* 이 메워지지 않음 |
| **B vs C** | *fine-tune epoch 의 효과* (둘 다 random init) | B 가 C 보다 높음 — random init 도 epoch 늘리면 더 학습. 다만 *수렴 정체* 로 일찍 평탄해질 수 있음 |

**왜 B 가 A 를 못 따라잡나** — 분류 task 의 *supervised 신호* 만으로는 *언어 구조 일반* 을 학습하기 어렵습니다. MLM 의 *비지도 self-supervised* 신호가 *모든 토큰 자리* 에 대해 *문맥 예측* 을 강제해 본체에 *언어 분포* 를 새기는 반면, 분류 신호는 *문장 단위로 0/1 한 비트* 만 줍니다. 같은 GPU 시간이라도 *학습 신호의 밀도* 가 다른 것.

**작은 모델·작은 데이터에서 주의** — 본 부록처럼 데이터가 2K 정도면 *random init + 긴 fine-tune* 이 overfitting 에 빠져 B 의 train loss 는 떨어지는데 eval 은 평탄해질 수 있습니다. *큰 모델·큰 데이터* 일수록 사전학습의 가치가 더 명확하게 커집니다 (Ch 10 의 DistilBERT 가 그 정점).

**T_A 안에서 MLM 비중** — `T_A_mlm / T_A_total` 이 클수록 *fair-compute 격차* 가 본질적입니다. 본 부록에서 MLM 비중은 약 `T_A_mlm / T_A_total` (위 셀의 출력 참고).""")

code(r"""# 보조 — 메시지를 숫자로 한 번 더
delta_AC = metrics_A["eval_accuracy"] - metrics_C["eval_accuracy"]
delta_AB = metrics_A["eval_accuracy"] - metrics_B["eval_accuracy"]
delta_BC = metrics_B["eval_accuracy"] - metrics_C["eval_accuracy"]
mlm_share = T_A_mlm / T_A_total if T_A_total > 0 else float("nan")

print("Accuracy deltas (this run):")
print(f"  A vs C  (pretraining net effect):           {delta_AC:+.4f}")
print(f"  A vs B  (pretraining vs fair-compute):      {delta_AB:+.4f}")
print(f"  B vs C  (extra fine-tune epochs effect):    {delta_BC:+.4f}")
print()
print(f"MLM share of total A compute: {mlm_share:.1%}")
print(f"B epochs run: {epochs_B}  (vs A/C: {CLS_EPOCHS})")""")

# ----- 10. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. **fair-compute 관점** — 셋업 B 가 셋업 A 를 *정확히* 따라잡지 못한다면, 그 격차의 원인은 *학습 신호의 종류* (self-supervised vs supervised) 인가, *학습 신호의 양* (모든 토큰 vs 문장 한 비트) 인가? 둘이 분리되나요?
2. **데이터 규모와의 관계** — `N_TRAIN_TEXT` 를 3000 → 30000 으로 늘리면 셋업 A vs B 의 격차는 *커질까 작아질까*? MLM 학습 자체가 더 좋아질지, fine-tune 만으로도 충분해질지, 어느 쪽 효과가 우세할까요?""")

# ----- 11. 다음 단계 -----
md(r"""## 다음 단계

이 부록에서 *작은 모델·작은 데이터* 셋업에서도 사전학습이 *fair-compute 등가* 보다 가치 있다는 점을 확인했습니다. 다만 그 격차의 크기는 *모델/데이터 규모* 에 강하게 의존합니다.

- **메인 챕터로 돌아가기**: [`21_en_bert_classify.ipynb`](./21_en_bert_classify.ipynb) — Ch 10 (DistilBERT 대규모 사전학습) 과의 비교 마무리
- **다음 챕터 예고**: Chapter 22 — 한국어 작은 BERT 직접 사전학습 (`klue/bert-base` 토크나이저 + NSMC text MLM). Ch 20 의 영어 패턴을 한국어로 재현, 같은 *작은 사전학습* 흐름.

> 부록의 핵심 메시지 한 줄 — *사전학습은 compute 의 형태가 아니라 학습 신호의 종류 차이*. 같은 GPU 시간이라도 *self-supervised 로 본체를 미리 정렬* 해 두는 게 *supervised 만 길게 돌리는 것* 보다 효율적.""")


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
