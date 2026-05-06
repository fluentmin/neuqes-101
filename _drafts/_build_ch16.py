"""Build 16_ko_multiclass/16_ko_multiclass.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "16_ko_multiclass" / "16_ko_multiclass.ipynb"

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
md(r"""# Chapter 16. 한국어 BERT Multi-class — KLUE-YNAT (뉴스 7분류)

**목표**: Ch 15 의 한국어 binary 셋업을 그대로 두고 **클래스 수만 K=2 → K=7** 로 늘립니다. 모델·토크나이저·hyperparams 가 *완전히 동일* 하고, 변하는 건 출력 헤드 차원과 데이터.

이 챕터는 Ch 12 (영어 multi-class, Yelp 5클래스) 의 한국어 버전이고, *Phase 1 → Phase 2* 의 task 일반화가 어떻게 자연스럽게 이어지는지 보여줍니다 — softmax+CE 셋업은 K 가 무엇이든 같은 코드.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 13분 (모델 다운로드 캐시 ~10s + 2 에폭 학습 ~10분 + 평가/시각화)

---

## 학습 흐름

1. 🚀 **실습**: KLUE-YNAT 5,000건으로 klue/bert-base 파인튜닝 → 뉴스 헤드라인 7카테고리 분류
2. 🔬 **해부**: 7×7 혼동 행렬, top-1 확률 분포, 카테고리별 precision/recall/F1
3. 🛠️ **샘플 단위 해석**: 자신있는 / 망설이는 헤드라인 직접 읽어보고 모델이 어디서 헷갈리는지 확인

---

> 📒 **사전 학습 자료**: Ch 12 (영어 multi-class, Yelp 5클래스), Ch 15 (한국어 binary, NSMC). 이번 챕터는 두 챕터의 *결합* — Ch 15 의 한국어 셋업 + Ch 12 의 multi-class 처리.""")

# ----- 2. 추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 | 토크나이저 | 데이터 | Output Head | Activation | Loss |
|---|---|---|---|---|---|---|
| 12 | DistilBERT | WordPiece (영어) | Yelp 5클래스 | `Linear(H, 5)` | softmax | `CrossEntropyLoss` |
| 15 | klue/bert-base | WordPiece (한국어) | NSMC binary | `Linear(H, 2)` | softmax | `CrossEntropyLoss` |
| **16 ← 여기** | klue/bert-base | 같음 | **KLUE-YNAT (뉴스 7분류)** | **`Linear(H, 7)`** | softmax | `CrossEntropyLoss` |
| 17 (다음) | klue/bert-base | 같음 | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | sigmoid (per-label) | `BCEWithLogitsLoss` (per-label) |

전체 20챕터 표는 [루트 README.md](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표)를 참고하세요.""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from Ch 15)

| 축 | Ch 15 (한국어 binary) | Ch 16 (한국어 multi-class) |
|---|---|---|
| **Task** | 이진 분류 (K=2) | **7-클래스 분류 (K=7)** ← *유일한 변화* |
| `num_labels` | 2 | **7** |
| 데이터 | NSMC (영화 리뷰) | **KLUE-YNAT (뉴스 헤드라인)** |
| 라벨 형식 | int 0/1 | **int 0-6** |
| 평가 metric | binary precision/recall/F1 + AUC | **accuracy + macro precision/recall/F1 + multi-class AUC (OvR)** |
| 모델 / 토크나이저 / problem_type / Activation / Loss / hyperparams | (모두 동일) | (모두 동일) |

> **변경점 한 가지 원칙** — Phase 2 안에선 *task 차원* (K=2 → K=7) 만 바뀝니다. 한국어 셋업·hyperparams 는 Ch 15 와 *완전히 같음*. 새 챕터의 학습 부담은 *7클래스 평가 metric 의 해석* 에만 집중.

### 한국어 환경에서도 multi-class 일반화는 *그대로* 작동

Ch 11 → Ch 12 (영어 binary → multi-class) 에서 `num_labels` 를 2 → 5 로만 바꿔도 모든 셋업이 그대로 동작했습니다. Ch 15 → Ch 16 도 정확히 같은 패턴 — *softmax+CE 의 진짜 강점*: K 가 무엇이든 같은 식으로 일반화됩니다.""")

# ----- 4. Loss 노트 -----
md(r"""## 📐 Loss 노트 — `CrossEntropyLoss` 가 K=7 에서 보이는 모습

수식은 Ch 12 와 동일:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i, y_i}$$

K=7 의 random baseline loss = $\log 7 = 1.946$. 학습 첫 step 에서 loss 가 ~1.9 정도 보이면 모델이 *균등 추측 단계* — 이후 loss 가 떨어지는 곡선이 학습이 *실제로 진행되는지* 진단 신호.

**숫자로 감 잡기 (K=7, 정답 = 클래스 5)**:

| logits | softmax → $\hat p_5$ | 손실 |
|---|---|---|
| 모두 0 (균등) | $1/7 \approx 0.143$ | **1.946** ← random |
| 정답 클래스만 +2 | $\approx 0.471$ | 0.752 |
| 정답 클래스만 +5 | $\approx 0.985$ | 0.015 |
| 다른 클래스 +5 (정답 0) | $\approx 0.005$ | **5.302** ← 자신 있게 틀림 |

**클래스 균형 영향** — KLUE-YNAT 의 클래스 분포는 *완벽 균형이 아닙니다* (스포츠/세계가 정치/IT보다 많음). class_weight 없이 학습하면 *다수 클래스* 에 편향될 가능성. 평가 단계에서 *macro F1* 을 같이 봐야 소수 클래스 정확도가 묻히지 않음.""")

# ----- 5. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트

Ch 15 와 *완전히 동일* — `klue/bert-base` 한국어 WordPiece. 토크나이저는 K 변화에 무관 (라벨 처리는 모델 측 일).

> **Phase 2 안에서는 토크나이저 고정** — Ch 15·16·17·18 모두 같은 한국어 WordPiece. Phase 3 (Ch 19-20) 에서 비로소 *직접 학습한 워드레벨 토크나이저* 가 등장.

### 헤드라인 토큰화 예시

NSMC 영화 리뷰는 보통 *짧은 한 줄* (~20 토큰), KLUE-YNAT 뉴스 헤드라인은 *조금 더 정형* 된 한국어 (~25-30 토큰). 같은 한국어지만 *문체가 다른* 두 도메인 — 도메인 적응이 어떻게 이뤄지는지 확인할 좋은 비교.""")

# ----- 6. install + import -----
code(r"""!pip install -q transformers datasets""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score, confusion_matrix,
)

plt.rcParams["axes.unicode_minus"] = False

print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CPU runtime — training will be very slow. Switch to T4 recommended.")""")

# ----- 7. nvidia-smi -----
md(r"""**baseline VRAM**:""")
code(r"""!nvidia-smi""")

# ----- 8. 데이터 -----
md(r"""## 1. 🚀 데이터 — KLUE-YNAT (뉴스 헤드라인 7분류)

**KLUE** = Korean Language Understanding Evaluation 벤치마크. **YNAT** = Yonhap News Agency Topic. 연합뉴스 헤드라인 한 줄 + 7카테고리 라벨.

| 라벨 | 카테고리 |
|---|---|
| 0 | IT과학 |
| 1 | 경제 |
| 2 | 사회 |
| 3 | 생활문화 |
| 4 | 세계 |
| 5 | 스포츠 |
| 6 | 정치 |

`datasets.load_dataset("klue", "ynat")` 로 정상 로드 (parquet 기반).""")

code(r"""ds = load_dataset("klue", "ynat")
print(f"splits: {list(ds.keys())}")
print(f"sizes: {[(k, len(v)) for k, v in ds.items()]}")
print(f"label names: {ds['train'].features['label'].names}")

# 클래스 분포
import collections
cnt = collections.Counter(ds["train"]["label"])
LABEL_NAMES = ds["train"].features["label"].names
print(f"\nClass distribution (train):")
for k in range(len(LABEL_NAMES)):
    n = cnt[k]
    print(f"  {LABEL_NAMES[k]:>8}  (label {k}): {n:>5}  ({n / len(ds['train']):.1%})")

print(f"\nfirst 3 samples:")
for ex in ds["train"].select(range(3)):
    print(f"  label={ex['label']} ({LABEL_NAMES[ex['label']]:>8})  title={ex['title']!r}")""")

code(r"""# T4 30분 룰: 5K train / 1K eval (KLUE 의 validation split 에서 sample)
SEED = 42
train_full = ds["train"].shuffle(seed=SEED).select(range(5000))
eval_full  = ds["validation"].shuffle(seed=SEED).select(range(1000))

# title 컬럼명을 transformers 표준 'text' 로 통일
train_full = train_full.rename_column("title", "text")
eval_full  = eval_full.rename_column("title", "text")

print(f"sampled train: {len(train_full)}")
print(f"sampled eval:  {len(eval_full)}")

# 토큰 길이 분포 미리 보기
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
sample_lens = [len(tokenizer.encode(t)) for t in train_full["text"][:200]]
print(f"\nToken length (sample 200): mean={np.mean(sample_lens):.1f}, median={np.median(sample_lens):.0f}, max={max(sample_lens)}")""")

# ----- 9. 토큰화 -----
md(r"""## 2. 토큰화 — Ch 15 패턴 그대로

라벨 형식만 binary int → 0-6 int. 한 줄 차이.""")

code(r"""def tokenize_fn(batch):
    out = tokenizer(batch["text"], truncation=True, max_length=128)
    out["labels"] = [int(l) for l in batch["label"]]
    return out

train_tok = train_full.map(tokenize_fn, batched=True).remove_columns(
    [c for c in train_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)
eval_tok  = eval_full.map(tokenize_fn,  batched=True).remove_columns(
    [c for c in eval_full.column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
)

print(train_tok)
print(f"\nFirst sample label: {train_tok[0]['labels']}  (int 0-6)")""")

# ----- 10. 모델 -----
md(r"""## 3. 모델 로드 — `num_labels=7` 만 바뀜

Ch 15 셋업에서 K=2 → K=7 한 줄 변화.""")

code(r"""model = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=len(LABEL_NAMES),
    problem_type="single_label_classification",
    id2label={i: name for i, name in enumerate(LABEL_NAMES)},
    label2id={name: i for i, name in enumerate(LABEL_NAMES)},
)

def param_summary(m):
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

total, trainable = param_summary(model)
print(f"Parameters:           {total:>13,}  ({total/1e6:.1f} M)")
print(f"Trainable parameters: {trainable:>13,}  ({trainable/total:.1%})")
print(f"Classifier:           {model.classifier}")
print(f"id2label:             {model.config.id2label}")""")

md(r"""**Ch 15 와의 파라미터 수 비교** — 7클래스로 늘어났는데도 모델은 *거의 안 무거워짐*:

| 부분 | Ch 15 (K=2) | Ch 16 (K=7) |
|---|---|---|
| BERT body (12 layer) | 110,617,344 | 110,617,344 |
| classifier `Linear(768, K)` | 1,538 | **5,383** |
| 합계 | 110,618,882 | **110,622,727** |

분류 헤드만 K 에 비례해 늘어나지만 BERT body 가 ~110M 이라 K 가 5 늘어도 전체 차이는 0.003%. **K 가 늘어났다고 학습이 *훨씬* 무거워지지는 않는다** — multi-class BERT 의 매력.""")

code(r"""!nvidia-smi""")

# ----- 11. 학습 -----
md(r"""## 4. 학습 — Ch 15 와 동일한 hyperparams

`compute_metrics` 만 multi-class 용으로 (Ch 12 의 패턴 그대로).""")

code(r"""def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 안정 softmax (K=7)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_full = exp / exp.sum(axis=1, keepdims=True)
    preds = probs_full.argmax(axis=1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_precision": float(p),
        "macro_recall":    float(r),
        "macro_f1":        float(f1),
    }
    # multi-class AUC: One-vs-Rest
    try:
        out["auc_ovr"] = float(roc_auc_score(labels, probs_full, multi_class="ovr"))
    except ValueError:
        out["auc_ovr"] = float("nan")
    return out""")

code(r"""training_args = TrainingArguments(
    output_dir="./ch16_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    fp16=True,
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
print(f"\nTraining done — mean train loss: {train_result.training_loss:.4f}")
print(f"random baseline loss (K=7): {np.log(7):.4f}")""")

code(r"""!nvidia-smi""")

# ----- 12. 평가 -----
md(r"""## 5. 🔬 평가 — softmax 확률 분포 + 혼동 패턴

Ch 12 의 평가 패턴을 한국어 환경에서 재현. 7클래스라 혼동 행렬이 7×7 — *어떤 카테고리가 어떤 카테고리와 헷갈리는지* 보는 데 핵심.""")

code(r"""eval_metrics = trainer.evaluate()
print("klue/bert-base KLUE-YNAT — evaluation:")
for k, v in eval_metrics.items():
    if k.startswith("eval_") and isinstance(v, float):
        print(f"  {k:>20}: {v:.4f}")""")

code(r"""preds_output = trainer.predict(eval_tok)
logits = preds_output.predictions
labels = preds_output.label_ids.astype(int)

exp = np.exp(logits - logits.max(axis=1, keepdims=True))
probs_full = exp / exp.sum(axis=1, keepdims=True)
preds = probs_full.argmax(axis=1)

top1_prob = probs_full.max(axis=1)
correct = (preds == labels)

print(f"logits shape:    {logits.shape}")
print(f"top-1 prob range: [{top1_prob.min():.4f}, {top1_prob.max():.4f}]")
print(f"top-1 prob mean: correct={top1_prob[correct].mean():.4f}, wrong={top1_prob[~correct].mean():.4f}")""")

code(r"""# 클래스별 분류 리포트
print(classification_report(
    labels, preds,
    target_names=LABEL_NAMES,
    digits=4, zero_division=0,
))""")

# ----- 13. 혼동 행렬 -----
md(r"""### 5-1. 혼동 행렬 — 어디서 헷갈리는가

행은 정답 카테고리, 열은 예측. 색은 *행 정규화 (recall)*, 숫자는 *원본 카운트*. 대각선이 진할수록 그 카테고리 재현율이 좋음.""")

code(r"""sns.set_theme(style="white", context="talk")
cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(
    cm_norm, annot=cm, fmt="d",
    cmap="Blues", vmin=0, vmax=1,
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    cbar_kws={"label": "row-normalized (recall)"}, ax=ax,
)
ax.set_xlabel("Predicted category")
ax.set_ylabel("Actual category")
ax.set_title("Confusion Matrix — KLUE-YNAT (7 categories)")
plt.tight_layout()
plt.show()""")

md(r"""**해석 가이드**

- **대각선 셀** = 그 카테고리의 재현율. 모든 셀이 0.85+ 면 잘 학습된 것.
- **오답 패턴**:
  - 정치 ↔ 경제: 둘 다 정책·법안·국제 이슈 다뤄 *경계가 모호* — 자연스러운 혼동
  - 생활문화 ↔ 사회: 사회 이슈 vs 일상·문화 보도 — 헤드라인 한 줄로는 사람도 헷갈리는 경계
  - IT과학 ↔ 경제: 기업·산업 뉴스가 양쪽에 걸침 (예: "삼성전자 4분기 실적 발표")
- **먼 클래스 혼동** (스포츠 ↔ 정치 등) 이 자주 보이면 라벨 노이즈나 학습 부족 신호.""")

# ----- 14. top-1 prob KDE -----
md(r"""### 5-2. Top-1 확률 분포 — 모델 자신감 진단

K=7 에선 *어느 한 클래스에 압도적 자신* 있는 경우 vs *2-3 후보 사이에서 갈등* 하는 경우가 나뉩니다. correct/wrong 으로 갈라 그려 calibration 확인.""")

code(r"""sns.set_theme(style="whitegrid", context="talk")
df_top = pd.DataFrame({
    "top1_prob": top1_prob,
    "outcome":   np.where(correct, "correct", "wrong"),
})

fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(
    data=df_top, x="top1_prob", hue="outcome",
    fill=True, common_norm=False, alpha=0.5,
    palette={"correct": "#5BD17F", "wrong": "#E55050"},
    clip=(1/7, 1.0), ax=ax,
)
ax.axvline(1/7, color="black", lw=1.0, ls=":", alpha=0.5)
ax.text(1/7, ax.get_ylim()[1]*0.95, "  uniform = 1/K", va="top", fontsize=10, alpha=0.6)
ax.set_title("Top-1 probability — distribution split by correctness (K=7)")
ax.set_xlabel("top-1 predicted probability  max_k P(y=k)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()""")

md(r"""**해석**

- 잘 학습된 모델은 *correct* 곡선이 1.0 가까이 몰림. *wrong* 은 더 낮은 영역 (0.4-0.7) 에 분산.
- correct/wrong 둘 다 1.0 근처에 압축돼 있으면 *over-confident* — 틀린 답에도 자신만만한 위험 신호. K 가 클수록 (7클래스) 이런 경향이 더 잘 드러남.
- *random baseline* 인 1/K = 0.143 근처 봉우리가 보이면 모델이 *판단 자체를 못 하는* 샘플 — 학습 데이터 부족 또는 헤드라인이 너무 짧은 경우.""")

# ----- 15. 샘플 단위 해석 -----
md(r"""### 5-3. 샘플 단위 해석 — 실제 헤드라인이 어떻게 분류되나

가장 자신있는 샘플 / 망설이는 샘플 / 자신있게 틀린 샘플 세 종류를 골라 직접 읽어 봅니다. 헤드라인 한 줄 만으로 모델이 어떤 카테고리 신호를 잡는지 감각.""")

code(r"""texts = list(eval_full["text"])

idx_top    = int(np.argmax(top1_prob))
idx_unc    = int(np.argmin(np.abs(top1_prob - 1/len(LABEL_NAMES) * 2)))   # 1/7 의 2배 근처 (거의 모름)
wrong_mask = ~correct
idx_wrong  = int(np.argmax(top1_prob * wrong_mask)) if wrong_mask.any() else -1

samples = [
    ("most confident overall", idx_top),
    ("most uncertain (~2/K)", idx_unc),
    ("most confident WRONG",   idx_wrong),
]

for label_str, idx in samples:
    if idx < 0:
        continue
    print("=" * 78)
    print(f"sample #{idx}  ({label_str})")
    print("=" * 78)
    print(f"text:        {texts[idx]}")
    print(f"true label:  {labels[idx]}  ({LABEL_NAMES[labels[idx]]})")
    print(f"prediction:  {preds[idx]}  ({LABEL_NAMES[preds[idx]]})  match: {'✓' if correct[idx] else '✗'}")
    print(f"top-1 prob:  {top1_prob[idx]:.4f}")
    # top-3 클래스 모두 보기
    top3 = np.argsort(probs_full[idx])[::-1][:3]
    print(f"top-3 distribution:")
    for k in top3:
        print(f"  {LABEL_NAMES[k]:>8}: {probs_full[idx, k]:.4f}")
    print()""")

md(r"""**관찰 포인트**

- *가장 자신있는* 샘플은 보통 카테고리 *시그널 단어* 가 명확 (예: "주가" → 경제, "월드컵" → 스포츠).
- *망설이는 샘플* 의 top-3 분포를 보면 모델이 *어느 카테고리 사이에서 갈팡질팡* 하는지 보임. 정치/경제/사회 셋이 비슷한 확률이면 헤드라인 자체가 다중 카테고리에 걸침.
- *자신있게 틀린* 샘플은 보통 *반어*, *비유*, *카테고리 간 경계 사례* — 학습 데이터에 비슷한 패턴이 없었거나 라벨 자체가 모호. 이걸 보면 "모델이 *바보* 라서 틀린 게 아니라 *데이터가 어렵다* " 는 감각 잡힘.""")

# ----- 16. library -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | 다음 챕터에서 |
|---|---|---|
| `load_dataset("klue", "ynat")` | KLUE 벤치마크 YNAT (한국어 뉴스 7분류) | Ch 17 에서 같은 데이터로 multi-label 합성 |
| `ds["train"].features["label"].names` | datasets.ClassLabel 의 사람-읽는 이름 | id2label 자동 매핑에 사용 |
| `seaborn.heatmap(..., xticklabels=한국어)` | 혼동 행렬 한국어 라벨 표시 | Ch 17 도 사용 |
| `sklearn.metrics.precision_recall_fscore_support(..., average="macro")` | 클래스별 평균 metric (불균형에 강함) | 분류 챕터마다 |
| `roc_auc_score(..., multi_class="ovr")` | multi-class AUC (One-vs-Rest) | Ch 17 multi-label 평가 |""")

# ----- 17. checkpoints -----
md(r"""## 🎯 체크포인트 질문

1. K=7 학습 첫 step 의 loss 가 약 1.95 정도라면 모델이 무엇을 학습한 상태인가요?
2. macro F1 이 weighted F1 보다 *훨씬* 낮으면 무엇을 의심해야 하나요?
3. 혼동 행렬에서 *정치 ↔ 경제* 혼동이 많은데 *스포츠 ↔ 정치* 혼동은 거의 없는 이유는?
4. 같은 klue/bert-base 가 NSMC binary (Ch 15) 와 KLUE-YNAT 7분류 (Ch 16) 에서 *분류 헤드만* 다른데, 왜 두 task 모두 잘 동작하나요?""")

# ----- 18. FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) 한국어 multi-class 데이터셋이 KLUE-YNAT 외에 어떤 게 있나요?

| 데이터셋 | 도메인 | 클래스 수 | 크기 |
|---|---|---|---|
| **KLUE-YNAT** (이번 챕터) | 뉴스 헤드라인 | 7 | 45K train |
| KLUE-TC (KLUE Topic Classification) | 짧은 문서 토픽 | 7 | 같은 KLUE 시리즈 |
| AI Hub 뉴스 분류 | 뉴스 본문 | 50+ | 100K+ (가입 필요) |
| 모두의 말뭉치 신문 코퍼스 | 신문 기사 | 다양 | 매우 큼 (라이선스 확인 필요) |
| Naver shopping 카테고리 분류 | 상품 설명 | 100+ | 1M+ (실무에선 흔히 다룸) |

KLUE-YNAT 가 *입문에 편한 이유* — datasets.load_dataset 한 줄, 깔끔한 라벨, 균형 분포에 가까움, 헤드라인 한 줄이라 max_length 짧음.

### Q2. (이론) 혼동 행렬에서 *대칭* 인 혼동과 *비대칭* 인 혼동은 무슨 차이인가요?

- **대칭 혼동** (예: 정치↔경제 양방향 비슷): 두 카테고리 *경계가 모호* 하다는 신호. 헤드라인 한 줄로는 *사람도 헷갈리는* 데이터.
- **비대칭 혼동** (예: 정치 → 경제 는 흔한데 경제 → 정치 는 드뭄): 모델이 *한쪽으로 편향* 됨. 가능한 원인:
  - 학습 데이터 *클래스 불균형* (한쪽이 많아 그쪽으로 답을 미는 경향)
  - *시그널 단어* 가 한쪽에 더 강하게 학습됨 (예: "정부", "정책" 이 정치 카테고리에 너무 강하게 매핑)

비대칭 혼동이 보이면 `class_weight` 적용 또는 *under-represented 클래스 oversampling* 으로 처치.

### Q3. (실무) `class_weight` 를 multi-class CE 에 적용하려면?

```python
import torch
from torch import nn

# 클래스별 빈도 → 가중치 (적은 클래스에 큰 가중)
class_counts = np.bincount(train_full["label"], minlength=len(LABEL_NAMES))
class_weights = len(train_full) / (len(LABEL_NAMES) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

KLUE-YNAT 는 *심한* 불균형이 아니라 (5K-8K 범위) class_weight 효과가 *작은 폭* — 완전 균형이 아닌 데이터에서도 보통 +1-2%p macro F1 정도 개선. 사용 결정은 *macro F1 vs accuracy* 의 격차로.

### Q4. (이론) 헤드라인 한 줄 (~30 토큰) 이 분류에 *충분* 한가요?

대부분의 경우 충분합니다 — *카테고리* 라는 task 가 *키워드 + 표현 스타일* 로 풀리고, 그 신호가 헤드라인에 *압축* 되어 들어 있어요. 예: "월드컵 한국 vs 일본 16강전 시작" → 스포츠 신호 명확.

부족한 경우:
- *제목과 본문이 정반대* 인 풍자 기사 (드뭄)
- *정치/경제 처럼 경계가 모호* 한 카테고리 (위 Q2)
- *지나치게 일반적* 인 헤드라인 ("어제 뉴스 정리")

이런 케이스엔 *기사 본문* 까지 같이 입력하면 정확도 +5-8%p 가능. 단 max_length 가 길어져 학습 비용 ↑.

### Q5. (실무) 한국어 BERT 가 *짧은 헤드라인* 에 *왜 그렇게* 잘 동작하나요?

`klue/bert-base` 의 사전학습 코퍼스에 *뉴스 + 위키* 가 큰 비중을 차지합니다. 이미 모델 weight 가 한국어 뉴스 텍스트의 *언어 분포* 를 학습한 상태에서 fine-tune 하니, *적은 데이터·짧은 학습* 으로도 좋은 성능.

영어 BERT 가 NLI 같은 *추론 task* 에서 사전학습 효과가 큰 것과 같은 원리 — *task 의 도메인이 사전학습 코퍼스와 가까울수록* fine-tune 효과가 큼.

### Q6. (실무) 클래스 수가 50, 100 으로 늘면 같은 코드가 동작하나요?

코드는 그대로 동작합니다 (`num_labels=100` 만 바꾸면 됨). 하지만 *학습 동역학* 이 달라집니다:

| K | random baseline loss | 학습 도전 |
|---|---|---|
| 7 (이번 챕터) | $\log 7 = 1.95$ | 무난 |
| 50 | $\log 50 = 3.91$ | 학습 데이터가 *클래스당 충분* 해야 |
| 100 | $\log 100 = 4.60$ | 클래스 불균형이 심하면 자주 collapse |
| 1000 | $\log 1000 = 6.91$ | hierarchical softmax 등 특수 기법 고려 |

K 가 50+ 가 되면 *클래스당 학습 샘플* 이 핵심 — 클래스당 100 개 미만이면 BERT 정확도 떨어짐. KLUE-YNAT 는 클래스당 5K-8K 라 *풍족한* 셋업.""")

# ----- 19. 삽질 -----
md(r"""## 🚀 삽질 코너 (선택)

다음 코드를 돌려보면 어떤 결과가 나올까요?

```python
# K=7 인데 num_labels=2 로 모델 만들기 (Ch 15 그대로 복붙)
model_wrong = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=2,   # ← 잘못 (실제는 K=7)
)
# ... 같은 학습 코드 ...
```

힌트: 학습 시 `CrossEntropyLoss(logits.shape=(B, 2), targets in 0-6)` 가 호출되는데 target 값이 logits 의 클래스 차원 범위를 *벗어남* → IndexError 또는 *runtime cuda assert*. 모델 헤드가 라벨 범위와 *일치해야* 한다는 단순하지만 흔한 실수.""")

# ----- 20. next -----
md(r"""## 다음 챕터 예고

**Chapter 17. 한국어 BERT Multi-label — KLUE-YNAT 합성 multi-label**

- 같은 데이터·같은 모델, 단 *task 만* single-label → multi-label
- KLUE-YNAT 헤드라인 *두 개를 결합* 해 인공 multi-label 데이터 합성 (Ch 13 의 측면 합성과 비슷한 패턴)
- `num_labels=7` 그대로, 단 `problem_type="multi_label_classification"` 으로 전환
- Activation: per-label sigmoid, Loss: per-label `BCEWithLogitsLoss`
- Ch 13 의 한국어 버전. 한 헤드라인이 *여러 카테고리에 걸칠 수 있는* 상황을 시뮬레이션

> **변하는 축**: Phase 2 안에서 *task 차원* (single-label → multi-label). 모델·토크나이저·hyperparams 는 그대로.""")


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT.relative_to(REPO)}  ({len(cells)} cells)")
