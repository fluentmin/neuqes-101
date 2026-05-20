"""Build 21_gpt_tinystories/21_gpt_tinystories.ipynb — GPT prototype (HF API).

prototype 노트북: GPT2LMHeadModel from scratch + TinyStories 작은 subset 학습 + 생성.
Trainer + DataCollatorForLanguageModeling(mlm=False) 패턴으로 BERT 챕터들과 톤 일관성 유지.
정식 챕터 분할(해부 / 학습 / 한국어)은 prototype 검증 후 결정.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "21_gpt_tinystories" / "21_gpt_tinystories.ipynb"

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
md(r"""# Chapter 21 (prototype). GPT — TinyStories 로 작은 언어모델 학습

**목표**: BERT(encoder, MLM) 와 결이 다른 **GPT (decoder-only, causal LM)** 를 \
`GPT2LMHeadModel` 로 *from scratch* (사전학습 없이) 띄우고, **TinyStories** 작은 subset 으로 \
`Trainer` 학습 → `model.generate()` 로 "Once upon a time…" 짧은 동화 생성.

> ⚠️ **이 노트북은 prototype 입니다.** 정식 챕터 분할 전에, \
> 한 노트북에 *셋업 + 학습 + 생성* 을 압축해 "T4 30분 안에 GPT 가 진짜 동작하는가" 를 확인합니다. \
> 검증 결과가 좋으면 정식 챕터로 분할: 해부 (Ch 21) / 본격 학습 (Ch 22) / 한국어 (Ch 23 선택).

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 25-30분 (데이터 로드 ~2분 + BPE 학습 ~3분 + 모델 학습 ~18분 + 생성 ~30초)

---

## 학습 흐름

1. 🚀 **실습**: TinyStories 30k stories → BPE 토크나이저 학습 → 작은 `GPT2LMHeadModel`(~3M params) 학습 → 생성
2. 🔬 **해부**: `GPT2Config` 의 핵심 필드, causal attention, weight tying 이 BERT 와 어떻게 다른가
3. 🛠️ **변형**: `model.generate` 의 sampling hyperparam (temperature, top_k, top_p) 비교

---

> 📒 **사전 학습 자료**: Ch 7-16 (BERT MLM/분류) 의 `AutoModelForXxx + Trainer` 패턴. \
> 이번 챕터는 같은 패턴인데 모델 클래스만 `BertForMaskedLM` → `GPT2LMHeadModel`, \
> `DataCollatorForLanguageModeling(mlm=True)` → `mlm=False` 로 바뀜.""")

# ----- 2. 변화추적표 -----
md(r"""## 📊 변화추적표

| Ch | 모델 클래스 | 토크나이저 | 데이터 | Output Head | Loss |
|---|---|---|---|---|---|
| 7-14 | `DistilBertForXxx` (encoder) | WordPiece (사전학습) | 영어 분류 | `Linear(H, K)` | CE / BCE / MSE |
| 15-16 | `BertForSequenceClassification` (klue) | WordPiece (사전학습) | 한국어 분류 | `Linear(H, K)` | CE |
| 19-20 (예정) | — (토크나이저 자체) | BPE *직접 학습* | TinyStories | — | — |
| **21 ← 여기** | **`GPT2LMHeadModel`** (decoder, **from scratch**) | **BPE *직접 학습*** | **TinyStories** | **`Linear(H, V)` (weight tied 자동)** | **CE (next-token, `mlm=False`)** |
| 22 (다음) | 같음, 학습 더 길게 | 같음 | TinyStories 키움 | 같음 | 같음 |

**이번 챕터의 변화**: 모델 패밀리 — encoder → **decoder (causal)**. \
Loss 수식은 BERT MLM 의 CE 와 동일, 마스킹 위치만 다름 (BERT: 무작위 15% / GPT: 모든 토큰의 다음 위치).""")

# ----- 3. 변경점 -----
md(r"""## 🔄 변경점 (Diff from BERT 챕터들)

| 축 | BERT (Ch 7-16) | GPT (Ch 21) |
|---|---|---|
| **모델 패밀리** | Encoder, bidirectional attention | **Decoder, causal attention** ← *핵심 변화* |
| 사전학습 | 사전학습된 가중치 로드 (`from_pretrained`) | **무작위 초기화 from scratch** (`GPT2LMHeadModel(config)`) |
| 학습 목표 | MLM (15% 토큰 마스킹) | **Next-token prediction** (모든 토큰) |
| 토크나이저 | WordPiece (사전학습된 것) | **BPE (코퍼스에서 직접 학습)** |
| Data collator | `DataCollatorForLanguageModeling(mlm=True)` | **`mlm=False`** — labels = input_ids 자동 |
| 결과 | 분류 확률 | **`model.generate()` 로 생성된 텍스트** |

**왜 이렇게 한 번에 많이 바꾸나** — prototype 이라 한 노트북에 압축. 정식 챕터로 분할할 때는 \
"변경점 한 가지 원칙" 에 맞춰 모델 축 / 데이터 축 / loss 축으로 쪼갤 예정입니다.""")

# ----- 4. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업""")

code(r"""%pip -q install -U transformers tokenizers datasets accelerate""")

code(r"""import torch, time, os, random
import numpy as np

assert torch.cuda.is_available(), "Runtime > Change runtime type > GPU (T4) 로 바꿔주세요"
device = torch.device("cuda")
print("torch     :", torch.__version__)
print("device    :", torch.cuda.get_device_name(0))
print("VRAM total:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

# 재현성
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)""")

# ----- 5. 데이터 -----
md(r"""## 1. TinyStories 데이터 로드

`roneneldan/TinyStories` 는 GPT-3.5 / GPT-4 가 *4세 어린이가 이해할 단어만* 으로 \
생성한 짧은 영어 동화 모음입니다 (Eldan & Li 2023). 어휘·문법이 단순해 \
**3-5M 파라미터** 짜리 작은 모델로도 grammatical 한 생성이 나옵니다.

prototype 이라 학습 split 의 처음 **30,000 stories** 만 씁니다.""")

code(r"""from datasets import load_dataset

N_TRAIN = 30_000      # 더 길게 돌리려면 키우세요 (full 은 약 2.1M stories)
N_VAL   = 500

raw_train = load_dataset("roneneldan/TinyStories", split=f"train[:{N_TRAIN}]")
raw_val   = load_dataset("roneneldan/TinyStories", split=f"validation[:{N_VAL}]")
print("train:", raw_train)
print("val  :", raw_val)
print("\n=== sample story ===")
print(raw_train[0]["text"][:400])""")

# ----- 6. 토크나이저 -----
md(r"""## 2. BPE 토크나이저 학습 → `PreTrainedTokenizerFast`

GPT-2 와 같은 종류 (byte-level BPE) 의 작은 vocab 을 직접 학습합니다. \
정식 챕터 (Ch 19-20) 에서 다룰 토크나이저 학습의 미니 버전.

- `vocab_size=2048` — 작은 모델에 맞춰 작게
- 특수 토큰: `<|endoftext|>` 하나만 (GPT-2 컨벤션, **bos = eos = pad** 겸용)""")

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
print(f"BPE 학습 완료: {time.time()-t0:.1f}s, vocab={bpe.get_vocab_size()}")

# HF 인터페이스로 wrap — bos/eos/pad 모두 <|endoftext|> 로
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=bpe,
    bos_token=EOS,
    eos_token=EOS,
    pad_token=EOS,
)

print("\n=== encode/decode 시연 ===")
sample = "Once upon a time, a little rabbit went to the forest."
enc = tokenizer(sample)
print("ids        :", enc["input_ids"])
print("tokens     :", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print("decode     :", tokenizer.decode(enc["input_ids"]))
print("vocab_size :", tokenizer.vocab_size)
print("eos_token  :", tokenizer.eos_token, " id =", tokenizer.eos_token_id)""")

# ----- 7. 토큰화 + group_texts -----
md(r"""## 3. 토큰화 + `group_texts` (HF 표준 CLM 전처리)

HuggingFace 의 causal LM 학습 표준 패턴 (`run_clm.py`) 그대로:

1. 전체 코퍼스를 토큰화 (배치 단위)
2. 모든 토큰을 EOS 로 이어붙여 1D 스트림으로 만든 뒤 `block_size` 단위로 잘라 chunk 화
3. 각 chunk 가 한 학습 sample — `DataCollatorForLanguageModeling(mlm=False)` 가 \
   `labels = input_ids` 를 자동으로 채워 next-token prediction loss 가 됨""")

code(r"""BLOCK_SIZE = 128

def tokenize_fn(batch):
    return tokenizer(batch["text"])

# 토큰화 (text 컬럼 제거)
tok_train = raw_train.map(tokenize_fn, batched=True, remove_columns=["text"], desc="tokenize train")
tok_val   = raw_val.map(tokenize_fn,   batched=True, remove_columns=["text"], desc="tokenize val")

# 각 sample 끝에 EOS 붙이기 (story 경계 표시)
def add_eos(batch):
    new_ids, new_mask = [], []
    for ids in batch["input_ids"]:
        ids = ids + [tokenizer.eos_token_id]
        new_ids.append(ids)
        new_mask.append([1] * len(ids))
    return {"input_ids": new_ids, "attention_mask": new_mask}

tok_train = tok_train.map(add_eos, batched=True, desc="add eos train")
tok_val   = tok_val.map(add_eos,   batched=True, desc="add eos val")

# group_texts — 모든 토큰을 이어붙여 BLOCK_SIZE 단위로 자름
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
print(f"≈ train tokens: {len(lm_train) * BLOCK_SIZE / 1e6:.2f} M")
print("\n첫 chunk decode (앞 200자):")
print(tokenizer.decode(lm_train[0]["input_ids"])[:200])""")

# ----- 8. 모델 -----
md(r"""## 4. `GPT2LMHeadModel` from scratch

`GPT2Config` 의 핵심 필드만 작게 잡고 가중치 초기화부터 (사전학습 X) 시작.

- `n_layer=4, n_head=4, n_embd=256` → 약 3M params, BERT 챕터들의 small DistilBERT 와 비슷한 스케일
- `n_positions = BLOCK_SIZE = 128` — 학습한 만큼만 context 사용
- bos/eos/pad token id 를 토크나이저와 동기화

**BERT 와의 차이가 코드로 드러나는 곳**:
- `BertForXxx` 가 아니라 `GPT2LMHeadModel` — 클래스 자체가 causal attention 내장
- `from_pretrained(...)` 없이 `GPT2LMHeadModel(config)` — 무작위 초기화
- HF 는 weight tying 을 자동 처리 (`tie_word_embeddings=True` 가 기본)""")

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

model = GPT2LMHeadModel(config)
n_params = model.num_parameters()
print(f"#params           : {n_params/1e6:.2f} M")
print(f"weight tying      : {config.tie_word_embeddings}  (lm_head ↔ wte 공유)")
print(f"fp32 weight size  : {n_params * 4 / 1024**2:.2f} MiB\n")
print(model)""")

# ----- 9. 학습 -----
md(r"""## 5. `Trainer` 로 학습

BERT 챕터들과 *완전히 같은* 패턴 — 바뀌는 건 모델 클래스와 `mlm=False` 두 곳뿐.

- `DataCollatorForLanguageModeling(mlm=False)` → labels = input_ids (next-token prediction)
- `max_steps=1500`, `bs=32`, `fp16=True` — T4 약 15-18 분
- `eval_steps=150` 으로 train/val loss 추이 관찰""")

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
    fp16=True,                           # T4 는 bf16 불가
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
    '''step 별 peak VRAM 기록 (로깅 윈도우 단위로 reset).'''
    def __init__(self):
        self.steps, self.peak_MiB = [], []
    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
    def on_log(self, args, state, control, logs=None, **kwargs):
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

print(f"\n=== summary ===")
print(f"elapsed       : {elapsed/60:.2f} min")
print(f"global_step   : {train_out.global_step}")
print(f"train_loss    : {train_out.training_loss:.4f}")
print(f"final peak    : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")""")

code(r"""import matplotlib.pyplot as plt
import math

# trainer.state.log_history 에서 train loss / eval loss 분리
log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]
eval_pts  = [(r["step"], r["eval_loss"]) for r in log if "eval_loss" in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# loss
ax1.plot([s for s,_ in train_pts], [l for _,l in train_pts], "-",
         color="tab:blue", alpha=0.6, label="train")
ax1.plot([s for s,_ in eval_pts],  [l for _,l in eval_pts],  "s-",
         color="tab:red", label="eval")
ax1.axhline(math.log(tokenizer.vocab_size), ls=":", color="gray",
            label=f"uniform baseline = ln({tokenizer.vocab_size}) ≈ {math.log(tokenizer.vocab_size):.2f}")
ax1.set_xlabel("step"); ax1.set_ylabel("cross-entropy loss")
ax1.set_title("TinyGPT-2 on TinyStories — loss")
ax1.grid(True, alpha=0.3); ax1.legend()

# VRAM
ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
         label="peak VRAM (per log window)")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.set_title(f"VRAM trace  (bs=32, fp16, n_pos={BLOCK_SIZE})")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()""")

# ----- 10. 생성 -----
md(r"""## 6. `model.generate()` 로 생성

학습된 모델에 prompt 를 주고 `model.generate(do_sample=True, ...)` 로 이어지는 텍스트를 sampling.

- `temperature` ↓ → 보수적, ↑ → 다양
- `top_k` → 상위 k 개만 후보
- `top_p` (nucleus) → 누적 확률 p 까지의 후보만 (top_k 대안)""")

code(r"""model.eval()

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens=120, temperature=0.8, top_k=50, top_p=None):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

prompts = [
    "Once upon a time,",
    "There was a little girl named Lily.",
    "The cat and the dog were best friends.",
]

for p in prompts:
    print("=" * 70)
    print("[prompt]", p)
    print("-" * 70)
    print(generate_text(p, max_new_tokens=120, temperature=0.8, top_k=50))
    print()""")

# ----- 11. 변형 -----
md(r"""## 🛠️ 변형 — sampling hyperparam 비교

같은 prompt 에 `temperature` / `top_k` / `top_p` 만 바꿔 생성 스타일 변화 관찰.""")

code(r"""prompt = "Once upon a time, a little rabbit"
configs = [
    {"label": "T=0.3, top_k=20  (보수적)",   "temperature": 0.3, "top_k": 20,  "top_p": None},
    {"label": "T=0.8, top_k=50  (균형)",     "temperature": 0.8, "top_k": 50,  "top_p": None},
    {"label": "T=1.0, top_p=0.9 (nucleus)",  "temperature": 1.0, "top_k": 0,   "top_p": 0.9},
    {"label": "T=1.2, top_k=100 (다양)",     "temperature": 1.2, "top_k": 100, "top_p": None},
]
for c in configs:
    print("=" * 70)
    print(f"[{c['label']}]")
    print(generate_text(prompt, max_new_tokens=80,
                        temperature=c["temperature"], top_k=c["top_k"], top_p=c["top_p"]))
    print()""")

# ----- 12. FAQ -----
md(r"""## ❓ FAQ

### Q1. `GPT2LMHeadModel` 과 `BertForMaskedLM` 의 진짜 핵심 차이는?

내부 attention 의 *causal mask* 입니다. `GPT2Attention` 은 `lower-triangular mask` 를 항상 적용해 \
토큰 *i* 가 *j* 를 *i ≥ j* 일 때만 볼 수 있게 합니다. 이 마스크 덕분에 *next-token prediction* 만으로 \
학습 가능 — 미래를 못 보니 모든 위치에서 다음 토큰을 예측해도 cheating 이 안 됩니다. \
BERT 는 양방향 attention 이라 어떤 토큰의 hidden 도 좌·우 컨텍스트를 다 보므로, \
일부 토큰을 [MASK] 로 가려야만 의미 있는 학습 신호가 생깁니다.

코드 수준에서는 `DataCollatorForLanguageModeling` 의 `mlm` 플래그 하나로 갈립니다:

```python
DataCollatorForLanguageModeling(tokenizer, mlm=True,  mlm_probability=0.15)  # BERT
DataCollatorForLanguageModeling(tokenizer, mlm=False)                         # GPT
```

### Q2. Loss 가 4 근처에서 멈추는데 정상인가요?

vocab=2048 의 uniform random baseline 이 `ln(2048) ≈ 7.625` 이니 **4.0 이면 random 대비 충분히 학습된 상태**입니다. \
TinyStories 처럼 어휘·구조가 단순한 데이터에서 3M 파라미터 모델은 보통 **2.5-3.5 가 한계** — \
그 이하로 가려면 (a) `N_TRAIN` 키우기 (b) `max_steps` 늘리기 (c) `n_layer / n_embd` 키우기 셋 중 하나.

### Q3. 생성된 문장이 어색합니다. 어떻게 좋아지나요?

이번 prototype 셋업 (15-18 분 학습) 은 *grammatical 한 문장* 까지가 목표입니다. *재미있는 스토리* 까지 가려면:

- `N_TRAIN` 30k → 200k (학습 토큰 자체가 부족)
- `max_steps` 1500 → 5000
- `n_layer` 4 → 6, `n_embd` 256 → 384

T4 에서 위 셋업이면 약 1 시간, eval loss 가 2.5 부근까지 떨어지면서 생성 품질도 한 단계 좋아집니다.

### Q4. 왜 `from_pretrained` 없이 `GPT2LMHeadModel(config)` 인가요?

BERT 챕터들은 `bert-base-uncased` 같은 *사전학습된* 가중치를 받아 fine-tune 했습니다. \
이번 챕터는 (1) 코퍼스가 작고 (TinyStories) (2) 토크나이저를 직접 학습한 새 BPE 라 \
사전학습 GPT-2 의 가중치와 vocab 이 맞지 않습니다. \
또한 *from scratch* 학습이 어떻게 도는지 직접 보는 것이 prototype 의 핵심 — \
실무에서는 보통 사전학습 GPT-2 / GPT-Neo 등을 fine-tune 합니다.

### Q5. `n_positions = BLOCK_SIZE` 로 같이 두는 이유는?

`n_positions` 는 position embedding 의 길이 (학습 가능한 파라미터). \
`BLOCK_SIZE` 보다 작으면 학습 시 IndexError, 크면 사용 안 되는 position 만큼 파라미터 낭비. \
둘을 같이 두면 학습한 인덱스 범위와 모델 한계가 일치합니다. \
더 긴 context 가 필요하면 학습 때부터 둘을 같이 키우거나, ALiBi / RoPE 같은 길이 외삽 가능한 \
positional 방식의 모델 (e.g. `LlamaForCausalLM`) 로 바꿔야 합니다.

### Q6. `tokenizer.pad_token = tokenizer.eos_token` 으로 둬도 되나요?

GPT-2 컨벤션입니다. **단, attention_mask 가 반드시 함께** 가야 패딩 위치를 무시합니다. \
이번 노트북은 `group_texts` 로 모든 chunk 길이가 `BLOCK_SIZE` 로 같아 패딩 자체가 없습니다. \
서로 다른 길이의 sample 로 학습한다면 `DataCollatorForLanguageModeling` 이 패딩과 \
attention_mask 를 함께 만들어 줍니다.

### Q7. `Trainer` 대신 직접 학습 루프를 짜야 할 때는?

(1) 비표준 sampler (예: random chunk sampling, contrastive batch) (2) `compute_loss` 가 \
복잡하게 여러 head 의 loss 를 조합 (Ch 14 의 auxiliary loss 가 그런 케이스) (3) gradient accumulation \
이상의 미세한 backward 제어. \
이번 셋업은 표준 CLM 이라 `Trainer` 가 깔끔합니다.""")

# ----- 13. 다음 단계 -----
md(r"""## 🚀 다음 단계 (prototype 검증 후)

이 노트북이 T4 30 분 안에 정상 돌고 grammatical 한 생성이 나오면, 다음 분할로 정식 챕터화:

- **Ch 21 (정식)** — GPT 아키텍처 해부. `GPT2Config` 의 필드별 의미, causal vs bidirectional attention \
  비교, weight tying, position embedding 까지. 학습은 매우 짧은 sanity check 만.
- **Ch 22** — TinyStories 본격 학습 + 생성. 데이터·step·모델 키워서 더 좋은 생성 품질.
- **Ch 23 (선택)** — 같은 셋업을 *from-scratch `nn.Module`* 로 구현. HF 클래스의 내부가 \
  실제로 200 줄 PyTorch 로 펼쳐지는 모습을 확인. HF API ↔ from-scratch 동등성 시연.
- **Ch 24 (선택)** — 한국어 작은 GPT (KLUE / 위키 일부).

prototype 검증 체크리스트:
- [ ] T4 에서 25-30 분 안에 끝까지 실행되는가
- [ ] eval loss 가 학습 중 *단조감소* 하는가 (대략 7 → 3-4)
- [ ] 생성된 문장이 *grammatical* 한가 (반복·횡설수설 아님)
- [ ] peak VRAM 이 T4 16 GiB 안에 충분히 들어오는가 (예상 < 3 GiB)
- [ ] BPE 토크나이저가 합리적으로 단어를 쪼개는가""")


# ----- 마무리: 노트북 저장 -----
NOTEBOOK = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
