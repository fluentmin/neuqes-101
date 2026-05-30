"""Build 10_bert_binary_sigmoid/appendix_profiling.ipynb.

성능 부록 — (1) 정확도 + (2) 학습 프로파일 + (3) 추론 프로파일.
프로파일러는 Accelerate ProfileKwargs(= torch.profiler 래퍼) 하나로 통일.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "10_bert_binary_sigmoid" / "appendix_profiling.ipynb"

cells = []
_counter = 0


def _cid():
    global _counter
    _counter += 1
    return f"cell{_counter:03d}"


def md(text: str):
    cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": text})


def code(text: str):
    cells.append({
        "cell_type": "code", "id": _cid(), "execution_count": None,
        "metadata": {}, "outputs": [], "source": text,
    })


# ----- 제목 -----
md(r"""# Ch 10 부록 — 정확도 + 성능 프로파일 (학습 & 추론)

Ch 10 의 **DistilBERT binary (sigmoid + BCE)** 를 실제로 학습해 **정확도** 를 확인하고, \
그 **학습** 과 **추론** 이 각각 GPU 에서 어디에 시간·메모리를 쓰는지 **프로파일** 합니다.

세 가지를 한 노트북에서:
1. **정확도** — 짧게 학습한 뒤 eval set 에서 accuracy / precision / recall / F1 / AUC (Ch 10 본편과 같은 5종)
2. **학습 프로파일** — forward / backward / optimizer 의 시간·메모리 분해
3. **추론 프로파일** — forward only. 학습 대비 시간·메모리가 얼마나 줄고 병목이 어떻게 다른지

**프로파일러**: **Accelerate `ProfileKwargs`** 하나로 통일합니다. 이건 `torch.profiler` 의 얇은 래퍼라 \
출력(op 테이블·schedule·메모리·chrome trace)이 torch.profiler 와 동일하면서, HF 생태계 \
(Trainer/Accelerate/DeepSpeed/FSDP)와 single↔multi GPU 에 같은 코드로 붙습니다. \
FLOPS/params 도 같은 프로파일에서 뽑습니다.

> 멀티 GPU / DeepSpeed / Nsight 같은 *분산* 프로파일링은 별도 부록에서 다룹니다 (이 부록은 단일 GPU).

**환경**: Google Colab **T4 GPU 권장** (CPU 도 동작, CUDA 줄은 자동 skip). 약 8-12분.

---

> 📒 **선행**: Ch 10 (DistilBERT binary, sigmoid+BCE).""")

# ----- 0. 환경 -----
md(r"""## 0. 환경 셋업""")

code(r"""%pip -q install -U transformers datasets accelerate scikit-learn""")

code(r"""import torch, time
import numpy as np

# device 자동 감지 — Colab CUDA / 로컬 MPS / CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

CUDA = device.type == "cuda"
print("torch  :", torch.__version__)
print("device :", device)
if CUDA:
    print("gpu    :", torch.cuda.get_device_name(0))

def sync():
    # 정확한 시간 측정을 위해 가속기 작업이 끝날 때까지 대기 (CUDA / MPS 모두 비동기)
    if CUDA:
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()""")

# ----- 1. 무엇을 보나 -----
md(r"""## 1. 이 부록이 보는 것

| 축 | 질문 | 어디서 |
|---|---|---|
| **정확도** | 모델이 잘 맞히나? | §3 — 학습 후 eval metric 5종 |
| **학습 프로파일** | 학습 1 step 이 어디서 느린가? | §4 — forward/backward/optimizer 분해 |
| **추론 프로파일** | 추론은 학습과 무엇이 다른가? | §5 — forward only, 메모리·시간 비교 |

프로파일러는 "어디가 느린지/무거운지", 정확도는 "그 모델이 쓸모 있는지" 를 알려줍니다. \
둘을 같이 봐야 *"빠르면서 정확한"* 지점을 찾습니다.

**도구는 하나** — Accelerate `ProfileKwargs`. torch.profiler 의 래퍼이고, 출력 읽는 법은 \
모두 torch.profiler 기준입니다 (Accelerate 는 "켜는 법"만 HF 답게 만들어줄 뿐).""")

# ----- 2. 대상 준비 -----
md(r"""## 2. 대상 준비 — DistilBERT binary + train/eval split

Ch 10 과 동일한 `distilbert-base-uncased` + `num_labels=1` + `BCEWithLogitsLoss` 셋업. \
정확도를 보려면 학습/평가 데이터가 필요하니 train 4,000 / eval 1,000 으로 나눕니다 \
(label 균형을 위해 `shuffle`).""")

code(r"""from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

MODEL = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
N_TRAIN, N_EVAL = 4000, 1000

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def load_split(split, n):
    raw = load_dataset("fancyzhx/yelp_polarity", split=split).shuffle(seed=42).select(range(n))
    enc = raw.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN, padding="max_length"),
        batched=True, remove_columns=raw.column_names,
    )
    enc = enc.add_column("labels", [[float(l)] for l in raw["label"]])   # (B,1) float for BCE
    enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return enc

train_ds = load_split("train", N_TRAIN)
eval_ds  = load_split("test",  N_EVAL)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(eval_ds,  batch_size=32, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, problem_type="multi_label_classification",
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"#params : {model.num_parameters()/1e6:.1f} M")
print(f"train   : {len(train_ds)} / eval : {len(eval_ds)}")""")

# ----- 3. 정확도 -----
md(r"""## 3. 정확도 — 짧게 학습하고 평가

먼저 **학습 전(무작위 분류 헤드)** 정확도를 보고, 1 에폭 학습한 뒤 **학습 후** 와 비교합니다. \
binary 5종 metric: accuracy / precision / recall / F1 / AUC.""")

code(r"""from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits_all.append(model(**batch).logits.float().cpu())
        labels_all.append(batch["labels"].cpu())
    logits = torch.cat(logits_all).squeeze(-1)
    labels = torch.cat(labels_all).squeeze(-1).int().numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p, "recall": r, "f1": f1,
        "auc": roc_auc_score(labels, probs),
    }, probs, labels

before, _, _ = evaluate(model, eval_loader)
print("[before training]", {k: round(v, 4) for k, v in before.items()})""")

code(r"""# 1 에폭 학습
model.train()
t0 = time.time()
for step, batch in enumerate(train_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if step % 50 == 0:
        print(f"step {step:4d}/{len(train_loader)}  loss {out.loss.item():.4f}")
print(f"train done: {time.time()-t0:.1f}s")

after, probs, labels = evaluate(model, eval_loader)
print("\n[after training]", {k: round(v, 4) for k, v in after.items()})""")

code(r"""import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

keys = list(after.keys())
x = np.arange(len(keys))
ax1.bar(x - 0.2, [before[k] for k in keys], 0.4, label="before", color="tab:gray")
ax1.bar(x + 0.2, [after[k]  for k in keys], 0.4, label="after",  color="tab:blue")
ax1.set_xticks(x); ax1.set_xticklabels(keys, rotation=20)
ax1.set_ylim(0, 1.05); ax1.set_ylabel("score")
ax1.set_title("Accuracy metrics — before vs after 1 epoch")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.hist(probs[labels == 0], bins=30, alpha=0.6, label="true 0 (neg)", color="tab:red")
ax2.hist(probs[labels == 1], bins=30, alpha=0.6, label="true 1 (pos)", color="tab:green")
ax2.axvline(0.5, ls=":", color="gray", label="threshold 0.5")
ax2.set_xlabel("predicted sigmoid prob"); ax2.set_ylabel("count")
ax2.set_title("Score distribution after training")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()""")

# ----- 4. 학습 프로파일 (Accelerate) -----
md(r"""## 4. 학습 프로파일 — Accelerate `ProfileKwargs`

*이 학습이 어디에 시간·메모리를 쓰는지* 봅니다. **Accelerate `ProfileKwargs`** 로 프로파일을 켜는데, \
이건 내부적으로 `torch.profiler` 라 출력·읽는 법이 동일합니다.

`ProfileKwargs` 핵심 인자 (그대로 torch.profiler 로 전달됨):
- `activities` — CPU / CUDA 중 무엇을 추적할지 (CUDA 면 GPU 커널까지)
- `schedule_option` — 장기 학습에서 *몇 step 만* (`wait`/`warmup`/`active`)
- `profile_memory=True` — **각 device(CPU/GPU) 메모리도 컬럼으로**
- `with_flops=True` — op 별 FLOPs 추정 (§6 에서 사용)
- `record_function("name")` — 코드 구간에 직접 라벨 (torch.profiler 기능, 코드에 직접 넣음)""")

code(r"""from accelerate import Accelerator, ProfileKwargs
from torch.profiler import record_function

# Accelerate ProfileKwargs = torch.profiler 래퍼. single↔multi GPU 같은 코드.
# (accelerate 없는 순수 PyTorch 면 torch.profiler.profile 을 직접 — API 동일)
profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"] if CUDA else ["cpu"],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    schedule_option={"wait": 1, "warmup": 1, "active": 3},   # 5 step 만
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
model, optimizer, train_loader, eval_loader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)

model.train()
with accelerator.profile() as prof:
    for step, batch in enumerate(train_loader):
        with record_function("forward"):
            out = model(**batch); loss = out.loss
        with record_function("backward"):
            accelerator.backward(loss)
        with record_function("optimizer"):
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        prof.step()
        if step >= 4:
            break

print("프로파일 수집 완료 (Accelerate ProfileKwargs = torch.profiler)")""")

md(r"""### 4-1. op 별 시간 — `prof.key_averages().table()`

- **Self CUDA / Self CPU**: 그 op *자체* 시간 (하위 제외)
- **CUDA total / CPU total**: 하위 포함
- `sort_by` 로 가장 비싼 op 를 찾습니다 (보통 `addmm`=linear, `bmm`=attention matmul).""")

code(r"""time_key = "self_cuda_time_total" if CUDA else "self_cpu_time_total"
print(prof.key_averages().table(sort_by=time_key, row_limit=12))""")

md(r"""### 4-2. 각 device 메모리 — `profile_memory=True` 의 결과

`activities` 에 CPU/CUDA 를 넣고 `profile_memory=True` 면 **CPU Mem / Self CPU Mem** (+ CUDA 면 \
**CUDA Mem / Self CUDA Mem**) 컬럼이 같이 나옵니다. `Self ... Mem` 이 **음수면 그 op 가 메모리를 푼다** 는 뜻.""")

code(r"""mem_key = "self_cuda_memory_usage" if CUDA else "self_cpu_memory_usage"
print(prof.key_averages().table(sort_by=mem_key, row_limit=10))""")

md(r"""### 4-3. forward / backward / optimizer 비중

`record_function` 라벨의 시간을 직접 재서 막대로 (버전 독립적으로 `perf_counter` + `sync()`).""")

code(r"""def time_train_phases(n_steps=10):
    model.train()
    it = iter(train_loader)
    b = next(it)   # prepare 된 loader → 이미 device 에
    out = model(**b); accelerator.backward(out.loss)
    optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()   # warmup

    fwd = bwd = opt = 0.0
    for _ in range(n_steps):
        b = next(it)
        t0 = time.perf_counter()
        out = model(**b); loss = out.loss; sync(); t1 = time.perf_counter()
        accelerator.backward(loss); sync(); t2 = time.perf_counter()
        optimizer.step(); optimizer.zero_grad(set_to_none=True); sync(); t3 = time.perf_counter()
        fwd += t1 - t0; bwd += t2 - t1; opt += t3 - t2
    return {"forward": fwd/n_steps*1e3, "backward": bwd/n_steps*1e3, "optimizer": opt/n_steps*1e3}

train_phases = time_train_phases(n_steps=10)
for k, v in train_phases.items():
    print(f"{k:>10}: {v:6.1f} ms/step")
print(f"{'TOTAL':>10}: {sum(train_phases.values()):6.1f} ms/step")""")

code(r"""fig, ax = plt.subplots(figsize=(6, 3))
names = list(train_phases.keys()); vals = [train_phases[k] for k in names]
ax.barh(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.1f} ms", va="center")
ax.set_xlabel("ms / step"); ax.set_title(f"Training step breakdown (bs={BATCH_SIZE}, {device.type})")
ax.invert_yaxis(); plt.tight_layout(); plt.show()""")

md(r"""보통 **backward ≈ forward 의 2배**, optimizer 는 작습니다.""")

md(r"""### 4-4. chrome trace 로 timeline 보기

`prof.export_chrome_trace` 로 저장한 JSON 을 `chrome://tracing` 또는 [Perfetto](https://ui.perfetto.dev) 에 \
드래그하면 op/kernel 의 시간축 배치와 **GPU idle gap** 이 보입니다.""")

code(r"""prof.export_chrome_trace("trace_train.json")
import os
print("saved trace_train.json", f"({os.path.getsize('trace_train.json')/1024:.0f} KB)")
print("→ chrome://tracing 또는 https://ui.perfetto.dev 에 드래그")""")

# ----- 5. 추론 프로파일 -----
md(r"""## 5. 추론 프로파일 — 학습과 무엇이 다른가

추론은 **forward only** 입니다. backward·optimizer 가 없고, `torch.inference_mode()` 안에서는 \
autograd graph 와 activation 을 저장하지 않아 **메모리가 크게 줄고** batch 를 더 키울 수 있습니다. \
프로파일은 학습과 똑같이 `accelerator.profile()` 로 켭니다.""")

code(r"""model.eval()
with accelerator.profile() as prof_inf:
    with torch.inference_mode():
        for step, batch in enumerate(eval_loader):
            _ = model(**batch).logits
            prof_inf.step()
            if step >= 4:
                break

print(prof_inf.key_averages().table(sort_by=time_key, row_limit=10))
print("\n→ 학습 테이블(§4-1)과 비교: backward 계열 op 가 사라짐")""")

md(r"""### 5-1. 시간 구성 — 학습 step vs 추론 forward

학습 step 은 forward + **backward + optimizer**, 추론은 **forward 뿐**. 누적 막대로 한눈에.""")

code(r"""def time_inference(n_steps=10):
    model.eval()
    it = iter(eval_loader)
    with torch.inference_mode():
        b = next(it); model(**b); sync()   # warmup
        t = 0.0
        for _ in range(n_steps):
            b = next(it)
            t0 = time.perf_counter(); model(**b); sync(); t += time.perf_counter() - t0
    return t / n_steps * 1e3

infer_ms = time_inference(n_steps=10)
fwd, bwd, opt = train_phases["forward"], train_phases["backward"], train_phases["optimizer"]
train_total = fwd + bwd + opt
print(f"train step : {train_total:6.1f} ms  (fwd {fwd:.1f} + bwd {bwd:.1f} + opt {opt:.1f})")
print(f"inference  : {infer_ms:6.1f} ms  (fwd only)")
print(f"→ 추론이 학습 step 의 {infer_ms/train_total*100:.0f}% 시간")

fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(["train"], [fwd], color="tab:blue", label="forward")
ax.barh(["train"], [bwd], left=[fwd], color="tab:orange", label="backward")
ax.barh(["train"], [opt], left=[fwd + bwd], color="tab:green", label="optimizer")
ax.barh(["inference"], [infer_ms], color="tab:cyan", label="inference (fwd only)")
ax.text(train_total, 0, f" {train_total:.1f} ms", va="center")
ax.text(infer_ms, 1, f" {infer_ms:.1f} ms", va="center")
ax.set_xlabel("ms / step")
ax.set_title("Time composition — train step vs inference (same batch)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.show()""")

md(r"""### 5-2. 메모리 구성 — 무엇이 VRAM 을 차지하나

학습 peak = **weight + gradient + optimizer state(Adam m,v) + activation**. \
추론 peak = **weight + 소량 activation**. 파라미터 수로 고정분을 계산하고 나머지를 activation 으로 근사.""")

code(r"""if CUDA:
    b = next(iter(eval_loader))

    model.train()
    torch.cuda.reset_peak_memory_stats()
    out = model(**b); accelerator.backward(out.loss)
    optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()
    train_peak = torch.cuda.max_memory_allocated() / 1024**2

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(**b); sync()
    infer_peak = torch.cuda.max_memory_allocated() / 1024**2

    P = model.num_parameters()
    w = P * 4 / 1024**2; g = P * 4 / 1024**2; adam = P * 8 / 1024**2   # fp32
    act_tr = max(0.0, train_peak - (w + g + adam))
    act_in = max(0.0, infer_peak - w)

    print(f"train peak : {train_peak:7.1f} MiB")
    print(f"infer peak : {infer_peak:7.1f} MiB   → 추론이 학습의 {train_peak/infer_peak:.1f}x 적게")

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar("train", w, color="tab:blue")
    ax.bar("train", g, bottom=w, color="tab:orange")
    ax.bar("train", adam, bottom=w + g, color="tab:green")
    ax.bar("train", act_tr, bottom=w + g + adam, color="tab:red")
    ax.bar("inference", w, color="tab:blue")
    ax.bar("inference", act_in, bottom=w, color="tab:red")
    handles = [
        Patch(color="tab:blue", label="weights"),
        Patch(color="tab:orange", label="gradients (train only)"),
        Patch(color="tab:green", label="optimizer Adam m,v (train only)"),
        Patch(color="tab:red", label="activation (approx)"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_ylabel("VRAM (MiB)")
    ax.set_title("Memory composition — train vs inference")
    plt.tight_layout(); plt.show()
else:
    train_peak = infer_peak = None
    print("CUDA 환경에서만 메모리 구성 비교가 의미 있습니다 (현재:", device.type, ")")""")

md(r"""### 5-3. `batch_size` 스윕 — 패턴 차이가 가장 잘 드러나는 곳

같은 batch_size 라도 추론은 throughput 이 높고 VRAM 이 완만하게 증가 → **같은 GPU 로 추론은 훨씬 큰 batch** 가능.""")

code(r"""def sweep_train_vs_infer(batch_sizes=(8, 16, 32, 64, 128)):
    rows = []
    for bs in batch_sizes:
        loader_bs = DataLoader(eval_ds, batch_size=bs, shuffle=False)
        try:
            b = {k: v.to(device) for k, v in next(iter(loader_bs)).items()}
            # --- train ---
            model.train()
            out = model(**b); out.loss.backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()   # warmup
            if CUDA:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(3):
                out = model(**b); out.loss.backward()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            sync(); tr_ms = (time.perf_counter() - t0) / 3 * 1e3
            tr_pk = torch.cuda.max_memory_allocated() / 1024**2 if CUDA else float("nan")
            # --- inference ---
            model.eval()
            with torch.inference_mode():
                model(**b); sync()
                if CUDA:
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(3):
                    model(**b)
                sync(); in_ms = (time.perf_counter() - t0) / 3 * 1e3
            in_pk = torch.cuda.max_memory_allocated() / 1024**2 if CUDA else float("nan")

            rows.append(dict(bs=bs, train_thr=bs / tr_ms * 1e3, infer_thr=bs / in_ms * 1e3,
                             train_pk=tr_pk, infer_pk=in_pk))
            print(f"bs={bs:4d} | train {bs/tr_ms*1e3:7.0f} s/s {tr_pk:7.0f} MiB | "
                  f"infer {bs/in_ms*1e3:7.0f} s/s {in_pk:7.0f} MiB")
        except torch.cuda.OutOfMemoryError:
            if CUDA:
                torch.cuda.empty_cache()
            print(f"bs={bs:4d} | OOM")
            break
    return rows

sweep_rows = sweep_train_vs_infer()""")

code(r"""import pandas as pd

sw = pd.DataFrame(sweep_rows)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(sw["bs"], sw["train_thr"], "o-", color="tab:blue", label="train")
axes[0].plot(sw["bs"], sw["infer_thr"], "s-", color="tab:cyan", label="inference")
axes[0].set_xscale("log", base=2); axes[0].set_xticks(sw["bs"]); axes[0].set_xticklabels(sw["bs"])
axes[0].set_xlabel("batch_size"); axes[0].set_ylabel("throughput (samples/sec)")
axes[0].set_title("Throughput — train vs inference"); axes[0].grid(True, alpha=0.3); axes[0].legend()

if CUDA:
    axes[1].plot(sw["bs"], sw["train_pk"], "o-", color="tab:blue", label="train")
    axes[1].plot(sw["bs"], sw["infer_pk"], "s-", color="tab:cyan", label="inference")
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(sw["bs"]); axes[1].set_xticklabels(sw["bs"])
    axes[1].set_xlabel("batch_size"); axes[1].set_ylabel("peak VRAM (MiB)")
    axes[1].set_title("Peak VRAM — train vs inference"); axes[1].grid(True, alpha=0.3); axes[1].legend()
else:
    axes[1].text(0.5, 0.5, "VRAM panel needs CUDA", ha="center", va="center"); axes[1].axis("off")

plt.tight_layout(); plt.show()""")

md(r"""### 5-4. 패턴 분석 — 학습 vs 추론

| 측면 | 학습 (train) | 추론 (inference) |
|---|---|---|
| 연산 | forward + **backward + optimizer** | **forward only** |
| 시간 비중 | backward ≈ forward 의 ~2배 (지배적) | forward 전부 |
| autograd graph | 생성·보관 | `inference_mode` 로 미생성 |
| 메모리 구성 | weight + grad + optimizer(Adam m,v) + activation | weight + 소량 activation |
| peak VRAM | 큼 (4종 전부) | 작음 (대략 1/3-1/4) |
| batch 한계 | 작음 (OOM 빠름) | 큼 (같은 GPU 로 몇 배) |
| throughput | 낮음 | 높음 |
| 주 병목 | backward 커널, activation 메모리 | forward 커널, 결과 `.cpu()` 전송 |

**왜 이렇게 갈리나**
- 학습은 backward 를 위해 forward 의 중간 activation 을 *전부 보관* → 메모리가 batch 에 비례해 가파르게 증가
- optimizer state(Adam 의 m, v)는 파라미터당 2배라 weight 의 약 3배가 *고정 비용*
- 추론은 `inference_mode` 에서 graph·activation 을 안 만들어 weight + 한 구간 activation 만

**실무 함의**
- 추론 서빙은 학습보다 *작은* GPU 로 가능, batch 크게 잡아 throughput↑
- 학습 OOM 우선순위: `fp16` → batch↓ + `gradient_accumulation` → gradient checkpointing → ZeRO(optimizer state 분산)
- 추론 OOM 은 드물지만 batch 가 매우 크면 결과 텐서(`.cpu()` 전) 누적 주의""")

# ----- 6. FLOPS / params -----
md(r"""## 6. FLOPS / params

시간은 하드웨어마다 다르지만 **FLOPs·params** 는 모델 고유의 절대 비용입니다. \
§4 에서 `with_flops=True` 로 켰으니 같은 `prof` 에서 FLOPs 정렬을 바로 뽑습니다.""")

code(r"""print(f"total params     : {model.num_parameters()/1e6:.1f} M")
print(f"trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f} M\n")

# §4 의 prof (with_flops=True) 재활용
print(prof.key_averages().table(sort_by="flops", row_limit=8))""")

md(r"""**MFU(Model FLOPs Utilization)** = 달성 FLOPS / GPU 이론 peak — "비싼 GPU 를 몇 % 쓰나"(참고 지표). \
T4 의 fp16 peak ≈ 65 TFLOPS. 입문 단계에선 "낮으면 GPU 가 논다" 정도만 알면 충분하고, \
정밀한 layer 단위 MFU 는 DeepSpeed Flops Profiler 가 강력합니다 (분산 부록에서).""")

# ----- 7. 진단 -----
md(r"""## 7. 출력 → 병목 진단

| 관찰 | 해석 | 다음 행동 |
|---|---|---|
| `addmm`/`bmm` 이 시간 대부분 | compute-bound (정상 BERT) | `fp16`, batch↑, 작은 모델 |
| step breakdown 합 ≪ wall-clock | **DataLoader 대기** | `num_workers↑`, `pin_memory=True`, 토큰화 캐싱 |
| train peak ≫ infer peak | backward activation 때문 (정상) | 학습만 `fp16`/gradient checkpointing |
| **추론**이 느린데 GPU idle | 추론 batch 가 작거나 `.cpu()` 전송 과다 | 추론 batch↑, threshold 비교는 GPU 에서 |
| MFU 낮음 (<10%) | GPU 가 논다 | batch↑, seq↑, fp16 |

**핵심**: 학습과 추론은 병목이 다릅니다. 학습은 backward·메모리, 추론은 forward·전송. 따로 프로파일해야 정확합니다.""")

# ----- 8. 더 깊이 -----
md(r"""## 8. 더 깊이 (안내)

- **순수 PyTorch** — accelerate 없이 `torch.profiler.profile(...)` 을 직접 써도 됩니다 (ProfileKwargs 가 받던 인자 그대로)
- **TensorBoard Profiler plugin** — `tensorboard_trace_handler` 로 저장 후 GPU util/step breakdown/추천 시각화
- **DeepSpeed Flops Profiler / Comms Logger** — layer 단위 FLOPs, NCCL 통신 비용
- **NVIDIA Nsight Systems** — CUDA stream/NCCL/kernel 시스템 timeline
- **Holistic Trace Analysis (HTA)** — 멀티 GPU trace 자동 분석

→ 멀티 GPU·분산 프로파일링은 **별도 부록**.""")

# ----- 9. FAQ -----
md(r"""## ❓ FAQ

### Q1. `activities` 에 CPU/CUDA 만 넣으면 메모리도 추적되나요?

`profile_memory=True` 를 함께 켜면 **각 device 메모리가 별도 컬럼**으로 나옵니다 — CPU → `CPU Mem`/`Self CPU Mem`, \
CUDA → `CUDA Mem`/`Self CUDA Mem`. `activities` 는 "어느 device 를 볼지"만 정하고, 메모리는 `profile_memory` 가 켜야 합니다. \
참고로 profiler 의 메모리는 *op 단위* 이고, 학습 전체 peak 는 `torch.cuda.max_memory_allocated()` 가 더 직접적입니다(§5-2).

### Q2. 왜 `torch.profiler` 를 따로 안 쓰고 Accelerate `ProfileKwargs` 하나만 쓰나요?

`ProfileKwargs` 가 `torch.profiler` 의 얇은 래퍼라 출력(key_averages 테이블·schedule·메모리·chrome trace)이 \
**완전히 동일** 합니다. HF 환경(Trainer/Accelerate/DeepSpeed/FSDP)에서 single↔multi GPU 코드가 같아 이걸 메인으로 씁니다. \
accelerate 가 없는 순수 PyTorch 라면 `torch.profiler.profile(activities=..., schedule=..., profile_memory=...)` 를 \
직접 쓰면 되고, 인자·출력이 그대로입니다 (§8).

### Q3. 학습과 추론을 왜 따로 프로파일하나요?

병목이 다르기 때문입니다. 학습은 backward(forward 의 ~2배)와 activation 메모리가 지배적이고, \
추론은 forward 와 결과 전송(`.cpu()`)이 지배적입니다. `inference_mode` 덕분에 추론 peak 메모리는 학습의 1/3~1/4 라 \
**추론은 학습보다 훨씬 큰 batch** 를 쓸 수 있습니다 (§5-3).

### Q4. `schedule_option` 의 wait/warmup/active 는 왜 필요한가요?

전체를 추적하면 trace 가 GB 로 폭증하고 느려집니다. `wait`(건너뜀)→`warmup`(기록 후 버림, 캐시 안정화)→`active`(기록) \
사이클로 대표 step 만 봅니다. 첫 step 은 cuDNN autotune 으로 비정상적으로 느려서 warmup 으로 거릅니다.

### Q5. 측정 시간이 매번 다릅니다.

`sync()`(=`torch.cuda.synchronize()`)를 빼면 GPU 작업이 안 끝났는데 시간을 재서 비정상적으로 빠르게 나옵니다. \
이 부록은 phase 측정마다 `sync()` 를 넣고 여러 step 평균을 냅니다. 첫 실행은 워밍업으로 느리니 한 번 더 돌려 보세요.

### Q6. 정확도가 낮게 나옵니다. 프로파일 탓인가요?

아닙니다. 프로파일은 측정만 하고 학습에 영향이 없습니다. 이 부록은 1 에폭·4,000건이라 Ch 10 본편(2 에폭)보다 \
정확도가 약간 낮을 수 있습니다. epoch/데이터를 늘리면 오르지만 학습 시간이 길어집니다 (프로파일과 무관).

### Q7. `Trainer` 를 쓸 때는?

`TrainerCallback.on_step_end` 에서 `prof.step()` 을 호출하는 콜백을 만들어 끼우거나, 이 부록처럼 \
Accelerate `ProfileKwargs` + `accelerator.profile()` 로 감쌉니다.

```python
class ProfCallback(TrainerCallback):
    def __init__(self, prof): self.prof = prof
    def on_step_end(self, args, state, control, **kwargs): self.prof.step()
```

### Q8. 로컬 Mac(MPS)에서도 그대로 되나요?

**실행은 됩니다** — 맨 위 device 감지가 MPS 를 잡고, CUDA 전용 코드(VRAM 측정)는 `if CUDA:` 로 가려 자동 skip, \
`sync()` 도 `torch.mps.synchronize()` 로 분기합니다. 다만 세 가지가 달라집니다:

1. **op 테이블이 제한적** — profiler `activities` 가 CPU 만이라(MPS 커널 시간은 torch.profiler 가 아직 잘 못 잡음) \
   GPU 커널 단위 분석은 약하고 CPU dispatch 위주로 보입니다.
2. **VRAM 측정 skip** — §5-2 메모리 구성·§5-3 VRAM 패널은 CUDA 전용이라 MPS 면 throughput 만 나옵니다 \
   (unified memory 라 `max_memory_allocated` 개념이 다름).
3. **fp16 효과 미미** — MPS 는 fp16 가속이 CUDA 만큼 크지 않습니다.

즉 **시간 비교·정확도·학습/추론 패턴 분석은 잘 되고**, GPU 커널 레벨 프로파일·정밀 VRAM 은 CUDA(T4)에서 보세요. \
인프라별(MPS/CPU/Single/Multi) 프로파일 차이는 별도 비교 부록에서 다룰 예정입니다.""")

# ----- 10. 다음 -----
md(r"""## 다음

- 여기서 찾은 병목을 실제로 고쳐(예: `num_workers↑`, `fp16`) **before/after** 를 같은 프로파일러로 재 보면 효과가 보입니다.
- 모델을 키우면(BERT-base → large) compute·메모리 패턴이 극적으로 변합니다 — *모델 크기 스케일링* 실험으로.
- 멀티 GPU 로 가면 통신(NCCL)·분산 특화 프로파일링이 필요 — *분산 프로파일링 부록* 으로.""")


NOTEBOOK = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
