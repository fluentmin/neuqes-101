"""Build 10_bert_binary_sigmoid/appendix_profiling.ipynb.

성능 부록 — (1) 정확도 측정 + (2) 학습 프로파일 + (3) 추론 프로파일 모두.
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

**도구 (단일 T4 GPU 에서 실측 가능)**: PyTorch Profiler · Accelerate `ProfileKwargs` · FLOPS/params.
멀티 GPU / DeepSpeed / Nsight 같은 *분산* 프로파일링은 별도 부록에서 다룹니다.

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
    # 정확한 시간 측정을 위해 GPU 작업이 끝날 때까지 대기
    if CUDA:
        torch.cuda.synchronize()""")

# ----- 1. 무엇을 보나 -----
md(r"""## 1. 이 부록이 보는 것

| 축 | 질문 | 어디서 |
|---|---|---|
| **정확도** | 모델이 잘 맞히나? | §3 — 학습 후 eval metric 5종 |
| **학습 프로파일** | 학습 1 step 이 어디서 느린가? | §4-5 — forward/backward/optimizer 분해 |
| **추론 프로파일** | 추론은 학습과 무엇이 다른가? | §6 — forward only, 메모리·시간 비교 |

프로파일러는 "어디가 느린지/무거운지" 를 알려주고, 정확도는 "그 모델이 쓸모 있는지" 를 알려줍니다. \
둘을 같이 봐야 *"빠르면서 정확한"* 지점을 찾습니다.""")

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
    raw = load_dataset("yelp_polarity", split=split).shuffle(seed=42).select(range(n))
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

# (1) before/after metric 비교
keys = list(after.keys())
x = np.arange(len(keys))
ax1.bar(x - 0.2, [before[k] for k in keys], 0.4, label="before", color="tab:gray")
ax1.bar(x + 0.2, [after[k]  for k in keys], 0.4, label="after",  color="tab:blue")
ax1.set_xticks(x); ax1.set_xticklabels(keys, rotation=20)
ax1.set_ylim(0, 1.05); ax1.set_ylabel("score")
ax1.set_title("Accuracy metrics — before vs after 1 epoch")
ax1.legend(); ax1.grid(True, alpha=0.3)

# (2) 정답 0/1 그룹의 sigmoid 확률 분포 (Ch 10 스타일)
ax2.hist(probs[labels == 0], bins=30, alpha=0.6, label="true 0 (neg)", color="tab:red")
ax2.hist(probs[labels == 1], bins=30, alpha=0.6, label="true 1 (pos)", color="tab:green")
ax2.axvline(0.5, ls=":", color="gray", label="threshold 0.5")
ax2.set_xlabel("predicted sigmoid prob"); ax2.set_ylabel("count")
ax2.set_title("Score distribution after training")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()""")

# ----- 4. 학습 프로파일 (PyTorch Profiler) -----
md(r"""## 4. 학습 프로파일 — 도구 ① PyTorch Profiler

이제 *이 학습이 어디에 시간을 쓰는지* 봅니다. `profile()` 컨텍스트로 몇 step 을 감쌉니다.

핵심 인자:
- `activities` — CPU / CUDA 중 무엇을 추적할지 (CUDA 면 GPU 커널까지)
- `schedule` — 장기 학습에서 *몇 step 만* (전체는 trace 가 GB 로 폭증)
- `profile_memory=True` — **각 device(CPU/GPU) 메모리도 컬럼으로** 추적
- `record_function("name")` — 코드 구간에 직접 라벨""")

code(r"""from torch.profiler import profile, schedule, record_function, ProfilerActivity

activities = [ProfilerActivity.CPU]
if CUDA:
    activities.append(ProfilerActivity.CUDA)

sched = schedule(wait=1, warmup=1, active=3)   # 5 step 만

model.train()
with profile(activities=activities, schedule=sched,
             record_shapes=True, profile_memory=True) as prof:
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with record_function("forward"):
            out = model(**batch); loss = out.loss
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        prof.step()
        if step >= 4:
            break

print("프로파일 수집 완료")""")

md(r"""### 4-1. op 별 시간 — `key_averages().table()`

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
    b = {k: v.to(device) for k, v in next(it).items()}
    model(**b).loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()  # warmup

    fwd = bwd = opt = 0.0
    for _ in range(n_steps):
        b = {k: v.to(device) for k, v in next(it).items()}
        t0 = time.perf_counter()
        out = model(**b); loss = out.loss; sync(); t1 = time.perf_counter()
        loss.backward(); sync(); t2 = time.perf_counter()
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

md(r"""보통 **backward ≈ forward 의 2배**, optimizer 는 작습니다. chrome trace 로 timeline 도 볼 수 있습니다.""")

code(r"""prof.export_chrome_trace("trace_train.json")
import os
print("saved trace_train.json", f"({os.path.getsize('trace_train.json')/1024:.0f} KB)")
print("→ chrome://tracing 또는 https://ui.perfetto.dev 에 드래그")""")

# ----- 5. Accelerate -----
md(r"""## 5. 도구 ② Accelerate `ProfileKwargs`

HF 생태계(Trainer/Accelerate/DeepSpeed/FSDP)에서 **가장 적은 코드** 로 같은 PyTorch Profiler 를 켭니다. \
single ↔ multi GPU 에 같은 코드라 분산 확장 시 그대로 씁니다.""")

code(r"""from accelerate import Accelerator, ProfileKwargs

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"] if CUDA else ["cpu"],
    record_shapes=True, profile_memory=True, with_flops=True,
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
acc_model, acc_opt, acc_loader = accelerator.prepare(model, optimizer, train_loader)

acc_model.train()
with accelerator.profile() as prof_acc:
    for step, batch in enumerate(acc_loader):
        out = acc_model(**batch)
        accelerator.backward(out.loss)
        acc_opt.step(); acc_opt.zero_grad(set_to_none=True)
        if step >= 3:
            break

print(prof_acc.key_averages().table(sort_by=time_key, row_limit=10))""")

# ----- 6. 추론 프로파일 -----
md(r"""## 6. 추론 프로파일 — 학습과 무엇이 다른가

추론은 **forward only** 입니다. backward·optimizer 가 없고, `torch.inference_mode()` 안에서는 \
autograd graph 와 activation 을 저장하지 않아 **메모리가 크게 줄고** batch 를 더 키울 수 있습니다.

같은 모델·같은 batch 로 추론을 프로파일하고 §4 의 학습과 비교합니다.""")

code(r"""model.eval()
with profile(activities=activities, schedule=schedule(wait=1, warmup=1, active=3),
             record_shapes=True, profile_memory=True) as prof_inf:
    with torch.inference_mode():
        for step, batch in enumerate(eval_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch).logits
            prof_inf.step()
            if step >= 4:
                break

print(prof_inf.key_averages().table(sort_by=time_key, row_limit=10))""")

md(r"""### 6-1. 시간 구성 — 학습 step vs 추론 forward

학습 step 은 forward + **backward + optimizer** 인데, 추론은 **forward 뿐** 입니다. \
누적 막대로 "추론은 학습의 어느 한 조각만" 이라는 게 한눈에 보입니다.""")

code(r"""def time_inference(n_steps=10, batch_size=BATCH_SIZE):
    model.eval()
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    it = iter(loader)
    with torch.inference_mode():
        b = {k: v.to(device) for k, v in next(it).items()}; model(**b); sync()  # warmup
        t = 0.0
        for _ in range(n_steps):
            b = {k: v.to(device) for k, v in next(it).items()}
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

md(r"""### 6-2. 메모리 구성 — 무엇이 VRAM 을 차지하나

학습 peak = **weight + gradient + optimizer state(Adam m,v) + activation**. \
추론 peak = **weight + 소량 activation** (`inference_mode` 라 grad/graph 없음). \
파라미터 수로 weight/grad/optimizer 의 고정분을 계산하고, 나머지를 activation 으로 근사해 누적 막대로 봅니다.""")

code(r"""if CUDA:
    b = {k: v.to(device) for k, v in next(iter(eval_loader)).items()}

    model.train()
    torch.cuda.reset_peak_memory_stats()
    out = model(**b); out.loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()
    train_peak = torch.cuda.max_memory_allocated() / 1024**2

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(**b); sync()
    infer_peak = torch.cuda.max_memory_allocated() / 1024**2

    # 고정분 계산 (fp32 기준): weight = P*4B, grad = P*4B, Adam(m,v) = P*8B
    P = model.num_parameters()
    w = P * 4 / 1024**2
    g = P * 4 / 1024**2
    adam = P * 8 / 1024**2
    act_tr = max(0.0, train_peak - (w + g + adam))   # 나머지를 activation 으로 근사
    act_in = max(0.0, infer_peak - w)

    print(f"train peak : {train_peak:7.1f} MiB")
    print(f"infer peak : {infer_peak:7.1f} MiB   → 추론이 학습의 {train_peak/infer_peak:.1f}x 적게")

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7, 4))
    # train stacked
    ax.bar("train", w, color="tab:blue")
    ax.bar("train", g, bottom=w, color="tab:orange")
    ax.bar("train", adam, bottom=w + g, color="tab:green")
    ax.bar("train", act_tr, bottom=w + g + adam, color="tab:red")
    # inference stacked
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

md(r"""### 6-3. `batch_size` 스윕 — 패턴 차이가 가장 잘 드러나는 곳

같은 batch_size 라도 추론은 throughput 이 높고 VRAM 이 완만하게 증가합니다 → **같은 GPU 로 추론은 훨씬 큰 batch** 가능. \
학습/추론을 batch 별로 재서 두 곡선으로 겹쳐 봅니다.""")

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

md(r"""### 6-4. 패턴 분석 — 학습 vs 추론

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
- optimizer state(Adam 의 m, v)는 파라미터당 2배라 weight 의 약 3배가 *고정 비용* 으로 항상 잡힘
- 추론은 `inference_mode` 에서 graph·activation 을 안 만들어 weight + 한 구간 activation 만 → 메모리가 완만

**실무 함의**
- 추론 서빙은 학습보다 *작은* GPU 로 가능하고, batch 를 크게 잡아 throughput 을 끌어올림
- 학습 OOM 우선순위: `fp16` → batch↓ + `gradient_accumulation` → gradient checkpointing(activation↓) → ZeRO(optimizer state 분산)
- 추론 OOM 은 드물지만, batch 가 매우 크면 결과 텐서(`.cpu()` 전)가 누적되니 주의""")

# ----- 7. FLOPS / params -----
md(r"""## 7. 도구 ③ FLOPS / params

시간은 하드웨어마다 다르지만 **FLOPs·params** 는 모델 고유의 절대 비용입니다.""")

code(r"""print(f"total params     : {model.num_parameters()/1e6:.1f} M")
print(f"trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f} M\n")

with profile(activities=activities, with_flops=True) as prof_flops:
    b = {k: v.to(device) for k, v in next(iter(eval_loader)).items()}
    model.eval()
    with torch.inference_mode():
        _ = model(**b)
model.train()

print(prof_flops.key_averages().table(sort_by="flops", row_limit=8))""")

md(r"""**MFU(Model FLOPs Utilization)** = 달성 FLOPS / GPU 이론 peak — "비싼 GPU 를 몇 % 쓰나"(참고 지표). \
T4 의 fp16 peak ≈ 65 TFLOPS. 입문 단계에선 "낮으면 GPU 가 논다" 정도만 알면 충분하고, \
정밀한 layer 단위 MFU 는 DeepSpeed Flops Profiler 가 강력합니다 (분산 부록에서).""")

# ----- 8. 진단 -----
md(r"""## 8. 출력 → 병목 진단

| 관찰 | 해석 | 다음 행동 |
|---|---|---|
| `addmm`/`bmm` 이 시간 대부분 | compute-bound (정상 BERT) | `fp16`, batch↑, 작은 모델 |
| step breakdown 합 ≪ wall-clock | **DataLoader 대기** | `num_workers↑`, `pin_memory=True`, 토큰화 캐싱 |
| train peak ≫ infer peak | backward activation 때문 (정상) | 학습만 `fp16`/gradient checkpointing |
| **추론**이 느린데 GPU idle | 추론 batch 가 작거나 `.cpu()` 전송 과다 | 추론 batch↑, threshold 비교는 GPU 에서 |
| MFU 낮음 (<10%) | GPU 가 논다 | batch↑, seq↑, fp16 |

**핵심**: 학습과 추론은 병목이 다릅니다. 학습은 backward·메모리, 추론은 forward·전송. 따로 프로파일해야 정확합니다.""")

# ----- 9. 더 깊이 -----
md(r"""## 9. 더 깊이 (안내)

- **TensorBoard Profiler plugin** — `tensorboard_trace_handler` 로 저장 후 GPU util/step breakdown/추천 시각화
- **DeepSpeed Flops Profiler / Comms Logger** — layer 단위 FLOPs, NCCL 통신 비용
- **NVIDIA Nsight Systems** — CUDA stream/NCCL/kernel 시스템 timeline
- **Holistic Trace Analysis (HTA)** — 멀티 GPU trace 자동 분석

→ 멀티 GPU·분산 프로파일링은 **별도 부록**.""")

# ----- 10. FAQ -----
md(r"""## ❓ FAQ

### Q1. `activities` 에 CPU/CUDA 만 넣으면 메모리도 추적되나요?

`profile_memory=True` 를 함께 켜면 **각 device 메모리가 별도 컬럼**으로 나옵니다 — CPU → `CPU Mem`/`Self CPU Mem`, \
CUDA → `CUDA Mem`/`Self CUDA Mem`. `activities` 는 "어느 device 를 볼지"만 정하고, 메모리는 `profile_memory` 가 켜야 합니다. \
참고로 profiler 의 메모리는 *op 단위* 이고, 학습 전체 peak 는 `torch.cuda.max_memory_allocated()` 가 더 직접적입니다.

### Q2. 학습과 추론을 왜 따로 프로파일하나요?

병목이 다르기 때문입니다. 학습은 backward(forward 의 ~2배)와 activation 메모리가 지배적이고, \
추론은 forward 와 결과 전송(`.cpu()`)이 지배적입니다. `inference_mode` 덕분에 추론 peak 메모리는 학습의 1/3~1/4 수준이라 \
**추론은 학습보다 훨씬 큰 batch** 를 쓸 수 있습니다 (§6 비교 참고).

### Q3. 정확도가 낮게 나옵니다. 프로파일 탓인가요?

아닙니다. 프로파일은 측정만 할 뿐 학습에 영향이 없습니다. 이 부록은 1 에폭·4,000건이라 Ch 10 본편(2 에폭)보다 \
정확도가 약간 낮을 수 있습니다. 정확도를 올리려면 epoch/데이터를 늘리세요 — 단 그러면 학습 시간이 길어집니다(프로파일과 무관).

### Q4. `schedule` 의 wait/warmup/active 는 왜 필요한가요?

전체를 추적하면 trace 가 GB 로 폭증하고 느려집니다. `wait`(건너뜀)→`warmup`(기록 후 버림, 캐시 안정화)→`active`(기록) \
사이클로 대표 step 만 봅니다. 첫 step 은 cuDNN autotune 으로 비정상적으로 느려서 warmup 으로 거릅니다.

### Q5. 측정 시간이 매번 다릅니다.

`sync()`(=`torch.cuda.synchronize()`)를 빼면 GPU 작업이 안 끝났는데 시간을 재서 비정상적으로 빠르게 나옵니다. \
이 부록은 phase 측정마다 `sync()` 를 넣고 여러 step 평균을 냅니다. 첫 실행은 워밍업으로 느리니 한 번 더 돌려 보세요.

### Q6. Trainer 를 쓸 때는?

`TrainerCallback.on_step_end` 에서 `prof.step()` 을 호출하는 콜백을 만들어 끼우거나, §5 의 Accelerate `ProfileKwargs` 가 더 간단합니다.

```python
class ProfCallback(TrainerCallback):
    def __init__(self, prof): self.prof = prof
    def on_step_end(self, args, state, control, **kwargs): self.prof.step()
```""")

# ----- 11. 다음 -----
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
