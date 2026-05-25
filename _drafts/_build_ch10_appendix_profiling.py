"""Build 10_bert_binary_sigmoid/appendix_profiling.ipynb — 성능 프로파일러 사용법 투어."""
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
md(r"""# Ch 10 부록 — 성능 프로파일러 사용법 투어

Ch 10 에서 학습한 **DistilBERT binary (sigmoid + BCE)** 를 그대로 두고, \
"이 학습이 GPU 에서 어디에 시간·메모리를 쓰는가" 를 들여다보는 **프로파일링 도구 투어** 입니다.

학습 코드를 더 빠르게 만드는 게 목표가 아니라, **도구를 켜고 출력을 읽는 법** 을 익히는 게 목표입니다. \
병목을 *찾는* 눈을 기르면 그 다음 최적화(`fp16`, `num_workers`, batch 키우기 등)는 자연스럽게 따라옵니다.

**다루는 도구 (단일 T4 GPU 에서 실측 가능한 것)**
1. **PyTorch Profiler** — 가장 범용. step별 op/kernel 시간, 메모리, chrome trace
2. **Accelerate `ProfileKwargs`** — HF/Trainer 환경에서 가장 쉬운 1차 스캔
3. **FLOPS / params** — `torch.profiler(with_flops=True)` + `model.num_parameters()`

> 멀티 GPU / DeepSpeed / Nsight 같은 *분산* 프로파일링은 별도 부록에서 다룹니다 (이 부록은 단일 GPU 입문).

**환경**: Google Colab **T4 GPU 권장**. CPU 에서도 동작 (CUDA 줄만 자동 skip). 약 5-8분.

---

> 📒 **선행**: Ch 10 (DistilBERT binary, sigmoid+BCE). 이 부록은 그 학습 셋업을 압축 재현해 위에 프로파일러를 얹습니다.""")

# ----- 0. 환경 -----
md(r"""## 0. 환경 셋업""")

code(r"""%pip -q install -U transformers datasets accelerate""")

code(r"""import torch, time
import torch.nn.functional as F

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

# ----- 1. 왜 프로파일링 -----
md(r"""## 1. 왜 프로파일링인가

학습이 느리거나 OOM 이 날 때, **추측 대신 측정** 으로 병목을 찾습니다. 전형적인 병목 셋:

| 병목 | 증상 | 프로파일러에서 보이는 모습 |
|---|---|---|
| **GPU compute** | GPU 사용률 높은데 느림 | matmul/attention 커널이 시간 대부분 차지 |
| **DataLoader** | GPU 사용률 낮고 idle gap | step 사이 CPU 대기, GPU 노는 구간 |
| **메모리** | OOM / 작은 batch 만 가능 | activation peak 가 backward 직후 폭증 |

이 부록은 위 셋을 **읽어내는 도구** 를 하나씩 켜 봅니다.""")

# ----- 2. 대상 준비 -----
md(r"""## 2. 프로파일 대상 준비 — Ch 10 학습을 압축 재현

Ch 10 과 동일한 `distilbert-base-uncased` + `num_labels=1` + `BCEWithLogitsLoss` 셋업. \
단 프로파일이 목적이라 데이터는 작게(512건), 학습은 몇 step 만 돕니다. \
정확도는 의미 없고 **속도·메모리 패턴만** 봅니다.""")

code(r"""from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

MODEL = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Ch 10 셋업: num_labels=1 + multi_label → BCEWithLogitsLoss 자동 매핑
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, problem_type="multi_label_classification",
).to(device)

# 프로파일용 작은 데이터 (정확도 무관)
raw = load_dataset("yelp_polarity", split="train[:512]")

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

ds = raw.map(tok, batched=True, remove_columns=raw.column_names)
ds = ds.add_column("labels", [[float(l)] for l in raw["label"]])   # (B,1) float for BCE
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"#params : {model.num_parameters()/1e6:.1f} M")
print(f"data    : {len(ds)} samples, batch_size={BATCH_SIZE}, max_len={MAX_LEN}")""")

code(r"""# 한 학습 step (forward + backward + optimizer)
def train_step(batch):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return out.loss

# 동작 확인
model.train()
loss = train_step(next(iter(loader)))
print("one step OK, loss =", float(loss))""")

# ----- 3. PyTorch Profiler -----
md(r"""## 3. 도구 ① PyTorch Profiler

가장 범용적인 도구. `profile()` 컨텍스트로 코드 구간을 감싸면 op/kernel 단위 시간·메모리를 모읍니다.

핵심 인자:
- `activities` — CPU / CUDA 중 무엇을 추적할지
- `schedule` — 장기 학습에서 *몇 step 만* 추적 (전체 추적은 trace 가 GB 단위로 폭증)
- `record_shapes` / `profile_memory` — 입력 shape / 메모리까지
- `record_function("name")` — 코드 구간에 직접 라벨 (forward/backward 구분 등)""")

code(r"""from torch.profiler import profile, schedule, record_function, ProfilerActivity

activities = [ProfilerActivity.CPU]
if CUDA:
    activities.append(ProfilerActivity.CUDA)

# schedule: 1 step 대기 → 1 step warmup → 3 step 기록 (총 5 step 만)
sched = schedule(wait=1, warmup=1, active=3)

model.train()
with profile(activities=activities, schedule=sched,
             record_shapes=True, profile_memory=True) as prof:
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with record_function("forward"):
            out = model(**batch); loss = out.loss
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        prof.step()
        if step >= 4:    # wait1 + warmup1 + active3
            break

print("프로파일 수집 완료")""")

md(r"""### 3-1. 결과 읽기 — `key_averages().table()`

op 별 집계 테이블. 주요 컬럼:
- **Self CUDA / Self CPU** — 그 op *자체* 에서 쓴 시간 (하위 op 제외)
- **CUDA total / CPU total** — 하위 op 까지 포함한 총 시간
- **# of Calls** — 호출 횟수

`sort_by` 로 정렬 기준을 바꿔 가장 비싼 op 를 찾습니다.""")

code(r"""sort_key = "self_cuda_time_total" if CUDA else "self_cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=12))""")

md(r"""**예상 모습** — 상위에 `aten::addmm`(linear), `aten::bmm`/attention matmul, `aten::native_layer_norm` 같은 \
Transformer 의 핵심 연산이 올라옵니다. GPU 면 `Self CUDA` 가, CPU 면 `Self CPU` 가 시간의 대부분.""")

md(r"""### 3-2. forward / backward / optimizer 비중

`record_function` 으로 붙인 라벨의 시간을 직접 재서 막대로 봅니다. \
(profiler 의 라벨별 집계는 버전마다 API 가 달라, 여기서는 `time.perf_counter` + `sync()` 로 직접 측정)""")

code(r"""def time_phases(n_steps=10):
    model.train()
    it = iter(loader)
    # warmup 1 step
    b = {k: v.to(device) for k, v in next(it).items()}
    model(**b).loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    sync()

    fwd = bwd = opt = 0.0
    for _ in range(n_steps):
        b = {k: v.to(device) for k, v in next(it).items()}
        t0 = time.perf_counter()
        out = model(**b); loss = out.loss; sync()
        t1 = time.perf_counter()
        loss.backward(); sync()
        t2 = time.perf_counter()
        optimizer.step(); optimizer.zero_grad(set_to_none=True); sync()
        t3 = time.perf_counter()
        fwd += t1 - t0; bwd += t2 - t1; opt += t3 - t2
    return {"forward": fwd / n_steps * 1e3,
            "backward": bwd / n_steps * 1e3,
            "optimizer": opt / n_steps * 1e3}

# loader 가 충분한 batch 를 갖도록 n_steps 는 batch 수보다 작게
phases = time_phases(n_steps=min(10, len(loader) - 2))
for k, v in phases.items():
    print(f"{k:>10}: {v:6.1f} ms/step")
print(f"{'total':>10}: {sum(phases.values()):6.1f} ms/step")""")

code(r"""import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.2))
names = list(phases.keys())
vals = [phases[k] for k in names]
ax.barh(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
for i, v in enumerate(vals):
    ax.text(v, i, f" {v:.1f} ms", va="center")
ax.set_xlabel("time per step (ms)")
ax.set_title(f"Train step breakdown  (DistilBERT, bs={BATCH_SIZE}, {device.type})")
ax.invert_yaxis()
plt.tight_layout()
plt.show()""")

md(r"""보통 **backward 가 forward 의 ~2배**, optimizer 는 작습니다. \
backward 가 압도적이면 compute-bound — `fp16` 이나 batch 조정이 후보. \
반대로 step breakdown 합보다 실제 wall-clock 이 훨씬 길면 그 차이가 **DataLoader 대기**(다음 셀에서).""")

md(r"""### 3-3. 메모리 — 어떤 op 가 메모리를 많이 쓰나

`profile_memory=True` 로 모은 메모리 사용량 정렬.""")

code(r"""mem_key = "self_cuda_memory_usage" if CUDA else "self_cpu_memory_usage"
print(prof.key_averages().table(sort_by=mem_key, row_limit=10))""")

md(r"""### 3-4. chrome trace 로 timeline 보기

`export_chrome_trace` 로 저장한 JSON 을 `chrome://tracing` 또는 [Perfetto](https://ui.perfetto.dev) 에 \
드래그하면 op/kernel 의 **시간축 배치** 와 **GPU idle gap** 이 보입니다.""")

code(r"""prof.export_chrome_trace("trace_pytorch.json")
import os
print("saved:", "trace_pytorch.json", f"({os.path.getsize('trace_pytorch.json')/1024:.0f} KB)")
print("→ chrome://tracing 또는 https://ui.perfetto.dev 에 드래그해서 열어보세요")""")

# ----- 4. Accelerate -----
md(r"""## 4. 도구 ② Accelerate `ProfileKwargs`

HF 생태계(Trainer / Accelerate / DeepSpeed / FSDP)에서 **가장 적은 코드** 로 같은 PyTorch Profiler 를 켭니다. \
`ProfileKwargs` 에 옵션을 담아 `Accelerator` 에 넘기고, `accelerator.profile()` 컨텍스트로 감싸면 끝.

single ↔ multi GPU 전환에도 같은 코드라, 나중에 분산으로 확장할 때 그대로 씁니다.""")

code(r"""from accelerate import Accelerator, ProfileKwargs

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"] if CUDA else ["cpu"],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
)
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

acc_model, acc_opt, acc_loader = accelerator.prepare(model, optimizer, loader)

acc_model.train()
with accelerator.profile() as prof_acc:
    for step, batch in enumerate(acc_loader):
        out = acc_model(**batch)
        accelerator.backward(out.loss)
        acc_opt.step(); acc_opt.zero_grad(set_to_none=True)
        if step >= 3:
            break

sort_key = "self_cuda_time_total" if CUDA else "self_cpu_time_total"
print(prof_acc.key_averages().table(sort_by=sort_key, row_limit=10))""")

md(r"""출력은 §3 의 PyTorch Profiler 와 같은 형식입니다 — Accelerate 는 그 위의 얇은 래퍼라, \
"HF 환경에서 빠르게 1차 스캔" 할 때 편합니다. 세밀한 제어(`schedule`, `on_trace_ready`)도 \
`ProfileKwargs` 에 `schedule_option=...` 으로 그대로 넘길 수 있습니다.""")

# ----- 5. FLOPS / params -----
md(r"""## 5. 도구 ③ FLOPS / params (모델 비용의 절대량)

시간은 하드웨어·구현에 따라 변하지만, **FLOPs(연산량)와 params(파라미터 수)** 는 모델 고유의 절대량입니다. \
"이 모델이 본질적으로 얼마나 무거운가" 를 봅니다.""")

code(r"""# params
n_params = model.num_parameters()
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total params     : {n_params/1e6:.1f} M")
print(f"trainable params : {n_trainable/1e6:.1f} M")

# FLOPs — torch.profiler 의 with_flops 로 op 별 추정
with profile(activities=activities, with_flops=True) as prof_flops:
    b = {k: v.to(device) for k, v in next(iter(loader)).items()}
    with torch.inference_mode():
        model.eval(); _ = model(**b)
model.train()

print()
print(prof_flops.key_averages().table(sort_by="flops", row_limit=8))""")

md(r"""### MFU — GPU 를 몇 % 쓰고 있나 (참고 지표)

**MFU(Model FLOPs Utilization)** = 실제 달성 FLOPS / GPU 이론 peak FLOPS. \
"비싼 GPU 를 제대로 쓰고 있나" 의 척도입니다 (보통 잘 튜닝된 학습이 30-50%).

```
MFU = (모델 1 step FLOPs) / (1 step 시간) / (GPU peak FLOPS)
```

예: T4 의 fp16 peak ≈ 65 TFLOPS, fp32 ≈ 8.1 TFLOPS. \
입문 단계에선 "이런 지표가 있고, 낮으면 GPU 가 논다는 뜻" 정도만 알면 충분합니다. \
정밀한 layer 단위 FLOPs/MFU 는 DeepSpeed Flops Profiler 가 더 강력합니다 (분산 부록에서).""")

# ----- 6. 무엇을 진단하나 -----
md(r"""## 6. 출력을 어떻게 병목 진단으로 연결하나

| 관찰 | 해석 | 다음 행동 |
|---|---|---|
| `addmm`/`bmm` 이 시간 대부분 | compute-bound (정상적인 BERT) | `fp16`, batch 키우기, 작은 모델 |
| step breakdown 합 ≪ wall-clock | **DataLoader 대기** | `num_workers↑`, `pin_memory=True`, 토큰화 캐싱 |
| backward 메모리 peak 가 큼 | activation 메모리 한계 | `fp16`, gradient checkpointing, batch↓ |
| chrome trace 에 GPU idle gap | host(CPU) 대기 또는 sync 과다 | 입력 준비 비동기화, 불필요한 `.item()`/`.cpu()` 제거 |
| MFU 낮음 (<10%) | GPU 가 논다 | batch↑, seq 길이↑, fp16 |

**핵심**: 프로파일러는 "어디가 느린지" 만 알려줍니다. *왜* 와 *어떻게 고칠지* 는 이 표처럼 해석이 필요합니다.""")

# ----- 7. 더 깊이 -----
md(r"""## 7. 더 깊이 (안내)

이 부록은 단일 GPU 입문용입니다. 더 강력한 도구들:

- **TensorBoard Profiler plugin** — `tensorboard_trace_handler` 로 trace 저장 후 TensorBoard 의 \
  "PyTorch Profiler" 탭에서 GPU util / step breakdown / 추천(recommendation)까지 시각화
- **DeepSpeed Flops Profiler** — layer 단위 FLOPs/latency, achieved TFLOPS (ZeRO 학습에 특화)
- **DeepSpeed Comms Logger** — NCCL all-reduce/all-gather 통신 비용 (멀티 GPU)
- **NVIDIA Nsight Systems** — CUDA stream / NCCL / kernel 의 시스템 timeline
- **Holistic Trace Analysis (HTA)** — 멀티 GPU trace 자동 분석 (idle/comm/compute 비율)

→ 멀티 GPU·분산 프로파일링은 **별도 부록**에서 다룹니다.""")

# ----- 8. FAQ -----
md(r"""## ❓ FAQ

### Q1. `schedule` 의 wait/warmup/active 는 왜 필요한가요?

전체 학습을 다 추적하면 trace 파일이 수 GB 로 폭증하고 학습도 느려집니다. \
`wait`(추적 안 함) → `warmup`(추적하되 버림, 캐시 안정화) → `active`(실제 기록) 사이클로 \
**대표적인 몇 step 만** 기록합니다. 첫 step 은 cuDNN autotune 때문에 비정상적으로 느려서 warmup 으로 걸러냅니다.

### Q2. CPU 에서도 의미가 있나요?

네. `ProfilerActivity.CUDA` 만 빠지고 CPU op 시간은 그대로 측정됩니다. \
다만 GPU idle gap / kernel 분석은 GPU 에서만 의미 있습니다. 이 부록은 `CUDA` 플래그를 자동 감지해 처리합니다.

### Q3. `record_function` 없이도 forward/backward 를 구분할 수 있나요?

profiler 가 자동으로 `aten::*` op 를 잡지만, "forward 단계 / backward 단계" 같은 *논리적 구간* 은 \
`record_function("name")` 으로 직접 라벨해야 trace 와 테이블에서 묶여 보입니다. \
backward 의 grad 연산은 보통 이름에 `Backward` 가 붙어 구분되기도 합니다.

### Q4. Trainer 를 쓸 때는 어떻게 프로파일하나요?

`TrainerCallback` 의 `on_step_end` 에서 `prof.step()` 을 호출하도록 콜백을 만들어 끼웁니다:

```python
class ProfCallback(TrainerCallback):
    def __init__(self, prof): self.prof = prof
    def on_step_end(self, args, state, control, **kwargs): self.prof.step()

with profile(activities=..., schedule=..., on_trace_ready=tensorboard_trace_handler("./tb")) as prof:
    trainer.add_callback(ProfCallback(prof))
    trainer.train()
```

또는 이 부록의 §4 처럼 Accelerate `ProfileKwargs` 가 더 간단합니다.

### Q5. 측정값이 매번 다른데 정상인가요?

네. 첫 실행은 cuDNN autotune·캐시 워밍업으로 느립니다. `warmup` step 을 두고, \
`time_phases` 처럼 여러 step 평균을 내면 안정됩니다. `sync()`(= `torch.cuda.synchronize()`)를 \
빼먹으면 GPU 작업이 안 끝났는데 시간을 재서 비정상적으로 빠르게 나오니 주의하세요.

### Q6. profiler 자체의 오버헤드는 없나요?

있습니다 (특히 `record_shapes`, `with_stack`, `profile_memory` 를 켜면). \
그래서 측정은 `schedule` 로 몇 step 만 하고, *절대 시간* 보다 *상대 비중* 을 신뢰하는 게 좋습니다.""")

# ----- 9. 다음 -----
md(r"""## 다음

- 여기서 찾은 병목(예: DataLoader 대기, backward 메모리)을 실제로 고쳐 보고 **before/after** 를 \
  같은 프로파일러로 재 보면 효과가 눈에 보입니다.
- 모델을 키우면(예: BERT-base → large) compute 와 메모리 패턴이 극적으로 변합니다 — \
  *모델 크기 스케일링* 실험으로 이어집니다.
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
