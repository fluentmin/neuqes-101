"""인프라별 PyTorch 프로파일 trace 생성 + 비교 (T4 / MPS / CPU / ...).

Ch 10 프로파일 부록(`appendix_profiling`)의 학습 §4 와 동일한 워크로드
(DistilBERT binary, bs=16/seq=128, record_function forward/backward/optimizer,
schedule wait1·warmup1·active3)를 device 만 바꿔 프로파일합니다.

입력은 더미(동일 shape)라 데이터 다운로드가 필요 없습니다 — op 시간은 데이터 값과 무관.

재현
----
    # 1) 각 환경에서 trace 생성 (device 자동 감지 / 강제)
    python profile_infra.py profile --out trace_t4.json              # CUDA 자동
    python profile_infra.py profile --device cpu --out trace_cpu.json
    python profile_infra.py profile --device mps --out trace_mps.json

    # 2) 모아서 비교 (표 + 그림)
    python profile_infra.py compare trace_t4.json trace_mps.json trace_cpu.json --fig cmp.png

`.json` / `.json.gz` 둘 다 읽습니다.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict


# ----------------------------- profile -----------------------------
def pick_device(force):
    import torch
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_profile(a):
    import torch
    from transformers import AutoModelForSequenceClassification
    from torch.profiler import profile, schedule, record_function, ProfilerActivity

    dev = pick_device(a.device)
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, problem_type="multi_label_classification"
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    V = model.config.vocab_size
    batch = dict(
        input_ids=torch.randint(0, V, (a.bs, a.seq)).to(dev),
        attention_mask=torch.ones(a.bs, a.seq, dtype=torch.long).to(dev),
        labels=torch.randint(0, 2, (a.bs, 1)).float().to(dev),
    )

    model.train()
    t0 = time.time()
    with profile(activities=activities, record_shapes=True, profile_memory=True,
                 schedule=schedule(wait=1, warmup=1, active=3)) as prof:
        for _ in range(5):
            with record_function("forward"):
                out = model(**batch); loss = out.loss
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer"):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            sync()
            prof.step()
    prof.export_chrome_trace(a.out)
    print(f"[{dev}] profiled 5 steps in {time.time()-t0:.1f}s → {a.out}")


# ----------------------------- compare -----------------------------
def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        t = json.load(f)
    return t["traceEvents"] if isinstance(t, dict) else t


def summarize(ev):
    cat = defaultdict(float)
    phase = defaultdict(float)
    cpuop = defaultdict(float)
    for e in ev:
        if e.get("ph") != "X":
            continue
        c = e.get("cat", "?")
        cat[c] += e.get("dur", 0)
        if c == "user_annotation" and e.get("name") in ("forward", "backward", "optimizer"):
            phase[e["name"]] += e.get("dur", 0)
        if c == "cpu_op":
            cpuop[e["name"]] += e.get("dur", 0)
    top = max(cpuop.items(), key=lambda x: x[1]) if cpuop else ("-", 0.0)
    return cat, phase, top


def cmd_compare(a):
    rows = [(p.split("/")[-1], *summarize(load(p))) for p in a.traces]

    print(f"\n{'trace':28} {'GPUkern(ms)':>11} {'fwd':>7} {'bwd':>7} {'opt':>7}  top cpu_op")
    print("-" * 90)
    for name, cat, ph, top in rows:
        gk = cat.get("kernel", 0) / 1e3
        print(f"{name[:28]:28} {(f'{gk:.0f}' if gk else '-'):>11} "
              f"{ph.get('forward',0)/1e3:7.0f} {ph.get('backward',0)/1e3:7.0f} "
              f"{ph.get('optimizer',0)/1e3:7.0f}  {top[0][:30]} ({top[1]/1e3:.0f} ms)")

    if not a.fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # GPU kernel vs cpu_op
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [r[1].get("kernel", 0) / 1e3 for r in rows], 0.4, label="GPU kernel", color="tab:green")
    axes[0].bar(x + 0.2, [r[1].get("cpu_op", 0) / 1e3 for r in rows], 0.4, label="cpu_op", color="tab:gray")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8, rotation=10)
    axes[0].set_ylabel("total dur (ms)"); axes[0].set_title("GPU kernel lane vs cpu_op")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
    # phases (log)
    ph_names = ["forward", "backward", "optimizer"]
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        axes[1].bar(np.arange(3) + (i - (len(rows)-1)/2) * w,
                    [max(r[2].get(p, 0) / 1e3, 1e-3) for p in ph_names], w, label=r[0])
    axes[1].set_yscale("log"); axes[1].set_xticks(range(3)); axes[1].set_xticklabels(ph_names)
    axes[1].set_ylabel("CPU wall, 3 steps (ms, log)"); axes[1].set_title("record_function phases")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout(); plt.savefig(a.fig, dpi=110)
    print(f"\nfigure → {a.fig}")


# ----------------------------- cli -----------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("profile", help="현재(또는 지정) device 에서 trace 생성")
    pp.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None, help="강제 device (기본: 자동)")
    pp.add_argument("--model", default="distilbert-base-uncased")
    pp.add_argument("--bs", type=int, default=16)
    pp.add_argument("--seq", type=int, default=128)
    pp.add_argument("--out", default="trace.json")
    pp.set_defaults(func=cmd_profile)

    pc = sub.add_parser("compare", help="여러 trace 파싱·비교 (표 + 옵션 그림)")
    pc.add_argument("traces", nargs="+", help="trace .json / .json.gz 들")
    pc.add_argument("--fig", default=None, help="비교 그림 저장 경로 (예: cmp.png)")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
