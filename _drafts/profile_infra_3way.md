# 프로파일 비교 레포트 — T4(CUDA) vs MPS vs CPU

> 같은 워크로드(DistilBERT binary 학습 1 step, bs=16/seq=128, `record_function` forward/backward/optimizer, active 3 step)를 **세 인프라**에서 프로파일한 chrome trace 를 파싱·비교합니다.
>
> - `trace_train.json` — Google Colab **T4 (CUDA)**, 실제 yelp 데이터
> - `trace_mps.json` — 로컬 **Apple Silicon (MPS)**, 실제 yelp 데이터
> - `trace_cpu.json` — 로컬 **CPU only** (M1, 8 threads), 더미 입력(동일 shape — op 시간은 데이터값 무관)

![3-way comparison](./profile_infra_3way.png)

---

## 1. 가장 큰 차이 — GPU 커널 레인의 유무

| 카테고리 | T4 (CUDA) | MPS | CPU |
|---|---:|---:|---:|
| `kernel` (GPU 커널) | **500 ms** | — | — |
| `cuda_runtime` | 428 ms | — | — |
| `gpu_memcpy/memset` | 1.4 ms | — | — |

`volta_sgemm_*`·`fmha_*` 같은 GPU 커널은 **T4 에서만** 잡힙니다. MPS·CPU trace 에는 GPU 레인이 없는데, **이유가 다릅니다**:
- **CPU**: 애초에 GPU 가 없음 → 모든 연산이 `cpu_op`.
- **MPS**: GPU(Metal)는 쓰지만, `torch.profiler` 에 **`ProfilerActivity.MPS` 가 없어** 커널 타임스탬프를 trace 에 *수집하지 못함*. (CUDA 는 CUPTI 로 `ProfilerActivity.CUDA` 를 통해 받음 — MPS 엔 그 연동이 아직 없음)

## 2. forward / backward / optimizer — 시간이 "어디에" 잡히나

아래는 profiler 의 `record_function` 구간 **CPU wall** (active 3 step 합)입니다 — **GPU 실행 시간이 아니라 그 구간의 CPU 측 시간**입니다.

| 구간 (record_function, CPU wall) | T4 (CUDA) | MPS | CPU |
|---|---:|---:|---:|
| forward | 35 ms | 30 ms | **980 ms** |
| backward | 55 ms | 30 ms | **958 ms** |
| optimizer | 396 ms | 778 ms | 279 ms |

**해석 — 같은 30~35 ms 라도 의미가 다릅니다**:
- **T4 / MPS (비동기 가속기)**: `model(**batch)` 는 GPU 큐에 명령을 *던지고 즉시 리턴*합니다. sync 가 없는 이 구간 값(30-35 ms)은 **CPU 의 dispatch(큐잉) 시간**일 뿐, GPU 실행 시간이 아닙니다.
  - **T4**: GPU 실행 시간은 `kernel` 레인(500 ms)·`gpu_user_annotation` forward(133 ms)로 *따로* 잡혀 보입니다.
  - **MPS**: GPU 커널 시간은 trace 에 안 잡힙니다(§1). 대신 **wall-clock 은 `torch.mps.synchronize()` 로 측정 가능** — 노트북 §4-3 `time_train_phases`(sync 포함) 기준 실제 **forward ≈ 82 ms / backward ≈ 128 ms / optimizer ≈ 69 ms**. trace 의 30 ms 는 그중 *dispatch* 만 본 값입니다.
- **CPU (동기)**: 가속기가 없어 dispatch 라는 개념이 없고, `record_function` 값(forward 980 ms)이 **곧 실제 연산 시간**입니다.

> 요약: **"MPS forward/backward 실측"은 가능합니다(wall-clock, `mps.synchronize` 사용).** 단 *profiler trace 의 op/커널 단위 GPU 시간* 은 `ProfilerActivity.MPS` 부재로 수집되지 않습니다. 둘은 다른 얘기입니다.

## 3. 1위 op 의 정체 — 가속기 vs CPU 의 정반대 시그니처

| 인프라 | top cpu_op | 의미 |
|---|---|---|
| T4 | `aten::item` 382 ms | **GPU→CPU 동기화** (가속기가 멈춤) |
| MPS | `aten::item` 736 ms | 동기화, T4 의 ~2배 (dispatch 무거움) |
| CPU | `AddmmBackward`/`aten::mm` 559-596 ms, `scaled_dot_product_attention` 394 ms | **실제 행렬 연산** |

→ **가속기(T4/MPS)는 `aten::item`(동기화)이 1위**, **CPU 는 `aten::mm`(실제 compute)이 1위**. CPU 에는 `aten::item` 이 top 에 없습니다 — CPU-CPU 라 동기화가 사실상 공짜.

## 4. 시그니처로 환경 역추적

trace 만 보고 어느 인프라인지 맞히기:

| 단서 | 결론 |
|---|---|
| `volta_sgemm`·`cuda_runtime`·`gpu_memcpy` 존재 | **T4 (CUDA)** |
| GPU 레인 없음 + `aten::item` 1위 + `*_for_mps` op | **MPS** (가속기지만 GPU 커널 시간 미수집) |
| GPU 레인 없음 + `aten::mm`/`AddmmBackward` 1위 + `aten::item` 없음 | **CPU only** (순수 연산) |

## 5. 종합

| 항목 | T4 (CUDA) | MPS | CPU |
|---|---|---|---|
| GPU 커널 레인(trace) | ✅ 500 ms | ❌ 미수집 (`ProfilerActivity.MPS` 부재) | — (GPU 없음) |
| GPU 실행시간 가시성 | trace 에 직접 | **wall-clock 만** (`mps.synchronize`) | N/A |
| forward/backward (실제) | GPU 가 빠르게 (kernel 500 ms) | wall ≈ 82 / 128 ms | CPU 가 느리게 980 / 958 ms |
| 1위 비용 | 동기화(item) | 동기화(item) ~2배 | 실제 matmul |
| 주 병목 | sync + optimizer | sync + dispatch | **compute (forward/backward)** |

### 시사점

1. **CPU 는 forward/backward(matmul) 자체가 병목** — 가속기가 없으니 당연. 큰 모델·batch 면 비현실적으로 느려짐 → GPU 가 필요한 이유가 trace 로 드러남.
2. **가속기(T4/MPS)에선 동기화(`aten::item`)가 발목** — 학습 루프의 불필요한 `.item()`/`.cpu()` 를 줄이면 가속기가 덜 멈춥니다.
3. **MPS 는 wall-clock 은 잴 수 있지만 op/커널 단위 GPU 분해는 안 됨** — `ProfilerActivity.MPS` 가 없어서. "전체가 얼마 걸렸나"는 `mps.synchronize` 로 알 수 있어도, "어느 커널이 비싼가"는 CUDA 에서 봐야 합니다.
4. **trace 시그니처로 환경을 역추적**할 수 있습니다(§4) — 프로파일 읽기의 핵심 훈련.

---

*세 chrome trace 를 PyTorch Profiler 스키마로 파싱해 집계. forward/backward/optimizer 는 `record_function` 의 CPU wall(가속기는 dispatch, CPU 는 실제 연산). MPS wall-clock(82/128/69 ms)은 노트북 §4-3 `time_train_phases`(`mps.synchronize` 포함) 측정값. 수치는 active 3 step 합계.*
