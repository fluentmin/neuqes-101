# 32_diffusion_intro — 작은 mask-diffusion LM 직접 구현 (Phase 5 첫 챕터)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/32_diffusion_intro/32_diffusion_intro.ipynb)

## 한 줄 목표
Phase 5 의 첫 챕터. Ch 24-31 의 *GPT (decoder, autoregressive, 왼→오 순차 생성)* 패러다임에서, *Diffusion LM (encoder/bidirectional, masked-denoise, 문장 전체 병렬 생성)* 패러다임으로 전환합니다. 핵심: **BERT MLM (Ch 20-23) 의 고정 15% 마스킹을 가변 0-100% 로 일반화하고, 한 번에 복원하는 대신 여러 번 반복 denoise 하면 그게 generation 입니다.** 작은 BERT-style 모델을 from scratch 로 TinyStories 에 diffusion 목표로 학습 → 전부 `[MASK]` 에서 시작해 *병렬 denoise* 로 텍스트를 생성하는 과정을 직접 구현·관찰합니다.

## Phase 5 도입

| 축 | Phase 4 (GPT, Ch 24-31) | **Phase 5 (Diffusion, Ch 32-34)** |
|---|---|---|
| attention | Causal (과거만) | **Bidirectional (양방향)** |
| 학습 목표 | next-token 예측 | **가변 마스킹 denoising** |
| 생성 순서 | 왼→오 한 토큰씩 순차 | **문장 전체를 동시에 반복 denoise** |
| 생성 step | 토큰 수 = step 수 | **자유 조절 (4 / 16 / 32 ...)** |
| 출발 상태 | prompt 토큰 | **전부 `[MASK]`** |
| 본체 계보 | GPT (Ch 24) | **BERT (Ch 20) — MLM 일반화** |

## 다루는 핵심 개념
- **mask-diffusion = MLM 일반화** — 고정 15% (BERT) → 가변 $t \sim U(0,1)$ (diffusion). Ch 1 부터 추적한 마스킹 thread 의 클라이맥스
- **`DiffusionCollator`** (직접 구현) — 매 배치 `t` 를 뽑아 그 비율로 `[MASK]` 치환. Ch 20 의 고정 15% collator 와 정면 대비
- **`1/t` 재가중 denoising loss** (`compute_loss` 오버라이드) — 마스킹 비율 무관하게 척도 정렬, log-likelihood upper bound
- **`BertForMaskedLM(config)` from scratch** — bidirectional encoder 가 diffusion 의 denoiser (Ch 20 과 같은 패턴, 목적만 다름)
- **reverse process generation** (`diffusion_generate`) — 전부 `[MASK]` → low-confidence remasking 으로 반복 denoise. 채우는 순서가 *위치가 아니라 confidence*
- **denoise 궤적 시각화** — 마스크가 *병렬로* 단어로 채워지는 과정 직접 관찰 (AR 의 왼→오와 핵심 대비)
- **조건부 생성 (infilling)** — prompt 고정 + 나머지 denoise. 양방향이라 중간 채우기도 가능 (AR 불가)
- **denoise step 수 trade-off** — 1 (빠르고 거침) ↔ 32 (느리고 정교), 추론 시점 조절
- **AR vs Diffusion 비교** — 같은 TinyStories, 생성 메커니즘만 다름. Ch 24 (GPT) 와 나란히
- **`[MASK]` 토큰** — WordPiece (`bert-base-uncased`) 내장. forward/reverse 양쪽의 캔버스

## Loss
masked-diffusion denoising loss — BERT MLM 의 CrossEntropyLoss 를 *가변 마스킹 비율 $t$* 로 일반화하고 *$1/t$ 재가중*:

$$L = \mathbb{E}_{t \sim U(0,1)} \left[ \frac{1}{t} \cdot \frac{1}{L} \sum_{i:\, x_t^{(i)} = \texttt{[MASK]}} -\log P_\theta\!\left(x_0^{(i)} \mid x_t\right) \right]$$

가려진 자리만 loss 계산 (`-100` 트릭, Ch 20-23 과 동일). `1/t` 재가중 덕분에 random baseline 이 어떤 $t$ 든 `ln(30522) ≈ 10.33` 으로 정렬 — Ch 20 MLM 과 같은 척도.

## 데이터
`roneneldan/TinyStories` (Eldan & Li 2023, arXiv:2305.07759) — Ch 24 (GPT) 와 *완전히 동일*. 학습 split 의 처음 30,000 stories. `block_size=128`, 특수 토큰 없이 순수 스트림으로 chunk 화. 데이터를 Ch 24 와 같게 둔 이유는 *생성 방식만 다른* AR vs Diffusion 비교를 위함.

## 모델
**`BertForMaskedLM`** with `hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128`. 약 **13M params** (대부분 임베딩). 완전 random init from scratch — bidirectional encoder 가 diffusion denoiser 역할.

## Hyperparams
- `max_steps=1500`, `per_device_train_batch_size=32`, `learning_rate=3e-4`
- `lr_scheduler_type="cosine"`, `warmup_steps=100`, `weight_decay=0.01`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가), `eval_steps=150`
- `remove_unused_columns=False` (collator 가 만드는 `labels`/`t` 보존), `label_names=["labels"]`
- 생성: `length=48, steps=16` 기본 (변형에서 `steps` 를 1-32 로 비교)

## 환경
Google Colab **T4 GPU 필수**. 약 25-30분 (데이터 로드 약 2분 + 토큰화 약 3분 + 학습 전 denoise 약 30초 + 모델 학습 약 13-15분 + 학습 후 denoise + 궤적 + AR 비교 약 3분).

device 자동 감지 (CUDA / MPS / CPU) - 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | 생성/학습 방식 | Loss |
|---|---|---|---|---|---|
| 24 | 작은 GPT2 (직접, scratch) | BPE (직접 학습) | TinyStories | autoregressive (왼→오) | CE (next-token) |
| 31 | SFT base + GRPO | BBPE | verifiable-reward | autoregressive + RL | GRPO loss |
| **32** | **작은 BERT-style (직접, scratch)** | **WordPiece (`bert-base-uncased`)** | **TinyStories** | **parallel denoise (가변 마스킹 + 반복)** | **masked-diffusion loss (`1/t` 재가중)** |
| 33 (다음) | LLaDA-8B-Instruct (사전학습) | LLaDA tokenizer | 다국어 추론 시연 | parallel denoise (추론만) | — |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[33_llada](../33_llada/) — `GSAI-ML/LLaDA-8B-Instruct` (arXiv:2502.09992), 8B params 실전 mask-diffusion LLM 추론 시연. 본 챕터에서 직접 구현한 *가변 마스킹 + 반복 denoise* 의 대규모 버전을 사전학습 모델로 체감하고 autoregressive LLM 과 비교합니다. Ch 34 (Trida-7B) 에서 한국 산 diffusion 모델 + AR 직접 비교로 Phase 5 마무리.
