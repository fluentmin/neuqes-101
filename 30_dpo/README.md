# 30_dpo — DPO / 사람 선호로 정렬 (Phase 4 학습 단계 4, alignment)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/30_dpo/30_dpo.ipynb)

## 한 줄 목표
Ch 28 (KoGPT2 SFT) 의 *다음 단계*. SFT 가 *지시를 따르게* (행동 정렬) 만들었다면, **DPO (Direct Preference Optimization)** 는 SFT 모델을 *preference 쌍 (chosen / rejected)* 으로 학습해 **사람의 선호에 맞춰 정렬** 합니다 — *좋은 답의 확률은 올리고, 나쁜 답은 내림*. 바뀌는 건 *데이터 (instruction-response → preference 쌍)* + *trainer (`SFTTrainer` → `trl.DPOTrainer`)* + *loss (next-token CE → DPO sigmoid)* + *frozen reference 모델 추가*.

## DPO = RLHF (PPO) 의 간소화
- 전통 RLHF = SFT → reward model 학습 → PPO (actor + critic + reward + reference, **4 모델**). T4 메모리에 무리
- **DPO = reward model 없이 preference 쌍으로 *직접* 정책 최적화**. policy + frozen reference (**2 모델**) 만. PPO 대비 간단·안정 → 본 커리큘럼이 PPO 대신 DPO 채택

| 방식 | 필요 모델 | 학습 | T4 |
|---|---|---|---|
| PPO (RLHF) | actor + critic + reward + reference (4) | 강화학습 | ✗ |
| **DPO (본 챕터)** | **policy + frozen reference (2)** | **지도학습** | **✓** |

## DPO Loss
$$L_{\text{DPO}} = -\log \sigma\big( \beta \cdot [ (\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)) - (\log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x)) ] \big)$$

- $y_w$ = chosen, $y_l$ = rejected, $\pi_{\text{ref}}$ = frozen reference (SFT 모델 복사·freeze)
- **implicit reward** $r(x,y) = \log\pi_\theta - \log\pi_{\text{ref}}$ — *정책이 reference 보다 이 답을 얼마나 더 선호하나*
- **margin** $= r(y_w) - r(y_l)$ 가 클수록 loss ↓. chosen 우위를 rejected 보다 크게
- **β** = reference 에서 벗어나는 정도 제어 (KL 제약, 기본 0.1). 크면 빨리 정렬·붕괴 위험 / 작으면 안전·느림
- **frozen reference 필요성** — 원본 SFT 에서 너무 멀어지지 않게 (reward hacking·degeneration 방지) 하는 *닻*

## `labels = -100` thread 연결
DPO 도 *response 부분만* log-prob 계산 (prompt 제외). MLM(15%) → CausalLM(거의 전부) → SFT(response 만) → **DPO(chosen/rejected 각각 response 만)**. *prompt 는 조건, 답변만 비교 대상* 이라는 원리가 alignment 까지 이어집니다.

## GPT 시대 학습 4단계 — 본 챕터의 위치

| 단계 | 용어 | 본 챕터? | 본 커리큘럼 |
|---|---|---|---|
| 1 | Pretraining | | Ch 24 (영어), Ch 26 (한국어) |
| 2 | Continual pretraining | | Ch 25 (영어), Ch 27 (한국어) |
| 3 | SFT (Instruction tuning) | | Ch 28 |
| **4** | **Alignment (DPO)** | **✅ ← 여기** | **Ch 30 (DPO), Ch 31 (GRPO)** |

## 다루는 핵심 개념
- **`trl.DPOTrainer` + `trl.DPOConfig`** — DPO 특화 trainer (첫 등장). response-only log-prob → implicit reward → margin → sigmoid loss 자동
- **`DPOConfig(beta=0.1)`** — reference 제약 강도 (KL). DPO 의 핵심 하이퍼파라미터
- **`DPOTrainer(ref_model=None)`** — reference 자동 복사·freeze (frozen reference 개념)
- **`prompt` / `chosen` / `rejected` 데이터 형식** — preference 쌍 표준
- **DPO loss 직관 시각화** (§3) — 한 샘플로 response-only log-prob → implicit reward → margin → loss 를 *손으로 계산*, margin↔loss 곡선
- **DPO 전·후 reward margin 비교** (§5, 핵심 데모) — margin 분포가 *양수 (chosen 선호) 로 이동* 하는지 + reward accuracy
- **frozen reference 직접 생성** — `copy.deepcopy` + `requires_grad_(False)` (§3 시연)

## 데이터
`maywell/ko_Ultrafeedback_binarized` — 한국어 preference 데이터셋 (`prompt` / `chosen` / `rejected`). 짧은 샘플 필터 + 약 1,500 subset, Ch 28 SFT 와 같은 instruction 포맷으로 prompt 감쌈.

## 모델
**policy** = SFT 모델 (노트북 단독 실행을 위해 base KoGPT2 로 시작 — 정석은 Ch 28 SFT 체크포인트). **reference** = 같은 모델 복사 + freeze. 토크나이저 `PreTrainedTokenizerFast` (Ch 27 이후 고정, AutoTokenizer 함정 회피).

## Hyperparams
- `num_train_epochs=1`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch 16)
- `learning_rate=5e-6` (DPO 는 SFT 보다 작은 lr), `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`
- `beta=0.1` (DPO 기본), `max_length=512`
- `fp16=True` (T4 는 bf16 불가)

## VRAM 주의
DPO 는 *policy + frozen reference 두 모델* 을 메모리에 올립니다 (SFT 의 약 2배). T4 (16GB) 에서는 batch 작게 (2) + grad accum (8) + `fp16=True` 로 관리. `ref_model=None` 으로 주면 `DPOTrainer` 가 reference 를 자동 생성.

## 라이브러리 주의 — `trl` 버전
`trl` 은 버전마다 `DPOTrainer` / `DPOConfig` API 변동이 큽니다 (`max_prompt_length` 같은 인자가 버전에 따라 사라지기도). 본 노트북은 *버전 간 안정적인 핵심 경로* (`prompt`/`chosen`/`rejected` 데이터 + `beta` + `max_length` + `ref_model=None`) 만 사용. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.

## 환경
Google Colab **T4 GPU 필수**. 약 22-30분 (preference 데이터 로드·필터 약 2분 + 모델 로드 약 2분 + DPO loss 시각화 약 1분 + DPO 학습 약 15-22분 + 전·후 margin 비교 약 3분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 28 | KoGPT2 (125M, SFT) | KoAlpaca instruction-response | response 토큰 (답변만) | CE (response-only) - SFT | `SFTTrainer` |
| 29 | Ch 28 SFT 모델 (평가) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) | - |
| **30** | **SFT 모델 (policy) + frozen ref** | **preference 쌍 (chosen/rejected)** | **chosen 선호 ↑ / rejected ↓** | **DPO sigmoid (β=0.1)** | **`DPOTrainer`** |
| 31 (다음) | SFT 모델 + verifier | verifiable-reward prompts (수학·코드) | group relative advantage | `GRPO loss` | `GRPOTrainer` |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[31_grpo](../31_grpo/) (예정) — **GRPO**. DPO 는 *사람이 비교한 preference 쌍*, GRPO 는 *verifier 가 자동 채점한 reward* (수학·코드 정답 자동 검증). 같은 prompt 에 여러 답을 rollout → 그룹 안 상대 비교 (group relative advantage). alignment 의 *두 방식 비교* (주관적 선호 vs 객관적 정답).
