# 31_grpo — GRPO / 검증 가능한 보상으로 정렬 (Phase 4 마지막, alignment 두 번째 방식)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/31_grpo/31_grpo.ipynb)

## 한 줄 목표
Ch 30 (DPO) 의 *다음 방식*. DPO 가 *사람/AI 가 비교한 preference 쌍* 으로 정렬했다면, **GRPO (Group Relative Policy Optimization)** 는 *정답을 자동 검증(verifier) 할 수 있는 task (수학·코드)* 에서 모델이 *여러 답을 생성(rollout)* 하고 *verifier 가 채점* 해 *잘한 답 방향* 으로 RL 합니다. **DeepSeek-R1 이 순수 RL 로 reasoning 을 끌어낸 방법**. 바뀌는 건 *신호 출처 (preference 쌍 → verifier reward)* + *trainer (`DPOTrainer` → `trl.GRPOTrainer`)* + *데이터 (chosen/rejected → prompt + 정답)* + *rollout (한 prompt 에 여러 답 생성)*.

## PPO vs DPO vs GRPO — alignment 세 갈래
| 방법 | 신호 출처 | 필요 모델 | 데이터 | T4 |
|---|---|---|---|---|
| PPO (전통 RLHF) | reward model 점수 | actor + critic + RM + reference (4) | prompt + 학습된 RM | ✗ |
| DPO (Ch 30) | preference 쌍 (사람/AI 비교) | policy + frozen reference (2) | `(prompt, chosen, rejected)` | ✓ |
| **GRPO (본 챕터)** | **verifier (정답 자동 검증)** | **policy 만** (+ 옵션 ref) | **`(prompt, 정답)` 검증 가능** | **✓** |

**왜 critic 도 reward model 도 없이 되나**: *group 평균* 이 baseline (critic 대체), *verifier* 가 reward (reward model 대체). GRPO 는 *RL 루프(rollout)는 유지* 하면서 *critic·RM 만* 없앤 PPO 간소화.

## GRPO 메커니즘 (group relative advantage)
1. **rollout** — 한 prompt 에 여러 답 생성 (group, 예: 4-8개)
2. **verifier reward** — 각 답을 자동 채점 (수학: 정답 매칭 → 1/0)
3. **group relative advantage** — $A_i = (r_i - \text{mean}(r)) / (\text{std}(r) + \varepsilon)$. *group 평균이 baseline* → critic 불필요
4. **정책 갱신** — advantage 양수(평균보다 잘함) 확률 ↑, 음수 ↓

수치 예시: reward $[1,0,1,0]$ → mean=0.5, std=0.5 → advantage $[+1,-1,+1,-1]$. group 전체가 같으면 (전부 정답·오답) std=0 → advantage 0 → 학습 신호 없음 (다양성 필요).

## verifiable reward 의 의미
- *정답을 자동 검증* 할 수 있는 task (수학·코드·형식) 는 *사람 채점 없이 무한히* RL 신호 생성 가능 (verifier 가 *공짜 reward model*)
- **DeepSeek-R1**: 정답이라는 객관 reward 만으로 *사람 데모 없이* reasoning(chain-of-thought) 능력 발현. Ch 29 부록의 *pass@1·cons@64* 와 같은 뿌리
- 한계: *검증 가능한 task 에만*. 열린 질문(글쓰기·대화)은 DPO(선호) 나 LLM-as-judge

## alignment 의 두 방식 — DPO vs GRPO
| | DPO (Ch 30) | GRPO (Ch 31) |
|---|---|---|
| 신호 | 사람/AI 가 *비교* 한 preference | verifier 가 *자동 검증* 한 정답 |
| 성격 | 주관적 선호 | 객관적 정답 |
| 적합 task | 정답 없는 열린 질문 | 정답 자동 확인 가능 |

## GPT 시대 학습 4단계 — 본 챕터의 위치

| 단계 | 용어 | 본 챕터? | 본 커리큘럼 |
|---|---|---|---|
| 1 | Pretraining | | Ch 24 (영어), Ch 26 (한국어) |
| 2 | Continual pretraining | | Ch 25 (영어), Ch 27 (한국어) |
| 3 | SFT | | Ch 28 |
| **4** | **Alignment (GRPO)** | **✅ ← 여기** | **Ch 30 (DPO), Ch 31 (GRPO)** |

## 다루는 핵심 개념
- **`trl.GRPOTrainer` + `trl.GRPOConfig`** — GRPO 특화 trainer (첫 등장). rollout → verifier 채점 → group advantage → 정책 갱신 자동
- **`reward_funcs` (verifier)** — 생성 답을 채점하는 callable (또는 list). `(completions, **kwargs)` → `list[float]`
- **`GRPOConfig(num_generations=4)`** — group size (rollout 답 개수)
- **`GRPOConfig(beta=0.0)`** — KL 제약 (0 = ref-free, 메모리 절약)
- **verifier + group advantage 손계산** (§3) — 한 prompt 에 여러 답 채점 → `(r-mean)/std` 로 advantage 재현
- **GRPO 전·후 정확도 비교** (§5, 핵심 데모) — eval 셋의 verifier pass rate 상승 확인
- **학습 곡선** (§6) — reward(group 평균)·reward_std(다양성)·loss

## 데이터
합성 **산술 (arithmetic)** — `(prompt, answer)`. `prompt` 는 모델 입력 (`"3 + 5 = ?"`), `answer` 는 *verifier 채점용* (모델 입력 아님). 가장 깨끗한 verifiable task (정답이 정수 하나 → 문자열 매칭). 코드는 샌드박스·시간 부담으로 산술 위주. Ch 28 SFT 와 같은 instruction 포맷으로 prompt 감쌈. train 256 / eval 64.

## 모델
**policy** = SFT 모델 (노트북 단독 실행을 위해 base KoGPT2 로 시작 — 정석은 Ch 28 SFT 체크포인트). ref-free (beta=0) 이라 reference 불필요. 토크나이저 `PreTrainedTokenizerFast` (Ch 27 이후 고정, AutoTokenizer 함정 회피).

## Hyperparams
- `num_train_epochs=1`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`
- `num_generations=4` (group size), `max_completion_length=24` (짧은 산술 답), `temperature=1.0` (rollout 다양성)
- `learning_rate=1e-5`, `beta=0.0` (ref-free), `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`
- `use_vllm=False` (Colab 호환, HF generate 로 rollout), `fp16=True` (T4 는 bf16 불가)

## rollout 주의 (T4 시간·메모리)
GRPO 는 *매 step 여러 답을 생성(rollout)* 하므로 무겁습니다 (DPO 보다 generation 비용 큼). T4 + 30분 룰: group size 작게 (4) + 짧은 generation + 작은 batch + 적은 step. 시간 빡빡하면 `N_TRAIN`·step 더 축소. ref-free (beta=0) 로 reference 메모리 절약.

## 라이브러리 주의 — `trl` 버전
`trl` 은 버전마다 `GRPOTrainer` / `GRPOConfig` API 변동이 큽니다 (`max_completion_length` 는 있으나 `max_prompt_length` 는 버전에 따라 없음). 본 노트북은 *버전 간 안정적인 핵심 경로* (`num_generations` + `reward_funcs` + `max_completion_length` + `prompt` 컬럼) 만 사용. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.

## 환경
Google Colab **T4 GPU 필수**. 약 22-30분 (데이터 준비 약 1분 + 모델 로드 약 2분 + verifier·advantage 손계산 약 2분 + GRPO 학습 약 15-22분 + 전·후 정확도 비교 약 3분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 29 | Ch 28 SFT 모델 (평가) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) | - |
| 30 | SFT 모델 (policy) + frozen ref | preference 쌍 (chosen/rejected) | chosen 선호 ↑ / rejected ↓ | DPO sigmoid (β=0.1) | `DPOTrainer` |
| **31** | **SFT 모델 (policy) + verifier** | **prompt + 정답 (검증 가능, 수학)** | **group relative advantage** | **GRPO loss (group baseline)** | **`GRPOTrainer`** |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## Phase 4 회고 + Phase 5 예고
Ch 24-31 **Phase 4 (GPT 시대)** 완성 — encoder(BERT)→decoder(GPT), 학습 4단계(pretraining → continual pretraining → SFT → alignment(DPO/GRPO)), 영/한 대칭. 관통 thread: `labels=-100` response-only, alignment 의 두 방식(주관 선호 vs 객관 정답), PPO 의 두 갈래 간소화.

**Phase 5 (Ch 32-34) — Diffusion LM**: Phase 1-4 의 모든 모델이 *autoregressive (한 토큰씩)* 였다면, Phase 5 는 *전체 시퀀스를 병렬 denoise* 하는 새 생성 패러다임. *한 토큰씩* 이라는 대전제를 깨는 커리큘럼의 새 막.
