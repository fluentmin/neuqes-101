# 29_benchmark_eval — 분야별 벤치마크 평가 (생성형 LLM 은 어떻게 평가하는가)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/29_benchmark_eval/29_benchmark_eval.ipynb)

## 한 줄 목표
Ch 28 까지 *학습* (사전학습 → continual pretraining → SFT) 으로 생성형 모델을 만들었습니다. 이 챕터는 *그 모델을 어떻게 측정하는가* 를 다룹니다. 분류 시대 (Ch 1-23) 는 `accuracy` / `F1` 하나로 끝났지만, **생성형 LLM 은 task 마다 평가 방식이 다릅니다**. *왜 LLM 평가가 특별한가* + *벤치마크 task format 3가지* + *MC log-likelihood 평가 직접 구현* + *`lm-evaluation-harness` 소개* 를 다룹니다. **학습 (`Trainer`) 없이 추론·평가만** 합니다.

## 이 챕터의 뼈대 — 벤치마크 task format 3가지

| format | 평가 방식 | 생성? | 벤치마크 예 | 본 챕터 |
|---|---|---|---|---|
| **① Multiple-choice** | 각 선택지의 *log-likelihood* 를 비교해 argmax | **생성 안 함** | MMLU, KMMLU, HellaSwag, KoBEST | §2 직접 구현 |
| **② Generation + 추출** | 생성 후 정답 파싱 / 실행 | 생성함 | GSM8K, HumanEval | §3 시연 |
| **③ LLM-as-judge** | 강한 LLM 이 답변 채점 | 생성함 | LogicKor, MT-Bench | §6 개념 |

> **핵심 직관**: 객관식 (MC) 은 모델에게 답을 *쓰게 하지 않습니다*. 각 선택지를 모델이 얼마나 그럴듯하게 보는가 (log-likelihood) 를 측정해 argmax 합니다. MMLU·KMMLU·HellaSwag 대부분이 이 방식입니다.

## 분류 평가 vs 생성 평가

|  | 분류 (Ch 1-23) | 생성 LLM (Ch 24-) |
|---|---|---|
| metric | accuracy / F1 (정답 라벨 비교) | task 마다 다름 |
| 출력 | 고정 클래스 | 자유 토큰 시퀀스 |
| 평가 | 한 가지 | 여러 방식 혼합 |

## 다루는 핵심 개념
- **MC log-likelihood 평가** (§2, 핵심) — `model.logits` → `log_softmax` → 선택지 토큰 log-prob 합/평균 → argmax. `lm-eval-harness` 없이 *원리 그대로* 구현. 생성하지 않음
- **log-prob 정규화** (sum=`acc` vs mean=`acc_norm`) — 선택지 길이 편향 완화. HellaSwag 처럼 길이가 다를 때 중요
- **Generation + 정답 추출** (§3) — few-shot prompt → greedy 생성 → 정규식 숫자 추출 → 채점. 생성 평가의 형식 변동 어려움
- **zero-shot vs few-shot** (§4) — in-context learning. 가중치 안 바꾸고 *답변 형식 정렬* 로 점수 상승
- **`lm-evaluation-harness`** (§5) — 표준 도구. `lm_eval.simple_evaluate` / `HFLM`. 직접 구현과 *같은 원리, 표준화*
- **분야별 벤치마크 지도** (§6) — 지식 (MMLU/KMMLU) / 상식 (HellaSwag/ARC/KoBEST) / 수학 (GSM8K) / 코드 (HumanEval) / 진실성 (TruthfulQA) / 대화 (LogicKor/MT-Bench)
- **작은 모델의 한계 · scaling · 벤치마크 오염** (§7) — 0.5B 는 random 근처, 규모가 점수를 만듦, contamination 주의

## 사용 모델 · 벤치마크
- 모델: **`Qwen/Qwen2.5-0.5B-Instruct`** (494M, 한·영 지원, T4 OK). 작은 모델이라 점수는 낮지만 *파이프라인을 끝까지 돌려보기* 에 적합 — 그 자체가 §7 의 교훈
- 벤치마크: **KoBEST** (`skt/kobest_v1`) 의 HellaSwag (4지선다 상식추론, 50문항) + BoolQ (2지선다, 50문항) subset + 가벼운 산술 subset (생성 평가용)

## Loss / 학습
**없음.** 이 챕터는 추론·평가만 합니다. `Trainer` / `SFTTrainer` 가 등장하지 않습니다.

## 라이브러리 주의 — `lm-eval` 버전
`lm-eval` (lm-evaluation-harness) 은 버전마다 *task 이름·인자* 가 달라집니다 (`kobest_boolq` vs `kobest` 등). §5 의 `simple_evaluate` 실행은 *선택* 으로 두고, 실패하면 `lm_eval.tasks.TaskManager().all_tasks` 로 설치된 버전의 task 이름을 확인하면 됩니다. **§2-§4 의 직접 구현은 `lm-eval` 없이도 모두 동작** 합니다 (메인 경로).

## 환경
Google Colab **T4 GPU 권장** (학습이 없어 CPU 에서도 동작하지만 느림). 약 12-18분 (`lm-eval` 설치 약 2분 + 모델·데이터 로드 약 2분 + MC 평가 약 4-6분 + 생성 평가 약 2-3분 + zero/few-shot 비교 약 2분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능.

## 변화 추적

| Ch | 모델 | 단계 | 데이터 | 평가 |
|---|---|---|---|---|
| 27 | KoGPT2 (125M) | continual pretraining | 한국어 TinyStories | perplexity / 생성 샘플 |
| 28 | KoGPT2 (125M, SFT) | SFT | KoAlpaca | instruction following (정성) |
| **29** | **Qwen2.5-0.5B-Instruct** | **(평가 — 학습 없음)** | **KoBEST / 산술 subset** | **MC log-likelihood / 생성+추출 / LLM-judge (task format 별)** |
| 30 (다음) | Ch 28 SFT + ref | alignment (DPO) | preference (chosen/rejected) | preference 정렬 |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[30_dpo](../30_dpo/) (예정) — DPO alignment. 본 챕터가 *능력 측정* 이라면 DPO 는 *사람 선호로 정렬*. §2 의 `continuation_logprob` (response log-prob 계산) 이 DPO 의 chosen/rejected log-prob 비교로 *재료 그대로* 이어집니다.
