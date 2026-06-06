"""Build 30_dpo/30_dpo.ipynb — Phase 4, 학습 단계 4 (Alignment / DPO).

Ch 28 (KoGPT2 SFT) 의 *다음 단계*. SFT 가 *지시를 따르게* (행동 정렬) 만들었다면,
DPO 는 *사람의 선호에 맞춰 정렬* — preference 쌍 (chosen / rejected) 으로
*좋은 답의 확률은 올리고 나쁜 답의 확률은 내립니다*.

전통 RLHF = SFT -> reward model -> PPO (actor+critic+RM+ref 4 모델). T4 메모리에 무리.
DPO = reward model 없이 preference 쌍으로 *직접* 정책 최적화. policy + frozen reference
2 모델만. PPO 대비 간단·안정 -> 본 커리큘럼이 PPO 대신 DPO 채택.

두 thread 의 연장:
  - Thread 1 (`labels = -100` 자리): DPO 도 *response 부분만* log-prob 계산 (prompt 제외).
    Ch 28 SFT 의 마스킹이 alignment 단계까지 이어집니다.
  - Thread 2 ("파인튜닝" 의미): 행동 정렬 (SFT) -> 선호 정렬 (alignment).

데이터: `maywell/ko_Ultrafeedback_binarized` (prompt / chosen / rejected).
Trainer: `trl.DPOTrainer` + `DPOConfig` (새 등장). Loss: DPO sigmoid (beta=0.1).

trl 1.5.1 검증:
  - DPOTrainer(model, ref_model=None, args=DPOConfig(...), train_dataset, processing_class)
    ref_model=None -> trainer 가 frozen reference 를 자동 생성.
  - DPOConfig: beta, max_length (max_prompt_length 없음), loss_type=['sigmoid'].
  - log_history: rewards/{chosen,rejected,margins,accuracies}, logps/{chosen,rejected}.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "30_dpo"
OUT_NB = OUT_DIR / "30_dpo.ipynb"
OUT_README = OUT_DIR / "README.md"

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
md(r"""# Chapter 30. DPO — 사람 선호로 정렬 (Direct Preference Optimization, 학습 단계 4)

**목표**: Phase 4 의 *학습 단계 4 (Alignment, 선호 정렬)* 의 첫 챕터. Ch 28 에서 **SFT** 로 KoGPT2 를 *지시를 따르게* (행동 정렬) 만들었고, Ch 29 에서 *능력을 벤치마크로 측정* 했습니다. 이제 **사람의 선호에 맞춰 정렬 (alignment)** 합니다. **DPO (Direct Preference Optimization)** 는 SFT 모델을 *preference 쌍 (chosen / rejected)* 으로 학습 — *좋은 답의 확률은 올리고, 나쁜 답의 확률은 내립니다*. 바뀌는 건 **데이터 (instruction-response → preference 쌍)** + **trainer (`SFTTrainer` → `trl.DPOTrainer`)** + **loss (next-token CE → DPO sigmoid)** + **frozen reference 모델 추가** 입니다.

**환경**: Google Colab **T4 GPU 필수**. policy + reference *두 모델* 을 동시에 올리므로 batch 를 작게 + gradient accumulation 으로 VRAM 을 관리합니다.

**예상 소요 시간**: 약 22-30분 (preference 데이터 로드·필터 약 2분 + SFT 모델 로드 약 2분 + DPO loss 직관 시각화 약 1분 + DPOTrainer 학습 약 15-22분 + DPO 전·후 reward margin 비교 약 3분)

---

## 학습 흐름

1. 📊 **누적 추적표** (Ch 27/28/29 + **30 강조** + Ch 31 예고) + GPT 학습 4단계 표 (Ch 30 = 단계 4 alignment, DPO)
2. 🔄 **변경점 (Diff from Ch 28 SFT)** — *데이터 + trainer + loss + reference 모델* 이 변함
3. 🎯 **alignment 의 의미** — SFT (지시 따름) → alignment (선호·품질 정렬). RLHF 흐름 + DPO 가 PPO 간소화인 이유
4. 📐 **DPO Loss** — 수식 + 직관 + 수치 예시 표. β 의 역할, frozen reference 가 필요한 이유
5. 🎯 **`labels = -100` thread 연결** — DPO 도 *response 부분만* log-prob 계산
6. 🔤 **토크나이저 노트** — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)
7. 🚀 **실습**: preference 데이터 로드 → SFT 모델·reference 준비 → **DPO loss 직관 시각화 (margin)** → `DPOTrainer` 학습 → DPO 전·후 reward margin 비교
8. 📦 **등장 라이브러리** (`trl.DPOTrainer`·`DPOConfig` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)

---

> 📒 **사전 학습 자료**: Ch 28 (KoGPT2 SFT — 본 챕터의 *출발 모델*), Ch 29 (벤치마크 평가), Ch 27 (KoGPT2 토크나이저 함정). 본 챕터는 *alignment 의 두 thread 연장*: (1) `labels = -100` 의 *response-only* 가 DPO 의 log-prob 계산에서도 이어지고, (2) "파인튜닝" 의 의미가 *행동 정렬 (SFT)* 에서 *선호 정렬 (alignment)* 로 한 발 더 나아갑니다.""")

# ----- 2. 누적 추적표 + GPT 4단계 -----
md(r"""## 📊 누적 추적표

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 27 | KoGPT2 (125M) | 한국어 TinyStories 30K | next-token | `CrossEntropyLoss` - continual pretraining | `Trainer` |
| 28 | KoGPT2 (125M, SFT) | KoAlpaca instruction-response 쌍 | response 토큰 (답변만) | `CrossEntropyLoss` (response-only) - SFT | `SFTTrainer` |
| 29 | Ch 28 SFT 모델 (평가) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) | - |
| **30 ← 여기** | **SFT 모델 (policy) + frozen reference** | **preference 쌍 (chosen / rejected)** | **chosen 선호 ↑ / rejected 선호 ↓** | **DPO sigmoid loss (β=0.1)** | **`DPOTrainer`** |
| 31 (다음) | SFT 모델 + verifier | verifiable-reward prompts (수학·코드) | group relative advantage | `GRPO loss` | `GRPOTrainer` |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

---

## 🌏 GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 4, Alignment / DPO)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표. 본 챕터는 *단계 4 (Alignment)* — *사람의 선호에 맞춰 정렬* 하는 마지막 단계의 첫 방식 (DPO) 입니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체 + 일반 코퍼스 | next-token | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습 본체 + 새 데이터 | next-token | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | **SFT** (Supervised Fine-Tuning) | instruction-response 로 *행동 정렬* | response 토큰 | Ch 28 | |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference·verifier reward 로 *선호 정렬* | preference 쌍 / reward | **Ch 30 (DPO) ← 여기**, Ch 31 (GRPO) | ✅ |

### 단계 3 (SFT) → 단계 4 (Alignment) 의 결정적 변화

- **단계 3 (SFT)**: *"좋은 답변 하나"* 를 따라 학습 (정답 demonstration 모방). 모델은 *지시를 따르는 법* 을 배웁니다 — 하지만 *"여러 답변 중 어느 게 더 나은가"* 라는 *선호* 는 가르치지 못합니다
- **단계 4 (Alignment / DPO)**: *"같은 질문에 좋은 답 vs 나쁜 답"* 쌍으로 학습. 모델은 *사람이 선호하는 방향* 으로 정렬됩니다 — *더 도움되고, 더 안전하고, 더 품질 높은* 답 쪽으로

> SFT 가 *지시를 따르게* 했다면, alignment 는 *따르는 방식을 사람의 선호에 맞춥니다*. DPO 는 그 alignment 를 *reward model 없이, preference 쌍으로 직접* 해내는 방식입니다 — RLHF (PPO) 의 간소화. 그 핵심은 아래 §4 의 DPO loss 한 줄입니다.""")

# ----- 3. 변경점 (Diff from Ch 28) -----
md(r"""## 🔄 변경점 (Diff from Ch 28 SFT)

| 축 | Ch 28 (KoGPT2 SFT) | Ch 30 (본 챕터, DPO) |
|---|---|---|
| 본체 | KoGPT2 `skt/kogpt2-base-v2` (125M) | **SFT 모델 (= KoGPT2 SFT 산출) 을 policy 로** ← 출발점이 SFT 모델 |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 BBPE) | **(동일)** ← 고정 |
| **데이터** | instruction-response 쌍 (`prompt` / `completion`) | **preference 쌍 (`prompt` / `chosen` / `rejected`)** ← *변화 1* |
| **Trainer** | `trl.SFTTrainer` | **`trl.DPOTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **Loss** | next-token `CrossEntropyLoss` (response-only) | **DPO sigmoid loss** ← *변화 3* (log-likelihood ratio) |
| **reference 모델** | 없음 (policy 하나) | **frozen reference 추가** ← *변화 4* (SFT 모델 복사 + freeze) |
| 학습 신호 | 좋은 답변 하나 *모방* | **chosen 선호 ↑, rejected 선호 ↓** (*비교*) |
| lr | 2e-5 | **5e-6 - 1e-5** ← DPO 는 SFT 보다 작은 lr (reference 에서 천천히 벗어남) |

> **핵심**: SFT 는 *하나의 좋은 답* 을 모방했다면, DPO 는 *(좋은 답, 나쁜 답) 쌍* 을 *비교* 합니다. 그러려면 *(1) preference 데이터*, *(2) 비교를 loss 로 바꾸는 DPOTrainer*, *(3) "원본에서 얼마나 벗어났나" 의 기준이 되는 frozen reference* 가 필요합니다. 네 가지가 한꺼번에 바뀌지만, *목적은 하나* — *모델을 사람이 선호하는 방향으로* 정렬.""")

# ----- 4. alignment 의 의미 -----
md(r"""## 🎯 alignment 의 의미 — SFT(지시 따름) 에서 선호·안전성·품질 정렬로

**alignment (정렬)** 은 모델의 행동을 *사람이 원하는 방향* 에 맞추는 단계입니다. SFT 와의 차이를 한 줄로:

- **SFT**: *지시를 따르게* 만든다 (행동 정렬). "질문이 오면 답하라"
- **Alignment**: *따르는 방식을 사람의 선호에 맞춘다* (선호 정렬). "기왕 답할 거면, 더 도움되고·안전하고·품질 높게"

### RLHF 흐름 — 그리고 DPO 가 그 간소화인 이유

전통적인 **RLHF (Reinforcement Learning from Human Feedback)** 는 세 단계입니다:

```
1. SFT          : instruction-response 로 base 모델을 지시 따르게 (Ch 28)
2. Reward Model : (chosen, rejected) preference 로 '점수 매기는 모델' 을 별도 학습
3. PPO          : reward model 을 보상으로 policy 를 강화학습 (RL)
```

PPO 단계는 *네 개의 모델* 을 동시에 메모리에 올립니다:

| 모델 | 역할 |
|---|---|
| **actor** (policy) | 학습 대상 — 답변을 생성 |
| **critic** (value) | 각 상태의 가치 추정 (PPO advantage 용) |
| **reward model** | 생성된 답변에 점수 |
| **reference** | KL 제약 기준 (원본에서 벗어남 측정) |

**T4 (16GB) 에 네 모델은 무리** 입니다. 그래서 본 커리큘럼은 PPO 대신 **DPO** 를 채택합니다.

### DPO = reward model 없이 preference 로 *직접* 정책 최적화

DPO 의 통찰: *"reward model 을 따로 학습한 뒤 RL 로 최적화"* 하는 두 단계를, *"preference 쌍에서 곧바로 policy 를 최적화"* 하는 **한 단계로 합칠 수 있다**. 수학적으로 *최적 정책과 reward 의 관계* 를 닫힌 형태로 풀면, reward model 을 명시적으로 만들 필요 없이 *preference 만으로* policy 를 직접 학습할 수 있습니다 (아래 §4 의 loss).

| 방식 | 필요한 모델 | 단계 | T4 적합성 |
|---|---|---|---|
| **PPO (RLHF)** | actor + critic + reward + reference (**4개**) | SFT → RM → PPO | ✗ (메모리 초과) |
| **DPO (본 챕터)** | policy + frozen reference (**2개**) | SFT → DPO | ✓ (batch 작게 + grad accum) |

> **DPO 는 PPO 대비 간단·안정** 합니다 — reward model 학습도, RL 루프도, critic 도 없습니다. *policy + frozen reference 두 모델* 만으로, *preference 쌍* 에서 *지도학습처럼* (loss.backward()) 정렬합니다. 그래서 T4 한 장에서도 alignment 를 *직접 손으로* 돌려볼 수 있습니다.""")

# ----- 5. DPO Loss -----
md(r"""## 📐 DPO Loss — preference 를 log-likelihood ratio 로

DPO 의 loss 는 *chosen 의 (정책 대비 reference) log-prob 우위* 를 *rejected 보다 크게* 만듭니다:

$$L_{\text{DPO}} = -\log \sigma\!\Big( \beta \cdot \big[\, (\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)) - (\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x)) \,\big] \Big)$$

- $y_w$ = chosen (좋은 답), $y_l$ = rejected (나쁜 답), $x$ = prompt
- $\pi_\theta$ = policy (학습 대상), $\pi_{\text{ref}}$ = frozen reference (SFT 모델 복사·freeze)
- $\sigma$ = sigmoid, $\beta$ = reference 에서 벗어나는 정도 제어 (KL 제약 역할, 기본 0.1)

### 직관 — 두 개의 "정책 대비 reference 우위" 를 비교

각 답변에 대해 **"정책이 reference 보다 이 답변을 얼마나 더 좋아하나"** 를 측정합니다:

$$r_\theta(x, y) = \log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x) \qquad (\text{= implicit reward})$$

DPO 는 이 *implicit reward* 가 *chosen 에서 rejected 보다 크도록* 학습합니다. **margin** $= r_\theta(x, y_w) - r_\theta(x, y_l)$ 가 클수록 loss 가 작아집니다 (sigmoid → 1 → $-\log 1 = 0$).

### 수치 예시 — margin 이 loss 에 어떻게 (β=0.1)

implicit reward 차이 (margin) 가 커질수록 loss 가 어떻게 줄어드는지 (β·margin 을 sigmoid 에 넣고 $-\log$):

| 상황 | margin $= r_\theta(y_w) - r_\theta(y_l)$ | β·margin | $\sigma(\beta \cdot \text{margin})$ | $L = -\log \sigma$ |
|---|---|---|---|---|
| chosen 이 rejected 보다 *훨씬* 선호됨 | +20 | +2.0 | 0.881 | **0.127** (낮음 ✓) |
| chosen 이 rejected 보다 약간 선호됨 | +5 | +0.5 | 0.622 | **0.474** |
| 둘이 비슷 (정렬 안 됨) | 0 | 0.0 | 0.500 | **0.693** |
| rejected 가 *더* 선호됨 (틀림!) | −10 | −1.0 | 0.269 | **1.313** (높음 ✗) |

> *chosen 의 우위가 클수록 loss ↓*, *역전되면 loss 가 폭증* 합니다. 학습은 자연히 *chosen 의 implicit reward 를 올리고 rejected 를 내리는* 방향으로 흐릅니다. **§3 에서 실제 KoGPT2 로 margin 을 손으로 계산** 해 봅니다.

### β 의 역할 — reference 에서 벗어나는 정도

- **β 큼** (예: 0.5): reference 제약이 *느슨* → policy 가 preference 에 강하게 끌려가 *빨리 정렬되지만* reference 에서 멀어져 *붕괴 (degeneration)·reward hacking* 위험
- **β 작음** (예: 0.05): reference 제약이 *강함* → 안전하지만 *정렬이 느림*
- 기본값 **0.1** 이 무난한 출발점

### 왜 frozen reference 가 필요한가

reference 가 없으면 (또는 β=0), 모델은 *chosen 의 확률을 무한정 올리고 rejected 를 0 으로* 밀어붙입니다 — 그 과정에서 *원본 SFT 의 일반 능력이 붕괴* 합니다 (한 패턴만 반복하거나, 문법이 무너지는 등). **frozen reference 는 "원본에서 너무 멀어지지 마라" 는 닻** 입니다:

- $\log \pi_\theta - \log \pi_{\text{ref}}$ 가 *상대적* 비교 → policy 가 reference 근처에 머물도록 KL 제약을 거는 효과
- *정렬하면서도 SFT 의 능력을 보존* — reward hacking·degeneration 방지

> reference 는 *SFT 모델을 복사해 freeze* 한 것입니다 (gradient 안 흐름). 학습 중 *policy 만 움직이고 reference 는 고정* 되어, 둘의 log-prob 차이가 *"얼마나 멀어졌나"* 의 기준이 됩니다.""")

# ----- 6. labels=-100 thread 연결 -----
md(r"""## 🎯 `labels = -100` thread 연결 — DPO 도 *response 부분만*

Ch 28 SFT 의 핵심은 *prompt 를 `-100` 으로 가리고 response 부분만* loss 를 계산하는 것이었습니다. **DPO 도 똑같이 response 부분만 봅니다.**

DPO loss 의 $\log \pi(y \mid x)$ 는 *답변 $y$ 의 토큰들에 대한 log-likelihood 합* 입니다 — **prompt $x$ 부분은 제외**. chosen 도, rejected 도 *각자의 response 토큰에서만* log-prob 을 더합니다 (prompt 는 양쪽 공통이라 비교에서 상쇄되기도 하고, 애초에 학습 대상이 아님).

| 단계 | 챕터 | response-only log-prob 계산 자리 |
|---|---|---|
| MLM | Ch 20·21·22 | 가린 약 15% 토큰 |
| CausalLM | Ch 24·25·26·27 | 거의 전 토큰 (pad 만 제외) |
| **SFT** | Ch 28 | **response 부분만** (prompt = `-100`) |
| **DPO (본 챕터)** | **Ch 30** | **chosen / rejected 각각의 response 부분만** (prompt 제외) |

```
prompt:   ### 명령어:\n건강한 식습관을 알려줘\n\n### 응답:\n   <- 양쪽 공통, log-prob 계산 제외
chosen:   규칙적인 식사와 채소 섭취가 중요합니다.            <- 이 부분의 log π_θ, log π_ref
rejected: ㄴㄴ 몰라 아무거나 먹어                          <- 이 부분의 log π_θ, log π_ref
```

> `labels = -100` thread 가 *alignment 단계까지* 이어집니다. SFT 에서 "답변 부분만 학습" 이었다면, DPO 에서는 "답변 부분의 log-prob 만 비교". **prompt 는 늘 조건 (given), 답변만 학습·비교 대상 (target)** 이라는 원리가 Phase 4 전체를 관통합니다. `DPOTrainer` 가 이 마스킹을 자동으로 처리하므로 우리가 직접 `-100` 을 찍을 필요는 없습니다 (§3 에서 그 효과를 *손으로 재현* 해 확인).""")

# ----- 7. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)

본 챕터의 토크나이저는 *Ch 27·28 과 완전히 동일*. KoGPT2 BBPE (vocab 51,200) 를 그대로 가져옵니다. **KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 (Ch 27 §토크나이저 노트), `PreTrainedTokenizerFast` + special token 명시로 로드합니다.

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
```

### preference 데이터의 토큰화 — prompt / chosen / rejected

DPO 데이터는 *세 개의 텍스트* 로 구성됩니다 (`prompt`, `chosen`, `rejected`). `DPOTrainer` 는 내부적으로:

1. `prompt + chosen` 과 `prompt + rejected` 를 *각각* 토큰화
2. 두 시퀀스의 *prompt 부분은 공통* (같은 토큰), *response 부분만 다름*
3. response 부분의 토큰에서 log-prob 을 계산 (위 §의 response-only)

> 같은 KoGPT2 토크나이저이므로 *Ch 28 SFT 에서 본 instruction 포맷 토큰화* 가 그대로 적용됩니다. chosen / rejected 는 *같은 prompt 에 대한 다른 답변* 이라 *prompt 토큰열은 완전히 동일*, 답변 토큰열만 갈립니다 — DPO 가 비교하는 건 정확히 그 *답변 토큰열의 log-prob* 입니다.

토크나이저는 Ch 27 이후 *Phase 4 내내 고정* — Ch 31 (GRPO) 에서도 같은 KoGPT2 토크나이저를 씁니다.""")

# ----- 8. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업

`trl` 의 **`DPOTrainer`** 와 **`DPOConfig`** 가 이번 챕터에 새로 등장합니다. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.

> ⚠️ `trl` 은 버전마다 `DPOTrainer` / `DPOConfig` API 변동이 큽니다 (`max_prompt_length` 같은 인자가 버전에 따라 사라지기도 합니다). 본 노트북은 설치된 `trl` 버전을 셋업 셀에서 출력하고, *버전 간 안정적인 핵심 경로* (`prompt`/`chosen`/`rejected` 데이터 + `beta` + `max_length`) 만 사용합니다.""")

code(r"""%pip install -q -U trl transformers tokenizers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import trl
print(f"trl          : {trl.__version__}")

# device 자동 감지 - Colab T4 / 로컬 MPS / CPU 모두 지원
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device       : cuda  ({device_name})")
    print(f"VRAM total   : {vram_gib:.2f} GiB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("device       : mps  (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("device       : cpu  (training will be very slow - Colab T4 recommended)")

print(f"torch        : {torch.__version__}")

# 재현성
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음)
USE_FP16 = (device.type == "cuda")
print(f"use fp16     : {USE_FP16}")""")

# ----- 9. §1 preference 데이터 -----
md(r"""## 1. preference 데이터 로드 — `prompt` / `chosen` / `rejected`

**`maywell/ko_Ultrafeedback_binarized`** — 한국어 preference 데이터셋. 각 샘플은 *같은 `prompt` 에 대한 `chosen` (사람이 선호하는 좋은 답) 과 `rejected` (덜 선호되는 답)* 을 가집니다. 이게 DPO 의 표준 데이터 형식 (`prompt` / `chosen` / `rejected` 세 컬럼).

원본 답변은 *에세이 길이* 라 T4 + 30분 룰에는 깁니다. **짧은 샘플만 필터 + 약 1,500 샘플 subset** 으로 학습 시간을 통제합니다.""")

code(r"""from datasets import load_dataset

N_DPO = 1500          # T4 + 30분 룰 - subset
MAX_PROMPT_CHARS = 300
MAX_RESP_CHARS = 400  # 긴 에세이 답변을 잘라 시퀀스 길이 통제 (학습 안정 + 속도)

raw = load_dataset("maywell/ko_Ultrafeedback_binarized", split="train")
print("raw dataset:", raw)
print("\nfields:", raw.column_names)

# 짧고 chosen != rejected 인 샘플만 (길이 통제 + 비교가 의미 있는 쌍)
def keep(ex):
    p, c, r = ex["prompt"], ex["chosen"], ex["rejected"]
    return (
        bool(p.strip()) and bool(c.strip()) and bool(r.strip())
        and c.strip() != r.strip()
        and len(p) <= MAX_PROMPT_CHARS
    )

raw = raw.filter(keep)
raw = raw.shuffle(seed=SEED).select(range(min(N_DPO, len(raw))))
print(f"\nafter filter + subset: {len(raw):,} samples")""")

md(r"""### prompt 포맷 + 답변 길이 통제

Ch 28 SFT 와 *같은 instruction 포맷* (`### 명령어:\n...\n\n### 응답:\n`) 으로 prompt 를 감쌉니다 — SFT 와 추론·학습 포맷을 일치시켜야 정렬이 제대로 됩니다. chosen / rejected 답변은 너무 길면 잘라 시퀀스 길이를 통제합니다.""")

code(r"""RESPONSE_TEMPLATE = "### 응답:\n"   # Ch 28 SFT 와 동일한 답변 경계


def build_prompt(instruction: str) -> str:
    '''Ch 28 SFT 와 동일한 instruction 포맷. 학습·추론 포맷을 일치시켜야 정렬이 됨.'''
    return f"### 명령어:\n{instruction}\n\n{RESPONSE_TEMPLATE}"


def to_preference(ex):
    chosen = ex["chosen"].strip()[:MAX_RESP_CHARS]
    rejected = ex["rejected"].strip()[:MAX_RESP_CHARS]
    return {
        "prompt": build_prompt(ex["prompt"].strip()),
        "chosen": chosen,
        "rejected": rejected,
    }


dpo_ds = raw.map(to_preference, remove_columns=raw.column_names, desc="format")
print("formatted dataset:", dpo_ds)
print("\n=== preference sample 0 ===")
ex0 = dpo_ds[0]
print("--- prompt ---")
print(ex0["prompt"])
print("--- chosen (선호) ---")
print(ex0["chosen"][:200])
print("\n--- rejected (덜 선호) ---")
print(ex0["rejected"][:200])""")

# ----- 10. §2 SFT 모델 + reference -----
md(r"""## 2. SFT 모델 (policy) 로드 + reference 준비

DPO 는 *SFT 모델에서 출발* 합니다. Ch 28 의 SFT 체크포인트가 있으면 그것을 policy 로 쓰는 게 정석입니다 — 여기서는 노트북 단독 실행을 위해 **base KoGPT2 로 시작** 합니다 (보통은 *SFT 를 거친 모델* 에서 DPO 를 시작합니다 — 그래야 이미 지시를 따르는 상태에서 *선호만* 정렬).

토크나이저는 Ch 27·28 과 동일 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피).""")

code(r"""from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

t0 = time.time()
# 주의: KoGPT2 는 AutoTokenizer 가 영어 GPT2 토크나이저로 잘못 fallback (Ch 27).
# PreTrainedTokenizerFast 로 special token 을 직접 지정해 로드.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)

# policy = 학습 대상. 보통은 Ch 28 SFT 체크포인트를 쓰지만, 단독 실행을 위해 base 로 시작.
# (실무: SFT 모델에서 DPO 를 시작해야 '지시 따름' 위에 '선호' 만 정렬됩니다.)
SFT_MODEL = "skt/kogpt2-base-v2"   # Ch 28 SFT 체크포인트 경로가 있으면 여기에
policy = AutoModelForCausalLM.from_pretrained(SFT_MODEL).to(device)
policy.config.pad_token_id = tokenizer.pad_token_id
print(f"load done: {time.time()-t0:.1f}s")

n_params = policy.num_parameters()
print(f"\n=== policy model ===")
print(f"#params      : {n_params/1e6:.2f} M")
print(f"vocab_size   : {tokenizer.vocab_size:,}")
print(f"tokenizer    : {type(tokenizer).__name__}")
print(f"  eos_token  : {tokenizer.eos_token}  id={tokenizer.eos_token_id}")
print(f"  pad_token  : {tokenizer.pad_token}  id={tokenizer.pad_token_id}")""")

md(r"""### reference 모델 — SFT 모델 복사 + freeze

DPO 는 *policy + frozen reference 두 모델* 을 씁니다. reference 는 *학습 시작 시점의 SFT 모델을 복사해 freeze* 한 것 — policy 가 *원본에서 얼마나 멀어졌나* 의 기준 (§4 의 KL 제약 닻).

> `trl` 1.x 의 `DPOTrainer` 는 **`ref_model=None` 으로 주면 reference 를 자동 생성** 합니다 (policy 의 복사본을 freeze, 또는 PEFT 사용 시 adapter 를 끈 base 를 reference 로). 우리는 *명시적으로 어떻게 동작하는지* 보이기 위해 §3 에서 reference 를 직접 복사·freeze 해 margin 을 손으로 계산하고, §4 의 실제 학습에서는 `ref_model=None` 으로 `DPOTrainer` 에 맡깁니다.""")

code(r"""# §3 의 'DPO loss 직관 시각화' 용 - reference 를 직접 복사 + freeze.
# (§4 의 실제 DPOTrainer 학습은 ref_model=None 으로 trl 에 맡깁니다.)
ref_model = copy.deepcopy(policy).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

n_trainable_ref = sum(p.requires_grad for p in ref_model.parameters())
print(f"reference model: frozen  (trainable params = {n_trainable_ref})")
print("policy   : 학습 대상 (gradient 흐름)")
print("reference: 고정 (gradient 안 흐름) - KL 제약의 닻")""")

# ----- 11. §3 DPO loss 직관 시각화 -----
md(r"""## 3. 🎯 DPO loss 직관 시각화 — margin 을 손으로 계산

여기가 본 챕터의 *개념 핵심*. 한 preference 샘플에 대해 **chosen / rejected 각각의 (정책 대비 reference) log-prob 우위** 를 직접 계산하고, 그 *margin* 으로 DPO loss 를 손으로 구해 봅니다. `DPOTrainer` 가 내부에서 하는 일을 *축소판으로 재현* 하는 셈입니다.

### 절차

1. `prompt + chosen`, `prompt + rejected` 를 각각 토큰화 (prompt 길이를 기록)
2. policy·reference 로 **response 부분 토큰만** 의 log-prob 합을 계산 (prompt 제외 = `labels = -100` thread)
3. implicit reward $r(x,y) = \log\pi_\theta(y\mid x) - \log\pi_{\text{ref}}(y\mid x)$ 를 chosen·rejected 각각
4. margin $= r(y_w) - r(y_l)$, loss $= -\log\sigma(\beta\cdot\text{margin})$""")

code(r"""BETA = 0.1   # DPO 기본 beta


@torch.no_grad()
def response_logprob(model, prompt_text, response_text):
    '''response 부분 토큰만의 log-prob 합 (prompt 는 제외 = labels=-100 thread).'''
    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = torch.tensor([p_ids + r_ids], device=model.device)
    logits = model(ids).logits                       # (1, L, V)
    logp = F.log_softmax(logits[:, :-1], dim=-1)     # 다음 토큰 분포 (shift)
    tgt = ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]   # (L-1,)
    # response 부분만: prompt 마지막 토큰이 첫 response 토큰을 예측 -> p_len-1 부터
    resp_logp = tok_logp[len(p_ids) - 1:]
    return resp_logp.sum().item()


sample = dpo_ds[0]
prompt_text = sample["prompt"]
chosen_text = sample["chosen"]
rejected_text = sample["rejected"]

# policy / reference 의 response-only log-prob
pi_w = response_logprob(policy, prompt_text, chosen_text)
pi_l = response_logprob(policy, prompt_text, rejected_text)
ref_w = response_logprob(ref_model, prompt_text, chosen_text)
ref_l = response_logprob(ref_model, prompt_text, rejected_text)

# implicit reward = log pi_theta - log pi_ref
r_w = pi_w - ref_w
r_l = pi_l - ref_l
margin = r_w - r_l
loss = -math.log(1.0 / (1.0 + math.exp(-BETA * margin)))   # -log sigmoid(beta*margin)

print("=" * 60)
print("DPO loss - 한 샘플로 손계산 (response-only log-prob)")
print("=" * 60)
print(f"log pi_theta(chosen)    : {pi_w:10.3f}")
print(f"log pi_ref  (chosen)    : {ref_w:10.3f}")
print(f"log pi_theta(rejected)  : {pi_l:10.3f}")
print(f"log pi_ref  (rejected)  : {ref_l:10.3f}")
print("-" * 60)
print(f"implicit reward (chosen)   r_w = {r_w:8.3f}")
print(f"implicit reward (rejected) r_l = {r_l:8.3f}")
print(f"margin = r_w - r_l             = {margin:8.3f}")
print(f"DPO loss = -log sigmoid(beta*margin) = {loss:8.4f}   (beta={BETA})")""")

md(r"""**무엇을 보고 있나** — 위 출력은 `DPOTrainer` 가 *매 step, 매 샘플* 내부에서 하는 계산입니다:

- *학습 전* (policy = reference 와 동일) 이라면 `r_w ≈ r_l ≈ 0`, margin ≈ 0, loss ≈ $-\log 0.5 = 0.693$ 근처에서 출발합니다
- 학습이 진행되면 policy 가 *chosen 의 log-prob 은 올리고 (r_w ↑), rejected 는 내려 (r_l ↓)* margin 이 커지고 loss 가 줄어듭니다
- reference 는 *고정* 이라 `log pi_ref` 는 변하지 않습니다 — 변하는 건 *policy 의 log-prob* 뿐 (그래서 reference 가 "닻" 역할)

아래에서 margin 을 바꿔 가며 *loss 곡선* 을 그려, *왜 margin 이 클수록 loss 가 작아지는지* 를 한눈에 봅니다.""")

code(r"""# margin -> loss 곡선 (beta 별) + 이번 샘플의 위치 표시
margins = np.linspace(-30, 30, 200)
fig, ax = plt.subplots(figsize=(8, 4.5))
for b in [0.05, 0.1, 0.5]:
    losses = -np.log(1.0 / (1.0 + np.exp(-b * margins)))
    ax.plot(margins, losses, label=f"beta = {b}")

# 이번 샘플의 (margin, loss) 위치
ax.scatter([margin], [loss], color="red", zorder=5,
           label=f"this sample (margin={margin:.1f})")
ax.axvline(0, color="gray", ls="--", alpha=0.5)
ax.axhline(-math.log(0.5), color="gray", ls=":", alpha=0.5)
ax.text(0.5, -math.log(0.5) + 0.05, "loss at margin=0  (-log 0.5)",
        fontsize=8, color="gray")
ax.set_xlabel("margin = r(chosen) - r(rejected)")
ax.set_ylabel("DPO loss = -log sigmoid(beta * margin)")
ax.set_title("DPO loss vs preference margin - larger chosen advantage, lower loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""")

# ----- 12. §4 DPOTrainer 학습 -----
md(r"""## 4. `DPOTrainer` 로 DPO 학습 — *새 trainer, preference 정렬*

`trl.DPOTrainer` 는 본 챕터에 처음 등장합니다. §3 에서 손으로 한 *response-only log-prob → implicit reward → margin → sigmoid loss* 를 *매 step 자동* 으로 수행합니다. 설정은 `DPOConfig` (`TrainingArguments` 상속) 로 주며, **`beta`** 가 reference 제약 강도입니다.

> **VRAM 주의**: DPO 는 *policy + frozen reference 두 모델* 을 메모리에 올립니다 (SFT 의 약 2배). T4 (16GB) 에서는 **batch 를 작게 (2) + gradient accumulation (8)** 으로 effective batch 16 을 만들고 `fp16=True` 로 메모리를 아낍니다. `ref_model=None` 으로 주면 `DPOTrainer` 가 reference 를 자동 생성·freeze 합니다.""")

code(r"""from trl import DPOTrainer, DPOConfig

# DPO 학습 전 reward margin 분포를 기록 (§5 에서 학습 후와 비교)
@torch.no_grad()
def reward_margins(model, ref, dataset, n=64):
    '''dataset 일부에 대해 implicit reward margin (chosen-rejected) 분포를 계산.'''
    model.eval()
    out = []
    for ex in dataset.select(range(min(n, len(dataset)))):
        pw = response_logprob(model, ex["prompt"], ex["chosen"])
        pl = response_logprob(model, ex["prompt"], ex["rejected"])
        rw = response_logprob(ref, ex["prompt"], ex["chosen"])
        rl = response_logprob(ref, ex["prompt"], ex["rejected"])
        out.append((pw - rw) - (pl - rl))
    return np.array(out)


before_margins = reward_margins(policy, ref_model, dpo_ds, n=64)
acc_before = float((before_margins > 0).mean())
print(f"BEFORE DPO - reward margin (n={len(before_margins)})")
print(f"  mean margin     : {before_margins.mean():.3f}")
print(f"  reward accuracy : {acc_before:.3f}  (ratio of margin>0; approx. 0.5 before training)")""")

code(r"""dpo_config = DPOConfig(
    output_dir="./out_kogpt2_dpo",
    num_train_epochs=1,                     # alignment 는 1 epoch 으로 충분 (T4 룰)
    per_device_train_batch_size=2,          # policy + ref 두 모델 -> batch 작게
    gradient_accumulation_steps=8,          # effective batch = 16
    learning_rate=5e-6,                     # DPO 는 SFT 보다 작은 lr (천천히 정렬)
    weight_decay=0.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    beta=BETA,                              # <- reference 제약 강도 (KL), 기본 0.1
    max_length=512,                         # prompt + response 길이 상한
    fp16=USE_FP16,                          # T4 는 bf16 불가
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=2,
    seed=SEED,
)


class VRAMCallback(__import__("transformers").TrainerCallback):
    '''step 별 peak VRAM 기록 (로깅 윈도우 단위 reset). CUDA 에서만 유효.'''

    def __init__(self):
        self.steps, self.peak_MiB = [], []

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            self.steps.append(state.global_step)
            self.peak_MiB.append(peak)
            torch.cuda.reset_peak_memory_stats()


vram_cb = VRAMCallback()

# ref_model=None -> DPOTrainer 가 reference 를 자동 복사·freeze.
trainer = DPOTrainer(
    model=policy,
    ref_model=None,
    args=dpo_config,
    train_dataset=dpo_ds,
    processing_class=tokenizer,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== DPO summary ===")
print(f"elapsed     : {elapsed/60:.2f} min")
print(f"global_step : {train_out.global_step}")
print(f"train_loss  : {train_out.training_loss:.4f}")
if torch.cuda.is_available():
    print(f"final peak  : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")""")

# ----- 13. §5 DPO 전후 비교 -----
md(r"""## 5. 🆚 DPO 전·후 reward margin 비교 — *선호가 정렬됐는가*

본 챕터의 핵심 데모. *같은 preference 샘플들* 에 대해 *DPO 전* 과 *DPO 후* 의 **reward margin (chosen - rejected 의 implicit reward)** 분포를 비교합니다.

- **DPO 전**: policy ≈ reference → margin ≈ 0 근처, reward accuracy (margin>0 비율) ≈ 0.5
- **DPO 후**: policy 가 *chosen 을 더 선호* → margin 분포가 *양수 쪽으로 이동*, reward accuracy ↑

margin 분포가 *오른쪽 (양수) 으로 밀려났다면* 정렬이 일어난 직접 증거입니다.""")

code(r"""# DPO 후 margin 분포 (학습된 policy vs 동일한 frozen reference)
after_margins = reward_margins(policy, ref_model, dpo_ds, n=64)
acc_after = float((after_margins > 0).mean())

print(f"AFTER DPO - reward margin (n={len(after_margins)})")
print(f"  mean margin     : {after_margins.mean():.3f}  (before: {before_margins.mean():.3f})")
print(f"  reward accuracy : {acc_after:.3f}  (before: {acc_before:.3f})")

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(min(before_margins.min(), after_margins.min()),
                   max(before_margins.max(), after_margins.max()), 30)
ax.hist(before_margins, bins=bins, alpha=0.6, color="tab:gray",
        label=f"before DPO (acc={acc_before:.2f})")
ax.hist(after_margins, bins=bins, alpha=0.6, color="tab:green",
        label=f"after DPO (acc={acc_after:.2f})")
ax.axvline(0, color="red", ls="--", alpha=0.7, label="margin = 0")
ax.set_xlabel("reward margin = r(chosen) - r(rejected)")
ax.set_ylabel("count")
ax.set_title("DPO before vs after - margin shifts toward positive (chosen preferred)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""")

md(r"""**해석 가이드 — preference alignment 의 증거**

- **before (gray)**: policy 가 아직 reference 와 같아 margin 이 *0 근처에 모여* 있습니다. chosen 과 rejected 를 *구별하지 못함* (reward accuracy ≈ 0.5)
- **after (green)**: 분포가 *양수 쪽으로 이동* — policy 가 *chosen 의 implicit reward 를 rejected 보다 높게* 매깁니다. reward accuracy 가 0.5 보다 위로 올라갑니다

> **핵심**: DPO 는 *답변을 새로 생성하지 않고도*, *주어진 (chosen, rejected) 쌍의 상대적 선호* 를 policy 에 새깁니다. 그게 *implicit reward margin 의 양수 이동* 으로 나타납니다. KoGPT2 (125M) + 짧은 학습이라 이동 폭은 작을 수 있지만, *방향* 이 양수로 잡혔다면 DPO 의 핵심 (선호 정렬) 은 작동한 것입니다.

> ⚠️ KoGPT2 는 작은 base 모델이고 (정석은 SFT 모델에서 출발), DPO 데이터·시간도 작아 효과가 *미묘* 할 수 있습니다. 관전 포인트는 *생성 품질의 극적 향상* 이 아니라 ***reward margin 이 chosen 쪽으로 이동했는가*** 입니다. 품질은 *SFT 모델에서 출발 + 더 많은 preference + 더 큰 모델* 로 끌어올립니다 (FAQ 참고).""")

# ----- 14. §6 학습 곡선 -----
md(r"""## 6. 학습 곡선 — DPO loss / reward 지표

`DPOTrainer` 는 학습 중 *loss* 뿐 아니라 *reward margin·reward accuracy* 같은 DPO 고유 지표를 로깅합니다 (`trainer.state.log_history`). loss 가 내려가고 reward accuracy 가 올라가는지 확인합니다.""")

code(r"""log = trainer.state.log_history
steps = [r["step"] for r in log if "loss" in r]
losses = [r["loss"] for r in log if "loss" in r]
# trl 의 DPO 로깅 키 (버전에 따라 존재 여부 다를 수 있어 get 으로 안전 접근)
acc_key = "rewards/accuracies"
mgn_key = "rewards/margins"
accs = [(r["step"], r[acc_key]) for r in log if acc_key in r]
mgns = [(r["step"], r[mgn_key]) for r in log if mgn_key in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(steps, losses, "-", color="tab:blue", alpha=0.8, label="DPO loss")
ax1.set_xlabel("step"); ax1.set_ylabel("DPO sigmoid loss")
ax1.set_title("KoGPT2 DPO - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

if accs:
    ax2.plot([s for s, _ in accs], [a for _, a in accs], "o-",
             color="tab:green", label="reward accuracy")
if mgns:
    ax2b = ax2.twinx()
    ax2b.plot([s for s, _ in mgns], [m for _, m in mgns], "s--",
              color="tab:orange", alpha=0.7, label="reward margin")
    ax2b.set_ylabel("reward margin", color="tab:orange")
ax2.axhline(0.5, color="gray", ls=":", alpha=0.6)
ax2.set_xlabel("step"); ax2.set_ylabel("reward accuracy", color="tab:green")
ax2.set_title("DPO reward accuracy / margin  (fraction chosen > rejected)")
ax2.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

if torch.cuda.is_available() and vram_cb.steps:
    print(f"peak VRAM (max over training): {max(vram_cb.peak_MiB):.0f} MiB"
          f"  (policy + reference, bs=2, grad_accum=8, fp16)")""")

# ----- 15. 변형 -----
md(r"""## 🛠️ 변형 — β 조정 / 더 많은 preference / DPO 변종

본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. β 조정 — reference 제약 강도

```python
# dpo_config.beta = 0.5    # 제약 느슨 -> 빨리 정렬되지만 붕괴 위험 (reference 에서 멀어짐)
# dpo_config.beta = 0.05   # 제약 강함 -> 안전하지만 정렬 느림
# 0.1 이 무난한 출발점. reward accuracy 가 안 오르면 beta 를 약간 올려 보세요.
```

### 변형 2. 더 많은 preference / SFT 모델에서 출발

```python
# N_DPO = 5000              # subset 확대 (T4 시간 증가 주의)
# SFT_MODEL = "./out_kogpt2_sft"   # Ch 28 SFT 체크포인트에서 출발 (정석)
# SFT 모델에서 DPO 를 시작해야 '지시 따름' 위에 '선호' 만 정렬됩니다.
```

### 변형 3. DPO 변종 — IPO / KTO / ORPO

`trl` 은 DPO 의 여러 변종을 `loss_type` 으로 지원합니다:

```python
# dpo_config.loss_type = "ipo"   # IPO: sigmoid 대신 squared loss (overfit 완화)
# dpo_config.loss_type = "kto_pair"  # KTO 계열: chosen/rejected 가 쌍이 아니어도 됨
# ORPO: SFT + preference 를 한 번에 (reference 불필요) - trl 의 ORPOTrainer
# 각 변종은 'preference 를 어떻게 loss 로 바꾸나' 의 변주. 핵심 (chosen 선호 ↑) 은 동일.
```

> IPO 는 *DPO 의 overfitting* 을, KTO 는 *쌍이 아닌 개별 좋음/나쁨 라벨* 을, ORPO 는 *reference 없이 SFT 와 동시* 정렬을 노립니다. 모두 *preference 로 정렬* 한다는 점은 같고, *loss 형태·데이터 요구* 만 다릅니다.""")

# ----- 16. 등장 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 28 과 차이 |
|---|---|---|
| `trl.DPOTrainer` | DPO 특화 trainer (response-only log-prob → margin → sigmoid loss 자동) | **새로 등장** (Ch 28 은 `SFTTrainer`) |
| `trl.DPOConfig` | `DPOTrainer` 설정 (`TrainingArguments` 상속 + `beta`·`max_length` 등) | **새로 등장** |
| `DPOConfig(beta=0.1)` | reference 제약 강도 (KL) — DPO 의 핵심 하이퍼파라미터 | **새로 등장** |
| `DPOTrainer(ref_model=None)` | reference 자동 복사·freeze (명시 지정도 가능) | **새로 등장** (frozen reference 개념) |
| `prompt` / `chosen` / `rejected` 데이터 형식 | preference 쌍 표준 형식 | **새로 등장** (Ch 28 은 `prompt`/`completion`) |
| `torch.nn.functional.log_softmax` + `gather` | response 토큰의 log-prob 합 (§3 손계산) | **공유** (개념은 CausalLM loss 와 동일) |
| `copy.deepcopy(policy)` + `requires_grad_(False)` | frozen reference 직접 생성 (§3 시연용) | **새로 등장** |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 BBPE (AutoTokenizer 함정 회피) | **공유** (Ch 27 이후 고정) |

> `trl` 은 버전마다 `DPOTrainer` / `DPOConfig` API 변동이 큽니다 (`max_prompt_length` 같은 인자가 버전에 따라 사라지기도). 본 노트북은 *버전 간 안정적인 핵심 경로* (`prompt`/`chosen`/`rejected` 데이터 + `beta` + `max_length` + `ref_model=None`) 만 사용합니다. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.""")

# ----- 17. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. DPO 에서 *왜 frozen reference 가 필요한가요?* reference 없이 (또는 β=0 으로) chosen 의 확률만 무한정 올리면 어떤 문제가 생기나요?
2. *DPO 와 PPO (RLHF)* 는 둘 다 preference 로 정렬합니다. 그런데 DPO 가 *T4 한 장에서 가능* 한 이유는 무엇인가요? (필요한 모델 개수로 설명)
3. DPO loss 의 **β** 가 *크면* / *작으면* 각각 어떤 trade-off 가 있나요? (정렬 속도 vs reference 에서 벗어남)
4. preference 데이터 `(prompt, chosen, rejected)` 는 어떻게 만드나요? 공개 데이터셋 외에, SFT 모델로 *직접* 만들려면 어떤 절차가 필요할까요?""")

# ----- 18. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) DPO 와 RLHF (PPO) 는 정확히 뭐가 다른가요?

둘 다 *preference 로 모델을 정렬* 하지만, *경로* 가 다릅니다:

| 항목 | PPO (RLHF) | DPO |
|---|---|---|
| reward model | *별도로 학습* (preference → 점수 모델) | **없음** (preference 에서 직접) |
| 학습 방식 | 강화학습 (rollout + advantage + PPO clip) | **지도학습** (loss.backward()) |
| 필요 모델 | actor + critic + reward + reference (4개) | **policy + reference (2개)** |
| 안정성 | RL 특유의 불안정 (튜닝 까다로움) | **상대적으로 안정** |

DPO 의 통찰은 *"reward model 의 최적 정책을 닫힌 형태로 풀면, reward model 을 명시적으로 만들 필요 없이 preference 만으로 policy 를 직접 최적화할 수 있다"* 는 것입니다. 그래서 *RM 학습 + RL 루프* 두 단계가 *지도학습 한 단계* 로 줄어듭니다.

```python
# PPO: SFT -> reward model 학습 -> PPO (rollout + RL)  ... 4 모델
# DPO: SFT -> DPOTrainer(model, ref_model=None, ...).train()  ... 2 모델, 지도학습
```

### Q2. (실무) reference 모델 없이 DPO 를 할 수 있나요?

`DPOTrainer(ref_model=None)` 은 *reference 가 없는 게 아니라*, *policy 의 복사본을 자동으로 reference 로 freeze* 하는 것입니다 (또는 PEFT 사용 시 adapter 를 끈 base 가 reference). 즉 reference 는 *항상* 있습니다.

*진짜로 reference 를 빼면* (`reference_free` 류 옵션 또는 ORPO):
- KL 제약이 사라져 *원본에서 멀어지는 것을 막을 닻이 없어집니다*
- chosen 확률만 무한정 키우다 *모델이 붕괴 (한 패턴 반복, 문법 붕괴)* 할 위험
- ORPO 는 *reference 없이도* 작동하도록 *loss 를 다르게 설계* 한 변종 (SFT 와 preference 를 한 번에)

```python
# 보통은 자동 reference 로 충분:
trainer = DPOTrainer(model=policy, ref_model=None, args=cfg,
                     train_dataset=dpo_ds, processing_class=tokenizer)
# 메모리가 빠듯하면: PEFT(LoRA) 로 policy 를 학습 -> reference 는 adapter 끈 base (추가 메모리 거의 0)
```

### Q3. (이론) β 가 너무 크면 / 너무 작으면 어떻게 되나요?

β 는 *reference 에서 벗어나는 정도* 를 제어합니다 (KL 제약의 세기):

- **β 너무 큼** (예: 1.0): reference 제약이 *거의 없음* → policy 가 preference 에 강하게 끌려가 *빨리 정렬* 되지만, *원본 SFT 의 일반 능력이 붕괴* (degeneration)·*reward hacking* 위험. margin 만 키우려고 *답변 품질을 희생* 할 수 있습니다
- **β 너무 작음** (예: 0.01): reference 제약이 *매우 강함* → policy 가 reference 근처에 묶여 *거의 안 움직임* → 정렬이 느리거나 안 됨

```python
# 0.1 에서 시작. reward accuracy 가 안 오르면 0.2-0.3 으로,
# 답변이 망가지면 (반복/붕괴) 0.05 로 낮춰 보세요.
dpo_config.beta = 0.1
```

> 직관: β 는 *"preference 를 얼마나 공격적으로 따를 것인가 vs 원본을 얼마나 지킬 것인가"* 의 다이얼입니다.

### Q4. (실무) preference 데이터 `(chosen, rejected)` 는 어디서 / 어떻게 만드나요?

세 가지 경로:

1. **공개 데이터셋**: 본 챕터의 `maywell/ko_Ultrafeedback_binarized`, 영어는 `Anthropic/hh-rlhf`, `argilla/ultrafeedback-binarized-preferences` 등
2. **사람 라벨링**: 같은 prompt 에 *여러 답변* 을 생성 → 사람이 *더 나은 쪽을 chosen* 으로 표시 (RLHF 의 원형)
3. **AI 라벨링 (RLAIF)**: 강한 모델 (예: GPT-4) 이 *어느 답이 더 나은지 판정* → chosen/rejected 자동 생성

```python
# SFT 모델로 직접 만들기 (간이):
# 1. 같은 prompt 에 답변 2개 생성 (temperature 다르게)
# 2. 더 강한 모델/규칙/사람이 chosen 선택
# 3. {"prompt":..., "chosen":..., "rejected":...} 로 저장
```

핵심은 *chosen 이 rejected 보다 "사람이 선호하는" 방향* 이면 된다는 점 — 정답일 필요는 없고 *상대적 선호* 만 있으면 DPO 가 작동합니다.

### Q5. (이론) DPO 변종 (IPO, KTO, ORPO) 은 무엇인가요?

모두 *preference 정렬* 의 변주입니다 — *loss 형태·데이터 요구* 만 다릅니다:

| 변종 | 핵심 차이 | 언제 |
|---|---|---|
| **IPO** | sigmoid 대신 *squared loss* | DPO 의 *overfitting* 완화 |
| **KTO** | *쌍이 아닌* 개별 좋음/나쁨 라벨 | preference *쌍을 만들기 어려울* 때 |
| **ORPO** | *reference 없이* SFT + preference 동시 | reference 메모리 절약 + 단계 합치기 |

```python
# trl 에서 loss_type 으로 변종 선택 (버전에 따라 지원 범위 다름)
dpo_config.loss_type = "ipo"        # IPO
# KTO 는 KTOTrainer, ORPO 는 ORPOTrainer 로 별도 클래스인 경우도
```

> 본 챕터는 *원조 DPO (sigmoid loss)* 로 *원리* 에 집중합니다. 변종들은 *같은 목표 (chosen 선호 ↑), 다른 수단*.

### Q6. (실무) 작은 모델 (KoGPT2 125M) DPO 의 한계는?

DPO 의 효과는 *출발 모델의 능력* 에 크게 의존합니다:

- **base 에서 출발 (본 노트북)**: 모델이 아직 *지시를 잘 못 따르므로* preference 정렬 효과가 *미묘*. 정석은 *SFT 모델에서 출발*
- **작은 모델**: chosen/rejected 의 log-prob 차이를 *섬세하게* 다루기 어려워 margin 이동 폭이 작음
- **짧은 학습**: 1 epoch / 1.5K 샘플은 *방향* 을 보기엔 충분하지만 *극적 변화* 는 어려움

> 본 챕터의 목표는 *완성된 정렬 모델* 이 아니라 ***DPO 가 무엇을 최적화하는가 (reward margin) 를 눈으로 확인*** 하는 것입니다. §3 의 손계산과 §5 의 margin 이동이 핵심. 실전 품질은 *SFT 모델 + 큰 모델 + 많은 preference + LoRA* 의 영역.

### Q7. (이론) 다음 단계 GRPO (Ch 31) 는 DPO 와 뭐가 다른가요?

둘 다 alignment (단계 4) 지만, *선호의 출처* 가 다릅니다:

| 단계 | 선호의 출처 | 데이터 |
|---|---|---|
| **DPO (Ch 30)** | *사람이 비교* 한 preference 쌍 | `(prompt, chosen, rejected)` |
| **GRPO (Ch 31)** | *verifier 가 자동 채점* 한 reward | verifiable-reward prompts (수학·코드) |

> DPO 는 *주관적 선호* (어느 답이 더 좋은가 — 사람 판단) 를, GRPO 는 *객관적 정답* (수학 답이 맞나, 코드가 돌아가나 — 자동 검증) 을 신호로 씁니다. GRPO 는 *같은 prompt 에 여러 답을 rollout* 해 *그룹 안에서 상대 비교* (group relative advantage) 합니다 — Ch 31 에서 본격.

```python
# Ch 31 미리보기 (GRPO)
# from trl import GRPOTrainer, GRPOConfig
# reward_funcs = [정답 검증 함수]  # 예: 수학 답 일치 여부 -> 1.0 / 0.0
# 같은 prompt 에 여러 답을 생성 -> 그룹 평균 대비 advantage 로 학습
```""")

# ----- 19. 다음 챕터 예고 -----
md(r"""## 다음 챕터 예고

**Chapter 31. GRPO — verifier reward 로 정렬 (Group Relative Policy Optimization)**

- DPO 는 *사람이 비교한 preference 쌍* 으로 정렬했다면, GRPO 는 *verifier 가 자동 채점한 reward* 로 정렬 — 수학·코드처럼 *정답을 자동 검증* 할 수 있는 영역
- *같은 prompt 에 여러 답을 rollout* → *그룹 안에서 상대 비교* (group relative advantage) → reward 높은 답 쪽으로 정책 강화
- reward model 도, critic 도 없이 *그룹 평균을 baseline* 으로 advantage 를 만드는 *PPO 의 또 다른 간소화*
- alignment 의 *두 방식 비교*: **DPO (주관적 선호, 사람 비교) vs GRPO (객관적 정답, 자동 검증)**

**Phase 4 GPT 시대 4단계 흐름 정리**:

| 챕터 | 단계 | 본체 | 데이터 | 학습 신호 |
|---|---|---|---|---|
| Ch 24·26 | 1 (pretraining) | 작은 GPT scratch | TinyStories (영/한) | next-token |
| Ch 25·27 | 2 (continual pretraining) | gpt2 / KoGPT2 | TinyStories (동일) | next-token |
| Ch 28 | 3 (SFT) | KoGPT2 | KoAlpaca instruction-response | response 토큰 |
| **Ch 30 ← 여기** | **4 (alignment, DPO)** | **SFT 모델 + frozen ref** | **preference 쌍 (chosen/rejected)** | **chosen 선호 ↑, rejected ↓** |
| Ch 31 | 4 (alignment, GRPO) | SFT 모델 + verifier | verifiable-reward prompts | group relative advantage |

> **변하는 축** (Ch 28 → Ch 30): *학습 단계* (SFT → alignment). 본체·토크나이저는 SFT 모델을 잇고, *데이터 (preference 쌍) + trainer (`DPOTrainer`) + loss (DPO sigmoid) + reference 모델* 이 바뀝니다. `labels = -100` 의 *response-only* 원리는 DPO 의 log-prob 계산에서도 이어집니다 — Phase 4 를 관통하는 thread.""")


# ----- 노트북 저장 -----
NOTEBOOK = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT_NB.relative_to(REPO)}  ({len(cells)} cells)")


# ----- README.md 작성 -----
README = """# 30_dpo — DPO / 사람 선호로 정렬 (Phase 4 학습 단계 4, alignment)

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
$$L_{\\text{DPO}} = -\\log \\sigma\\big( \\beta \\cdot [ (\\log \\pi_\\theta(y_w|x) - \\log \\pi_{\\text{ref}}(y_w|x)) - (\\log \\pi_\\theta(y_l|x) - \\log \\pi_{\\text{ref}}(y_l|x)) ] \\big)$$

- $y_w$ = chosen, $y_l$ = rejected, $\\pi_{\\text{ref}}$ = frozen reference (SFT 모델 복사·freeze)
- **implicit reward** $r(x,y) = \\log\\pi_\\theta - \\log\\pi_{\\text{ref}}$ — *정책이 reference 보다 이 답을 얼마나 더 선호하나*
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
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}")
