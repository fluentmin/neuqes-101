"""Build 31_grpo/31_grpo.ipynb — Phase 4, 학습 단계 4 (Alignment / GRPO), Phase 4 마지막.

Ch 30 (DPO) 의 *다음 방식*. DPO 는 *사람/AI 가 비교한 preference 쌍* (chosen / rejected) 으로
정렬했다면, GRPO 는 정반대 접근 — *정답을 자동 검증(verifier) 할 수 있는 task* (수학·코드) 에서
모델이 *여러 답을 생성(rollout)* 하고 *verifier 가 채점* 해 *잘한 답 방향* 으로 RL.
DeepSeek-R1 의 reasoning 학습에 쓴 방법.

GRPO 메커니즘 (group relative):
  1. 한 prompt 에 여러 답 생성 (rollout group, 예: 4-8개)
  2. 각 답을 verifier 로 채점 (수학: 정답 매칭 -> reward 1/0)
  3. group 내 상대 advantage = (각 reward - group 평균) / group 표준편차
     -> group 평균이 baseline 역할 -> critic(value model) 불필요 (PPO 대비 핵심 간소화)
  4. advantage 양수인 답의 확률 올리고, 음수는 내림

PPO vs DPO vs GRPO:
  - PPO : reward model 점수 -> actor+critic+RM+ref (4 모델)
  - DPO : preference 쌍 -> policy + frozen ref (2 모델)
  - GRPO: verifier (정답 자동 검증) -> policy 만 (+ 옵션 ref)

trl 1.5.1 검증 (실측):
  - GRPOTrainer(model, reward_funcs, args=GRPOConfig(...), train_dataset, processing_class)
  - GRPOConfig: num_generations(=group size, default 8), max_completion_length (NOT max_prompt_length),
    beta(default 0.0 = ref-free), temperature, scale_rewards(default 'group'), loss_type(default 'dapo'),
    use_vllm(default False -> HF generate 로 rollout).
  - reward_func 시그니처: (completions, **kwargs) — 데이터셋 컬럼(answer 등)이 kwargs 로 list 전달.
    반환: list[float] (각 completion 의 reward).
  - 데이터셋은 'prompt' 컬럼 필요. 정답은 추가 컬럼(answer) 으로 두면 reward_func 의 kwargs 로 들어옴.
  - log_history(step): reward, reward_std, rewards/<func>/mean, frac_reward_zero_std,
    completions/mean_length, entropy, loss, kl(beta>0 시).
  - group relative advantage 손계산: (r - mean)/(std + 1e-4) -> [1,0,1,0] -> [+1,-1,+1,-1] 실측 일치.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "31_grpo"
OUT_NB = OUT_DIR / "31_grpo.ipynb"
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
md(r"""# Chapter 31. GRPO — 검증 가능한 보상으로 정렬 (Group Relative Policy Optimization, Phase 4 마지막)

**목표**: Phase 4 의 *마지막 챕터*. Ch 30 에서 **DPO** 로 *사람/AI 가 비교한 preference 쌍 (chosen / rejected)* 으로 정렬했다면, 본 챕터는 alignment 의 *두 번째 방식* — **GRPO (Group Relative Policy Optimization)** 입니다. GRPO 는 정반대 접근으로, *정답을 자동 검증(verifier) 할 수 있는 task (수학·코드)* 에서 모델이 *여러 답을 생성(rollout)* 하고 *verifier 가 채점* 해 *잘한 답 방향* 으로 강화학습 합니다. **DeepSeek-R1 이 순수 RL 로 reasoning 능력을 끌어낸 방법** 이 바로 이것입니다. 바뀌는 건 **신호 출처 (preference 쌍 → verifier reward)** + **trainer (`DPOTrainer` → `trl.GRPOTrainer`)** + **데이터 (chosen/rejected → prompt + 정답)** + **rollout (한 prompt 에 여러 답 생성)** 입니다.

**환경**: Google Colab **T4 GPU 필수**. GRPO 는 *매 step 여러 답을 생성(rollout)* 하므로 무겁습니다 — group size 를 작게 (4) + 짧은 generation + 작은 step 으로 시간을 통제합니다.

**예상 소요 시간**: 약 22-30분 (verifiable 데이터 준비 약 1분 + SFT 모델 로드 약 2분 + verifier·group advantage 손계산 시연 약 2분 + `GRPOTrainer` 학습 약 15-22분 + GRPO 전·후 정확도(verifier pass rate) 비교 약 3분)

---

## 학습 흐름

1. 📊 **누적 추적표** (Ch 28/29/30 + **31 강조**) + GPT 학습 4단계 표 (Ch 30 DPO·Ch 31 GRPO = 단계 4 alignment 의 두 방식)
2. 🔄 **변경점 (Diff from Ch 30 DPO)** — *신호 출처 + trainer + 데이터 + rollout* 이 변함
3. 🎯 **PPO vs DPO vs GRPO 대비 표** — 신호·모델·데이터. *왜 GRPO 가 critic 도 reward model 도 없이 되나*
4. 📐 **GRPO 메커니즘** — rollout group → verifier reward → group relative advantage 수식 + 수치 예시
5. 🔬 **verifiable reward 의 의미** — 정답 있는 task 는 사람 채점 없이 무한 RL 신호. DeepSeek-R1 의 reasoning
6. 🔤 **토크나이저 노트** — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)
7. 🚀 **실습**: verifiable 데이터 → SFT 모델 → **verifier + group advantage 손계산** → `GRPOTrainer` 학습 → GRPO 전·후 정확도 비교
8. 📦 **등장 라이브러리** (`trl.GRPOTrainer`·`GRPOConfig`·`reward_funcs` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)
9. 🎓 **Phase 4 회고 + Phase 5 (Diffusion LM) 예고**

---

> 📒 **사전 학습 자료**: Ch 30 (DPO — alignment 의 첫 방식), Ch 28 (KoGPT2 SFT — 본 챕터의 *출발 모델*), Ch 29 (벤치마크 평가 — 특히 부록의 *pass@1·cons@64* 가 verifiable reward 와 직접 연결). 본 챕터는 *alignment 의 두 방식 비교* 를 완성합니다: **DPO (주관적 선호, 사람/AI 비교) vs GRPO (객관적 정답, 자동 검증)**.""")

# ----- 2. 누적 추적표 + GPT 4단계 -----
md(r"""## 📊 누적 추적표

| Ch | 모델 | 데이터 | 학습 신호 | Loss | Trainer |
|---|---|---|---|---|---|
| 28 | KoGPT2 (125M, SFT) | KoAlpaca instruction-response 쌍 | response 토큰 (답변만) | `CrossEntropyLoss` (response-only) - SFT | `SFTTrainer` |
| 29 | Ch 28 SFT 모델 (평가) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) | - |
| 30 | SFT 모델 (policy) + frozen reference | preference 쌍 (chosen / rejected) | chosen 선호 ↑ / rejected 선호 ↓ | DPO sigmoid loss (β=0.1) | `DPOTrainer` |
| **31 ← 여기** | **SFT 모델 (policy) + verifier** | **prompt + 정답 (검증 가능, 수학)** | **group relative advantage** | **GRPO loss (group baseline)** | **`GRPOTrainer`** |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

---

## 🌏 GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 4, Alignment 의 두 번째 방식 / GRPO)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표. 본 챕터는 *단계 4 (Alignment)* 의 *두 번째 방식* — Ch 30 DPO 와 Ch 31 GRPO 가 *alignment 의 두 방식* 입니다.

| 단계 | 정확 용어 | 의미 | 학습 신호 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체 + 일반 코퍼스 | next-token | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습 본체 + 새 데이터 | next-token | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | **SFT** (Supervised Fine-Tuning) | instruction-response 로 *행동 정렬* | response 토큰 | Ch 28 | |
| 4 | **Alignment** (DPO / GRPO / RLHF) | preference·verifier reward 로 *선호·능력 정렬* | preference 쌍 / reward | Ch 30 (DPO), **Ch 31 (GRPO) ← 여기** | ✅ |

### alignment 의 두 방식 — DPO vs GRPO

- **DPO (Ch 30)**: *"같은 질문에 좋은 답 vs 나쁜 답"* preference 쌍으로 학습. 신호는 *사람/AI 가 비교* 한 *주관적 선호*. 열린 질문(글쓰기·대화·취향) 처럼 *정답이 없는* task 에 적합
- **GRPO (Ch 31, 본 챕터)**: *"이 답이 맞나"* 를 *자동 검증(verifier)* 해 reward. 신호는 *객관적 정답* (수학 답이 맞나, 코드가 테스트를 통과하나). *정답을 자동 확인할 수 있는* task 에 적합

> DPO 가 *사람의 선호* 를 따라간다면, GRPO 는 *정답이라는 객관 신호* 를 따라갑니다. 후자의 강점은 ***사람 채점 없이 무한히 RL 신호를 만들 수 있다*** 는 것 — 정답이 있는 task 라면 verifier 가 *공짜 reward model* 역할을 합니다. DeepSeek-R1 이 이걸로 *순수 RL 만으로 reasoning* 능력을 끌어냈습니다 (§6).""")

# ----- 3. 변경점 (Diff from Ch 30) -----
md(r"""## 🔄 변경점 (Diff from Ch 30 DPO)

| 축 | Ch 30 (DPO) | Ch 31 (본 챕터, GRPO) |
|---|---|---|
| 본체 | SFT 모델 (policy) + frozen reference | **SFT 모델 (policy)** ← *reference 는 옵션* (β=0 이면 불필요) |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 BBPE) | **(동일)** ← 고정 |
| **신호 출처** | preference 쌍 (사람/AI 가 비교) | **verifier reward (정답 자동 검증)** ← *변화 1* |
| **Trainer** | `trl.DPOTrainer` | **`trl.GRPOTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **데이터** | `(prompt, chosen, rejected)` 쌍 | **`(prompt, 정답)`** ← *변화 3* (검증 가능한 task) |
| **rollout** | 없음 (주어진 쌍을 비교) | **한 prompt 에 여러 답 생성** ← *변화 4* (group rollout) |
| **Loss** | DPO sigmoid loss | **GRPO loss (group relative advantage)** ← *변화 5* |
| advantage baseline | - (쌍 비교) | **group 평균** (critic 대신) |

> **핵심**: DPO 는 *주어진 (좋은 답, 나쁜 답) 쌍* 을 *비교* 했다면, GRPO 는 *모델이 직접 여러 답을 생성* 하고 *verifier 가 채점* 해 *그룹 안에서 상대 비교* 합니다. 가장 큰 변화는 *신호의 출처* — *사람/AI 가 만든 preference* 에서 *정답 자동 검증* 으로. 그래서 *정답이 있는 task (수학·코드)* 라면 *사람 없이 무한히 RL 신호* 를 만들 수 있습니다.""")

# ----- 4. PPO vs DPO vs GRPO -----
md(r"""## 🎯 PPO vs DPO vs GRPO — alignment 의 세 갈래 (본 챕터의 뼈대)

alignment 의 세 방법을 *신호 출처·필요 모델·데이터* 로 정리합니다. GRPO 의 위치를 이 표 하나로 잡습니다.

| 방법 | 신호 출처 | 필요 모델 | 데이터 | T4 |
|---|---|---|---|---|
| **PPO** (전통 RLHF) | reward model 점수 | actor + critic + reward model + reference (**4개**) | prompt + 학습된 RM | ✗ (메모리 초과) |
| **DPO** (Ch 30) | preference 쌍 (사람/AI 비교) | policy + frozen reference (**2개**) | `(prompt, chosen, rejected)` | ✓ |
| **GRPO** (Ch 31, 본 챕터) | **verifier (정답 자동 검증)** | **policy 만** (+ 옵션 reference) | **`(prompt, 정답)` — 검증 가능** | ✓ |

### 왜 GRPO 는 critic 도 reward model 도 없이 되나 — *group 평균이 baseline*

전통 PPO 는 *advantage* 를 계산하려고 **critic (value model)** 을 따로 둡니다 — "이 상태에서 기대되는 reward 가 얼마인가" 의 *baseline* 을 추정하기 위해서입니다. advantage = (실제 reward) − (critic 이 예측한 baseline).

GRPO 의 통찰: **같은 prompt 에 답을 여러 개 (group) 생성하면, *그 group 의 평균 reward* 가 곧 baseline 이 된다.** critic 을 학습할 필요가 없습니다 — *그룹 동료들의 평균* 이 "이 prompt 에서 보통 어느 정도 받나" 를 알려주니까요.

| 항목 | PPO | GRPO |
|---|---|---|
| baseline (advantage 기준) | **critic (value model)** 이 예측 | **group 평균 reward** (동료 비교) |
| reward 출처 | **reward model** (별도 학습) | **verifier** (정답 자동 검증, 학습 불필요) |
| 필요 모델 | actor + critic + RM + ref (4) | **policy** (+ 옵션 ref) |

> **GRPO 는 PPO 의 또 다른 간소화** 입니다. DPO 가 *reward model + RL 루프* 를 *지도학습 한 단계* 로 줄였다면, GRPO 는 *critic 을 group 평균* 으로, *reward model 을 verifier* 로 대체합니다. 둘 다 *PPO 의 4 모델* 을 덜어내는 길이지만, GRPO 는 *RL 루프(rollout)는 유지* 하면서 *critic 과 RM 만* 없앤 점이 다릅니다 — 그래서 *정답이 있는 task* 에서 강력합니다.""")

# ----- 5. GRPO 메커니즘 -----
md(r"""## 📐 GRPO 메커니즘 — rollout group → verifier reward → group relative advantage

GRPO 의 한 step 은 네 단계입니다:

1. **rollout**: 한 prompt $x$ 에 대해 policy 가 **여러 답 (group)** $\{y_1, \dots, y_G\}$ 을 생성 (예: $G=4$)
2. **verifier reward**: 각 답을 verifier 로 채점 → reward $\{r_1, \dots, r_G\}$ (수학: 정답이면 1, 아니면 0)
3. **group relative advantage**: group 내에서 *평균 대비 상대 위치* 로 advantage 를 계산:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \varepsilon}$$

4. **정책 갱신**: advantage 가 *양수* 인 답 (group 평균보다 잘함) 의 확률은 ↑, *음수* 인 답은 ↓

여기서 **group 평균이 baseline** 역할을 합니다 — "이 prompt 에서 동료들은 평균 얼마나 받았나" 보다 *잘했으면* advantage 양수. 그래서 *critic (value model) 이 불필요* 합니다 (위 §의 PPO 대비 핵심 간소화).

### 수치 예시 — group 4개 답, reward → advantage

한 prompt 에 4개 답을 생성하고 verifier 로 채점한 reward 가 $[1, 0, 1, 0]$ 라고 합시다 (2개 정답, 2개 오답):

| 답 | reward $r_i$ | $r_i - \text{mean}$ | advantage $A_i = (r_i - \text{mean}) / \text{std}$ | 정책 갱신 |
|---|---|---|---|---|
| $y_1$ | 1 | +0.5 | **+1.0** | 확률 ↑ (잘함) |
| $y_2$ | 0 | −0.5 | **−1.0** | 확률 ↓ (못함) |
| $y_3$ | 1 | +0.5 | **+1.0** | 확률 ↑ (잘함) |
| $y_4$ | 0 | −0.5 | **−1.0** | 확률 ↓ (못함) |

(mean = 0.5, std = 0.5) → 정답인 답은 *advantage +1* 로 강화, 오답은 *−1* 로 억제. **reward 자체가 아니라 *그룹 평균 대비 상대값* 으로 학습** 한다는 점이 핵심입니다.

다른 group $[1, 1, 1, 0]$ (3개 정답, 1개 오답) 이라면: mean=0.75, std≈0.43 → 정답 advantage ≈ **+0.58**, 오답 ≈ **−1.73**. *동료 대부분이 맞힌 상황에서 혼자 틀린 답* 이 더 크게 억제됩니다.

### 모든 답이 같으면 — 학습 신호 0

group 전체가 정답 $[1,1,1,1]$ 이거나 전체 오답 $[0,0,0,0]$ 이면 std = 0 → **advantage 가 전부 0** → 그 prompt 에서는 학습 신호가 없습니다. *그룹 안에 잘한 답과 못한 답이 섞여 있어야* 비교가 생깁니다. (그래서 group size 와 temperature 로 *답의 다양성* 을 확보하는 게 중요 — §의 변형.)

> **§3 에서 실제 verifier 와 group advantage 를 손으로 계산** 해 위 표를 재현합니다. `GRPOTrainer` 가 매 step·매 prompt 내부에서 하는 일이 정확히 이것입니다.""")

# ----- 6. verifiable reward 의 의미 -----
md(r"""## 🔬 verifiable reward 의 의미 — 정답 있는 task 는 무한 RL 신호

GRPO 의 진짜 힘은 *알고리즘* 보다 **reward 의 출처** 에 있습니다.

### verifiable reward = *자동 채점 가능한* 신호

- **DPO 의 신호**: *사람/AI 가 비교* 한 preference 쌍. 만들려면 *사람 라벨링* 이나 *강한 judge 모델 (GPT-4)* 이 필요 → *비용·확장 한계*
- **GRPO 의 신호 (verifiable)**: *정답을 자동 검증* (수학 답 일치, 코드 테스트 통과). 한 번 verifier 를 만들면 *사람 없이 무한히* reward 를 생성 → *확장 자유*

| | DPO (preference) | GRPO (verifiable reward) |
|---|---|---|
| reward 만드는 주체 | 사람 / judge 모델 | **verifier (규칙·테스트)** |
| 비용 | 라벨당 비용 (사람·API) | **거의 0** (검증은 자동) |
| 확장성 | 라벨 수에 묶임 | **정답만 있으면 무한 rollout** |
| 적용 범위 | 모든 task (주관 포함) | **검증 가능한 task 만** (수학·코드·형식) |

### DeepSeek-R1 — 순수 RL 로 reasoning

DeepSeek-R1 (그리고 R1-Zero) 은 *수학·코드처럼 정답을 자동 검증* 할 수 있는 문제에 GRPO 를 대규모로 적용해, **사람의 reasoning 데모(SFT) 없이도 모델이 스스로 *긴 사고 과정(chain-of-thought)* 을 만들어내게** 했습니다. 정답이라는 *객관 신호* 만으로, 모델이 *"천천히 단계를 밟아 풀면 정답률이 오른다"* 를 *스스로 발견* 한 것입니다.

> Ch 29 부록에서 본 **pass@1 vs cons@64** (한 번 맞히기 vs 여러 번 생성해 다수결) 가 여기 직접 연결됩니다. verifiable task 는 *여러 답을 생성해 정답을 골라낼 수 있으니*, GRPO 의 *group rollout + verifier* 와 자연스럽게 맞물립니다. *생성을 여러 번 해 정답을 확인* 하는 평가(cons@64)가, *생성을 여러 번 해 정답 방향으로 학습* 하는 GRPO 와 같은 뿌리입니다.

### 한계 — 검증 가능한 task 에만

verifiable reward 의 강점은 *검증 가능한 task* 에서만 성립합니다:

- ✅ **잘 맞음**: 수학 (답 일치), 코드 (테스트 통과), 형식 준수 (정규식·파서), 게임 (승패)
- ✗ **안 맞음**: 글쓰기·대화·요약·취향 — *"무엇이 정답인지" 자동 판정이 어려움*. 이런 *열린 질문* 은 DPO (사람 선호) 나 *LLM-as-judge* (Ch 29 부록) 가 적합

> 실무에서는 **두 신호를 섞습니다** — *검증 가능한 부분은 verifier (GRPO)*, *주관적 품질은 preference/judge (DPO)*. 본 챕터는 *verifiable reward 의 원리* 를 *산술 task* 로 가장 깨끗하게 보입니다.""")

# ----- 7. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 이후 고정)

본 챕터의 토크나이저는 *Ch 27·28·30 과 완전히 동일*. KoGPT2 BBPE (vocab 51,200) 를 그대로 가져옵니다. **KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 (Ch 27 §토크나이저 노트), `PreTrainedTokenizerFast` + special token 명시로 로드합니다.

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
```

### GRPO 데이터의 토큰화 — prompt 만 입력, 답은 *생성*

DPO 데이터는 `(prompt, chosen, rejected)` *세 텍스트* 였습니다. **GRPO 데이터는 `prompt` 하나만** 토큰화해 모델에 넣고, *답(completion)은 모델이 직접 생성(rollout)* 합니다. 정답은 *토큰화 대상이 아니라 verifier 가 채점할 때만* 쓰는 *추가 컬럼* 입니다.

1. `prompt` 를 토큰화 → policy 에 입력
2. policy 가 `num_generations` 개의 *completion 을 생성* (rollout) — 각 completion 도 KoGPT2 토크나이저로 디코딩
3. 디코딩된 텍스트를 *verifier(reward 함수)* 가 채점 → reward

> 같은 KoGPT2 토크나이저이므로 *Ch 28 SFT·Ch 30 DPO 에서 본 instruction 포맷 토큰화* 가 그대로 적용됩니다. 차이는 *답이 데이터에 있느냐 (DPO) vs 모델이 생성하느냐 (GRPO)* 입니다. 토크나이저는 *Phase 4 내내 고정* — Ch 27 이후 한 번도 바뀌지 않았습니다.""")

# ----- 8. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업

`trl` 의 **`GRPOTrainer`** 와 **`GRPOConfig`**, 그리고 **`reward_funcs`** (verifier 함수) 가 이번 챕터에 새로 등장합니다. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.

> ⚠️ `trl` 은 버전마다 `GRPOTrainer` / `GRPOConfig` API 변동이 큽니다 (인자 이름이 버전에 따라 바뀝니다 — 예: `max_completion_length` 는 있지만 `max_prompt_length` 는 버전에 따라 없음). 본 노트북은 설치된 `trl` 버전을 셋업 셀에서 출력하고, *버전 간 안정적인 핵심 경로* (`num_generations` + `reward_funcs` + `max_completion_length` + `prompt` 컬럼) 만 사용합니다.""")

code(r"""%pip install -q -U trl transformers tokenizers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

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

# ----- 9. §1 verifiable 데이터 -----
md(r"""## 1. verifiable 데이터 — `prompt` + 정답 (산술)

GRPO 데이터의 핵심은 **정답을 자동 검증할 수 있어야** 한다는 것입니다. 코드(테스트 실행) 는 무겁고 환경 의존이 크니, 본 챕터는 *가장 깨끗한 verifiable task* 인 **산술(arithmetic)** 로 시작합니다 — 정답이 *정수 하나* 라 *문자열 매칭만으로 채점* 됩니다.

각 샘플은 `(prompt, answer)` 두 컬럼입니다:
- `prompt`: 풀어야 할 문제 (예: `"3 + 5 = ?"`) — 모델에 입력
- `answer`: 정답 (예: `"8"`) — *verifier 가 채점할 때만* 사용 (모델 입력 아님)

> 합성 산술이라 *정답을 우리가 알고* 있으니, *verifier (정답 매칭) 가 완벽* 합니다. 이것이 verifiable reward 의 이상적 형태 — *reward 가 잡음 없이 정확*. (GSM8K 같은 실제 수학 데이터셋도 같은 방식이지만, 답 추출이 더 까다롭습니다 — FAQ 참고.)""")

code(r"""from datasets import Dataset

# Ch 28 SFT / Ch 30 DPO 와 같은 instruction 포맷으로 prompt 를 감쌉니다.
RESPONSE_TEMPLATE = "### 응답:\n"


def build_prompt(question: str) -> str:
    '''Ch 28 SFT 와 동일한 instruction 포맷 (학습·추론 포맷 일치).'''
    return f"### 명령어:\n{question}\n\n{RESPONSE_TEMPLATE}"


def make_arithmetic(n: int, max_operand: int = 9, seed: int = 0):
    '''산술 prompt + 정답. verifier 가 정답을 자동 검증할 수 있는 task.'''
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(1, max_operand)
        b = rng.randint(1, max_operand)
        op = rng.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        rows.append({
            "prompt": build_prompt(f"{a} {op} {b} = ?"),
            "answer": str(ans),          # 정답 (verifier 채점용)
        })
    return Dataset.from_list(rows)


N_TRAIN = 256       # T4 + 30분 룰 - rollout 이 무거우니 작게
grpo_ds = make_arithmetic(N_TRAIN, max_operand=9, seed=SEED)
eval_ds = make_arithmetic(64, max_operand=9, seed=SEED + 1)   # 전·후 비교용

print(f"train: {len(grpo_ds)} samples,  eval: {len(eval_ds)} samples")
print("\n=== sample 0 ===")
print("--- prompt (model input) ---")
print(grpo_ds[0]["prompt"])
print("--- answer (for verifier scoring, not model input) ---")
print(grpo_ds[0]["answer"])""")

# ----- 10. §2 모델 로드 -----
md(r"""## 2. SFT 모델 (policy) 로드

GRPO 는 *SFT 모델에서 출발* 합니다 (Ch 28 의 SFT 체크포인트가 정석). 노트북 단독 실행을 위해 **base KoGPT2 로 시작** 합니다 — 보통은 *이미 지시를 따르는 SFT 모델* 에서 GRPO 를 시작해야 *rollout 이 의미 있는 답* 을 내고 verifier 가 *섞인 reward* (잘한 답 + 못한 답) 를 줄 수 있습니다.

토크나이저는 Ch 27·28·30 과 동일 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피).""")

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

# ----- 11. §3 verifier + group advantage 손계산 -----
md(r"""## 3. 🎯 verifier (reward function) 정의 + group advantage 손계산

여기가 본 챕터의 *개념 핵심*. **verifier 함수** 를 정의하고, 한 prompt 에 *여러 답* 을 채점한 뒤 *group relative advantage* 를 손으로 계산해 §의 표를 재현합니다. `GRPOTrainer` 가 매 step·매 prompt 내부에서 하는 일을 *축소판으로 재현* 하는 셈입니다.

### verifier — 생성 답에서 정답 추출 → 매칭 → reward

`trl` 의 reward 함수 시그니처는 **`reward_func(completions, **kwargs)`** 입니다:
- `completions`: policy 가 생성한 답들의 *리스트* (group)
- `**kwargs`: 데이터셋의 *나머지 컬럼* 이 *리스트로* 전달 (우리의 `answer` 컬럼이 `answer=[...]` 로 들어옴)
- 반환: 각 completion 의 **reward 리스트** (`list[float]`)""")

code(r"""def extract_last_int(text: str):
    '''생성 답에서 마지막 정수를 추출 (없으면 None). 산술 task 의 정답 후보.'''
    matches = re.findall(r"-?\d+", text)
    return matches[-1] if matches else None


def reward_correct(completions, answer, **kwargs):
    '''verifier: 생성 답의 마지막 정수가 정답과 일치하면 1.0, 아니면 0.0.

    trl reward_func 시그니처:
      - completions: 생성된 답 리스트 (group)
      - answer     : 데이터셋의 'answer' 컬럼이 리스트로 전달 (정답)
      - 반환       : 각 completion 의 reward 리스트
    '''
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_last_int(comp)
        rewards.append(1.0 if (pred is not None and pred == str(gold)) else 0.0)
    return rewards


# verifier 시연 - 한 prompt("3 + 5 = ?", 정답 8) 에 4개 답 (일부 맞음/틀림)
demo_completions = [
    "The answer is 8.",         # 맞음 -> 1.0
    "answer: 7",                # 틀림 -> 0.0
    "8",                        # 맞음 -> 1.0
    "I don't know",             # 숫자 없음 -> 0.0
]
demo_answers = ["8", "8", "8", "8"]
demo_rewards = reward_correct(demo_completions, answer=demo_answers)

print("=" * 56)
print("verifier demo - prompt: '3 + 5 = ?', gold answer: 8")
print("=" * 56)
for c, r in zip(demo_completions, demo_rewards):
    print(f"  reward={r:.1f}  <- completion: {c!r}")
print(f"\nrewards (group): {demo_rewards}")""")

md(r"""### group relative advantage 손계산 — reward → advantage

verifier 가 매긴 reward $[1, 0, 1, 0]$ 를 *group 평균 대비 상대값* 으로 바꿉니다 (§의 수식):

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \varepsilon}$$

이게 `GRPOTrainer` 가 *critic 없이* advantage 를 만드는 방법 — *group 평균이 baseline*.""")

code(r"""def group_advantage(rewards, eps=1e-4):
    '''GRPO 의 group relative advantage = (r - mean) / (std + eps). critic 불필요.'''
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


rewards = np.array(demo_rewards)
adv = group_advantage(rewards)

print("=" * 60)
print("group relative advantage - by hand (group mean as baseline, no critic)")
print("=" * 60)
print(f"rewards          : {rewards}")
print(f"group mean       : {rewards.mean():.3f}   <- baseline (replaces critic)")
print(f"group std        : {rewards.std():.3f}")
print(f"advantage        : {np.round(adv, 3)}")
print("-" * 60)
for i, (r, a) in enumerate(zip(rewards, adv)):
    arrow = "prob UP (above avg)" if a > 0 else ("prob DOWN (below avg)" if a < 0 else "no signal")
    print(f"  y{i+1}: reward={r:.0f}  advantage={a:+.2f}  -> {arrow}")

# 다른 group 들도 - 동료 구성에 따라 advantage 가 어떻게 달라지나
print("\nadvantage for various group compositions:")
for rw in [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]]:
    a = group_advantage(rw)
    note = "  (all same -> no learning signal)" if np.allclose(a, 0) else ""
    print(f"  rewards={rw} -> advantage={np.round(a, 2)}{note}")""")

md(r"""**무엇을 보고 있나** — 위 두 출력은 `GRPOTrainer` 가 *매 step, 매 prompt* 내부에서 하는 계산입니다:

- **verifier** 가 *사람 없이 자동* 으로 reward 를 매깁니다 (정답 매칭). preference 라벨이 필요 없습니다
- **group advantage** 가 *critic 없이* 만들어집니다 — *그룹 동료들의 평균* 이 baseline. 평균보다 잘한 답은 +, 못한 답은 −
- **group 전체가 같으면 (전부 정답·전부 오답) advantage = 0** → 학습 신호 없음. *그룹 안에 다양성* (잘한 답 + 못한 답) 이 있어야 GRPO 가 작동합니다

> 이 두 부품 — *verifier (reward)* 와 *group advantage (baseline)* — 이 GRPO 의 전부입니다. 아래 §4 에서 `GRPOTrainer` 에 이 verifier 를 넘기면, 나머지 (rollout · advantage · 정책 갱신) 는 자동입니다.""")

# ----- 12. §4 GRPOTrainer 학습 -----
md(r"""## 4. `GRPOTrainer` 로 GRPO 학습 — *새 trainer, verifier 로 정렬*

`trl.GRPOTrainer` 는 본 챕터에 처음 등장합니다. §3 에서 손으로 한 *verifier reward → group advantage* 를, *매 step* *rollout (여러 답 생성) → 채점 → advantage → 정책 갱신* 으로 자동 수행합니다. 설정은 `GRPOConfig` (`TrainingArguments` 상속) 로 주며, **`num_generations`** 가 group size 입니다.

> **rollout 주의 (T4 시간·메모리)**: GRPO 는 *매 step 여러 답을 생성* 하므로 무겁습니다 (DPO 보다 generation 비용이 큼). T4 + 30분 룰을 지키려면: **group size 작게 (`num_generations=4`) + 짧은 generation (`max_completion_length` 작게) + 작은 batch + 적은 step**. 시간이 빡빡하면 `N_TRAIN` 이나 step 을 더 줄이세요.

> **`trl` 버전 주의**: `GRPOConfig` 는 `max_completion_length` 를 받지만 `max_prompt_length` 는 버전에 따라 없습니다. `beta` 기본값은 *0.0 (reference 없이, ref-free)* — KL 제약을 켜려면 `beta>0` 으로 주고 reference 가 메모리에 추가됩니다. 본 노트북은 *ref-free (beta=0)* 로 메모리를 아낍니다.""")

code(r"""from trl import GRPOTrainer, GRPOConfig


# GRPO 전·후 비교용 - eval 셋에서 정확도(verifier pass rate) 측정
@torch.no_grad()
def eval_accuracy(model, dataset, n=64, n_sample=2, max_new=24):
    '''각 prompt 에 n_sample 개 답을 생성해 verifier pass rate (정확도) 계산.'''
    model.eval()
    correct, total = 0, 0
    for ex in dataset.select(range(min(n, len(dataset)))):
        enc = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=True, temperature=1.0,
            top_p=0.95, num_return_sequences=n_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        for g in gen:
            text = tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_last_int(text)
            correct += int(pred is not None and pred == str(ex["answer"]))
            total += 1
    return correct / max(total, 1)


acc_before = eval_accuracy(policy, eval_ds, n=64, n_sample=2)
print(f"BEFORE GRPO - arithmetic accuracy (verifier pass rate): {acc_before:.3f}")""")

code(r"""GROUP_SIZE = 4   # num_generations - rollout group size (T4 룰: 작게)

grpo_config = GRPOConfig(
    output_dir="./out_kogpt2_grpo",
    num_train_epochs=1,
    per_device_train_batch_size=GROUP_SIZE,   # group rollout 이 한 batch 에 들어가도록
    gradient_accumulation_steps=4,
    num_generations=GROUP_SIZE,               # <- 한 prompt 당 생성 답 개수 (group size)
    max_completion_length=24,                 # 짧은 산술 답 - generation 비용 통제
    temperature=1.0,                          # rollout 다양성 (group 안에 정답·오답 섞이게)
    learning_rate=1e-5,
    beta=0.0,                                 # 0 = ref-free (reference 없이, 메모리 절약)
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                            # T4 는 bf16 불가
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    use_vllm=False,                           # vLLM 없이 HF generate 로 rollout (Colab 호환)
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

# reward_funcs 에 verifier 를 넘기면 rollout -> 채점 -> group advantage -> 정책 갱신 자동.
trainer = GRPOTrainer(
    model=policy,
    reward_funcs=reward_correct,   # <- verifier (callable 또는 list). 데이터의 answer 컬럼이 kwargs 로 전달
    args=grpo_config,
    train_dataset=grpo_ds,
    processing_class=tokenizer,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== GRPO summary ===")
print(f"elapsed     : {elapsed/60:.2f} min")
print(f"global_step : {train_out.global_step}")
print(f"train_loss  : {train_out.training_loss:.4f}")
if torch.cuda.is_available():
    print(f"final peak  : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")""")

# ----- 13. §5 GRPO 전후 비교 -----
md(r"""## 5. 🆚 GRPO 전·후 정확도 비교 — *verifier pass rate 가 올랐는가*

본 챕터의 핵심 데모. *같은 eval 셋* (학습에 안 쓴 산술 문제) 에 대해 *GRPO 전* 과 *후* 의 **정확도 (verifier pass rate)** 를 비교합니다.

- **GRPO 전**: policy 가 산술을 잘 못 풀어 pass rate 낮음
- **GRPO 후**: *정답 방향* 으로 정책이 강화되어 pass rate ↑ (정답을 더 자주 생성)

정확도가 *올랐다면* verifiable reward 로 능력이 정렬된 직접 증거입니다.""")

code(r"""acc_after = eval_accuracy(policy, eval_ds, n=64, n_sample=2)

print(f"AFTER  GRPO - arithmetic accuracy (verifier pass rate): {acc_after:.3f}")
print(f"BEFORE GRPO - arithmetic accuracy                     : {acc_before:.3f}")
print(f"delta                                                 : {acc_after - acc_before:+.3f}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
bars = ax.bar(["before GRPO", "after GRPO"], [acc_before, acc_after],
              color=["tab:gray", "tab:green"], alpha=0.85)
for b, v in zip(bars, [acc_before, acc_after]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
            ha="center", va="bottom")
ax.set_ylabel("accuracy (verifier pass rate)")
ax.set_ylim(0, 1)
ax.set_title("GRPO before vs after - arithmetic accuracy")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.show()""")

md(r"""**해석 가이드 — verifiable reward alignment 의 증거**

- **before (gray)**: policy 가 산술을 잘 못 풀어 pass rate 가 낮습니다 (base KoGPT2 는 산술에 약함)
- **after (green)**: *정답 방향* 으로 정책이 강화되어 pass rate 가 오릅니다 — 모델이 *정답을 더 자주 생성*

> **핵심**: GRPO 는 *preference 라벨 없이*, *verifier 가 자동 채점한 reward* 만으로 능력을 정렬합니다. group 안에서 *정답이 평균보다 잘한 답* 으로 강화되며, 그 효과가 *정확도(pass rate) 상승* 으로 나타납니다.

> ⚠️ KoGPT2 (125M) 는 작은 base 모델이고 (정석은 SFT 모델에서 출발), 학습 step·group size 도 작아 효과가 *미묘* 할 수 있습니다. 관전 포인트는 *극적 향상* 이 아니라 ***정확도가 정답 방향으로 올랐는가*** 입니다. 또한 *group 안에 정답·오답이 섞여야* (std>0) 학습 신호가 생기므로, base 모델이 *가끔이라도 정답을 내야* GRPO 가 작동합니다 — §6 의 reward 곡선에서 확인.""")

# ----- 14. §6 학습 곡선 -----
md(r"""## 6. 학습 곡선 — reward / reward std / completion 길이

`GRPOTrainer` 는 학습 중 *loss* 뿐 아니라 *reward (group 평균)·reward_std·completion 길이* 같은 GRPO 고유 지표를 로깅합니다 (`trainer.state.log_history`). reward 가 오르고, reward_std 가 *0 이 아닌* (= group 안에 다양성이 있는) 구간에서 학습이 일어났는지 확인합니다.""")

code(r"""log = trainer.state.log_history
steps = [r["step"] for r in log if "loss" in r]
losses = [r["loss"] for r in log if "loss" in r]


def series(key):
    return [(r["step"], r[key]) for r in log if key in r]


reward_s = series("reward")          # group 평균 reward
reward_std_s = series("reward_std")  # group reward 표준편차 (다양성 지표)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

if reward_s:
    ax1.plot([s for s, _ in reward_s], [v for _, v in reward_s], "o-",
             color="tab:green", label="reward (group mean)")
if reward_std_s:
    ax1.plot([s for s, _ in reward_std_s], [v for _, v in reward_std_s], "s--",
             color="tab:orange", alpha=0.7, label="reward std (group diversity)")
ax1.set_xlabel("step"); ax1.set_ylabel("reward")
ax1.set_title("GRPO - reward and reward std")
ax1.grid(True, alpha=0.3); ax1.legend()

if steps and losses:
    ax2.plot(steps, losses, "-", color="tab:blue", alpha=0.8, label="GRPO loss")
ax2.set_xlabel("step"); ax2.set_ylabel("GRPO loss")
ax2.set_title("GRPO - loss")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()

if torch.cuda.is_available() and vram_cb.steps:
    print(f"peak VRAM (max over training): {max(vram_cb.peak_MiB):.0f} MiB"
          f"  (policy only, ref-free, num_generations={GROUP_SIZE}, fp16)")""")

# ----- 15. 변형 -----
md(r"""## 🛠️ 변형 — group size / format reward / 코드 verifier / 다른 task

본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. group size (`num_generations`)

```python
# grpo_config.num_generations = 8   # group 키우면 baseline (group 평균) 추정이 안정 -> advantage 정밀
#                                   # 단 rollout 비용 = group size 에 비례 (T4 시간 증가)
# 4 가 T4 출발점. group 안에 정답·오답이 섞이려면 너무 작지 않아야 함 (2 는 비교가 빈약).
```

### 변형 2. format reward 추가 — 여러 verifier 조합

`reward_funcs` 는 *리스트* 로 줄 수 있습니다. 정답 reward + *형식 reward* (예: 정해진 형식으로 답했나) 를 합칠 수 있습니다:

```python
def reward_format(completions, **kwargs):
    '''정답을 '#### 숫자' 형식으로 냈으면 보너스 (형식 준수도 verifiable).'''
    return [0.2 if re.search(r"####\s*-?\d+", c) else 0.0 for c in completions]

# 여러 verifier 를 리스트로 -> reward 가 합산됨 (reward_weights 로 가중치도 가능)
trainer = GRPOTrainer(model=policy, reward_funcs=[reward_correct, reward_format], ...)
```

### 변형 3. 코드 verifier

산술 대신 *코드 생성* task 면, verifier 가 *생성 코드를 실행해 테스트 통과 여부* 를 채점합니다:

```python
def reward_code(completions, test_cases, **kwargs):
    '''생성 코드를 샌드박스에서 실행 -> 테스트 통과하면 1.0 (주의: 샌드박스 필수).'''
    return [1.0 if run_tests_safely(c, t) else 0.0 for c, t in zip(completions, test_cases)]
```

> 코드 실행은 *보안 샌드박스* 가 필수이고 T4 + 30분 룰엔 무거워 본 챕터는 산술로 한정했습니다. 원리는 동일 — *자동 검증 → reward*.

### 변형 4. GSM8K 등 실제 수학 데이터

```python
# from datasets import load_dataset
# gsm = load_dataset("openai/gsm8k", "main", split="train")
# 정답 추출이 더 까다로움 (답이 '#### 42' 형식) -> verifier 의 정답 파싱을 맞춰야 함
```

> 모든 변형의 공통점: *verifier 를 어떻게 정의하나* 가 핵심입니다. *무엇을 reward 로 줄지* = *어떤 능력을 정렬할지*. GRPO 알고리즘 자체는 동일합니다.""")

# ----- 16. 등장 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 30 과 차이 |
|---|---|---|
| `trl.GRPOTrainer` | GRPO 특화 trainer (rollout → verifier 채점 → group advantage → 정책 갱신 자동) | **새로 등장** (Ch 30 은 `DPOTrainer`) |
| `trl.GRPOConfig` | `GRPOTrainer` 설정 (`TrainingArguments` 상속 + `num_generations`·`max_completion_length`·`beta` 등) | **새로 등장** |
| `reward_funcs` (verifier) | 생성 답을 채점하는 callable (또는 list). `(completions, **kwargs)` → `list[float]` | **새로 등장** (DPO 는 preference 데이터, reward 함수 없음) |
| `GRPOConfig(num_generations=4)` | group size — 한 prompt 당 생성 답 개수 (rollout) | **새로 등장** |
| `GRPOConfig(beta=0.0)` | KL 제약 강도. 0 = ref-free (reference 없이, 메모리 절약) | **새로 등장** (DPO 의 beta 와 의미 비슷하나 기본 0) |
| group relative advantage | `(r - mean) / (std + eps)` — group 평균이 baseline (critic 대체) | **새로 등장** (DPO 는 쌍 비교, advantage 없음) |
| `model.generate(num_return_sequences=k)` | rollout — 한 prompt 에 여러 답 생성 | **새로 등장** (DPO 는 생성 불필요) |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 BBPE (AutoTokenizer 함정 회피) | **공유** (Ch 27 이후 고정) |

> `trl` 은 버전마다 `GRPOTrainer` / `GRPOConfig` API 변동이 큽니다 (`max_prompt_length` 같은 인자가 버전에 따라 없음). 본 노트북은 *버전 간 안정적인 핵심 경로* (`num_generations` + `reward_funcs` + `max_completion_length` + `prompt` 컬럼) 만 사용합니다. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.""")

# ----- 17. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. GRPO 는 *PPO 와 달리 critic (value model) 이 없습니다*. 그런데도 advantage 를 계산할 수 있는 이유는 무엇인가요? (*group 평균* 이라는 단어를 써서 설명)
2. *DPO 와 GRPO* 는 둘 다 alignment (단계 4) 입니다. *신호의 출처* 가 어떻게 다른가요? 각각 *어떤 task* 에 적합한가요?
3. 한 prompt 의 group reward 가 `[1, 1, 1, 1]` (전부 정답) 이면 *advantage 가 전부 0* 이 됩니다. 이게 *왜 학습 신호가 없는* 상태인지, 그리고 *어떻게 다양성을 확보* 하는지 설명해 보세요.
4. *reward hacking* 이란 무엇인가요? verifier 가 *정답 매칭* 일 때 일어날 수 있는 reward hacking 의 예를 하나 들어 보세요.""")

# ----- 18. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) GRPO 와 DPO, 언제 무엇을 쓰나요?

*신호의 출처* 가 다르고, 그에 따라 *적합한 task* 가 갈립니다:

| | DPO (Ch 30) | GRPO (Ch 31) |
|---|---|---|
| 신호 | 사람/AI 가 *비교* 한 preference 쌍 | verifier 가 *자동 검증* 한 reward |
| 적합 task | *정답이 없는* 열린 질문 (글쓰기·대화·취향) | *정답을 자동 확인* 가능 (수학·코드·형식) |
| 데이터 비용 | 라벨당 비용 (사람·judge) | 거의 0 (정답만 있으면 무한 rollout) |
| 학습 방식 | 지도학습 (생성 불필요) | RL (rollout - 매 step 생성) |

```python
# 정답이 있는 task -> GRPO
trainer = GRPOTrainer(model, reward_funcs=verifier, ...)   # verifier 가 자동 채점
# 정답이 없는 주관적 품질 -> DPO
trainer = DPOTrainer(model, ref_model=None, ...)           # preference 쌍으로 비교
```

> 실무에서는 *섞어 씁니다* — 검증 가능한 능력(수학·코드)은 GRPO, 주관적 품질(말투·안전성)은 DPO/judge.

### Q2. (실무) verifier 가 없는 task 는 GRPO 를 못 쓰나요?

GRPO 의 전제는 *reward 를 자동으로 매길 수 있어야* 한다는 것입니다. *정답을 자동 판정할 수 없는* task (예: "이 시가 아름다운가") 는 GRPO 의 *깨끗한 신호* 를 얻기 어렵습니다. 대안:

- **LLM-as-judge 를 verifier 로** (Ch 29 부록): 강한 모델이 *점수* 를 매겨 reward 로. 단 judge 의 편향·비용·잡음이 reward 에 섞임 (RLAIF)
- **부분적 verifier**: 형식·길이·금칙어 같은 *검증 가능한 부분만* reward 로 (format reward)
- **DPO 로 전환**: 비교가 더 쉬운 task 면 preference 쌍이 나음

```python
# judge 모델을 reward 로 (예시 - 비용·잡음 주의)
def reward_judge(completions, **kwargs):
    return [judge_model.score(c) for c in completions]   # 0-1 점수
```

> 핵심: *reward 가 신뢰할 만한가* 가 GRPO 성패를 가릅니다. 잡음 많은 reward 는 *잘못된 방향* 으로 정렬합니다.

### Q3. (이론) group size (`num_generations`) 는 결과에 어떤 영향을 주나요?

group 평균이 *baseline (critic 대체)* 이므로, group size 가 *baseline 추정의 안정성* 을 좌우합니다:

- **group 작음** (예: 2): rollout 싸지만, *평균(baseline) 추정이 불안정*. group 안에 *정답·오답이 섞일 확률* 도 낮아져 *advantage 0 (학습 신호 없음)* 인 prompt 가 많아짐
- **group 큼** (예: 8-16): baseline 안정 + 다양성 확보 → advantage 정밀. 단 *rollout 비용 = group size 에 비례* (T4 시간 ↑)

```python
grpo_config.num_generations = 4   # T4 출발점. 시간 여유 있으면 8 로
```

> 직관: group 은 *"이 prompt 에서 동료 몇 명에게 물어볼까"* 입니다. 많을수록 *평균이 믿을 만* 하지만 *물어보는 비용* 이 듭니다.

### Q4. (이론·실무) reward hacking 이란? GRPO 에서 어떻게 막나요?

**reward hacking** = 모델이 *진짜 목표가 아니라 reward 의 허점* 을 찾아 점수만 올리는 현상입니다. verifier 가 *정답 매칭* 일 때의 예:

- verifier 가 *"문자열에 정답 숫자가 들어 있으면 1.0"* 이면, 모델이 *"답은 1 2 3 4 5 6 7 8 9 ..."* 처럼 *모든 숫자를 나열* 해 정답을 포함시킬 수 있음 (풀이 없이 reward 획득)
- *마지막 정수만* 본다면, *엉뚱한 풀이 뒤에 정답만 붙이는* 식으로 우회

막는 법:

```python
# 1) verifier 를 엄격하게 - 정확한 형식 + 정답 둘 다 요구
def reward_strict(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        m = re.search(r"####\s*(-?\d+)\s*$", c.strip())   # 정해진 형식 + 끝에 위치
        out.append(1.0 if (m and m.group(1) == str(a)) else 0.0)
    return out
# 2) beta>0 으로 KL 제약 (reference 에서 멀어지면 페널티 - 붕괴/hacking 완화)
# 3) format reward 와 정답 reward 를 분리해 reward_weights 로 균형
```

> verifier 설계가 GRPO 의 *가장 중요한 부분* 입니다. *허점 없는 reward* = *원하는 능력* 으로 정렬.

### Q5. (이론) GRPO 가 DeepSeek-R1 의 reasoning 과 무슨 관계인가요?

DeepSeek-R1(-Zero) 은 *수학·코드처럼 정답을 자동 검증* 할 수 있는 문제에 GRPO 를 *대규모로* 적용했습니다. 핵심 발견:

- *사람의 reasoning 데모(SFT) 없이도*, **정답이라는 객관 reward 만으로** 모델이 *스스로 긴 사고 과정(chain-of-thought)* 을 만들어냄
- *"단계를 천천히 밟으면 정답률이 오른다"* 를 모델이 *RL 로 스스로 발견* (생성이 길어지고 self-check 가 나타남)

> Ch 29 부록의 **cons@64** (여러 번 생성해 다수결) 와 같은 뿌리입니다 — *여러 답을 생성해 정답을 골라내는* 평가가, *여러 답을 생성해 정답 방향으로 학습* 하는 GRPO 와 맞물립니다. verifiable task 라서 *생성을 무한히* 할 수 있다는 점이 둘의 공통 전제입니다.

### Q6. (실무) 왜 PPO 대신 GRPO 인가요? (특히 T4)

PPO 는 *actor + critic + reward model + reference* **4 모델** 을 동시에 메모리에 올립니다 — T4 (16GB) 에 무리입니다. GRPO 는:

- **critic 제거** → group 평균이 baseline
- **reward model 제거** → verifier 가 자동 채점 (학습 불필요)
- 남는 건 **policy** (+ 옵션 reference). T4 한 장에서 *rollout 만 감당* 하면 됨

```python
# PPO: actor + critic + reward model + reference (4 모델) -> T4 초과
# GRPO: policy 하나 (beta=0 ref-free) + verifier(함수) -> T4 가능
GRPOConfig(num_generations=4, beta=0.0, use_vllm=False)   # ref-free + HF generate
```

> 단 GRPO 도 *rollout (매 step 생성)* 은 PPO 와 공유하므로, *생성 비용* 은 듭니다. T4 에서는 group·step·generation 길이를 작게 잡아 통제합니다.

### Q7. (실무) 작은 모델 (KoGPT2 125M) GRPO 의 한계는?

GRPO 효과는 *출발 모델이 가끔이라도 정답을 내는지* 에 달렸습니다:

- **base 에서 출발 (본 노트북)**: 모델이 산술을 거의 못 풀면 group 이 *전부 오답* → std=0 → *advantage 0 (학습 신호 없음)*. 정석은 *SFT 모델에서 출발* (이미 어느 정도 푸는 상태)
- **작은 모델**: reasoning 능력 자체가 약해 GRPO 로 끌어올릴 *상한* 이 낮음 (R1 은 큰 모델이라 가능)
- **짧은 학습**: 방향을 보기엔 충분하나 극적 변화는 어려움

> 본 챕터의 목표는 *완성된 reasoning 모델* 이 아니라 ***GRPO 가 무엇을 최적화하는가 (verifier reward + group advantage) 를 눈으로 확인*** 하는 것입니다. §3 의 손계산과 §5 의 정확도 변화가 핵심. 실전은 *SFT 모델 + 큰 모델 + 많은 rollout + 엄격한 verifier* 의 영역입니다.""")

# ----- 19. Phase 4 회고 + Phase 5 예고 -----
md(r"""## 🎓 Phase 4 회고 + Phase 5 예고

### Phase 4 완성 — encoder 에서 decoder 로, pretraining 에서 alignment 로

Ch 24-31 의 **Phase 4 (GPT 시대)** 를 마칩니다. Phase 1-3 이 *encoder (BERT)* 로 *이해(분류)* 를 다뤘다면, Phase 4 는 *decoder (GPT)* 로 *생성* 과 *학습 4단계 전체* 를 통과했습니다:

| 단계 | 챕터 | 무엇을 | 학습 신호 |
|---|---|---|---|
| 1 **Pretraining** | Ch 24 (영어), Ch 26 (한국어) | scratch GPT 를 일반 코퍼스로 | next-token |
| 2 **Continual pretraining** | Ch 25 (영어), Ch 27 (한국어) | 사전학습 모델에 새 데이터 | next-token |
| 3 **SFT** | Ch 28 | 지시를 따르게 (행동 정렬) | response 토큰 |
| 4 **Alignment** (DPO) | Ch 30 | 사람 선호로 (주관적 비교) | preference 쌍 |
| 4 **Alignment** (GRPO) | **Ch 31 ← 여기** | 정답으로 (객관적 검증) | verifier reward + group advantage |

**Phase 4 를 관통한 thread**:
- **`labels = -100` 의 response-only**: SFT(답변만 학습) → DPO(답변만 비교) 로 이어짐
- **영/한 대칭**: pretraining·continual pretraining 을 영어(Ch 24·25)와 한국어(Ch 26·27) 로 대칭 진행
- **alignment 의 두 방식**: DPO(주관적 선호) 와 GRPO(객관적 정답) — *신호의 출처* 가 정렬 방법을 나눔
- **PPO 의 두 갈래 간소화**: DPO(reward model + RL 루프 제거), GRPO(critic + reward model 제거) — 둘 다 *T4 에서 alignment 를 손으로* 돌려볼 수 있게 함

### Phase 5 예고 — Diffusion LM (Ch 32-34), 새 패러다임

Phase 1-4 의 모든 모델은 *autoregressive* 였습니다 — *왼쪽에서 오른쪽으로, 한 토큰씩* 생성 (MLM 도 결국 토큰 단위 예측). **Phase 5 는 완전히 다른 생성 패러다임 — Diffusion Language Model** 을 다룹니다:

- **autoregressive (지금까지)**: 토큰을 *순차적* 으로 하나씩. 이전 토큰이 다음 토큰의 조건
- **diffusion (Phase 5)**: *전체 시퀀스를 한꺼번에* 두고, *잡음(masked/noised) 상태에서 병렬로 denoise* 해 점진적으로 완성. 이미지 diffusion (노이즈에서 그림으로) 의 텍스트 버전

> Phase 5 (Ch 32-34) 에서는 *왜 텍스트에 diffusion 을 적용하는가*, *autoregressive 대비 무엇이 다른가* (병렬 생성·양방향 문맥·되돌리기), 그리고 *작은 diffusion LM 을 직접 학습* 해 봅니다. *한 토큰씩* 이라는 Phase 1-4 의 대전제를 깨는, 커리큘럼의 새 막입니다.

**다음 챕터: Chapter 32 — Diffusion Language Model 입문 (autoregressive 가 아닌 병렬 denoise).**""")


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
README = """# 31_grpo — GRPO / 검증 가능한 보상으로 정렬 (Phase 4 마지막, alignment 두 번째 방식)

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
3. **group relative advantage** — $A_i = (r_i - \\text{mean}(r)) / (\\text{std}(r) + \\varepsilon)$. *group 평균이 baseline* → critic 불필요
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
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}")
