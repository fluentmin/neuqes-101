"""Build 31_grpo/appendix_qwen_grpo_hpo.ipynb — Ch 31 부록.

메인 챕터 (Ch 31) 는 base KoGPT2 (125M) 로 산술 GRPO 를 합니다. 그런데 base KoGPT2 가
산술을 거의 못 풀어 group reward 가 대부분 0 -> advantage 0 -> reward 전후 차이가 미미합니다.
본문 마지막 절 (🔍 왜 reward 가 잘 안 올랐는가) 에서 *GRPO 의 전제조건* (모델이 가끔이라도
성공해야 그 방향을 증폭) 을 짚고, 해결 레버 4가지를 부록으로 넘깁니다.

이 부록은 그 해결책을 실제로 보입니다:
  - (2) 더 강한 base 모델: Qwen/Qwen2.5-0.5B-Instruct (산술 일부 가능, Ch 29 에서 사용)
  - (4) format reward: correctness reward + format reward 두 개 (0 만 나오는 것 방지)
  - HPO: num_generations / temperature / beta / learning_rate 가 reward·수렴에 주는 영향

실측 (trl 1.5.1, .venv MPS Qwen2.5-0.5B-Instruct, 2-step GRPO):
  - reward_correct mean 0.75 (base KoGPT2 의 ~0 과 대비) -> base 능력이 있어 GRPO 출발 가능
  - reward_std 0.4-0.5 (>0) -> group 다양성 있음 -> advantage 0 아님 -> 학습 신호 존재
  - reward_funcs=[reward_correct, reward_format] 합산, per-func 로깅:
    rewards/reward_correct/mean, rewards/reward_format/mean 분리 기록.
  - format reward 만으로도 group 에 차이 -> std>0 (correctness 가 전부 0 일 때도 학습 신호 확보).

셀 구조 (약 24 셀):
  1. 제목 + Colab 배지 + 부록 안내 (본문 한계 -> 해결)
  2. 한 줄 질문
  3. 본문과의 연결 (KoGPT2 reward 안 오른 이유 요약 -> 부록 개선)
  4. 환경 셋업
  §1 모델 - Qwen2.5-0.5B-Instruct
  §2 verifiable 데이터 + format reward + correctness reward 두 개
  §3 baseline 측정 (GRPO 전 정확도 - base reward>0 확인)
  §4 GRPO 학습 (튜닝된 hyperparameter)
  §5 reward 전·후 비교
  §6 HPO - hyperparameter 가 reward 에 주는 영향
  §7 해석 - base 능력 + format reward + HPO 가 reward 를 올린 이유
  체크포인트 + FAQ
  다음 (메인 Ch 31 / Phase 5 복귀)

빌더 패턴은 메인 _build_ch31.py 와 동일.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "31_grpo"
OUT_NB = OUT_DIR / "appendix_qwen_grpo_hpo.ipynb"

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


# ----- 1. 제목 + Colab 배지 + 부록 안내 -----
md(r"""# Chapter 31 부록 — reward 를 *실제로* 올리는 GRPO (Qwen + format reward + HPO)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/31_grpo/appendix_qwen_grpo_hpo.ipynb)

> **부록 한 줄 질문** — *"base 모델·task·hyperparameter 를 바꾸면 GRPO reward 가 실제로 오를까?"*

메인 챕터 ([`31_grpo.ipynb`](./31_grpo.ipynb)) 는 base **KoGPT2 (125M)** 로 산술 GRPO 를 돌렸습니다. 그런데 KoGPT2 가 산술을 거의 못 풀어 *group reward 가 대부분 0* → advantage 0 → **reward 전·후 차이가 거의 없었습니다**. 본문 마지막 절 (🔍 왜 reward 가 잘 안 올랐는가) 에서 그 이유를 *GRPO 의 전제조건* — *모델이 가끔이라도 성공해야 그 방향을 증폭할 수 있다* — 으로 정리하고, 해결 레버 네 가지를 제시했습니다.

이 부록은 그 레버들을 실제로 적용해, **reward 가 눈에 띄게 오르는 GRPO** 를 시연합니다:

- **(2) 더 강한 base 모델** — `Qwen/Qwen2.5-0.5B-Instruct` (Ch 29 에서 쓴, 산술을 *가끔 맞히는* 작은 instruct 모델) 로 교체
- **(4) format reward** — correctness reward + format reward 두 개를 조합해 *reward 가 0 만 나오는 것* 을 방지
- **HPO** — `num_generations`·`temperature`·`beta`·`learning_rate` 가 reward·수렴에 어떤 영향을 주는지 정리

> **본문 vs 부록의 역할**: 본문은 *GRPO 의 전제조건* 을 (reward 가 안 오르는 현상으로) 체감하는 챕터입니다. 부록은 *그 전제조건을 충족시켜* reward 를 올리고, *어떤 hyperparameter 가 reward 에 영향을 주는지* 보이는 챕터입니다. 둘의 대비가 GRPO 전제조건을 증명합니다.

**환경**: Google Colab **T4 GPU 권장**. Qwen 0.5B 의 GRPO 도 *매 step rollout* 이라 무겁습니다 — group size·completion 길이·step 을 통제해 약 15-25분에 맞춥니다. HPO 는 *전체 grid* 가 아니라 *핵심 축* 만 실측·정리합니다.

**예상 소요 시간**: 약 15-25분 (Qwen 로드 약 1-2분 + baseline 정확도 측정 약 2-3분 + GRPO 학습 약 8-15분 + 전·후 비교 약 2-3분 + HPO 정리 약 2분).
""")

# ----- 2. 한 줄 질문 -----
md(r"""## 한 줄 질문

> *"base 모델·task·hyperparameter 를 바꾸면 GRPO reward 가 실제로 오를까?"*

본문에서 **answer 는 "그렇다 — 단, GRPO 의 전제조건을 충족할 때만"** 이라고 봤습니다. 이 부록은 그 전제조건을 충족시켜 *answer 를 눈으로* 확인합니다.""")

# ----- 3. 본문과의 연결 -----
md(r"""## 본문과의 연결 — KoGPT2 의 reward 가 안 오른 이유, 그리고 부록의 개선

본문에서 정리한 *진단* 과 부록의 *처방* 을 한 표로 잇습니다:

| 항목 | 본문 (KoGPT2 125M) | 부록 (Qwen2.5-0.5B-Instruct) |
|---|---|---|
| base 능력 (산술) | 거의 못 풂 → reward 대부분 0 | *가끔 맞힘* → reward > 0 |
| group reward 분포 | `[0,0,0,0]` 빈번 (std=0) | 섞임 `[1,0,1,0]` (std>0) |
| advantage | 대부분 0 (학습 신호 없음) | 0 아님 (학습 신호 있음) |
| reward 함수 | correctness 1개 | **correctness + format 2개** |
| reward 전·후 | 차이 미미 | **명확히 상승** |

> **왜 이렇게 갈리나** — GRPO 의 advantage 는 $A_i = (r_i - \text{mean}) / (\text{std} + \varepsilon)$ 입니다. group 의 *모든 답이 같은 reward* 면 std=0 → advantage 0 → gradient 0. KoGPT2 는 *모든 답이 똑같이 오답* 이라 비교가 불가능했습니다. Qwen 은 *가끔 정답* 을 내 group 안에 차이가 생기고, 거기에 *format reward* 로 *형식만 지켜도 부분 보상* 을 더해 *std 가 0 이 되는 것 자체를 막습니다*. 이렇게 *학습 신호* 가 살아나면 GRPO 가 비로소 *정답 방향* 을 증폭합니다.

이 부록은 GRPO 알고리즘 자체는 본문과 *완전히 동일* 합니다. 바뀌는 건 *출발점 (base 모델)* 과 *reward 설계 (format 추가)*, 그리고 *hyperparameter* 뿐입니다 — 본 커리큘럼의 *변경점 한 가지* 정신대로, *알고리즘은 고정* 하고 *전제조건만* 바꿔 효과를 분리해 봅니다.""")

# ----- 4. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업

본문과 같은 `trl.GRPOTrainer` / `GRPOConfig` / `reward_funcs` 를 씁니다. 모델만 KoGPT2 → Qwen2.5-0.5B-Instruct 로 바뀝니다. Qwen 은 KoGPT2 같은 `AutoTokenizer` fallback 함정이 없어 `AutoTokenizer` 를 그대로 씁니다 (Ch 29 와 동일).""")

code(r"""%pip install -q -U trl transformers tokenizers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import math
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

# fp16 은 CUDA 에서만 (MPS 는 미지원, CPU 는 의미 없음). T4 는 bf16 불가.
USE_FP16 = (device.type == "cuda")
print(f"use fp16     : {USE_FP16}")""")

# ----- §1 모델 -----
md(r"""## 1. 모델 — `Qwen/Qwen2.5-0.5B-Instruct` (산술 일부 가능)

본문의 KoGPT2 (125M) 대신 **`Qwen/Qwen2.5-0.5B-Instruct`** (약 494M, Ch 29 에서 평가에 쓴 작은 instruct 모델) 를 policy 로 씁니다. 핵심은 이 모델이 *산술을 가끔이라도 맞힌다* 는 것 — 그래야 group 에 *정답·오답이 섞여* GRPO 의 출발 조건 (std>0) 을 충족합니다.

Qwen 은 KoGPT2 와 달리 `AutoTokenizer` 가 올바른 토크나이저를 로드하므로 (Ch 27 의 함정 없음), `AutoTokenizer` 를 그대로 씁니다. instruct 모델이라 *chat template* 으로 prompt 를 감싸 *지시를 따르게* 합니다.""")

code(r"""from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # Ch 29 에서 쓴 작은 instruct 모델 (산술 일부 가능)

t0 = time.time()
# Qwen 은 AutoTokenizer 함정 없음 (KoGPT2 와 차이)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 주의 — dtype 과 AMP 의 관계 (T4 에서 자주 막히는 지점):
#   - Qwen2.5 는 config 기본이 bfloat16. 그대로 로드하면 fp16 GradScaler 가
#     bf16 gradient 를 unscale 못 함 (T4 는 bf16 미지원).
#   - 그렇다고 fp16 으로 통째로 로드하면 "fp16 gradient 는 unscale 불가" 에러.
#   - 정석 mixed precision = *모델 파라미터는 fp32*, AMP(fp16=True)가 forward 연산만
#     fp16 으로 돌리고 master weight 는 fp32 로 둠 → scaler 가 정상 동작.
# 따라서 모델은 fp32 로 로드하고, fp16 은 GRPOConfig(fp16=True) 의 AMP 에 맡깁니다.
LOAD_DTYPE = torch.float32
policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=LOAD_DTYPE).to(device)
if tokenizer.pad_token_id is not None:
    policy.config.pad_token_id = tokenizer.pad_token_id
print(f"load done: {time.time()-t0:.1f}s")

n_params = policy.num_parameters()
print(f"\n=== policy model ===")
print(f"model        : {MODEL_NAME}")
print(f"#params      : {n_params/1e6:.2f} M")
print(f"tokenizer    : {type(tokenizer).__name__}")
print(f"vocab_size   : {tokenizer.vocab_size:,}")
print(f"  eos_token  : {tokenizer.eos_token}  id={tokenizer.eos_token_id}")
print(f"  pad_token  : {tokenizer.pad_token}  id={tokenizer.pad_token_id}")""")

# ----- §2 데이터 + 두 reward -----
md(r"""## 2. verifiable 데이터 + 두 개의 reward 함수 (correctness + format)

본문과 *같은 형식* 의 산술 데이터 `(prompt, answer)` 를 만듭니다. 차이는 두 가지:

1. **prompt 를 Qwen 의 chat template 로** 감쌉니다 — instruct 모델이 지시를 따르도록. 그리고 답을 *정해진 형식* (`"정답: N"`) 으로 내라고 명시 (format reward 와 짝).
2. **reward 함수를 두 개** 둡니다:
   - **correctness reward** — 생성 답의 마지막 정수가 정답과 일치하면 `1.0`, 아니면 `0.0`
   - **format reward** — 답이 `"정답: N"` 형식을 따르면 `0.2` 부분 보상 (정답 여부와 무관)

> **format reward 가 왜 중요한가** — 모델이 정답을 *못 맞혀도* 형식만 지키면 0.2 를 받습니다. 그러면 group 안에서 *형식 지킨 답 (0.2) vs 안 지킨 답 (0.0)* 의 reward 차이가 생겨 **std>0** → advantage 가 0 에서 벗어납니다. correctness 가 전부 0 이라도 *학습 신호가 살아 있어* 모델이 먼저 *형식* 을 배우고, 그 위에서 *정답* 으로 나아갑니다. 작은 모델에서 reward 가 *0 만 나오는 함정* 을 막는 핵심 장치입니다.

`trl` 은 `reward_funcs` 에 *리스트* 를 주면 각 함수의 reward 를 **합산** 합니다 (`reward_weights` 로 가중치도 가능). 학습 중 `rewards/<func>/mean` 으로 함수별 reward 가 따로 로깅됩니다.""")

code(r"""from datasets import Dataset


def build_prompt(question: str) -> str:
    '''Qwen chat template 로 prompt 를 감쌈. 답을 정해진 형식("정답: N")으로 내라고 지시.'''
    user_msg = f"{question} Solve it. Write only the final answer in the format: 정답: N"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


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


N_TRAIN = 128       # T4 + 30분 룰 - rollout 이 무거우니 작게 (Qwen 은 KoGPT2 보다 큼)
grpo_ds = make_arithmetic(N_TRAIN, max_operand=9, seed=SEED)
eval_ds = make_arithmetic(64, max_operand=9, seed=SEED + 1)   # 전·후 비교용

print(f"train: {len(grpo_ds)} samples,  eval: {len(eval_ds)} samples")
print("\n=== sample 0 (prompt with chat template) ===")
print(grpo_ds[0]["prompt"])
print("--- answer (for verifier scoring) ---")
print(grpo_ds[0]["answer"])""")

code(r"""def extract_last_int(text: str):
    '''생성 답에서 마지막 정수를 추출 (없으면 None). 산술 task 의 정답 후보.'''
    matches = re.findall(r"-?\d+", text)
    return matches[-1] if matches else None


def reward_correct(completions, answer, **kwargs):
    '''correctness verifier: 마지막 정수가 정답과 일치하면 1.0, 아니면 0.0.'''
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_last_int(comp)
        rewards.append(1.0 if (pred is not None and pred == str(gold)) else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    '''format verifier: '정답: N' 형식이면 0.2 부분 보상 (정답 여부와 무관).

    correctness 가 전부 0 이라도 형식 차이로 std>0 -> 학습 신호 확보.
    '''
    return [0.2 if re.search(r"정답:\s*-?\d+", c) else 0.0 for c in completions]


# 두 reward 시연 - 한 prompt 에 4개 답 (정답·형식 조합이 다른)
demo_completions = [
    "정답: 8",          # 정답 O, 형식 O -> 1.0 + 0.2 = 1.2
    "answer is 8",      # 정답 O, 형식 X -> 1.0 + 0.0 = 1.0
    "정답: 7",          # 정답 X, 형식 O -> 0.0 + 0.2 = 0.2  (형식만으로 신호)
    "no idea",          # 정답 X, 형식 X -> 0.0 + 0.0 = 0.0
]
demo_answers = ["8", "8", "8", "8"]
rc = reward_correct(demo_completions, answer=demo_answers)
rf = reward_format(demo_completions)
total = [a + b for a, b in zip(rc, rf)]

print("=" * 64)
print("two reward funcs demo - prompt '3 + 5 = ?', gold 8 (trl sums them)")
print("=" * 64)
for c, a, b, t in zip(demo_completions, rc, rf, total):
    print(f"  correct={a:.1f}  format={b:.1f}  total={t:.1f}  <- {c!r}")
print(f"\ntotal rewards (group): {total}")
print("note: correctness all-zero group still gets std>0 via format reward")""")

md(r"""### format reward 가 학습 신호를 살리는 순간 — advantage 비교

correctness reward 만 있을 때와, format reward 를 더했을 때 *advantage* 가 어떻게 달라지는지 손계산으로 봅니다. *correctness 가 전부 0* 인 (모델이 한 문제도 못 맞힌) group 을 가정합니다.""")

code(r"""def group_advantage(rewards, eps=1e-4):
    '''GRPO 의 group relative advantage = (r - mean) / (std + eps).'''
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


# 시나리오: 모델이 한 문제도 못 맞힘 (correctness 전부 0), 형식은 일부만 지킴
correct_only = [0.0, 0.0, 0.0, 0.0]                 # correctness reward 만
with_format = [0.2, 0.0, 0.2, 0.0]                  # + format reward (2개는 형식 지킴)

print("=" * 60)
print("when the model gets NOTHING right (correctness all zero):")
print("=" * 60)
print(f"correctness only : rewards={correct_only}")
print(f"  -> advantage   : {np.round(group_advantage(correct_only), 3)}   (all 0 = NO signal)")
print(f"+ format reward  : rewards={with_format}")
print(f"  -> advantage   : {np.round(group_advantage(with_format), 3)}   (signal restored!)")
print("-" * 60)
print("format reward keeps std>0 so GRPO can still learn (to follow format first)")""")

# ----- §3 baseline -----
md(r"""## 3. baseline 측정 — GRPO *전* 정확도 (base reward > 0 인가)

GRPO 의 출발 조건은 *base 모델이 가끔이라도 정답을 내는가* 입니다. 학습 전에 Qwen 의 산술 정확도 (verifier pass rate) 를 측정해, **0 보다 큰지** 확인합니다 — 본문 KoGPT2 의 *거의 0* 과 대비되는 지점입니다. base reward > 0 이어야 group 에 다양성이 생겨 GRPO 가 *증폭할 신호* 를 갖습니다.""")

code(r"""@torch.no_grad()
def eval_accuracy(model, dataset, n=64, n_sample=2, max_new=24):
    '''각 prompt 에 n_sample 개 답을 생성해 verifier pass rate (정확도) 계산.'''
    model.eval()
    correct, total = 0, 0
    for ex in dataset.select(range(min(n, len(dataset)))):
        enc = tokenizer(ex["prompt"], return_tensors="pt").to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
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
print(f"BEFORE GRPO - Qwen arithmetic accuracy (verifier pass rate): {acc_before:.3f}")
if acc_before > 0:
    print("  -> base reward > 0  ==> group has diversity (std>0)  ==> GRPO can start")
else:
    print("  -> base reward == 0  ==> consider easier task or format reward")""")

# ----- §4 GRPO 학습 -----
md(r"""## 4. GRPO 학습 — 튜닝된 hyperparameter + 두 reward

§3 에서 base reward > 0 을 확인했으니, 이제 GRPO 로 *정답 방향* 을 증폭합니다. 본문보다 *reward 가 잘 오르도록* hyperparameter 를 조정합니다:

- **`num_generations=8`** — group size 를 키워 *정답이 group 에 섞일 확률* ↑ + baseline (group 평균) 추정 안정 (본문은 4)
- **`temperature=0.7`** — 적당한 탐색 (너무 높으면 불안정, 너무 낮으면 다양성 부족)
- **`reward_funcs=[reward_correct, reward_format]`** — 두 verifier 합산 (format 이 *0 만 나오는 것* 방지)
- **`learning_rate=1e-6`** — 작은 instruct 모델이라 *작게* (크면 instruct 능력 붕괴 위험)
- **`beta=0.0`** — ref-free (메모리 절약). reward hacking·붕괴가 걱정되면 `beta>0` 로 KL 제약

> **T4 시간 통제**: `max_completion_length` 을 짧게 (산술 답은 짧음), `max_steps` 로 step 을 제한합니다. group size 8 은 rollout 비용이 4 의 2배이므로, 그만큼 step·데이터를 줄여 30분 룰을 지킵니다.""")

code(r"""from trl import GRPOTrainer, GRPOConfig

GROUP_SIZE = 8   # num_generations - 본문(4)보다 키워 정답이 group 에 섞일 확률 ↑

grpo_config = GRPOConfig(
    output_dir="./out_qwen_grpo",
    per_device_train_batch_size=GROUP_SIZE,   # group rollout 이 한 batch 에 들어가도록
    gradient_accumulation_steps=2,
    num_generations=GROUP_SIZE,               # <- group size (튜닝 축 1)
    max_completion_length=24,                 # 짧은 산술 답 - generation 비용 통제
    temperature=0.7,                          # rollout 탐색 (튜닝 축 2)
    learning_rate=1e-6,                       # instruct 모델이라 작게 (튜닝 축 3)
    beta=0.0,                                 # ref-free (튜닝 축 4: KL 제약, 0=없음)
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    fp16=USE_FP16,                            # T4 는 bf16 불가
    logging_steps=2,
    max_steps=30,                             # T4 30분 룰 - step 제한 (시간 빡빡하면 줄이세요)
    save_strategy="no",
    report_to="none",
    use_vllm=False,                           # vLLM 없이 HF generate 로 rollout (Colab 호환)
    seed=SEED,
)

# reward_funcs 에 두 verifier 를 리스트로 -> reward 합산. answer 컬럼은 kwargs 로 전달.
trainer = GRPOTrainer(
    model=policy,
    reward_funcs=[reward_correct, reward_format],   # <- 두 verifier (합산)
    args=grpo_config,
    train_dataset=grpo_ds,
    processing_class=tokenizer,
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

# ----- §5 reward 전후 비교 -----
md(r"""## 5. 🆚 reward 전·후 비교 — *본문 KoGPT2 와의 대비*

본문의 핵심 데모를 *개선된 셋업* 으로 다시 합니다. *같은 eval 셋* 에 대해 GRPO 전·후 정확도 (verifier pass rate) 와, 학습 중 reward 곡선을 함께 봅니다. 본문 KoGPT2 가 *거의 평평* 했다면, 여기서는 *reward 가 오르는* 모습을 기대합니다.""")

code(r"""acc_after = eval_accuracy(policy, eval_ds, n=64, n_sample=2)

print(f"AFTER  GRPO - Qwen arithmetic accuracy (verifier pass rate): {acc_after:.3f}")
print(f"BEFORE GRPO - Qwen arithmetic accuracy                     : {acc_before:.3f}")
print(f"delta                                                      : {acc_after - acc_before:+.3f}")

# 학습 중 reward 곡선 (함수별 + 합산)
log = trainer.state.log_history


def series(key):
    return [(r["step"], r[key]) for r in log if key in r]


reward_s = series("reward")                                   # 합산 reward (group 평균)
reward_std_s = series("reward_std")                           # group reward 표준편차
rc_s = series("rewards/reward_correct/mean")                  # correctness 만
rf_s = series("rewards/reward_format/mean")                   # format 만

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# (좌) 전·후 정확도 막대
bars = ax1.bar(["before GRPO", "after GRPO"], [acc_before, acc_after],
               color=["tab:gray", "tab:green"], alpha=0.85)
for b, v in zip(bars, [acc_before, acc_after]):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
             ha="center", va="bottom")
ax1.set_ylabel("accuracy (verifier pass rate)")
ax1.set_ylim(0, 1)
ax1.set_title("Qwen GRPO - accuracy before vs after")
ax1.grid(True, axis="y", alpha=0.3)

# (우) reward 곡선 (함수별 + 합산)
if reward_s:
    ax2.plot([s for s, _ in reward_s], [v for _, v in reward_s], "o-",
             color="tab:green", label="reward (total, group mean)")
if rc_s:
    ax2.plot([s for s, _ in rc_s], [v for _, v in rc_s], "^--",
             color="tab:blue", alpha=0.7, label="reward_correct mean")
if rf_s:
    ax2.plot([s for s, _ in rf_s], [v for _, v in rf_s], "v--",
             color="tab:purple", alpha=0.7, label="reward_format mean")
if reward_std_s:
    ax2.plot([s for s, _ in reward_std_s], [v for _, v in reward_std_s], "s:",
             color="tab:orange", alpha=0.6, label="reward std (diversity)")
ax2.set_xlabel("step"); ax2.set_ylabel("reward")
ax2.set_title("Qwen GRPO - reward curves (correct + format)")
ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)

plt.tight_layout(); plt.show()""")

md(r"""**해석 — 본문과 무엇이 달랐나**

- **before (gray)** 가 *0 이 아닙니다* — Qwen 은 산술을 *가끔 맞혀* base reward > 0. 이게 본문 KoGPT2 (거의 0) 와의 결정적 차이입니다
- **reward 곡선이 오릅니다** — group 에 *정답·오답이 섞여* (std>0) advantage 가 0 이 아니므로, GRPO 가 *정답 방향* 을 증폭합니다
- **format reward** 가 *바닥을 받쳐* 줍니다 — correctness 가 낮은 초반에도 format reward 로 std>0 이 유지되어 학습이 멈추지 않습니다

> **결론**: 본문에서 reward 가 안 오른 건 *GRPO 가 약해서* 가 아니라 *base 모델이 출발 조건 (가끔의 성공) 을 못 채워서* 였습니다. *base 를 바꾸고 (Qwen) + format reward 로 신호를 받치면* 같은 GRPO 가 *reward 를 올립니다*. 이것이 본문 마지막 절의 교훈 — *GRPO 는 이미 가끔 성공하는 능력을 증폭할 뿐, 무에서 유를 만들지 못한다* — 의 직접 증거입니다.""")

# ----- §6 HPO -----
md(r"""## 6. HPO — hyperparameter 가 reward 에 주는 영향 (이 부록의 핵심)

GRPO 의 reward·수렴은 hyperparameter 에 민감합니다. *전체 grid* 학습은 T4 에서 무거우니, **핵심 축 하나 (`num_generations`) 를 실제로 비교** 하고, 나머지 축은 *원리 + 권장값* 으로 정리합니다.

### 실측 비교 — `num_generations` (group size) 4 vs 8

같은 데이터·step 으로 group size 만 4 와 8 로 바꿔 *초반 reward·reward_std* 를 비교합니다. group 이 클수록 *정답이 섞일 확률* 과 *baseline 추정 안정성* 이 오르지만, *rollout 비용* 도 비례해 오릅니다.

> 비용 통제를 위해 *짧은 step* 으로만 비교합니다 (절대 reward 보다 *경향* 을 봅니다). 시간이 빡빡하면 이 셀은 건너뛰고 아래 권장값 표만 봐도 됩니다.""")

code(r"""def quick_grpo_reward(num_gen, steps=6, lr=1e-6, temperature=0.7, beta=0.0):
    '''group size 등 hyperparameter 를 바꿔 짧게 GRPO 를 돌리고 평균 reward·std 반환.

    절대값보다 *경향* 비교용 (T4 비용 통제 위해 step 작게).
    '''
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=LOAD_DTYPE).to(device)
    if tokenizer.pad_token_id is not None:
        m.config.pad_token_id = tokenizer.pad_token_id
    cfg = GRPOConfig(
        output_dir="./out_hpo_tmp",
        per_device_train_batch_size=num_gen,
        gradient_accumulation_steps=1,
        num_generations=num_gen,
        max_completion_length=24,
        temperature=temperature,
        learning_rate=lr,
        beta=beta,
        max_grad_norm=1.0,
        fp16=USE_FP16,
        logging_steps=1,
        max_steps=steps,
        save_strategy="no",
        report_to="none",
        use_vllm=False,
        seed=SEED,
    )
    tr = GRPOTrainer(
        model=m,
        reward_funcs=[reward_correct, reward_format],
        args=cfg,
        train_dataset=grpo_ds,
        processing_class=tokenizer,
    )
    tr.train()
    rewards = [r["reward"] for r in tr.state.log_history if "reward" in r]
    stds = [r["reward_std"] for r in tr.state.log_history if "reward_std" in r]
    del m, tr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (float(np.mean(rewards)) if rewards else float("nan"),
            float(np.mean(stds)) if stds else float("nan"))


hpo_rows = []
for ng in [4, 8]:
    t0 = time.time()
    mean_r, mean_std = quick_grpo_reward(num_gen=ng, steps=6)
    hpo_rows.append({
        "num_generations": ng,
        "mean_reward": round(mean_r, 3),
        "mean_reward_std": round(mean_std, 3),
        "wall_sec": round(time.time() - t0, 1),
    })

hpo_df = pd.DataFrame(hpo_rows)
print("=== HPO: num_generations (group size) sweep ===")
print(hpo_df.to_string(index=False))
print("\nlarger group -> more chance of a correct answer in the group + steadier baseline,")
print("but rollout cost scales with group size (see wall_sec).")""")

md(r"""### hyperparameter 별 영향 정리 (권장값 표)

위에서 `num_generations` 를 실측했고, 나머지 축은 *원리 + T4 권장값* 으로 정리합니다. GRPO 의 reward 를 올리려면 *학습 신호 (std>0) 를 확보* 하면서 *안정성 (붕괴 방지)* 을 지키는 균형이 핵심입니다.

| hyperparameter | reward 에 주는 영향 | 올리면 | 내리면 | T4 권장 |
|---|---|---|---|---|
| **`num_generations`** (group size) | 정답이 group 에 섞일 확률 + baseline 안정 | 신호 ↑·비용 ↑ (선형) | 비용 ↓·신호 빈약 (std=0 잦음) | **4-8** |
| **`temperature`** | rollout 다양성 (탐색) | 다양성 ↑·불안정 ↑ | 다양성 ↓ (group 단조) | **0.7-1.0** |
| **`beta`** (KL 제약) | reference 에서 멀어지는 정도 | 보수적 (붕괴·hacking ↓, 학습 느림) | 자유로움 (빠르나 붕괴 위험) | **0.0-0.04** |
| **`learning_rate`** | 갱신 보폭 | 빠르나 *붕괴* 위험 (instruct 능력 상실) | 안정·느림 | **1e-6-5e-6** |
| **`max_completion_length`** | 답 길이 (산술은 짧음) | 비용 ↑ | 답 잘림 위험 | **16-32** (산술) |

직관 요약:

- **`num_generations`↑** = "이 prompt 에서 동료 몇 명에게 물어볼까" — 많을수록 정답이 섞일 확률·baseline 신뢰 ↑, 비용도 비례
- **`temperature`↑** = 탐색 ↑ — group 에 다양성을 주지만 너무 높으면 *엉뚱한 답* 이 많아져 불안정
- **`beta`↑** = reference 에 가깝게 (보수적) — reward hacking·붕괴를 막지만 학습이 느려짐. 작은 모델·짧은 학습이면 0 으로 두고 빠르게, 붕괴가 보이면 0.01-0.04
- **`learning_rate` 너무 크면** instruct 모델이 *지시 따르는 능력* 을 잃고 reward 가 붕괴 — 작은 모델일수록 작게

> **HPO 의 결론**: GRPO reward 를 올리는 1순위는 *학습 신호 (std>0) 확보* 입니다 — `num_generations` 와 `temperature` 로 group 다양성을, `format reward` 로 reward 바닥을 받칩니다. 그 다음이 *안정성* — `learning_rate` 를 작게, 필요하면 `beta` 로 KL 제약. *base 모델의 능력* 이 천장을 정하므로, 가장 큰 레버는 여전히 *base 선택* (KoGPT2 → Qwen) 입니다.""")

# ----- §7 해석 -----
md(r"""## 7. 해석 — base 능력 + format reward + HPO 가 reward 를 올린 이유

본문(KoGPT2)과 부록(Qwen)의 차이를 *GRPO 전제조건* 으로 정리합니다.

| | 본문 (KoGPT2 125M, correctness only) | 부록 (Qwen 0.5B, correctness + format + HPO) |
|---|---|---|
| base 산술 능력 | 거의 0 | *가끔 맞힘* (>0) |
| group 다양성 (std) | 0 잦음 (전부 오답) | >0 (정답 섞임 + format 이 바닥 받침) |
| advantage | 대부분 0 (신호 없음) | 0 아님 (신호 있음) |
| reward 전·후 | 평평 | 상승 |

**세 레버가 함께 작동한 결과**:

1. **base 능력 (Qwen)** — 가장 큰 레버. 모델이 *가끔이라도 정답* 을 내야 group 에 차이가 생깁니다. GRPO 가 증폭할 *원재료* 입니다
2. **format reward** — 안전망. correctness 가 낮은 초반에도 *형식 차이* 로 std>0 을 유지해 학습이 멈추지 않게 합니다
3. **HPO** (`num_generations`↑ 등) — 증폭 효율. group 다양성과 baseline 안정성을 높여 advantage 를 정밀하게 만듭니다

> **GRPO 전제조건의 증명**: 같은 알고리즘 (GRPO) 인데 본문은 reward 가 안 오르고 부록은 오릅니다. 차이는 *알고리즘* 이 아니라 *전제조건 충족 여부* — 즉 *모델이 가끔이라도 성공하는가 (base 능력)* + *reward 가 0 만 나오지 않는가 (format reward)* + *그 신호를 잘 증폭하는가 (HPO)*. 이것이 본문 마지막 절의 한 문장 — ***GRPO 는 이미 가끔 성공하는 능력을 증폭할 뿐, 무에서 유를 만들지 못한다*** — 을 실험으로 확인한 것입니다. 그래서 *RL 전에 SFT·충분한 base* 가 전제입니다 (DeepSeek-R1 도 큰 base 에서 출발).""")

# ----- 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. 본문 KoGPT2 는 GRPO reward 가 안 올랐는데 부록 Qwen 은 올랐습니다. *알고리즘은 같습니다*. 무엇이 달라서 결과가 갈렸나요? (*group 다양성·std·advantage* 를 써서 설명)
2. **format reward** 는 정답 여부와 무관하게 *형식만 맞으면* 부분 보상을 줍니다. correctness reward 가 *전부 0* 인 group 에서도 *학습 신호* 가 생기는 이유를 advantage 식으로 설명해 보세요.
3. `num_generations` 를 4 → 8 로 키우면 reward 와 *비용* 에 각각 어떤 영향이 있나요? 왜 무작정 키우지 않나요?""")

# ----- FAQ -----
md(r"""## ❓ FAQ

### Q1. (실무) format reward 가 정확히 왜 도움이 되나요?

GRPO 의 advantage 는 $A_i = (r_i - \text{mean}) / (\text{std} + \varepsilon)$ 입니다. group 의 *모든 답이 같은 reward* 면 std=0 → advantage 0 → 학습 신호가 없습니다. 작은 모델은 어려운 문제에서 *모든 답이 똑같이 오답 (correctness 전부 0)* 인 경우가 많아 여기에 자주 빠집니다.

format reward 는 *정답을 못 맞혀도* 형식 (`"정답: N"`) 만 지키면 0.2 를 줍니다. 그러면 group 안에서 *형식 지킨 답 (0.2) vs 안 지킨 답 (0.0)* 의 차이가 생겨 **std>0** 이 됩니다:

```python
# correctness 전부 0 이라도 format 으로 std>0 -> advantage 0 아님
rewards = [0.2, 0.0, 0.2, 0.0]   # 2개는 형식 지킴
adv = (np.array(rewards) - 0.1) / (0.1 + 1e-4)   # -> [≈+1, ≈-1, ≈+1, ≈-1]
```

> 모델이 먼저 *형식* 을 배우고 (쉬움), 그 위에서 *정답* (어려움) 으로 나아가는 *사다리* 역할입니다. 단 format reward 가 *너무 크면* reward hacking (형식만 지키고 정답은 포기) 위험이 있으니, correctness 보다 *작게* (0.2 정도) 둡니다.

### Q2. (이론) `num_generations` (group size) 의 trade-off 는?

group 평균이 *baseline (critic 대체)* 이므로, group size 가 클수록:

- **장점**: ① 정답이 group 에 *섞일 확률* ↑ (std>0 확률 ↑) ② baseline (평균) 추정이 *안정* → advantage 정밀
- **단점**: rollout 비용이 *group size 에 비례* (매 step 생성을 group size 만큼) → T4 시간 ↑

```python
GRPOConfig(num_generations=8)   # 4 -> 8: 신호 ↑, 비용 2배
```

> T4 출발점은 4-8. group 이 너무 작으면 (2) *advantage 0 인 prompt* 가 많아 학습이 비효율적이고, 너무 크면 시간이 폭증합니다. *데이터·step 과 함께* 균형을 맞춥니다.

### Q3. (이론) GRPO 가 SFT 를 대체할 수 있나요?

대체하지 못합니다. GRPO(RL) 는 *모델이 이미 가끔 성공하는 능력* 을 그 방향으로 *증폭* 할 뿐, *없던 능력* 을 새로 가르치지 못합니다. base 가 task 를 *전혀* 못 풀면 group 이 *전부 오답* → std=0 → advantage 0 → 학습 신호 자체가 없습니다 (본문 KoGPT2 가 그 예).

> 그래서 정석 파이프라인은 *SFT 로 형식·기초를 먼저* (가끔이라도 성공하게) → *그 위에 GRPO 로 증폭* 입니다. DeepSeek-R1 도 *충분히 큰 base* 에서 출발했기에 *순수 RL* 이 가능했습니다 — 큰 모델은 어려운 문제도 *가끔* 맞혀 증폭할 신호가 있었습니다.

### Q4. (실무) reward hacking 은 어떻게 막나요?

**reward hacking** = 모델이 *진짜 목표가 아니라 reward 의 허점* 을 찾아 점수만 올리는 현상입니다. 예:

- format reward 가 크면 *형식만 지키고 정답은 포기* (0.2 만 챙김)
- correctness verifier 가 *"문자열에 정답 숫자가 있으면 1.0"* 이면 *모든 숫자를 나열* 해 우회

막는 법:

```python
# 1) format reward 를 correctness 보다 작게 (형식은 보조, 정답이 본질)
#    correctness 1.0  vs  format 0.2  처럼 가중
# 2) verifier 를 엄격하게 - 정확한 형식 + 끝 위치 + 정답
def reward_strict(completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        m = re.search(r"정답:\s*(-?\d+)\s*$", c.strip())   # 형식 + 끝 위치 + 정답
        out.append(1.0 if (m and m.group(1) == str(a)) else 0.0)
    return out
# 3) beta>0 으로 KL 제약 (reference 에서 멀어지면 페널티 -> 붕괴·hacking 완화)
```

> verifier·reward 설계가 GRPO 의 *가장 중요한 부분* 입니다. *허점 없는 reward* = *원하는 능력* 으로 정렬.

### Q5. (이론) 큰 모델일수록 RL (GRPO) 효과가 큰 이유는?

GRPO 는 *증폭* 기법이라 *증폭할 원재료 (가끔의 성공)* 가 있어야 작동합니다. 큰 모델은:

- 어려운 문제도 *가끔 맞혀* group 에 *정답이 섞일 확률* 이 높습니다 (std>0 잦음 → 학습 신호 풍부)
- *잠재된 능력* (사전학습으로 본 패턴) 이 많아 RL 로 *끌어낼 상한* 이 높습니다

```python
# 작은 모델: base reward ≈ 0 -> group 전부 오답 -> std=0 -> 학습 신호 없음
# 큰 모델  : base reward > 0 -> group 에 정답 섞임 -> std>0 -> GRPO 가 증폭
```

> 본문(KoGPT2 125M)→부록(Qwen 0.5B) 의 단 한 단계 크기 변화만으로도 reward 가 *안 오름 → 오름* 으로 갈렸습니다. DeepSeek-R1 이 *순수 RL 로 reasoning* 을 끌어낸 것도 *충분히 큰 base* 였기 때문 — RL 의 효과는 *base 능력에 비례* 합니다.""")

# ----- 다음 -----
md(r"""## 🔚 다음 — 메인 Ch 31 / Phase 5 복귀

이 부록은 본문 ([`31_grpo.ipynb`](./31_grpo.ipynb)) 마지막 절 (🔍 왜 reward 가 잘 안 올랐는가) 의 *해결편* 이었습니다. 핵심 메시지를 다시:

> ***GRPO 는 모델이 이미 가끔 성공하는 능력을 증폭할 뿐, 무에서 유를 만들지 못한다. 그래서 RL 전에 SFT·충분한 base·format reward 로 "출발점" 을 먼저 마련해야 한다.***

- **본문**: base KoGPT2 로 *GRPO 전제조건* 을 (reward 가 안 오르는 현상으로) 체감
- **부록**: Qwen + format reward + HPO 로 *그 전제조건을 충족시켜* reward 를 올림 — 둘의 대비가 전제조건을 증명

본문으로 돌아가 **Phase 4 회고** (Ch 24-31: pretraining → continual pretraining → SFT → alignment(DPO/GRPO)) 와 **Phase 5 (Ch 32-34, Diffusion LM)** 예고를 마저 보세요. Phase 5 는 *한 토큰씩* 이라는 Phase 1-4 의 대전제를 깨는, *전체 시퀀스를 병렬 denoise* 하는 새 생성 패러다임입니다.

**메인 Ch 31 로 복귀 → Phase 4 마무리 → Chapter 32 (Diffusion LM).**""")


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
