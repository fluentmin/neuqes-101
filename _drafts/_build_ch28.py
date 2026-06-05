"""Build 28_sft/28_sft.ipynb — Phase 4, 학습 단계 3 (SFT / Instruction Tuning).

Ch 27 (KoGPT2 continual pretraining) 의 *다음 단계*. 같은 KoGPT2 본체
(`skt/kogpt2-base-v2`, 125M), 같은 next-token CrossEntropyLoss — 다만
*데이터 형식 (instruction-response 쌍)* + *prompt 마스킹 (`labels[:prompt_len] = -100`)*
이 바뀝니다. 그게 GPT 시대 *학습 단계 3 (SFT)* 의 본질 — *행동 정렬*.

두 thread 의 클라이맥스:
  - Thread 1: `labels = -100` 자리. MLM(15% 만) -> CausalLM(거의 전부) -> SFT(답변만).
    `trl` collator 의 labels 마스킹을 직접 print 로 시각화.
  - Thread 2: "파인튜닝" 의미 변화의 완성. BERT task head 부착 ->
    GPT SFT behavior alignment. *진짜* instruction following.

데이터: KoAlpaca (`beomi/KoAlpaca-v1.1a`, instruction/output 필드).
Trainer: `trl.SFTTrainer` (첫 등장). Collator: trl 내장 completion-only 마스킹.
포맷: KoGPT2 는 chat template 없어 직접 포맷 (`### 명령어:\n...\n\n### 응답:\n...`).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "28_sft"
OUT_NB = OUT_DIR / "28_sft.ipynb"
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
md(r"""# Chapter 28. KoGPT2 SFT — Instruction Tuning (지시를 따르도록, 학습 단계 3)

**목표**: Phase 4 의 *학습 단계 3 (SFT, Supervised Fine-Tuning / Instruction Tuning)* 챕터. Ch 27 에서 *KoGPT2 본체를 한국어 TinyStories 로 continual pretraining* 했다면, 이번엔 같은 **KoGPT2 (`skt/kogpt2-base-v2`, 125M)** 본체를 **KoAlpaca instruction-response 쌍 데이터** 로 **SFT** 합니다. 본체도 같고, loss 종류 (next-token `CrossEntropyLoss`) 도 같습니다. 바뀌는 건 **데이터 형식 (연속 텍스트 → instruction-response 쌍)** + **`labels` 마스킹 (pad 만 → prompt 부분 전체)** + **trainer (`Trainer` → `SFTTrainer`)** 입니다. *그 한 줄 `labels[:prompt_len] = -100` 이 모델을 지시를 따르게 만듭니다*.

**환경**: Google Colab **T4 GPU 필수**.

**예상 소요 시간**: 약 20-28분 (KoAlpaca 로드·포맷 약 2분 + KoGPT2 로드 약 2분 + collator labels 마스킹 시각화 약 1분 + SFT 학습 약 12-18분 + SFT 전·후 instruction following 비교 약 3분)

---

## 학습 흐름

1. 📊 **누적 추적표** (Ch 25/26/27 + **28 강조** + Ch 29 예고) + GPT 학습 4단계 표 (Ch 28 = 단계 3)
2. 🔄 **변경점 (Diff from Ch 27)** — *데이터 형식 + trainer + labels 마스킹* 만 변함. 본체·토크나이저·loss 종류는 그대로
3. 🎯 **`labels = -100` thread 완성 표** — MLM 15% / CausalLM 거의 전부 / SFT 답변만. *세 단계를 한 화면에*. 이 챕터가 thread 의 종착점
4. ⚠️ **파인튜닝 의미 변화 완성** — BERT task head vs GPT SFT behavior alignment. *진짜* instruction following
5. 📐 **Loss** — next-token CE 동일, 단 *어느 자리에서 계산하는가* 가 핵심 변화 (response-only)
6. 🔤 **토크나이저 노트** — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 방식). instruction 포맷 토큰화 + response_template 위치
7. 🚀 **실습**: KoAlpaca 로드 → KoGPT2 로드 → **collator labels 마스킹 직접 시각화 (클라이맥스)** → `SFTTrainer` 학습 → SFT 전·후 instruction following 비교
8. 📦 **등장 라이브러리** (`trl.SFTTrainer` 첫 등장) / 🎯 **체크포인트** / ❓ **FAQ** (답변 포함)

---

> 📒 **사전 학습 자료**: Ch 27 (KoGPT2 continual pretraining — 본 챕터와 *같은 본체*), Ch 24-26 (GPT 사전학습), Ch 20-22 (MLM 의 `labels = -100`). 본 챕터는 Phase 4 의 두 thread (`labels = -100` 자리 / "파인튜닝" 의미 변화) 의 *클라이맥스*. **`### 응답:` 뒤만 학습한다** 는 한 줄이 *왜 GPT 하나가 모든 task 를 해내는가* 의 답입니다.""")

# ----- 2. 누적 추적표 + GPT 4단계 -----
md(r"""## 📊 누적 추적표

| Ch | 모델 | 토크나이저 | 데이터 | `labels = -100` 자리 | Loss |
|---|---|---|---|---|---|
| 25 | `gpt2` (124M, 사전학습) | BPE (gpt2 그대로, vocab 50,257) | 영어 TinyStories 30K | pad 만 | `CrossEntropyLoss` (next-token) - continual pretraining |
| 26 | 작은 GPT2 (한국어, 약 3M, scratch) | BBPE (직접 학습, vocab 약 4,000) | 한국어 TinyStories 30K | pad 만 | `CrossEntropyLoss` (next-token) |
| 27 | KoGPT2 `skt/kogpt2-base-v2` (125M) | BBPE (KoGPT2 그대로, vocab 51,200) | 한국어 TinyStories 30K | pad 만 | `CrossEntropyLoss` (next-token) - continual pretraining |
| **28 ← 여기** | **KoGPT2 `skt/kogpt2-base-v2` (125M, 동일)** | **BBPE (KoGPT2 그대로, vocab 51,200, 동일)** | **KoAlpaca instruction-response 쌍 (약 3-5K)** | **prompt 부분 (`### 응답:` 앞 전부)** | **`CrossEntropyLoss` (next-token, *답변 부분만*) — SFT** |
| 29 (다음) | Ch 28 SFT 모델 + 비교 | (동일) | 분야별 벤치마크 (KMMLU / HAERAE / MMLU ...) | - (평가만) | - (`lm-evaluation-harness`) |

전체 챕터 표는 [루트 README](https://github.com/yoon-gu/neuqes-101#챕터별-변화추적표) 를 참고하세요.

---

## 🌏 GPT 시대 학습 4단계 — 본 챕터의 위치 (단계 3, SFT)

Ch 24 에서 도입한 GPT 시대 학습 4단계 표. 본 챕터는 *단계 3 (SFT)* — *행동 정렬* 이 처음 일어나는 단계입니다.

| 단계 | 정확 용어 | 의미 | `labels = -100` 자리 | 본 커리큘럼 | 본 챕터? |
|---|---|---|---|---|---|
| 1 | **Pretraining** (사전학습) | random init 본체 + 일반 코퍼스 | pad 만 | Ch 24 (영어), Ch 26 (한국어) | |
| 2 | **Continual pretraining** (계속 사전학습) | 사전학습된 본체 + 새 데이터, *같은 CausalLM task* | pad 만 | Ch 25 (영어), Ch 27 (한국어) | |
| 3 | **SFT** (Supervised Fine-Tuning / Instruction tuning) | instruction-response 쌍으로 *행동 정렬*. *답변 부분만* 학습 | **prompt 부분** | **Ch 28 ← 여기** | ✅ |
| 4 | **Alignment** (DPO / RLHF / GRPO) | preference 또는 verifier reward 로 *선호 정렬* | (RL 내부, response 부분만) | Ch 30 (DPO), Ch 31 (GRPO) | |

### 단계 2 (Ch 27) → 단계 3 (Ch 28) 의 결정적 변화

- **단계 2 (continual pretraining)**: *연속된 일반 텍스트* 를 *거의 모든 자리* 에서 학습. 모델은 *다음 토큰 분포* 를 도메인에 맞게 다듬을 뿐 — *지시를 따르는 법* 은 배우지 않음
- **단계 3 (SFT)**: *instruction-response 쌍* 에서 *답변 토큰만* 학습. 모델은 *주어진 지시에 어떻게 응답하는가* 를 배움 — **행동 정렬 (behavior alignment)**

> *같은 본체, 같은 loss 종류, 단 데이터 형식 + 마스킹 자리만 바뀌어* 모델이 *지시를 따라가게* 됩니다. 그 한 줄이 `labels[:prompt_len] = -100`. 본 챕터는 그 한 줄을 *눈으로 확인* 하는 챕터입니다.""")

# ----- 3. 변경점 (Diff from Ch 27) -----
md(r"""## 🔄 변경점 (Diff from Ch 27)

| 축 | Ch 27 (KoGPT2 continual pretraining) | Ch 28 (본 챕터, KoGPT2 SFT) |
|---|---|---|
| 본체 | KoGPT2 `skt/kogpt2-base-v2` (125M) | **KoGPT2 `skt/kogpt2-base-v2` (125M, 동일)** ← 고정 |
| 토크나이저 | `PreTrainedTokenizerFast` (KoGPT2 BBPE, vocab 51,200) | **(동일)** ← 고정 |
| Loss 종류 | next-token `CrossEntropyLoss` | **next-token `CrossEntropyLoss` (종류 동일)** ← 고정 |
| **데이터 형식** | 연속된 일반 텍스트 (TinyStories) | **instruction-response 쌍 (KoAlpaca)** ← *변화 1* |
| **Trainer** | `transformers.Trainer` | **`trl.SFTTrainer`** ← *변화 2* (새 클래스, 첫 등장) |
| **`labels = -100` 자리** | pad 만 (거의 모든 자리 학습) | **prompt 부분 전체 (`### 응답:` 앞)** ← *변화 3, 핵심* |
| 효과 | 도메인 적응 (동화 풍) | **instruction 따라가기 (행동 정렬)** ← 메시지 |
| lr | 2e-5 | 2e-5 (SFT 표준, 동일 범위) |

> **핵심**: *본체·토크나이저·loss 종류는 그대로*. 바뀌는 건 *데이터 형식 + trainer + 마스킹 자리* 세 가지. 그중에서도 **`labels` 마스킹 자리** 가 *왜 모델이 instruction 을 따라가게 되는가* 의 진짜 원인입니다. Ch 27 의 collator 가 *거의 모든 자리* 를 학습했다면, Ch 28 은 *prompt 를 전부 가리고 답변만* 학습합니다 — *정확히 정반대 자리*.""")

# ----- 4. labels = -100 thread 완성 표 -----
md(r"""## 🎯 `labels = -100` thread 의 완성 — 세 단계를 한 화면에

커리큘럼 전체를 관통하는 thread 의 *종착점* 입니다. `labels = -100` 은 *그 자리를 loss 에서 제외* 하라는 의미 (`CrossEntropyLoss(ignore_index=-100)`). **어느 자리를 -100 으로 두느냐가 곧 모델이 무엇을 학습하느냐** 를 결정합니다.

| 단계 | 챕터 | task | `labels = -100` 자리 | loss 계산 자리 | 모델이 배우는 것 |
|---|---|---|---|---|---|
| **MLM** | Ch 20·21·22 | 양방향 빈칸 채우기 | 선택 안 된 약 85% (= 안 가린 자리) | **선택된 약 15% (가린 자리) 만** | 문맥으로 *가려진 단어* 복원 |
| **CausalLM** | Ch 24·25·26·27 | 다음 토큰 예측 | **pad 만** (거의 없음) | **거의 전 토큰** | *다음에 올 토큰* 분포 |
| **SFT (본 챕터)** | **Ch 28** | instruction 따라가기 | **prompt 부분 전체** (`### 응답:` 앞) | **답변 토큰만** | *지시에 대한 응답* 생성 |

### 세 단계를 한눈에 — *같은 `-100`, 다른 자리*

```
MLM (Ch 21):     [the] [MASK] [sat] [on] [the] [MASK]
labels:          -100   cat   -100  -100 -100   mat       <- 가린 15% 만 학습

CausalLM (Ch 27): [옛날] [옛날에] [작은] [소녀가] [살았어요]
labels:           옛날에  작은    소녀가  살았어요   <eos>      <- 거의 전부 학습 (shift)

SFT (Ch 28):     [### 명령어:] [피보나치 설명] [### 응답:] [피보나치는] [수열입니다]
labels:           -100  -100   -100  -100  -100   -100   피보나치는  수열입니다   <- 답변만 학습
```

> **MLM 은 일부 (15%) 만 가리고**, **CausalLM 은 거의 안 가리고**, **SFT 는 prompt 전부 가립니다**. 세 task 모두 *같은 `CrossEntropyLoss(ignore_index=-100)`* 를 쓰지만 *-100 의 자리* 가 다릅니다. **Ch 28 에서 이 thread 가 완성됩니다** — 아래 §3 에서 KoGPT2 의 실제 collator 출력을 print 로 *눈으로 확인* 합니다 (Ch 21 의 `[MASK]` 80/10/10 시각화의 SFT 판).

핵심 메시지: **모델이 instruction 을 "따라간다" 는 것은, instruction 토큰 자체는 학습하지 않고 그에 대한 *응답만* 학습한다는 의미**. 만약 prompt 도 함께 학습하면 모델은 *질문 자체를 외우는* 쪽으로 기웁니다 — 우리가 원하는 건 *질문에 답하는 법* 입니다. 그 차이가 한 줄 `labels[:prompt_len] = -100` 입니다.""")

# ----- 5. 파인튜닝 의미 변화 완성 -----
md(r"""## ⚠️ "파인튜닝" 의미 변화의 완성 — task head → behavior alignment

커리큘럼 전체에서 *fine-tune* 이라는 단어는 *세 의미* 로 쓰였습니다. Ch 28 이 그 세 번째 의미 (*행동 정렬*) 의 도착점입니다.

| 의미 | 무엇이 바뀌나 | 챕터 | 비유 |
|---|---|---|---|
| **① task adaptation** (BERT 파인튜닝) | 본체 + **새 head** (`Linear(H, K)`) + **새 task loss** | Ch 9-23 (분류) | *새 도구* 를 손에 붙임 |
| **② 데이터 적응** (GPT continual pretraining) | head 그대로, **같은 next-token task**, *새 데이터* | Ch 25·27 | *같은 도구* 로 *새 재료* 연습 |
| **③ 행동 정렬** (GPT SFT) | head 그대로, **같은 next-token CE**, *instruction 형식 데이터* + *prompt 마스킹* | **Ch 28 ← 여기** | *도구는 그대로*, *지시를 따르는 법* 을 깨움 |

### BERT 파인튜닝 (①) vs GPT SFT (③) — *진짜* instruction following

- **BERT 파인튜닝 (①)**: task 마다 *다른 head* 를 붙입니다. 감정분류 head, NER head, QA head... *task 하나당 모델 하나*. head 가 task 를 정의
- **GPT SFT (③)**: *head 는 LM head 하나 그대로*. *입력 프롬프트 형식* 만 바꾸면 *같은 모델* 이 번역도, 요약도, 질의응답도 합니다. **task 가 head 가 아니라 prompt 에 인코딩됨**

> *왜 GPT 하나가 모든 task 를 해내는가?* — 답은 SFT 입니다. 본체는 *입력 프롬프트만 바꾸면 다른 일* 을 하도록 *행동 정렬* 되어 있습니다. **SFT 가 그 능력을 깨우는 단계**. BERT 가 *task 마다 head 를 갈아끼우던* 시대에서, GPT 가 *prompt 하나로 모든 task* 를 하는 시대로의 전환 — 그 전환점이 본 챕터입니다.

이게 *진짜* behavior tuning 입니다 — *task 적응 (①)* 도, *데이터 적응 (②)* 도 아닌, *모델의 행동 자체* 를 instruction 을 따르도록 정렬하는 단계.""")

# ----- 6. Loss 노트 -----
md(r"""## 📐 Loss — next-token CE 동일, 단 *어느 자리에서 계산하는가*

본 챕터의 loss 는 Ch 27 과 *같은 종류* — next-token `CrossEntropyLoss`:

$$L_{\text{CLM}} = -\sum_{i} \log P(x_{i+1} \mid x_{\leq i})$$

다른 점은 **어느 위치 $i$ 에서 합산하느냐** 입니다. SFT 는 *답변 부분 토큰만* 합산합니다:

$$L_{\text{SFT}} = -\sum_{i \in \text{response}} \log P(x_{i+1} \mid x_{\leq i}) \qquad (\text{prompt 부분은 } -100 \text{ 으로 제외})$$

### 왜 prompt 도 학습하면 안 되나 — 숫자로 감 잡기

instruction `"### 명령어:\n2+2 는?\n\n### 응답:\n"` 뒤에 답변 `"4 입니다."` 가 오는 한 샘플을 생각해 봅시다. 토큰이 *prompt 12개 + 답변 4개* 라고 하면:

| 학습 방식 | loss 합산 자리 | 모델이 강화하는 것 |
|---|---|---|
| **전체 학습** (prompt 포함) | 16개 토큰 전부 | *"### 명령어:" → "2+2 는?"* 같은 *질문 자체의 패턴* 까지 외움 |
| **response-only** (본 챕터) | 답변 4개만 | *prompt 가 주어졌을 때 답변* 하는 능력만 |

전체 학습을 하면 loss 의 *대부분* 이 *prompt 토큰* 에서 나옵니다 (prompt 가 보통 더 김). 그러면 모델은 *질문을 받아쓰는 데* gradient 를 낭비합니다. 우리가 원하는 건 *질문을 외우는 게 아니라 답하는 법* — 그래서 prompt 를 `-100` 으로 가립니다.

### response-only 의 직관

| 토큰 위치 | label | loss 기여 | 의미 |
|---|---|---|---|
| `### 명령어:` ... `### 응답:` (prompt) | `-100` | **0** (제외) | *주어진 조건* — 외울 필요 없음 |
| `4` `입니다` `.` `<eos>` (답변) | 원본 token id | **포함** | *이걸 생성하는 법* 을 학습 |

> *prompt 는 조건 (given), 답변은 학습 대상 (target)*. 이 구분이 SFT 의 전부입니다. `trl` 의 collator 가 `### 응답:` 위치를 찾아 그 *앞을 전부 -100* 으로 만듭니다 — §3 에서 직접 봅니다.""")

# ----- 7. 토크나이저 노트 -----
md(r"""## 🔤 토크나이저 노트 — KoGPT2 `PreTrainedTokenizerFast` (Ch 27 방식 그대로)

본 챕터의 토크나이저는 *Ch 27 과 완전히 동일*. KoGPT2 BBPE (vocab 51,200) 를 그대로 가져옵니다. **단 KoGPT2 는 `AutoTokenizer` 가 영어 GPT2 로 잘못 fallback 하는 함정** 이 있어 (Ch 27 §토크나이저 노트), `PreTrainedTokenizerFast` + special token 명시로 로드합니다.

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)
```

### instruction 포맷 + response_template 위치

KoGPT2 는 *chat template 이 없습니다* (instruction-tuned 모델이 아니므로). 그래서 instruction-response 를 *직접 포맷* 합니다:

```
### 명령어:
{instruction}

### 응답:
{output}
```

여기서 **`### 응답:\n`** 가 **response_template** — *이 문자열 이후부터가 답변* 이라는 경계 표시입니다. `trl` collator 는 이 경계를 기준으로 *앞은 prompt (= -100), 뒤는 답변 (= 학습)* 으로 나눕니다.

### 같은 문장이 어떻게 토큰화되는가

instruction 포맷 `### 명령어:\n피보나치 설명\n\n### 응답:\n` 을 KoGPT2 BBPE 로 토큰화하면:

- `###` → `#`·`#`·`#` (3 토큰), `명령어` → `명령`·`어` (2 토큰), `:` → 1 토큰, `\n` → 1 토큰 ...
- 한국어 어절 (`피보나치`, `설명`) 은 KoGPT2 가 한국어 코퍼스로 학습한 *의미 있는 토큰* 으로 압축

> **response_template `### 응답:\n` 자체도 토큰 시퀀스** 입니다. collator 는 이 *토큰 시퀀스* 를 input_ids 안에서 찾아 그 *직후 위치* 부터 답변으로 간주합니다. 그래서 response_template 은 *데이터에 일관되게 등장하고, 본문과 충돌하지 않는* 문자열이어야 합니다 (`### 응답:` 처럼 특수한 마커가 적합).

다음 챕터 (Ch 29 벤치마크 평가) 에서도 *같은 KoGPT2 토크나이저* 를 그대로 사용합니다 — 토크나이저는 Ch 27 이후 고정.""")

# ----- 8. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업

`trl` (Transformer Reinforcement Learning) 라이브러리가 이번 챕터에 새로 등장합니다 — `SFTTrainer` 와 SFT 용 데이터 collator 를 제공. `transformers` / `datasets` / `accelerate` 와 함께 설치합니다.""")

code(r"""%pip install -q -U trl transformers tokenizers datasets accelerate""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import math
import os
import random
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

# ----- 9. §1 KoAlpaca 데이터 -----
md(r"""## 1. KoAlpaca instruction 데이터 로드 + 포맷

**`beomi/KoAlpaca-v1.1a`** — 한국어 instruction tuning 데이터셋. 각 샘플은 `instruction` (지시) 과 `output` (응답) 필드를 가집니다 (`url` 필드는 출처 — 학습에 사용 안 함). T4 + 30분 룰 안에서 **약 3,000 샘플** 만 subset 으로 사용합니다.

KoGPT2 는 chat template 이 없으니 *직접 포맷* — `### 명령어:\n{instruction}\n\n### 응답:\n{output}`. 여기서 **`### 응답:\n` 가 response_template** (답변 시작 경계).""")

code(r"""from datasets import load_dataset

N_SFT = 3000          # T4 + 30분 룰 - subset
MAX_CHARS = 600       # 너무 긴 답변은 잘라 평균 길이 통제 (학습 안정 + 속도)

raw = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
print("raw dataset:", raw)
print("\nfields:", raw.column_names)
print("\n=== sample 0 ===")
ex0 = raw[0]
print("instruction:", ex0["instruction"][:200])
print("output     :", ex0["output"][:200])

# instruction / output 모두 비어있지 않은 샘플만, 길이 통제 후 subset
def keep(ex):
    return bool(ex["instruction"].strip()) and bool(ex["output"].strip())

raw = raw.filter(keep)
raw = raw.shuffle(seed=SEED).select(range(min(N_SFT, len(raw))))
print(f"\nafter filter + subset: {len(raw):,} samples")""")

md(r"""### 포맷 함수 — `prompt` / `completion` 두 컬럼으로

`trl.SFTTrainer` 는 *`prompt` + `completion` 두 컬럼* 형식을 받으면 *completion (답변) 부분만 자동으로 학습 대상* 으로 잡습니다 (`completion_only_loss=True`). 그래서 우리는 instruction 을 prompt 쪽에, output 을 completion 쪽에 넣되, **response_template `### 응답:\n` 까지를 prompt 에 포함** 시켜 *답변 시작 경계* 를 명확히 합니다.""")

code(r"""RESPONSE_TEMPLATE = "### 응답:\n"   # 이 뒤부터가 '답변' (학습 대상)


def build_prompt(instruction: str) -> str:
    '''KoGPT2 용 instruction 포맷. RESPONSE_TEMPLATE 로 끝나 답변 경계를 명시.'''
    return f"### 명령어:\n{instruction}\n\n{RESPONSE_TEMPLATE}"


def to_prompt_completion(ex):
    output = ex["output"].strip()
    if len(output) > MAX_CHARS:
        output = output[:MAX_CHARS]
    return {
        "prompt": build_prompt(ex["instruction"].strip()),
        "completion": output,
    }


sft_ds = raw.map(to_prompt_completion, remove_columns=raw.column_names, desc="format")
print("formatted dataset:", sft_ds)
print("\n=== formatted sample 0 ===")
print("--- prompt ---")
print(sft_ds[0]["prompt"])
print("--- completion ---")
print(sft_ds[0]["completion"][:200])""")

# ----- 10. §2 KoGPT2 로드 -----
md(r"""## 2. KoGPT2 토크나이저·모델 로드 — *Ch 27 과 동일한 본체*

본 챕터의 본체는 *Ch 27 과 완전히 같은 KoGPT2*. 토크나이저도 같은 방식 (`PreTrainedTokenizerFast` + special token 명시 — `AutoTokenizer` 함정 회피). encode → decode 왕복으로 한국어가 깨지지 않는지 한 줄 검증합니다.""")

code(r"""from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

t0 = time.time()
# 주의: KoGPT2 는 AutoTokenizer 가 영어 GPT2 토크나이저로 잘못 fallback 합니다.
# SKT 공식 방식대로 PreTrainedTokenizerFast 로 special token 을 직접 지정해 로드.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>", eos_token="</s>", unk_token="<unk>",
    pad_token="<pad>", mask_token="<mask>",
)

model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2").to(device)
model.config.pad_token_id = tokenizer.pad_token_id
print(f"load done: {time.time()-t0:.1f}s")

# encode -> decode 왕복 검증 (한국어 깨짐 방지)
probe = "옛날 옛날에 작은 소녀가"
roundtrip = tokenizer.decode(tokenizer(probe)["input_ids"])
print(f"\nroundtrip check: {roundtrip!r}  ({'OK' if roundtrip == probe else 'BROKEN'})")

n_params = model.num_parameters()
print(f"\n=== model ===")
print(f"#params      : {n_params/1e6:.2f} M  (same body as Ch 27)")
print(f"vocab_size   : {tokenizer.vocab_size:,}")
print(f"tokenizer    : {type(tokenizer).__name__}")
print(f"  eos_token  : {tokenizer.eos_token}  id={tokenizer.eos_token_id}")
print(f"  pad_token  : {tokenizer.pad_token}  id={tokenizer.pad_token_id}")""")

# ----- 11. §3 클라이맥스: collator labels 마스킹 시각화 -----
md(r"""## 3. 🎯 collator 의 `labels` 마스킹 직접 시각화 — **이 챕터의 클라이맥스**

여기가 본 챕터의 핵심. `trl` 의 SFT collator 가 한 instruction-response 샘플을 받아 **prompt 부분을 전부 `-100` 으로, 답변 부분만 원본 token id 로** 만드는 것을 *눈으로* 확인합니다. Ch 21 의 `[MASK]` 80/10/10 시각화의 *SFT 판* 입니다.

### 동작 원리

1. `SFTTrainer` 가 *prompt + completion* 을 토큰화해 이어 붙이고, *completion 부분에 1, prompt 부분에 0* 인 `completion_mask` 를 만듭니다 (response_template `### 응답:\n` 가 prompt 의 끝).
2. collator 가 `labels = input_ids.clone()` 한 뒤 *`completion_mask == 0` 인 자리 (= prompt) 를 전부 `-100`* 으로 덮습니다.
3. 그래서 loss 는 *답변 토큰에서만* 계산됩니다 — `labels[:prompt_len] = -100` 의 효과.""")

code(r"""# trl 1.x 의 SFT collator. 버전마다 위치가 다를 수 있어 폴백 import.
try:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as SFTCollator
except Exception:
    from trl import DataCollatorForLanguageModeling as SFTCollator  # 일부 버전

# 한 샘플을 prompt / completion 으로 직접 토큰화 (SFTTrainer 내부와 같은 절차)
sample = sft_ds[0]
prompt_text = sample["prompt"]
completion_text = sample["completion"]

p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
c_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
c_ids = c_ids + [tokenizer.eos_token_id]   # SFTTrainer 는 답변 끝에 EOS 부착

input_ids = p_ids + c_ids
completion_mask = [0] * len(p_ids) + [1] * len(c_ids)   # 0 = prompt, 1 = 답변

print(f"prompt tokens     : {len(p_ids)}")
print(f"completion tokens : {len(c_ids)}  (incl. EOS)")
print(f"total tokens      : {len(input_ids)}")

# collator 적용 - prompt 부분이 -100 으로 마스킹됨
collator = SFTCollator(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)
batch = collator([{"input_ids": input_ids, "completion_mask": completion_mask}])
labels = batch["labels"][0].tolist()
ids = batch["input_ids"][0].tolist()

n_learn = sum(1 for l in labels if l != -100)
print(f"\nlabels learned    : {n_learn} / {len(labels)}  (prompt masked = {len(labels) - n_learn})")""")

code(r"""# 토큰별 표 - position | token | input_id | label | learn?
rows = []
for i, (tid, lab) in enumerate(zip(ids, labels)):
    rows.append({
        "pos": i,
        "token": repr(tokenizer.decode([tid])),
        "input_id": tid,
        "label": lab,
        "learn?": "Y (response)" if lab != -100 else "- (prompt, -100)",
    })
label_table = pd.DataFrame(rows)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)
print("=" * 78)
print("Per-token labels - prompt is masked (-100), only response is learned")
print("=" * 78)
print(label_table.to_string(index=False))""")

md(r"""**무엇을 보고 있나** — 위 표의 `learn?` 열을 보면:

- **prompt 부분** (`### 명령어:` ... `### 응답:\n` 까지) → `label = -100` → *loss 에서 제외*. 모델은 *이 질문 자체* 를 외우지 않습니다
- **답변 부분** (`### 응답:\n` *이후* 의 모든 토큰 + EOS) → `label = 원본 token id` → *loss 에 포함*. 모델은 *이 답변을 생성하는 법* 만 배웁니다

> Ch 21 의 `[MASK]` 시각화는 *문장의 약 15% 를 가렸다* 면, 여기서는 *prompt 전체를 가립니다* — **정반대 방향의 마스킹**. 그리고 이게 `labels = -100` thread 의 *세 번째이자 마지막 단계*. MLM(15% 만 학습) → CausalLM(거의 전부 학습) → **SFT(답변만 학습)**. 한 줄 `labels[:prompt_len] = -100` 의 효과를 *눈으로 확인* 했습니다.""")

code(r"""# 요약 시각화 - prompt vs response 토큰 수, loss 기여 비율
n_prompt = len(labels) - n_learn
n_resp = n_learn

fig, ax = plt.subplots(figsize=(9, 1.8))
ax.barh([0], [n_prompt], color="lightgray", edgecolor="gray",
        label=f"prompt (masked, -100): {n_prompt} tokens")
ax.barh([0], [n_resp], left=[n_prompt], color="tab:green", edgecolor="darkgreen",
        label=f"response (learned): {n_resp} tokens")
ax.set_yticks([])
ax.set_xlabel("token position")
ax.set_title("SFT labels: prompt masked (-100), only response contributes to loss")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2)
plt.tight_layout(); plt.show()""")

# ----- 12. §4 SFTTrainer 학습 -----
md(r"""## 4. `SFTTrainer` 로 SFT 학습 — *새 trainer, 같은 loss 종류*

`trl.SFTTrainer` 는 본 챕터에 처음 등장하는 클래스입니다. `transformers.Trainer` 를 상속해 *SFT 에 특화된 전처리* (prompt/completion 토큰화, EOS 부착, completion 마스킹) 를 자동으로 해 줍니다. 설정은 `SFTConfig` (── `TrainingArguments` 를 상속) 로 주며, **`completion_only_loss=True`** 가 *답변 부분만 학습* 하라는 핵심 옵션입니다.""")

code(r"""from trl import SFTTrainer, SFTConfig

# SFT 학습 전 generation 비교를 위해 '학습 전' 모델 상태를 기록해 둠 (§5 에서 사용)
PROMPTS = [
    "피보나치 수열을 설명해줘",
    "건강한 식습관 3가지를 알려줘",
    "파이썬으로 리스트를 뒤집는 방법은?",
    "아침에 일찍 일어나는 팁을 알려줘",
]
GEN_KWARGS = dict(max_new_tokens=80, do_sample=True, temperature=0.8,
                  top_k=50, repetition_penalty=1.3)


@torch.no_grad()
def generate_answer(active_model, instruction: str, **kwargs):
    '''instruction 을 포맷해 답변을 생성. RESPONSE_TEMPLATE 뒤부터를 답변으로 디코드.'''
    text = build_prompt(instruction)
    enc = tokenizer(text, return_tensors="pt").to(active_model.device)
    out = active_model.generate(
        **enc,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # 답변 부분만 잘라내기 (response_template 이후)
    if RESPONSE_TEMPLATE.strip() in full:
        return full.split(RESPONSE_TEMPLATE.strip(), 1)[-1].strip()
    return full[len(text):].strip()


torch.manual_seed(SEED)
model.eval()
before_outputs = []
print("=" * 70)
print("BEFORE SFT - raw KoGPT2 (no instruction tuning yet)")
print("=" * 70)
for p in PROMPTS:
    ans = generate_answer(model, p, **GEN_KWARGS)
    before_outputs.append(ans)
    print(f"\n[instruction] {p}")
    print(f"[answer] {ans[:240]}")""")

code(r"""sft_config = SFTConfig(
    output_dir="./out_kogpt2_sft",
    num_train_epochs=1,                     # SFT 는 1-3 epoch 이 표준 - T4 룰 안에서 1
    per_device_train_batch_size=2,          # KoGPT2 125M + instruction 은 시퀀스가 길어 작게
    gradient_accumulation_steps=8,          # effective batch = 16
    learning_rate=2e-5,                     # SFT 표준 lr
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    max_length=512,                         # instruction + response 길이 상한
    completion_only_loss=True,              # <- 핵심: 답변 부분만 loss (prompt = -100)
    packing=False,                          # 샘플 경계 유지 (마스킹이 정확하려면 packing 끔)
    fp16=USE_FP16,                          # T4 는 bf16 불가
    logging_steps=20,
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

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=sft_ds,
    processing_class=tokenizer,
    callbacks=[vram_cb],
)

t0 = time.time()
train_out = trainer.train()
elapsed = time.time() - t0

print(f"\n=== SFT summary ===")
print(f"elapsed     : {elapsed/60:.2f} min")
print(f"global_step : {train_out.global_step}")
print(f"train_loss  : {train_out.training_loss:.4f}")
if torch.cuda.is_available():
    print(f"final peak  : {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")""")

# ----- 13. §5 학습 곡선 -----
md(r"""## 5. 학습 곡선 — *답변 부분에서만 계산된* loss

아래 loss 는 *답변 토큰에서만* 계산된 값입니다 (prompt 는 `-100` 으로 제외). Ch 27 의 loss (거의 모든 자리) 와는 *합산 대상* 이 다르므로 절대값을 직접 비교하지는 않습니다.""")

code(r"""log = trainer.state.log_history
train_pts = [(r["step"], r["loss"]) for r in log if "loss" in r and "eval_loss" not in r]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

if train_pts:
    ax1.plot([s for s, _ in train_pts], [l for _, l in train_pts], "-",
             color="tab:blue", alpha=0.8, label="train (response-only)")
ax1.set_xlabel("step"); ax1.set_ylabel("cross-entropy loss (response tokens only)")
ax1.set_title("KoGPT2 SFT on KoAlpaca - loss")
ax1.grid(True, alpha=0.3); ax1.legend()

if vram_cb.steps:
    ax2.plot(vram_cb.steps, vram_cb.peak_MiB, "o-", color="tab:green",
             label="peak VRAM (per log window)")
    ax2.set_title("VRAM trace  (bs=2, grad_accum=8, fp16)")
else:
    ax2.text(0.5, 0.5, "VRAM trace available on CUDA only",
             ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("VRAM trace - CUDA only")
ax2.set_xlabel("step"); ax2.set_ylabel("VRAM (MiB)")
ax2.grid(True, alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()""")

# ----- 14. §6 SFT 전후 instruction following 비교 -----
md(r"""## 6. 🆚 SFT 전·후 instruction following 비교 — *행동 정렬이 일어났는가*

본 챕터의 핵심 데모. *같은 instruction* 을 *SFT 전 (raw KoGPT2)* 과 *SFT 후* 에 각각 넣어 답변을 비교합니다.

- **SFT 전 (raw KoGPT2)**: instruction 을 *지시로 인식하지 못하고* — 질문을 *이어쓰기* 하거나, 동화처럼 계속 쓰거나, 엉뚱한 방향으로 흐름
- **SFT 후**: instruction 을 *따라* — 질문에 *대답* 하는 구조화된 답변

이 차이가 *행동 정렬 (behavior alignment)* 의 직접 증거입니다.""")

code(r"""torch.manual_seed(SEED)
model.eval()
after_outputs = []
print("=" * 70)
print("AFTER SFT - KoGPT2 + KoAlpaca instruction tuning")
print("=" * 70)
for p in PROMPTS:
    ans = generate_answer(model, p, **GEN_KWARGS)
    after_outputs.append(ans)
    print(f"\n[instruction] {p}")
    print(f"[answer] {ans[:240]}")""")

code(r"""# BEFORE vs AFTER 나란히 비교
print("=" * 80)
print("BEFORE SFT (raw KoGPT2) vs AFTER SFT (KoGPT2 + KoAlpaca) - instruction following")
print("=" * 80)
comparison = []
for p, before, after in zip(PROMPTS, before_outputs, after_outputs):
    print(f"\nINSTRUCTION : {p}")
    print("-" * 80)
    print(f"BEFORE      : {before[:300]}")
    print(f"AFTER       : {after[:300]}")
    comparison.append({
        "instruction": p,
        "before (raw)": before[:80] + ("..." if len(before) > 80 else ""),
        "after (sft)": after[:80] + ("..." if len(after) > 80 else ""),
    })

print("\n\n=== compact table ===")
print(pd.DataFrame(comparison).to_string(index=False))""")

md(r"""**해석 가이드 — behavior alignment 의 증거**

- **BEFORE (raw KoGPT2)**: 같은 *125M 본체* 인데도 instruction 을 *지시로 받아들이지 못합니다*. `"피보나치 수열을 설명해줘"` 를 넣으면 *설명* 대신 *질문을 이어 쓰거나*, 일반 산문으로 흘러가거나, 동화체로 새는 경향
- **AFTER (KoGPT2 + KoAlpaca SFT)**: *같은 본체* 가 이제 instruction 을 *따라* — 질문에 *대답하는* 구조로 응답. 짧은 SFT (1 epoch, 약 3K 샘플) 만으로도 *행동의 방향* 이 바뀝니다

> **핵심**: 본체는 *한 토큰도 바꾸지 않은 같은 125M KoGPT2* 입니다 (continual pretraining 처럼 *데이터만* 바뀐 게 아니라, *데이터 형식 + 마스킹 자리* 가 바뀌었습니다). 그 결과 *모델의 행동 자체* 가 instruction 을 따르도록 정렬됐습니다. **이게 *왜 GPT 하나가 모든 task 를 해내는가* 의 답** — 입력 프롬프트 형식만 바꾸면 다른 일을 하도록, SFT 가 그 능력을 *깨웠습니다*.

> ⚠️ KoGPT2 는 125M 의 *작은* 모델이고 SFT 데이터·시간도 작아서, 답변 품질 자체는 거칠 수 있습니다 (사실 오류, 반복 등). 본 챕터의 관전 포인트는 *답변의 정확도* 가 아니라 ***instruction 을 따라가는 행동 자체가 생겼는가*** 입니다. 품질은 *더 큰 모델 + 더 많은 데이터 + LoRA* 로 끌어올립니다 (FAQ 참고).""")

# ----- 15. 변형 -----
md(r"""## 🛠️ 변형 — 더 많은 데이터 / 다른 response_template / LoRA

본 챕터에서 다루지 못한 변형들 — 직접 시도해 보고 싶다면 아래를 출발점으로:

### 변형 1. 더 많은 데이터 / epoch

```python
# N_SFT = 10000           # subset 확대 (T4 시간 증가 주의)
# sft_config.num_train_epochs = 3   # SFT 는 1-3 epoch 표준
# 더 많은 instruction 다양성 -> instruction following 능력 향상
```

### 변형 2. 다른 response_template

```python
# RESPONSE_TEMPLATE = "### Answer:\n"   # 영어 마커
# RESPONSE_TEMPLATE = "<|assistant|>\n" # chat-style 마커
# response_template 은 '답변 시작 경계' 표시일 뿐 - 데이터에 일관되게만 등장하면 됨.
# 단 본문과 충돌하지 않는 특수 문자열이어야 (collator 가 input_ids 안에서 이걸 찾음).
```

### 변형 3. LoRA / QLoRA — 더 큰 모델 SFT

```python
# from peft import LoraConfig
# peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"],
#                          lora_dropout=0.05, task_type="CAUSAL_LM")
# trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_ds,
#                      processing_class=tokenizer, peft_config=peft_config)
# 본체 weight 는 freeze, 작은 adapter 만 학습 -> 메모리 대폭 절감.
# 7B 급 모델 SFT 의 실무 표준 (QLoRA 는 4bit 양자화까지 더함). 본 커리큘럼 범위 밖.
```""")

# ----- 16. 등장 라이브러리 -----
md(r"""## 📦 이번 챕터에 등장한 라이브러리·함수

| 이름 | 한 줄 설명 | Ch 27 과 차이 |
|---|---|---|
| `trl.SFTTrainer` | SFT 특화 trainer (prompt/completion 전처리 + completion 마스킹 자동) | **새로 등장** (Ch 27 은 `transformers.Trainer`) |
| `trl.SFTConfig` | `SFTTrainer` 설정 (`TrainingArguments` 상속 + SFT 옵션) | **새로 등장** |
| `SFTConfig(completion_only_loss=True)` | *답변 부분만* loss (prompt = `-100`) — SFT 의 핵심 옵션 | **새로 등장** |
| `trl` 의 SFT collator (`DataCollatorForLanguageModeling`) | `completion_mask` 로 prompt 를 `-100` 마스킹 | **새로 등장** (Ch 27 은 `transformers.DataCollatorForLanguageModeling(mlm=False)`) |
| `prompt` / `completion` 데이터 형식 | instruction-response 쌍 표준 형식 | **새로 등장** (Ch 27 은 단일 `text` 컬럼) |
| `AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")` | KoGPT2 본체 로드 | **공유** (Ch 27 과 같은 본체) |
| `PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", ...)` | KoGPT2 BBPE 토크나이저 (AutoTokenizer 함정 회피) | **공유** (Ch 27 과 동일) |
| `model.generate(repetition_penalty=...)` | 반복 억제 sampling (작은 모델의 반복 완화) | **약간 다름** (반복 페널티 추가) |

> `trl` 은 버전마다 API 변동이 큰 라이브러리입니다 (`DataCollatorForCompletionOnlyLM` 처럼 버전에 따라 사라진 클래스도 있습니다). 본 노트북은 *`prompt`/`completion` 데이터 + `completion_only_loss=True`* 라는 *최신 trl 의 표준 경로* 를 씁니다 — 이 경로가 버전 간 가장 안정적입니다. 설치된 `trl` 버전은 셋업 셀의 출력에서 확인하세요.""")

# ----- 17. 체크포인트 -----
md(r"""## 🎯 체크포인트 질문

1. SFT 에서 *왜 prompt 부분을 `-100` 으로 가리나요?* 만약 prompt 도 학습 대상에 포함하면 (response-only 가 아니라 전체 학습) 모델은 무엇을 강화하게 되고, 왜 그게 instruction following 에 불리한가요?
2. *Continual pretraining (Ch 27)* 과 *SFT (Ch 28)* 는 *같은 본체 + 같은 next-token CrossEntropyLoss* 를 씁니다. 그런데도 둘은 다른 단계입니다 — 정확히 *무엇 두 가지* 가 다른가요? (데이터 형식 / `labels = -100` 자리)
3. `### 응답:\n` (response_template) 의 역할은 무엇인가요? collator 는 이 문자열을 어떻게 사용해 prompt 와 답변을 나누나요?
4. *"같은 모델이 입력 프롬프트만 바꾸면 다른 task 를 한다"* — BERT 시대 (task 마다 새 head) 와 비교해 이게 왜 가능한가요? SFT 가 그 능력에서 하는 역할은?""")

# ----- 18. FAQ -----
md(r"""## ❓ FAQ

### Q1. (이론) SFT 가 continual pretraining (Ch 27) 과 정확히 뭐가 다른가요?

*같은 본체, 같은 next-token `CrossEntropyLoss`* 를 쓴다는 점은 같습니다. 다른 건 **두 가지**:

| 항목 | Continual pretraining (Ch 27) | SFT (Ch 28) |
|---|---|---|
| **데이터 형식** | 연속된 일반 텍스트 (TinyStories) | **instruction-response 쌍** (KoAlpaca) |
| **`labels = -100` 자리** | pad 만 (거의 모든 자리 학습) | **prompt 부분 전체** (답변만 학습) |

```python
# Ch 27 (continual pretraining) - 거의 모든 자리 학습
# collator: labels = input_ids.clone()  (pad 만 -100)

# Ch 28 (SFT) - prompt 는 -100, 답변만 학습
SFTConfig(completion_only_loss=True)   # collator 가 prompt 를 자동 -100 마스킹
```

continual pretraining 은 *모델의 지식·도메인* 을 다듬고, SFT 는 *모델의 행동 (지시를 따르는 법)* 을 정렬합니다. 그래서 실무에서는 보통 *pretraining → (continual pretraining) → SFT → alignment* 순서로 *쌓아* 갑니다.

### Q2. (이론) 왜 prompt 를 `-100` 으로 가리나요? 가리지 않으면 어떻게 되나요?

*질문을 외우는 것* 과 *답하는 법을 배우는 것* 의 차이입니다. prompt 도 학습 대상에 넣으면:

1. loss 의 *상당 부분* 이 *prompt 토큰* 에서 나옵니다 (prompt 가 보통 답변보다 길거나 비슷). 모델은 *질문을 받아쓰는* 데 gradient 를 씁니다
2. 모델이 *주어진 질문 분포* 에 과적합 — *새로운 질문* 에 약해질 수 있습니다
3. 우리가 원하는 *"질문이 주어졌을 때 답하는 능력"* (조건부 생성) 이 희석됩니다

`labels[:prompt_len] = -100` 한 줄로 *prompt 는 조건 (given), 답변만 학습 대상 (target)* 으로 분리합니다. §3 에서 본 collator 출력이 정확히 이 효과입니다.

```python
# 개념적으로 (collator 가 자동으로 해 주는 일)
input_ids = tokenizer(prompt + response)["input_ids"]
labels = input_ids.copy()
prompt_len = len(tokenizer(prompt)["input_ids"])
labels[:prompt_len] = [-100] * prompt_len   # <- prompt 를 loss 에서 제외
```

### Q3. (실무) `SFTTrainer` 는 `transformers.Trainer` 와 뭐가 다른가요?

`SFTTrainer` 는 `transformers.Trainer` 를 *상속* 한 서브클래스입니다 — 학습 루프 (forward / backward / optimizer step) 는 *완전히 동일*. 다른 건 *데이터 전처리를 자동화* 한다는 점:

- *prompt + completion* 을 토큰화해 이어 붙이고, 답변 끝에 **EOS 를 자동 부착**
- `completion_only_loss=True` 면 *completion 마스킹* (`completion_mask` 생성 → prompt `-100`) 을 자동
- (옵션) `packing`, chat template 적용 등 SFT 편의 기능

```python
# transformers.Trainer (Ch 27) - 직접 토큰화 + group_texts + collator 설정
# trl.SFTTrainer (Ch 28) - prompt/completion 데이터만 주면 위 과정 자동
trainer = SFTTrainer(model=model, args=SFTConfig(completion_only_loss=True),
                     train_dataset=sft_ds, processing_class=tokenizer)
```

즉 *같은 학습 루프, 더 적은 보일러플레이트*. SFT 의 *마스킹 같은 디테일* 을 라이브러리가 처리해 줍니다.

### Q4. (이론) chat template 이 뭔가요? KoGPT2 는 왜 직접 포맷하나요?

**chat template** 은 *대화 메시지 (`{"role": "user", "content": ...}`) 를 모델이 학습한 형식의 문자열로 변환하는 규칙* 입니다. instruction-tuned 모델 (예: Llama-Instruct, Qwen-Chat) 은 토크나이저에 chat template 이 내장돼 있어 `tokenizer.apply_chat_template(messages)` 한 줄로 포맷됩니다.

**KoGPT2 는 *base 모델* (instruction tuning 안 됨) 이라 chat template 이 없습니다.** 그래서 우리가 *직접* 포맷합니다:

```python
def build_prompt(instruction):
    return f"### 명령어:\n{instruction}\n\n### 응답:\n"
```

> *우리가 정한 포맷* (`### 명령어:` / `### 응답:`) 으로 SFT 하면, *추론 시에도 같은 포맷* 으로 입력해야 합니다. SFT 가 *그 포맷을 모델의 chat template 으로* 가르치는 셈입니다. instruction-tuned 모델을 *직접 만드는* 과정이 곧 *그 모델의 chat template 을 정의* 하는 일입니다.

### Q5. (실무) 더 큰 모델을 SFT 하려면? LoRA / QLoRA 는 무엇인가요?

KoGPT2 (125M) 는 T4 에서 full fine-tuning 이 가능하지만, *7B 급 이상* 은 full SFT 가 *T4 메모리 (16GB) 를 초과* 합니다. 그래서 실무 표준은 **LoRA** (Low-Rank Adaptation):

- 본체 weight 는 *freeze* (그대로 두고)
- 각 layer 에 *작은 low-rank adapter 행렬 (`r=8-64`)* 만 추가해 *그것만 학습*
- 학습 파라미터가 *전체의 약 0.1-1%* → 메모리·시간 대폭 절감

**QLoRA** 는 여기에 *본체를 4bit 양자화* 까지 더해 *더 큰 모델 (예: 70B) 도 단일 GPU* 에서 SFT 가능하게 합니다.

```python
from peft import LoraConfig
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"],
                         lora_dropout=0.05, task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, args=sft_config, train_dataset=sft_ds,
                     processing_class=tokenizer, peft_config=peft_config)
```

본 챕터는 *full SFT* (LoRA 없이) 로 *마스킹의 원리* 에 집중했습니다. LoRA 는 *메모리 기법* 일 뿐 *마스킹·loss 원리는 동일* 합니다.

### Q6. (실무) SFT 후 답변 품질이 거친데요? (반복, 사실 오류)

KoGPT2 는 *125M 의 작은 base 모델* 이고, 본 챕터의 SFT 는 *약 3K 샘플 / 1 epoch* 로 *최소 규모* 입니다. 그래서:

- **반복** — `model.generate(repetition_penalty=1.3)`, `no_repeat_ngram_size=3` 등으로 완화
- **사실 오류 / 환각** — 작은 모델의 근본 한계. 더 큰 모델 + RAG (검색 증강) 로 보완
- **포맷 일관성** — 더 많은 데이터 / epoch 로 개선

> 본 챕터의 목표는 *답변의 정확도* 가 아니라 ***instruction 을 따라가는 행동 자체가 생겼는가*** 입니다. §6 의 BEFORE/AFTER 에서 *지시를 따르는 방향* 으로 바뀌었다면 SFT 의 핵심 (행동 정렬) 은 성공한 것입니다. 품질은 *모델 크기 + 데이터 + LoRA* 의 영역.

### Q7. (이론) 다음 단계 alignment (DPO, Ch 30) 는 SFT 와 뭐가 다른가요?

SFT 는 *"좋은 답변 하나" 를 따라 학습* 합니다 (정답 demonstration 모방). 하지만 *"여러 답변 중 어느 게 더 나은가"* 라는 *선호 (preference)* 는 가르치지 못합니다. **alignment (DPO 등)** 가 그 단계:

| 단계 | 데이터 | 학습 신호 |
|---|---|---|
| **SFT (Ch 28)** | instruction → *하나의* 좋은 답변 | 그 답변을 *따라 생성* |
| **DPO (Ch 30)** | instruction → *(chosen, rejected) 쌍* | chosen 을 *더 선호*, rejected 를 *덜 선호* |

> 흥미롭게도 DPO 도 *`labels = -100` thread 를 잇습니다* — chosen / rejected 각각의 *response 부분에서만* log-likelihood 를 계산합니다 (prompt 는 양쪽 공통이라 제외). 즉 *답변 부분만 본다* 는 본 챕터의 원리가 alignment 까지 이어집니다. DPO 는 Ch 30 에서 본격.

```python
# Ch 30 미리보기 (DPO)
# from trl import DPOTrainer, DPOConfig
# 데이터: {"prompt": ..., "chosen": ..., "rejected": ...}
# chosen 의 response 확률은 높이고, rejected 의 response 확률은 낮춤
```""")

# ----- 19. 다음 챕터 예고 -----
md(r"""## 다음 챕터 예고

**Chapter 29. 벤치마크 평가 — SFT 모델을 분야별 벤치마크로 측정**

- 본 챕터에서 만든 *SFT 모델* 을 *정량 벤치마크* 로 평가: 한국어 (KMMLU / HAERAE-Bench / LogicKor / KoBEST) + 영어 (MMLU / HellaSwag / GSM8K ...)
- `lm-evaluation-harness` 로 *task-format 별* 자동 평가 — *instruction following 이 점수로 드러나는가*
- *SFT 전 (base) vs SFT 후* 벤치마크 비교 — §6 의 정성적 BEFORE/AFTER 를 *정량* 으로
- 그 다음 Ch 30 (DPO) — *preference 정렬*. **`labels = -100` thread 가 DPO 에서도 이어집니다** — chosen/rejected *response 부분만* 계산

**Phase 4 GPT 시대 4단계 흐름 정리**:

| 챕터 | 단계 | 본체 | 데이터 | `labels = -100` 자리 |
|---|---|---|---|---|
| Ch 24·26 | 1 (pretraining) | 작은 GPT scratch | TinyStories (영/한) | pad 만 |
| Ch 25·27 | 2 (continual pretraining) | gpt2 / KoGPT2 | TinyStories (동일) | pad 만 |
| **Ch 28 ← 여기** | **3 (SFT)** | **KoGPT2 (동일)** | **KoAlpaca instruction-response** | **prompt 부분 (답변만 학습)** |
| Ch 30·31 | 4 (alignment) | SFT 모델 + ref | preference / verifier reward | response 부분만 (RL 내부) |

> **변하는 축** (Ch 27 → Ch 28): *학습 단계* (continual pretraining → SFT). 본체·토크나이저·loss 종류는 같고, *데이터 형식 + `labels = -100` 자리* 가 바뀝니다. 본 챕터에서 그 *마스킹 자리* 를 collator 출력으로 *눈으로 확인* 했습니다 — 그게 Phase 4 두 thread 의 클라이맥스입니다.""")


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
README = """# 28_sft — KoGPT2 SFT / Instruction Tuning (Phase 4 학습 단계 3)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/28_sft/28_sft.ipynb)

## 한 줄 목표
Ch 27 (KoGPT2 continual pretraining) 의 *다음 단계*. 같은 **KoGPT2 (`skt/kogpt2-base-v2`, 125M)** 본체를 **KoAlpaca instruction-response 쌍 데이터** 로 **SFT (Supervised Fine-Tuning / Instruction Tuning)** 합니다. 본체도 같고 loss 종류 (next-token `CrossEntropyLoss`) 도 같습니다 — 바뀌는 건 *데이터 형식 (연속 텍스트 → instruction-response 쌍)* + *`labels` 마스킹 (pad 만 → prompt 부분 전체)* + *trainer (`Trainer` → `trl.SFTTrainer`)*. **그 한 줄 `labels[:prompt_len] = -100` 이 모델을 지시를 따르게 만듭니다** — *행동 정렬 (behavior alignment)*.

## 이 챕터가 클라이맥스인 두 thread

### Thread 1 — `labels = -100` 자리의 완성
| 단계 | 챕터 | `labels = -100` 자리 | loss 계산 |
|---|---|---|---|
| MLM | Ch 20·21·22 | 선택 안 된 약 85% | 가린 약 15% 만 |
| CausalLM | Ch 24·25·26·27 | pad 만 | 거의 전 토큰 |
| **SFT (본 챕터)** | **Ch 28** | **prompt 부분 전체** | **답변 토큰만** |

§3 에서 `trl` collator 의 출력 `labels` 를 *토큰별 표* 로 직접 print — prompt 가 `-100`, 답변만 원본 token id 인 것을 *눈으로 확인*. (Ch 21 `[MASK]` 80/10/10 시각화의 SFT 판.)

### Thread 2 — "파인튜닝" 의미 변화의 완성 (behavior alignment)
- BERT 파인튜닝 (Ch 9-23) = task head 부착 (task 적응)
- GPT continual pretraining (Ch 25·27) = head 그대로, 같은 task, 새 데이터
- **GPT SFT (Ch 28) = head 그대로, 같은 next-token CE, 단 *데이터 형식 + prompt 마스킹* → *행동 정렬*** ← *진짜* instruction following

> *같은 모델이 입력 프롬프트만 바꾸면 다른 task* — 그게 *왜 GPT 하나가 모든 task 를 해내는가* 의 답. SFT 가 그 능력을 *깨우는* 단계.

## GPT 시대 학습 4단계 — 본 챕터의 위치

| 단계 | 용어 | 본 챕터? | 본 커리큘럼 |
|---|---|---|---|
| 1 | Pretraining | | Ch 24 (영어), Ch 26 (한국어) |
| 2 | Continual pretraining | | Ch 25 (영어), Ch 27 (한국어) |
| **3** | **SFT (Instruction tuning)** | **✅ ← 여기** | **Ch 28** |
| 4 | Alignment (DPO / GRPO) | | Ch 30·31 |

## 다루는 핵심 개념
- **`trl.SFTTrainer` + `trl.SFTConfig`** — SFT 특화 trainer (첫 등장). `transformers.Trainer` 를 상속, *prompt/completion 전처리 + EOS 부착 + completion 마스킹* 자동
- **`SFTConfig(completion_only_loss=True)`** — *답변 부분만* loss, prompt 는 `-100`. SFT 의 핵심 옵션
- **`prompt` / `completion` 데이터 형식** — instruction-response 쌍 표준. RESPONSE_TEMPLATE `### 응답:\\n` 가 답변 시작 경계
- **collator labels 마스킹 시각화** (§3, 클라이맥스) — 토큰별 `position | token | input_id | label | learn?` 표로 prompt=`-100`, 답변만 학습 직접 확인
- **SFT 전·후 instruction following 비교** (§6, 핵심 데모) — 같은 본체가 *지시를 따르는 방향* 으로 행동 정렬
- **`AutoModelForCausalLM` + `PreTrainedTokenizerFast`** — Ch 27 과 같은 KoGPT2 본체·토크나이저 (AutoTokenizer 함정 회피)
- **chat template** 의 의미 — KoGPT2 는 base 모델이라 없어 직접 포맷

## Loss
next-token `CrossEntropyLoss` — *Ch 27 과 같은 종류*. 다른 건 *어느 자리에서 합산하느냐*. SFT 는 *답변 토큰만* 합산 (prompt 는 `-100` 으로 제외).

수식: $L_{\\text{SFT}} = -\\sum_{i \\in \\text{response}} \\log P(x_{i+1} \\mid x_{\\leq i})$  (prompt 부분 제외)

## 데이터
`beomi/KoAlpaca-v1.1a` — 한국어 instruction tuning 데이터셋 (`instruction` / `output` 필드). 약 3,000 샘플 subset, `### 명령어:\\n{instruction}\\n\\n### 응답:\\n{output}` 로 직접 포맷.

## 모델
**`AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")`** — *Ch 27 과 같은 KoGPT2 본체* (125M params). LM head 그대로, next-token task 그대로.

## Hyperparams
- `num_train_epochs=1`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch 16)
- `learning_rate=2e-5` (SFT 표준), `lr_scheduler_type="cosine"`, `warmup_ratio=0.03`
- `max_length=512`, `completion_only_loss=True`, `packing=False`
- `fp16=True` (T4 는 bf16 불가)

## 라이브러리 주의 — `trl` 버전
`trl` 은 버전마다 API 변동이 큽니다 (`DataCollatorForCompletionOnlyLM` 처럼 버전에 따라 사라진 클래스도 있음). 본 노트북은 *`prompt`/`completion` 데이터 + `completion_only_loss=True`* 라는 *최신 trl 의 표준 경로* 를 사용 — 버전 간 가장 안정적. 설치된 `trl` 버전은 셋업 셀 출력에서 확인하세요.

## 환경
Google Colab **T4 GPU 필수**. 약 20-28분 (KoAlpaca 로드·포맷 약 2분 + KoGPT2 로드 약 2분 + collator 시각화 약 1분 + SFT 약 12-18분 + SFT 전·후 비교 약 3분).

device 자동 감지 (CUDA / MPS / CPU) — 로컬 Mac MPS 에서도 실행 가능 (학습 시간 약 2-3배 증가).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | `labels = -100` 자리 | Loss |
|---|---|---|---|---|---|
| 26 | 작은 GPT2 (한국어, 약 3M, scratch) | BBPE (직접 학습, vocab 약 4,000) | 한국어 TinyStories 30K | pad 만 | CE (next-token) |
| 27 | KoGPT2 (125M) | BBPE (KoGPT2 그대로, vocab 51,200) | 한국어 TinyStories 30K | pad 만 | CE (next-token) - continual pretraining |
| **28** | **KoGPT2 (125M, 동일)** | **BBPE (KoGPT2 그대로, 동일)** | **KoAlpaca instruction-response (약 3K)** | **prompt 부분 (답변만 학습)** | **CE (next-token, response-only) — SFT** |
| 29 (다음) | Ch 28 SFT 모델 + 비교 | (동일) | 분야별 벤치마크 | - (평가만) | - (`lm-evaluation-harness`) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표) 를 참고하세요.

## 다음 챕터
[29_benchmark_eval](../29_benchmark_eval/) (예정) — 본 챕터의 SFT 모델을 *분야별 벤치마크* (KMMLU / HAERAE / MMLU ...) 로 *정량* 평가. §6 의 정성적 BEFORE/AFTER 를 점수로. 그 다음 [30_dpo](../30_dpo/) (DPO alignment) — *preference 정렬*. `labels = -100` thread 가 DPO 에서도 chosen/rejected *response 부분만* 계산으로 이어집니다.
"""

OUT_README.write_text(README, encoding="utf-8")
print(f"Wrote {OUT_README.relative_to(REPO)}")
