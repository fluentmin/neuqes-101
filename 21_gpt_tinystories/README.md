# 21_gpt_tinystories — GPT (HF API) on TinyStories  *[prototype]*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_gpt_tinystories/21_gpt_tinystories.ipynb)

> ⚠️ **prototype 단계입니다.** 한 노트북에 *셋업 + 학습 + 생성* 을 압축해, T4 30분 안에 GPT 가 정상 동작하는지 확인하는 용도입니다. 검증이 통과하면 정식 챕터 Ch 21 (해부) / Ch 22 (본격 학습) / Ch 23 (from-scratch `nn.Module` 비교) / Ch 24 (한국어) 로 분할 예정.

## 한 줄 목표
BERT 챕터들과 *완전히 같은* `Trainer` 패턴으로, 모델 클래스만 `BertForMaskedLM` → **`GPT2LMHeadModel`** 으로 바꿔 **decoder-only GPT** 를 *from scratch* 학습. TinyStories 작은 subset 에서 grammatical 한 영어 동화를 생성합니다.

## 다루는 핵심 개념
- **`GPT2LMHeadModel(config)`** — `from_pretrained` 없이 무작위 초기화 from scratch
- **`GPT2Config`** 의 핵심 필드 — `n_layer / n_head / n_embd / n_positions`, `tie_word_embeddings` (자동 True)
- **causal attention** — encoder(BERT) 와의 본질적 차이. 모델 클래스가 내장 처리
- **`DataCollatorForLanguageModeling(mlm=False)`** — labels = input_ids 자동 (BERT 의 `mlm=True` 와 단 한 글자 차이)
- **`group_texts` 패턴** — HF `run_clm.py` 표준 CLM 전처리: 토큰 스트림을 `block_size` 단위로 자름
- **BPE 토크나이저 직접 학습** — `tokenizers.BPE` + ByteLevel pre-tokenizer, vocab 2048
- **GPT-2 special token 컨벤션** — `<|endoftext|>` 하나가 bos / eos / pad 겸용
- **`model.generate(do_sample=True, ...)`** — temperature, top_k, top_p 비교
- Random baseline loss = $\ln 2048 \approx 7.625$, TinyStories 3M 모델은 보통 2.5-3.5 가 한계

## Loss
**`CrossEntropyLoss` (next-token, `mlm=False`)** — BERT MLM 의 CE 와 수식적으로 동일. 마스킹 위치만 다름 (BERT: 무작위 15% / GPT: 모든 토큰의 다음 위치). 모델 forward 안에서 `logits` 와 `labels` 가 한 칸 shift 되어 처리됨.

## 데이터
`roneneldan/TinyStories` (Eldan & Li 2023) — GPT-4 가 4세 어린이 어휘로 생성한 짧은 영어 동화 약 2.1M 편. prototype 은 처음 **30,000편만** 사용 (≈ 4-6M 토큰).

## 모델
**`GPT2LMHeadModel`** with `n_layer=4, n_head=4, n_embd=256, n_positions=128`. 약 **3M params** (weight tying 자동 적용). BERT 챕터들의 DistilBERT (~66M) 와 다르게 *완전 무작위 초기화* 에서 시작.

## Hyperparams
- `max_steps=1500`, `per_device_train_batch_size=32`, `learning_rate=3e-4`
- `lr_scheduler_type="cosine"`, `warmup_steps=100`
- AdamW `betas=(0.9, 0.95)`, `weight_decay=0.1`, `max_grad_norm=1.0`
- `fp16=True` (T4 는 bf16 불가)
- `eval_strategy="steps"`, `eval_steps=150`

## 환경
Google Colab **T4 GPU 필수**. 약 25-30분 (데이터 로드 ~2분 + BPE 학습 ~3분 + 모델 학습 ~18분 + 생성 ~30초).

## 변화 추적

| Ch | 모델 클래스 | 토크나이저 | 데이터 | Collator | Loss |
|---|---|---|---|---|---|
| 7-16 | `BertForXxx` (encoder, `from_pretrained`) | WordPiece (사전학습) | 분류 데이터 | 태스크별 (분류용) | CE / BCE / MSE |
| **21 (prototype)** | **`GPT2LMHeadModel(config)`** (decoder, **from scratch**) | **BPE (직접 학습)** | **TinyStories 30k** | **`DataCollatorForLanguageModeling(mlm=False)`** | **CE (next-token)** |
| 22 (예정) | 같음, 학습 더 길게 | 같음 | TinyStories 키움 | 같음 | 같음 |
| 23 (예정) | 같음을 `nn.Module` 로 재구현 | 같음 | 같음 | 같음 | 같음 |

## prototype 검증 체크리스트
- [ ] T4 에서 25-30 분 안에 끝까지 실행
- [ ] eval loss 가 학습 중 단조감소 (대략 7 → 3-4)
- [ ] 생성된 문장이 grammatical (반복·횡설수설 아님)
- [ ] peak VRAM 이 T4 16 GiB 안에 충분 (예상 < 3 GiB)
- [ ] BPE 토크나이저가 합리적으로 단어를 쪼개는가

## 관련 이슈
[#8 — Ch 21-22 제안: TinyStories + 작은 GPT 로 generation 챕터 추가 (옵션 B)](https://github.com/yoon-gu/neuqes-101/issues/8)
