# 20_en_bert_pretrain — 작은 BERT 직접 사전학습 (영어 MLM scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/20_en_bert_pretrain/20_en_bert_pretrain.ipynb)

## 한 줄 목표
Phase 3 의 두 번째 챕터. Ch 19 에서 *토크나이저* 를 직접 학습해 봤다면, 이번엔 **모델 본체를 random init 해 MLM 사전학습** 합니다. 표준 BERT (110M) 의 1/10 크기 작은 BERT (~10M, hidden=256/layer=4) 를 `BertConfig` 로 직접 설계, `bert-base-uncased` 의 WordPiece 토크나이저는 그대로 가져와 Yelp text 5,000 문장으로 MLM 사전학습 → 체크포인트 저장 → Ch 21 에서 분류 fine-tune.

## 다루는 핵심 개념
- **MLM (Masked Language Modeling)** — 입력 토큰의 15% 를 `[MASK]` 로 가리고 원래 토큰을 맞추는 self-supervised task
- `transformers.BertConfig` 로 작은 BERT 직접 설계 (`hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024`)
- `BertForMaskedLM(config)` 로 *random init* (pretrained weight 없이) — `from_pretrained` 와 반대 흐름
- `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` — 매 batch 마다 동적 masking (80% [MASK] / 10% random / 10% keep)
- `group_texts` 패턴 (HF `run_mlm.py` 표준) — 가변 길이 텍스트를 고정 길이 `block_size=128` 블록 스트림으로
- MLM head 가 입력 임베딩과 *tied* — vocab 차원 출력이라 파라미터 절약
- random baseline loss `ln(vocab_size) ≈ 10.33`, perplexity 로 변환 가능 (`exp(loss)`)
- `model.save_pretrained()` / `tokenizer.save_pretrained()` 로 HF 표준 체크포인트 저장 — Ch 21 에서 `from_pretrained` 로 로드

## Loss
`CrossEntropyLoss` — 가려진 위치들의 *원래 토큰* 을 vocab 30,522 차원 softmax 로 예측. 라벨이 -100 인 위치는 자동 무시 (collator 가 처리).

수식: $L_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid x_{\setminus M})$ — 가려진 토큰 위치 $M$ 에서의 평균 음의 로그 우도.

## 데이터
`fancyzhx/yelp_polarity` train 5,000 문장 + test 500 문장 (eval). *라벨 무시* — MLM 은 self-supervised 라 라벨 불필요.

`block_size=128` 로 `group_texts` 후 train 약 1,000-1,500 블록 / eval 약 100-150 블록.

## 환경
Google Colab T4 GPU (fp16). 약 20-25분 (`bert-base-uncased` 토크나이저 로드 + 토큰화 -2분 + MLM 1-2 epoch ~15-20분 + 평가/저장).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 17 | klue/bert-base | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 multi-label | `Linear(H, 7)` | `BCEWithLogitsLoss` |
| 18 | klue/bert-base + 보조 | WordPiece (한국어, 사전학습) | KLUE-YNAT 합성 + 보조 라벨 | 메인(7) + 보조 | `BCEWithLogitsLoss + λ·L_aux` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — |
| **20** | **작은 BERT (직접, scratch)** | **`bert-base-uncased` 토크나이저 (가져옴)** | **Yelp text (라벨 무시)** | **MLM head** | **`CrossEntropyLoss` (masked)** |
| 21 (다음) | Ch 20 사전학습 BERT + 분류 헤드 | (Ch 20과 동일) | Yelp 이진화 | `Linear(H, 2)` | `CrossEntropyLoss` |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 산출물
`./ch20_small_bert_mlm/` 폴더에 `config.json + model.safetensors + tokenizer.json + vocab.txt + ...` 저장. Ch 21 에서 `AutoModelForSequenceClassification.from_pretrained("./ch20_small_bert_mlm", num_labels=2)` 한 줄로 *encoder body* 를 가져와 새 분류 헤드를 부착해 fine-tune.

## 다음 챕터
[21_en_bert_classify](../21_en_bert_classify/) — 이번 챕터 사전학습 모델을 Yelp 이진 분류로 fine-tune. **Ch 10 (DistilBERT 대규모 사전학습 모델 fine-tune) 과 직접 비교** — 작은 사전학습 BERT (10M, 5K 문장 MLM) vs 표준 사전학습 BERT (66M, 대규모 corpus) 의 정량 격차. random init baseline 도 함께 학습해 *사전학습의 순 효과* 분리.
