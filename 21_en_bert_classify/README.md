# 21_en_bert_classify — 작은 BERT 분류 (영어 Yelp 이진, scratch 사전학습 + fine-tune)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/21_en_bert_classify/21_en_bert_classify.ipynb)

## 한 줄 목표
Phase 3 의 세 번째 챕터. Ch 20 에서 *작은 BERT 를 직접 MLM 사전학습* 했다면, 이번엔 그 위에 **분류 헤드를 얹어 fine-tune**. Ch 10 (DistilBERT, 약 66M params, 수십억 토큰 사전학습) 과 같은 Yelp 이진 분류 셋업에 *우리가 만든 작은 BERT* (약 10M params, Yelp 5K 문장 MLM) 를 붙여 두 결과를 나란히 비교 — *사전학습 규모* 가 downstream 정확도에 얼마나 차이를 만드는지 정량으로.

self-contained 노트북: Ch 20 의 MLM 학습을 1 epoch 짧게 재현 → 같은 본체로 분류 fine-tune → Ch 10 결과와 비교. 본문은 *사전학습 → 분류 fine-tune* 메인 흐름에 집중. *사전학습 없이 같은 GPU compute 로 분류 fine-tune* 만 했을 때의 fair-compute 비교는 부록 노트북 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 에서 분리해 다룹니다.

## 다루는 핵심 개념
- `BertForMaskedLM` → `BertForSequenceClassification` 헤드 교체 — 본체 (`embeddings + encoder + pooler`) 는 그대로, MLM head 떼고 분류 head (`Linear(256, 2)`) 부착
- in-memory state_dict 전송: `cls_model.bert.load_state_dict(mlm_model.bert.state_dict())` — 디스크 없이 본체 가중치 복사
- 같은 `BertConfig` (hidden=256, layer=4, head=4, intermediate=1024, 약 10M params) 가 MLM 모델과 분류 모델 양쪽에 적용
- 사전학습 효과의 *순 측정* — random init baseline 과 비교
- **Ch 10 (DistilBERT 대규모 사전학습) vs Ch 21 (작은 BERT 자체 사전학습)** 의 정량 비교

## Loss
`CrossEntropyLoss` — 분류 fine-tune 표준 (K=2, softmax + CE). 라벨은 `int 0/1`, `problem_type="single_label_classification"`. random baseline loss = `ln(2) ≈ 0.693`.

수식: $L = -\frac{1}{N}\sum_{i=1}^{N} \log \hat p_{i, y_i}$ — Ch 11/15 와 같은 K-class softmax CE.

## 데이터
`fancyzhx/yelp_polarity` 이진 분류 (label 0/1, 5점 척도 자동 이진화). 5,000 train / 1,000 eval, seed 42 — Ch 10 과 같은 split.

MLM 사전학습 단계에서는 같은 5,000 문장의 *text 만* (라벨 무시) 사용해 `block_size=128` `group_texts` 패턴으로 1 epoch 학습.

## 환경
Google Colab T4 GPU (fp16). 약 25분 (MLM 1 epoch 약 10-12분 + 분류 fine-tune 2 epoch 약 8-10분 + 평가/시각화 약 2분).

## 변화 추적

| Ch | 모델 | 토크나이저 | 데이터 | Output | Loss |
|---|---|---|---|---|---|
| 10 | DistilBERT 파인튜닝 (약 66M) | `bert-base-uncased` WordPiece | Yelp 이진화 | `Linear(H, 1)` | `BCEWithLogitsLoss` |
| 19 | — (토크나이저 학습 전용) | WordPiece + WordLevel (둘 다 직접 학습) | Yelp text + NSMC text | — | — |
| 20 | 작은 BERT (직접, scratch) | `bert-base-uncased` 토크나이저 (가져옴) | Yelp text (라벨 무시) | MLM head | `CrossEntropyLoss` (masked) |
| **21** | **Ch 20 사전학습 BERT + 분류 헤드 (약 10M)** | (Ch 20과 동일) | **Yelp 이진화** | **`Linear(H, 2)`** | **`CrossEntropyLoss`** |
| 22 (다음) | 작은 BERT (직접, scratch) — 한국어 | `klue/bert-base` 토크나이저 (가져옴) | NSMC text | MLM head | `CrossEntropyLoss` (masked) |

전체 챕터 표는 [루트 README](../README.md#챕터별-변화추적표)를 참고하세요.

## 비교 표 — Ch 10 vs Ch 21

| 차원 | Ch 10 (DistilBERT) | Ch 21 (small BERT scratch) | 비고 |
|---|---|---|---|
| 본체 파라미터 | 약 66M | 약 10M | Ch 21 은 1/6 작음 |
| 사전학습 코퍼스 | Wikipedia + BookCorpus (약 33억 토큰) | Yelp 5K 문장 (약 70만 토큰) | 약 5000배 격차 |
| 사전학습 시간 | TPU 수일 | T4 약 10-12분 | |
| 분류 fine-tune 셋업 | (같음 — 5K/1K, batch 16, lr 2e-5, 2 epoch, fp16) | | 본체 외 통제 |
| 기대 accuracy | 약 92-95% | 약 75-85% | 비교는 실측치로 |

격차가 *사전학습 규모의 가치* 를 정량으로 보여줍니다. *작은 사전학습도 random init 보다는 분명히 낫다* 는 것, 그리고 *fair-compute (사전학습 compute 를 fine-tune 으로 옮겨도)* 격차가 메워지지 않는다는 것은 부록 [`appendix_compute_budget.ipynb`](./appendix_compute_budget.ipynb) 참조.

## 다음 챕터
[22_ko_bert_pretrain](../22_ko_bert_pretrain/) — Ch 20 의 영어 사전학습 패턴을 한국어로 재현. 같은 작은 BertConfig + `klue/bert-base` 토크나이저 + NSMC text MLM. Ch 22 → Ch 23 (한국어 분류) 가 이번 챕터 → Ch 21 (영어) 와 *대칭*.
