# Codex 핸드오프 노트

이 파일은 Claude → codex 로의 의도적 핸드오프 메시지를 담습니다. codex가 master 를 sync 한 뒤 한 번 훑어보면 됩니다. 처리 끝난 항목은 사용자(또는 codex)가 지워도 무방.

---

## 2026-05-04 — sklearn 1.5+ API 호환성 정리

**무엇이 바뀌었나** — 챕터 노트북·빌드 스크립트·폴더 README 전반에서 *deprecated/제거* 된 sklearn 인자 사용 제거. 코드는 모던 API로 동작하도록 정리하고, 설명문에서도 *역사적 deprecation 언급* 을 모두 빼고 *현재 동작* 만 풀어쓰기로 통일.

**영향 받은 챕터**: Ch 3 (예고문), Ch 4 (코드+FAQ), Ch 5 (코드+설명), Ch 6 (코드+FAQ Q7), Ch 7·10·11 (추적표 표기), Ch 11 (FAQ Q4 phrasing)

**핵심 룰 (앞으로 모든 챕터에 적용)**:

- `LogisticRegression(multi_class="multinomial")` ❌ → `LogisticRegression()` ✅
  - 이유: sklearn 1.5+ 에서 `multi_class` 인자가 `LogisticRegression` 에서 deprecated 됐고 1.7+ 에선 완전 제거. 모던 sklearn 은 데이터의 클래스 개수(K=2 binary / K≥3 multi-class)로 자동 분기.
  - 코드만 바꾸고 *왜 이 인자를 안 쓰는지의 역사적 설명은 노출하지 않음*.
- `LogisticRegression(multi_class="ovr")` ❌ → `OneVsRestClassifier(LogisticRegression())` ✅
  - OvR 학습은 wrapper class 로만 표현됨. multi-class·multi-label 양쪽에 통하는 표준 패턴.
- `roc_auc_score(..., multi_class="ovr")` ✅ — *이건 다른 API* 로 sklearn 1.8 도 지원. **변경 없음**.

**책(book/) 측 작업 안내**:

- 챕터 .tex 재생성 (`book/tools/notebook_to_tex.py`) 한 번 돌리시면 새 노트북 본문이 자동 반영됩니다.
- 주의 깊게 봐야 할 자리:
  - Ch 4 §실습 코드 블록 (방식 A/B 학습 셀) — 주석이 짧아짐
  - Ch 4 FAQ Q5 — 제목·본문 모두 다시 쓰여짐 ("multi_class 인자는 왜 안 쓰나요?" → "softmax 와 OvR 을 어떻게 구분하나요?")
  - Ch 6 FAQ Q7 — 같은 톤으로 단순화. 두 도구 비교 표가 한 컬럼으로 줄어듦
  - Ch 11 FAQ Q4 — phrasing 깔끔해짐
  - 추적표 행: `LogisticRegression()` (multinomial 자동) 형태로 통일

**검증**: 로컬 venv (sklearn 1.8) + nbconvert 로 Ch 3-6 모두 정상 실행 확인. Colab 환경(구 sklearn) 에서도 그대로 돌아감 — 모던 API 가 구버전과 호환되는 *상위 호환* 변경이라.

---

## 2026-05-06 — 용어 통일: "측면" → "항목" + Ch 13 해석 예시 셀 추가

**무엇이 바뀌었나** — Yelp aspect-based multi-label 데이터를 가리키던 *측면* 단어를 전부 **항목** 으로 일괄 치환. "측면" 이 학술 직역이라 한국어 산문에서 어색하다는 사용자 피드백.

**바뀐 단어**: 89건 일괄 치환 across 14개 파일 (빌드 스크립트 + 챕터 README + 루트 README). 코드 식별자 (`ASPECT_KEYWORDS`, `ASPECTS`, `extract_aspects()`) 는 영어로 그대로 둠 — NLP 표준 용어 + 영문 식별자 vs 한국어 산문 분리 원칙.

**영향 받은 챕터**: 5, 6, 7, 9 (+appendix), 12, 13, 14 + 챕터 폴더 README + 루트 README.

**일괄 치환 후 *어색한 자리* 3건 별도 보정**:

1. Ch 6 §삽질 — "항목이 가장 강한 것 *하나* 만" (어순 어색) → "*가장 강하게 활성된 항목 하나* 만"
2. Ch 14 FAQ Q4 — "긍정 항목 vs 부정 항목" (의미 부정확, 항목 detection task 인데 sentiment 처럼 들림) → "Ch 13 의 5개 항목 multi-label 셋업 그대로"
3. Ch 13 §평가 사이에 추가한 해석 예시 셀 도입부 — 번호 (4-0) 대신 무번호 제목 (앞뒤 4-1·4-2 와 충돌 안 나도록)

**Ch 13 신규 셀 — "샘플 단위 해석 — 모델 출력을 읽어내는 법"**:

§4 평가 metric 출력 직후, §4-1 KDE 시각화 직전에 새 markdown + code + markdown 3셀 삽입. 평가 셋에서 정답 항목이 *가장 많은* 샘플 + *가장 적은* 샘플 1개씩 골라 *true / prob / pred / match* 표를 출력하고, "predicted: [...]" / "true: [...]" 사람이 읽는 한 줄까지 보여 줌. 5차원 multi-hot 출력을 *문장 단위로* 어떻게 해석하는지 step-by-step 설명. metric 의 micro/macro F1 이 결국 이 샘플별 비교의 집계라는 점을 명시.

**책(book/) 측 작업 안내**:

- 노트북 다수에서 마크다운 본문이 변경됐으니 .tex 재생성 필수.
- 주의 깊게 봐야 할 자리:
  - 모든 챕터 본문에서 "측면" 단어가 한 곳도 남지 않았는지 .tex 에서 grep 으로 cross-check.
  - Ch 13 에 새로 추가된 *샘플 단위 해석* 섹션은 §4 안에 들어가는 새 figure-caption 단위가 아니라 그냥 본문 텍스트 + listing — 새 figure 로 잘못 잡지 않도록.
  - Ch 14 FAQ Q4 의 짧은 phrase 변경 (긍정/부정 → multi-label 셋업).

**검증**: 17개 노트북 lint clean. Ch 13 추가 셀은 기존 변수 (`labels`, `probs`, `preds`, `eval_full`) 만 재사용해 새 의존성 없음 — Colab T4 / 로컬 venv 양쪽 그대로 동작.

---

(이 파일은 _drafts/ 안에 있어 출판물·노트북에 노출되지 않습니다. codex 가 확인 끝낸 항목은 지우거나 'done' 표시 해 주시면 됩니다.)
