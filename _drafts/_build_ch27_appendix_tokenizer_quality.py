"""Build 27_ko_gpt2_continual_pretrain/appendix_korean_tokenizer_quality.ipynb.

Ch 27 부록 — 한국어 토크나이저 품질 분석.

메인 챕터 (Ch 27, KoGPT2 continual pretraining) 는 *토크나이저는 본체와 운명공동체*
라는 점과 *AutoTokenizer 가 KoGPT2 를 영어 GPT2 로 잘못 fallback 하는 함정* 을 다뤘습니다.
이 부록은 그 심화 — *왜 토크나이저가 모델의 한국어 품질을 예측하는 선행 지표인가* 를
여러 다국어 LLM 토크나이저를 실제 숫자로 비교해 보여줍니다.

핵심 지표 4가지:
- Fertility (한국어 문장당 토큰 수, tokens/char): 낮을수록 압축률↑·추론 빠름·context 효율↑
- vocab 한국어 점유율 (가-힣 포함 토큰 %): 높을수록 의미 단위 보존
- 자모/byte 분해 여부: 음절·형태소 = 좋음, 자모/byte 깨짐 = 나쁨
- OOV·신조어·외래어 처리: UNK 없이 자연스러운 subword 면 좋음

비교 토크나이저: KoGPT2 / klue / polyglot-ko / KcELECTRA (한국어 특화) +
Qwen2.5 / mBERT / bloom·xglm (다국어) + tiktoken o200k·cl100k (OpenAI) +
gpt2 (영어 BPE 극단 대조군). 로드 실패 건은 try/except 로 skip.

모델 본체는 로드하지 않고 토크나이저만 다뤄 CPU 로 충분합니다 (T4 metadata 는 Colab
일관성 유지용). 빌더 패턴은 메인 빌더 `_build_ch27.py` 와 같은 cells / _cid / md / code
/ NOTEBOOK json dump 구조이며, 부록 빌더 `_build_ch23_appendix_random_baseline.py` 의
구조를 참고했습니다.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "27_ko_gpt2_continual_pretrain"
OUT_NB = OUT_DIR / "appendix_korean_tokenizer_quality.ipynb"

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
md(r"""# Chapter 27 부록 — 한국어 토크나이저 품질 분석

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoon-gu/neuqes-101/blob/master/27_ko_gpt2_continual_pretrain/appendix_korean_tokenizer_quality.ipynb)

> **부록 한 줄 질문** — *"왜 한국어 LLM 개발자는 토크나이저 vocab 만 봐도 그 모델의 한국어 품질을 어느 정도 예측할까?"*

메인 챕터 ([`27_ko_gpt2_continual_pretrain.ipynb`](./27_ko_gpt2_continual_pretrain.ipynb)) 에서 두 가지를 배웠습니다.

1. **토크나이저는 본체와 운명공동체** — KoGPT2 본체는 자기 토크나이저의 vocab (51,200) 으로만 글을 봅니다. 토크나이저가 한국어를 잘 쪼개야 본체도 잘 배웁니다.
2. **`AutoTokenizer` 가 KoGPT2 를 영어 GPT2 로 잘못 fallback 하는 함정** — 그래서 `PreTrainedTokenizerFast` 로 special token 을 직접 지정해 로드했습니다.

이 부록은 그 심화입니다. 실무에서 한국어 LLM 을 고를 때, 가중치를 받아 직접 추론을 돌려 보기 *전에* 토크나이저 vocab 만 열어 봐도 그 모델의 한국어 품질을 꽤 예측할 수 있습니다. 그 직관을 **여러 다국어 LLM 토크나이저를 실제 숫자로 비교** 해 정량으로 확인합니다.

비교 대상은 한국어 특화 (KoGPT2·KLUE·polyglot-ko·KcELECTRA), 다국어 LLM (Qwen2.5·mBERT·BLOOM/XGLM), 최신 OpenAI (`tiktoken` 의 GPT-4o·GPT-4 토크나이저), 그리고 영어 BPE 극단 대조군 (`gpt2`) 입니다. 본체 가중치는 받지 않고 토크나이저만 다루므로 **GPU 불필요 — CPU 로 충분** 합니다. 약 5-10분 (토크나이저 다운로드 시간).

---""")

# ----- 2. 핵심 질문 -----
md(r"""## 🧭 핵심 질문 — 토크나이저가 왜 품질의 *선행* 지표인가

LLM 의 한국어 품질을 좌우하는 큰 축은 둘입니다.

- **사전학습 데이터** — 얼마나 많은·다양한 한국어를 봤는가 (Ch 22·26·27 의 주제).
- **토크나이저** — 그 한국어를 *어떻게 쪼개* 본체에 넣는가.

데이터는 가중치를 받아 돌려 보기 전엔 알기 어렵지만, **토크나이저는 vocab 파일 하나만 열면 바로 보입니다**. 그리고 토크나이저의 한국어 처리 방식은 본체가 한국어를 *얼마나 효율적으로 학습·추론* 하는지를 직접 제약합니다. 그래서 토크나이저는 *측정하기 쉬우면서도 본질에 가까운* 선행 지표가 됩니다.

이 부록에서 4가지 지표로 그 선행성을 보입니다.

| 지표 | 의미 | 좋은 한국어 토크나이저 |
|---|---|---|
| **Fertility** (tokens/char) | 같은 한국어 문장이 몇 토큰으로 쪼개지나 | **낮음** (압축률↑, 추론 빠름, context 효율↑) |
| **vocab 한국어 점유율** | vocab 중 한글 음절 포함 토큰 % | **높음** |
| **자모/byte 분해** | `옛날` → `옛`/`날` (음절) vs `ㅇㅖㅅ` (자모) vs `�` (byte) | 음절·형태소 = 좋음, 자모/byte = 나쁨 |
| **OOV·신조어·외래어** | `킹받네` / `recursion` 처리 | UNK 없이 자연스러운 subword |

§1-§4 에서 이 지표들을 하나씩 재고, §5 에서 종합해 *vocab 으로 품질을 예측* 하는 그림을 그립니다.

---""")

# ----- 3. 환경 셋업 -----
md(r"""## 🛠️ 환경 셋업

토크나이저만 비교하므로 모델 본체는 로드하지 않습니다. `tiktoken` 은 OpenAI 의 GPT-4o·GPT-4 토크나이저 비교를 위해 추가로 설치합니다. **GPU 불필요 (CPU 로 충분)** — Colab 런타임은 기본 그대로 두어도 됩니다.""")

code(r"""%pip install -q -U transformers tiktoken""")

code(r"""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import transformers
import tiktoken

plt.rcParams["axes.unicode_minus"] = False

print(f"transformers: {transformers.__version__}")
print(f"tiktoken:     {tiktoken.__version__}")
print("note: tokenizers only — no model weights, CPU is enough.")""")

# ----- 4. 토크나이저 로드 -----
md(r"""## 1. 🔤 토크나이저 로드 — 여러 다국어 LLM 을 한 dict 로

비교 대상을 한 dict 에 모읍니다. 각 토크나이저는 `encode(text) -> list[int]` 인터페이스로 통일하기 위해 얇은 wrapper 로 감쌉니다 (`transformers` 와 `tiktoken` 의 인터페이스가 다르기 때문).

**KoGPT2 는 메인 챕터에서 배운 방식 그대로** `PreTrainedTokenizerFast` + special token 직접 지정으로 로드합니다 (`AutoTokenizer` 의 영어 GPT2 fallback 함정 회피).

로드는 모두 `try/except` 로 감쌉니다. gated 모델 (meta-llama, google/gemma 등) 은 HF 로그인이 필요해 제외했고, 네트워크·버전 사정으로 일부가 실패해도 *로드된 것만* 으로 분석을 진행합니다 (한두 개 빠져도 결론은 흔들리지 않습니다).""")

code(r'''from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _bytelevel_decoder():
    """GPT-2 식 byte-level BPE 의 (unicode 문자 -> 원래 byte) 역매핑 테이블.
    polyglot-ko / Qwen / BLOOM / gpt2 의 vocab 토큰은 원래 UTF-8 byte 를 보기 좋은
    unicode 문자로 치환해 저장합니다 (예: 'ìĺ¤'). 한글 판별을 위해 이 매핑을 거꾸로
    돌려 byte 로 복원한 뒤 UTF-8 로 다시 디코드합니다."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_MAP = _bytelevel_decoder()


def decode_vocab_token(tok):
    """vocab 토큰 1개를 사람이 읽는 한국어 문자열로 복원.
    byte-level 치환 문자가 섞여 있으면 byte 로 되돌려 UTF-8 디코드,
    아니면 (WordPiece/SentencePiece) 표식만 떼어 그대로 사용."""
    if any(ch in _BYTE_MAP and ord(ch) > 127 for ch in tok):
        try:
            return bytes(_BYTE_MAP.get(ch, ord(ch) & 0xFF) for ch in tok).decode(
                "utf-8", errors="replace"
            )
        except Exception:
            pass
    return tok.replace("▁", " ").replace("##", "")


class HFWrapper:
    """transformers 토크나이저를 통일된 encode/tokenize 인터페이스로 감싸는 wrapper."""

    kind = "hf"

    def __init__(self, name, tok):
        self.name = name
        self.tok = tok
        self.vocab_size = tok.vocab_size

    def encode(self, text):
        # special token 제외 — fertility 는 본문 토큰 수만 세는 게 공정
        return self.tok.encode(text, add_special_tokens=False)

    def tokenize(self, text):
        # 사람이 읽을 토큰 문자열 (▁ / ## 등 표식 포함)
        return self.tok.tokenize(text)

    def vocab_tokens(self):
        # vocab 토큰을 사람이 읽는 문자열로 디코드해 리스트로 반환.
        # 주의: byte-level BPE (polyglot-ko, Qwen, BLOOM, gpt2) 의 get_vocab() 키는
        # byte 가 mangling 된 문자열(예: 'ìĺ¤') 이라 그대로는 한글 판별이 안 됩니다.
        # decode_vocab_token 으로 byte-level·WordPiece·SentencePiece 를 한 번에 복원.
        return [decode_vocab_token(t) for t in self.tok.get_vocab().keys()]


class TiktokenWrapper:
    """tiktoken Encoding 을 같은 인터페이스로 감싸는 wrapper."""

    kind = "tiktoken"

    def __init__(self, name, enc):
        self.name = name
        self.enc = enc
        self.vocab_size = enc.n_vocab

    def encode(self, text):
        return self.enc.encode(text)

    def tokenize(self, text):
        # token id -> bytes -> 문자열 (깨진 byte 는 �로 보임)
        return [
            self.enc.decode_single_token_bytes(t).decode("utf-8", errors="replace")
            for t in self.enc.encode(text)
        ]

    def vocab_tokens(self):
        out = []
        for i in range(self.enc.n_vocab):
            try:
                out.append(self.enc.decode_single_token_bytes(i).decode("utf-8", errors="replace"))
            except Exception:
                continue
        return out''')

code(r'''# 로드할 토크나이저 목록 — (라벨, 카테고리, 로더 함수)
# 카테고리: ko = 한국어 특화, multi = 다국어 LLM, openai = tiktoken, en = 영어 대조군


def _load_kogpt2():
    # 메인 챕터 방식 그대로 — AutoTokenizer fallback 함정 회피
    return PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token="</s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>",
    )


SPECS = [
    ("KoGPT2",       "ko",     "hf",       _load_kogpt2),
    ("KLUE-BERT",    "ko",     "hf",       lambda: AutoTokenizer.from_pretrained("klue/bert-base")),
    ("polyglot-ko",  "ko",     "hf",       lambda: AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")),
    ("KcELECTRA",    "ko",     "hf",       lambda: AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")),
    ("Qwen2.5",      "multi",  "hf",       lambda: AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")),
    ("mBERT",        "multi",  "hf",       lambda: AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")),
    ("BLOOM",        "multi",  "hf",       lambda: AutoTokenizer.from_pretrained("bigscience/bloom-560m")),
    ("GPT-4o (o200k)", "openai", "tiktoken", lambda: tiktoken.get_encoding("o200k_base")),
    ("GPT-4 (cl100k)", "openai", "tiktoken", lambda: tiktoken.get_encoding("cl100k_base")),
    ("gpt2 (en)",    "en",     "hf",       lambda: AutoTokenizer.from_pretrained("gpt2")),
]

tokenizers = {}   # 라벨 -> wrapper
category = {}     # 라벨 -> 카테고리

for label, cat, kind, loader in SPECS:
    try:
        obj = loader()
        wrapper = TiktokenWrapper(label, obj) if kind == "tiktoken" else HFWrapper(label, obj)
        tokenizers[label] = wrapper
        category[label] = cat
        print(f"[ok]   {label:16s} ({cat:6s})  type={type(obj).__name__:24s}  vocab={wrapper.vocab_size:,}")
    except Exception as e:
        msg = str(e).splitlines()[0][:80] if str(e) else type(e).__name__
        print(f"[skip] {label:16s} ({cat:6s})  -> {msg}")

print(f"\nloaded {len(tokenizers)} tokenizers: {list(tokenizers.keys())}")''')

# ----- 5. 검증 문장 셋 -----
md(r"""## 2. 📝 검증 문장 셋 — 한국어 4개 도메인

한국어의 결을 고루 보려고 4개 도메인을 고릅니다. 일상·뉴스·전문 용어·인터넷 신조어가 섞여 있어, 토크나이저가 *격식체* 만 잘 쪼개는지 아니면 *비격식·외래어·신조어* 까지 자연스럽게 처리하는지 드러납니다.""")

code(r'''SENTENCES = {
    "daily":    "오늘 날씨가 정말 좋아서 친구랑 한강에서 산책했어요.",
    "news":     "정부는 내년도 예산안을 국회에 제출했다고 밝혔다.",
    "technical": "트랜스포머 모델의 어텐션 메커니즘은 병렬 연산이 가능하다.",
    "slang":    "이 영화 진짜 킹받는데 recursion 개념이 너무 어려워요 ㅋㅋ",
}

for domain, text in SENTENCES.items():
    # 공백 제외 글자 수 — fertility 의 분모로 쓸 char 수
    n_char = len(text.replace(" ", ""))
    print(f"[{domain:9s}] chars(no space)={n_char:3d}  {text}")''')

# ----- 6. §1 Fertility 분석 -----
md(r"""## 3. §1 Fertility 분석 — 같은 문장이 몇 토큰으로 쪼개지나

**Fertility** 는 *입력 단위당 토큰 수* 입니다. 한국어는 영어와 달리 띄어쓰기로 단어를 세기 애매하므로 (한 어절이 형태소 여럿) **글자(char) 기준** 이 공정합니다. 즉 `tokens / chars(공백 제외)`.

- **낮을수록 좋음**: 같은 문장을 적은 토큰으로 표현 → 같은 context window 에 더 많은 한국어, 추론 시 토큰 수 ↓ (비용·지연 ↓), 학습 시 의미 단위 보존.
- 영어 BPE (`gpt2`) 처럼 한국어를 byte 로 깨면 fertility 가 폭증합니다.""")

code(r'''rows = []
for label, tk in tokenizers.items():
    for domain, text in SENTENCES.items():
        n_char = len(text.replace(" ", ""))
        n_tok = len(tk.encode(text))
        rows.append({
            "tokenizer": label,
            "category": category[label],
            "domain": domain,
            "chars": n_char,
            "tokens": n_tok,
            "tokens_per_char": round(n_tok / n_char, 3),
        })

fert_df = pd.DataFrame(rows)

# 토크나이저별 평균 fertility (4개 도메인 평균)
fert_mean = (
    fert_df.groupby(["tokenizer", "category"])["tokens_per_char"]
    .mean().round(3).reset_index()
    .sort_values("tokens_per_char")
    .reset_index(drop=True)
)
print("Mean fertility (tokens / char, averaged over 4 domains) — lower is better:")
print(fert_mean.to_string(index=False))''')

code(r'''# 도메인별 토큰 수 pivot — 한 문장이 토크나이저마다 몇 토큰인지
pivot = fert_df.pivot(index="tokenizer", columns="domain", values="tokens")
pivot = pivot[["daily", "news", "technical", "slang"]]
pivot = pivot.loc[fert_mean["tokenizer"]]  # fertility 오름차순 정렬
print("Token count per sentence (lower is better):")
print(pivot.to_string())''')

code(r'''# bar chart — 토크나이저별 평균 fertility (영어 라벨만, plt 에 한글 없음)
CAT_COLOR = {"ko": "#2E86C1", "multi": "#28B463", "openai": "#A569BD", "en": "#CB4335"}
order = fert_mean["tokenizer"].tolist()
colors = [CAT_COLOR[category[t]] for t in order]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(order, fert_mean["tokens_per_char"], color=colors)
ax.set_title("Korean fertility by tokenizer (tokens / char) - lower is better")
ax.set_xlabel("tokenizer")
ax.set_ylabel("tokens per char (mean over 4 domains)")
ax.tick_params(axis="x", rotation=30)
for i, v in enumerate(fert_mean["tokens_per_char"]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
ax.legend(handles, CAT_COLOR.keys(), title="category", loc="upper left")
plt.tight_layout()
plt.show()

best = fert_mean.iloc[0]
worst = fert_mean.iloc[-1]
print(f"lowest  fertility: {best['tokenizer']:16s} {best['tokens_per_char']:.3f}  ({best['category']})")
print(f"highest fertility: {worst['tokenizer']:16s} {worst['tokens_per_char']:.3f}  ({worst['category']})")
print(f"ratio (worst / best): {worst['tokens_per_char'] / best['tokens_per_char']:.1f}x")''')

# ----- 7. §2 토큰화 시각화 -----
md(r"""## 4. §2 토큰화 시각화 — 음절/형태소 vs 자모/byte

숫자만으로는 *왜* fertility 가 다른지 와닿지 않습니다. 한 문장을 각 토크나이저로 쪼개 **토큰 경계** 를 직접 봅니다. 한국어 토큰은 한글 폰트 문제로 `plt` 가 아니라 `print` 로 보여 줍니다 (그래서 토큰 텍스트 자체는 한국어 그대로).

세 가지 패턴이 눈에 보입니다.

- **음절·형태소 단위** (좋음): `오늘|날씨|가|...` 처럼 의미 덩어리로 쪼갬.
- **음절 단위 잘게** (보통): `오|늘|날|씨|...` 처럼 한 글자씩.
- **byte 깨짐** (나쁨): `ìĺ¤|ëĬ|...` 또는 `�` 처럼 한국어 글자가 UTF-8 byte 로 부서짐 (영어 BPE 의 전형).""")

code(r'''def show_tokenization(text, sep="|"):
    """한 문장을 모든 토크나이저로 쪼개 토큰 경계를 print 로 보여준다."""
    n_char = len(text.replace(" ", ""))
    print(f"sentence: {text}")
    print(f"chars(no space): {n_char}\n")
    for label, tk in tokenizers.items():
        toks = tk.tokenize(text)
        # ▁(SentencePiece 공백 표식) 를 보기 좋게 치환, 너무 길면 자름
        shown = sep.join(t.replace("▁", "_") for t in toks)
        if len(shown) > 90:
            shown = shown[:90] + " ..."
        print(f"[{label:16s}] n={len(toks):3d}  {shown}")


# 일상 문장 — 음절/형태소 vs byte 차이가 가장 또렷
show_tokenization(SENTENCES["daily"])''')

code(r'''# 신조어·외래어 문장 — OOV·신조어·외래어 처리 비교 (킹받는 / recursion / ㅋㅋ)
show_tokenization(SENTENCES["slang"])''')

md(r"""위 출력에서 한국어 특화 토크나이저는 `킹받`, `recursion`, `ㅋㅋ` 를 UNK 없이 subword 로 자연스럽게 흡수합니다. 반면 영어 BPE (`gpt2`) 는 한국어 글자마다 여러 byte 토큰으로 깨져 (`�` 또는 `ìĺ¤` 형태) 토큰 수가 폭증합니다. 이 byte 깨짐이 §1 의 fertility 폭증으로 직결됩니다.""")

# ----- 8. §3 vocab 한국어 점유율 -----
md(r"""## 5. §3 vocab 한국어 점유율 — vocab 만 열어 보는 선행 지표

여기가 *vocab 만 봐도 안다* 의 핵심입니다. 추론을 한 번도 돌리지 않고, **vocab 전체를 순회해 한글 음절(가-힣)을 포함한 토큰의 비율** 과 평균 토큰 길이만 셉니다.

- **한국어 점유율 높음** = vocab 예산을 한국어 의미 단위에 많이 배정 → 같은 문장을 적은 토큰으로 (fertility ↓ 와 직결).
- **평균 토큰 길이 김** = 한 토큰이 더 긴 글자 덩어리를 담음 → 압축률 ↑.

`tiktoken` 은 vocab 순회 방식이 달라 (`decode_single_token_bytes`) wrapper 에서 처리해 둔 그대로 셉니다.""")

code(r'''def has_hangul_syllable(s):
    """문자열에 한글 음절(가-힣)이 하나라도 있으면 True."""
    return any("가" <= ch <= "힣" for ch in s)


def clean_token(t):
    """디코드된 vocab 토큰에서 남은 표식·공백을 떼어 글자만 남긴다.
    (vocab_tokens 가 이미 convert_tokens_to_string 으로 디코드했으므로 대부분
    앞 공백 정도만 정리하면 됩니다. fallback raw 토큰의 ▁/## 도 함께 제거.)"""
    return t.replace("▁", "").replace("##", "").strip()


vocab_rows = []
for label, tk in tokenizers.items():
    toks = tk.vocab_tokens()
    cleaned = [clean_token(t) for t in toks]
    n_total = len(cleaned)
    n_ko = sum(1 for c in cleaned if has_hangul_syllable(c))
    # 한글 음절을 포함한 토큰의 평균 글자 길이 (한국어 압축력)
    ko_lens = [len(c) for c in cleaned if has_hangul_syllable(c)]
    avg_ko_len = float(np.mean(ko_lens)) if ko_lens else 0.0
    vocab_rows.append({
        "tokenizer": label,
        "category": category[label],
        "vocab_size": n_total,
        "ko_tokens": n_ko,
        "ko_share_%": round(100 * n_ko / n_total, 2) if n_total else 0.0,
        "avg_ko_token_len": round(avg_ko_len, 2),
    })

vocab_df = pd.DataFrame(vocab_rows).sort_values("ko_share_%", ascending=False).reset_index(drop=True)
print("Korean share of vocabulary (higher is better):")
print(vocab_df.to_string(index=False))''')

code(r'''# bar chart — vocab 한국어 점유율
order2 = vocab_df["tokenizer"].tolist()
colors2 = [CAT_COLOR[category[t]] for t in order2]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(order2, vocab_df["ko_share_%"], color=colors2)
ax.set_title("Korean share of vocabulary by tokenizer (%) - higher is better")
ax.set_xlabel("tokenizer")
ax.set_ylabel("Korean tokens / vocab (%)")
ax.tick_params(axis="x", rotation=30)
for i, v in enumerate(vocab_df["ko_share_%"]):
    ax.text(i, v + 0.8, f"{v:.0f}", ha="center", fontsize=9)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CAT_COLOR.values()]
ax.legend(handles, CAT_COLOR.keys(), title="category", loc="upper right")
plt.tight_layout()
plt.show()''')

# ----- 9. §4 영어 대조군 극단 -----
md(r"""## 6. §4 영어 대조군 극단 — `gpt2` 의 한국어 byte 깨짐 (Ch 27 함정의 정량 버전)

메인 챕터의 *`AutoTokenizer` 가 KoGPT2 를 영어 GPT2 로 fallback 하면 한국어가 깨진다* 는 경고를 **숫자로** 확인합니다. 영어 전용 BPE (`gpt2`) 는 한국어 글자를 학습한 적이 없어 UTF-8 byte 단위로 쪼갭니다. 한국어 한 글자는 보통 3 byte 라, fertility 가 한국어 특화 토크나이저의 *몇 배* 로 폭증합니다.""")

code(r'''if "gpt2 (en)" in tokenizers and any(category[t] == "ko" for t in tokenizers):
    gpt2_fert = fert_mean.loc[fert_mean["tokenizer"] == "gpt2 (en)", "tokens_per_char"].iloc[0]
    ko_labels = [t for t in tokenizers if category[t] == "ko"]
    ko_fert = fert_mean[fert_mean["tokenizer"].isin(ko_labels)]
    print("gpt2 (English BPE) vs Korean-specialized tokenizers — fertility (tokens/char):\n")
    print(f"  gpt2 (en):       {gpt2_fert:.3f}")
    for _, r in ko_fert.iterrows():
        ratio = gpt2_fert / r["tokens_per_char"]
        print(f"  {r['tokenizer']:16s} {r['tokens_per_char']:.3f}   -> gpt2 is {ratio:.1f}x more tokens")
    print()
    print("So an English-only BPE spends several times more tokens on the same Korean text:")
    print("  - context window fills up faster (fewer Korean chars per window)")
    print("  - inference cost / latency scale with token count -> several x worse")
    print("  - the model must learn Korean from raw bytes -> very inefficient")
    print("\nThis is the quantitative version of the Ch 27 AutoTokenizer fallback trap.")
else:
    print("gpt2 or Korean tokenizers not loaded — skipping the extreme contrast.")''')

# ----- 10. §5 종합 -----
md(r"""## 7. §5 종합 — vocab 으로 품질 예측하기

두 선행 지표를 한 그림에 모읍니다.

- x 축: **vocab 한국어 점유율 (%)** — 높을수록 오른쪽.
- y 축: **fertility (tokens/char)** — 낮을수록 아래.

**오른쪽 아래** (한국어 vocab ↑ + fertility ↓) 에 있을수록 *한국어 LLM 의 토크나이저로 유리* 합니다. KoGPT2·polyglot-ko 같은 한국어 특화 토크나이저가 그쪽에, `gpt2` 같은 영어 BPE 가 왼쪽 위 (vocab 한국어 0% + fertility 폭증) 에 위치합니다. 다국어 LLM 은 보통 그 사이 어딘가입니다 — 한국어를 포기하진 않았지만 vocab 예산을 100여 개 언어에 나눠 쓴 결과입니다.""")

code(r'''merged = pd.merge(
    fert_mean.rename(columns={"tokens_per_char": "fertility"}),
    vocab_df[["tokenizer", "ko_share_%", "avg_ko_token_len"]],
    on="tokenizer",
)

fig, ax = plt.subplots(figsize=(10, 7))
for cat, color in CAT_COLOR.items():
    sub = merged[merged["category"] == cat]
    ax.scatter(sub["ko_share_%"], sub["fertility"], s=140, color=color, label=cat, edgecolors="black", zorder=3)
for _, r in merged.iterrows():
    ax.annotate(r["tokenizer"], (r["ko_share_%"], r["fertility"]),
                xytext=(6, 4), textcoords="offset points", fontsize=9)

ax.set_xlabel("Korean share of vocab (%)  -> better to the right")
ax.set_ylabel("fertility (tokens / char)  -> better lower")
ax.set_title("Tokenizer quality map for Korean — bottom-right is best")
ax.invert_yaxis()  # 위로 갈수록 fertility 큼(나쁨)
ax.legend(title="category", loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()''')

code(r'''# 종합 순위 표 — fertility 오름차순 (좋은 순)
ranking = merged.sort_values("fertility").reset_index(drop=True)
ranking.insert(0, "rank", ranking.index + 1)
ranking = ranking[["rank", "tokenizer", "category", "fertility", "ko_share_%", "avg_ko_token_len"]]
print("Overall ranking for Korean (sorted by fertility, lower is better):")
print(ranking.to_string(index=False))''')

# ----- 11. 해석 -----
md(r"""## 8. 🔬 해석 — 왜 vocab 이 선행 지표인가

산점도와 순위 표가 보여 준 패턴을 메커니즘으로 풀어 봅니다.

- **fertility ↓ = 추론·context 효율 ↑** — 같은 4,096 토큰 context 에 한국어 특화 토크나이저는 영어 BPE 의 *몇 배* 분량을 담습니다. 추론 비용·지연이 토큰 수에 비례하므로, fertility 가 낮은 토크나이저는 *같은 문서를 더 싸고 빠르게* 처리합니다. 긴 한국어 문서·RAG·대화에서 차이가 누적됩니다.
- **한국어 vocab ↑ = 학습 효율 ↑** — vocab 예산을 한국어 의미 단위 (`날씨`, `예산안`, `메커니즘`) 에 많이 배정하면 본체가 *의미 덩어리* 를 직접 봅니다. 반대로 글자·byte 로 잘게 쪼개면 본체가 *철자부터* 한국어를 조립해야 합니다.
- **자모/byte 분해 = 비효율의 근원** — `오늘` 이 `오|늘` (음절) 이 아니라 `�` (byte) 로 깨지면, 본체는 의미를 배우기 전에 *byte → 글자 → 단어* 복원부터 배워야 합니다. 같은 데이터로도 학습이 더디고, 추론에서 토큰이 폭증합니다.

> **단, vocab 은 필요조건이지 충분조건이 아닙니다.** 좋은 토크나이저는 한국어 품질의 *상한* 을 열어 줄 뿐, 그 상한을 채우는 건 *충분한·다양한 한국어 사전학습 데이터* 입니다 (Ch 22·26·27 의 메시지). 한국어 vocab 점유율이 높아도 한국어를 거의 학습하지 않은 모델은 여전히 한국어를 못합니다. 반대로 fertility 가 좋아도 (Ch 23 부록의 negative transfer 처럼) 사전학습 도메인이 task 와 어긋나면 효과가 제한됩니다. **좋은 토크나이저 + 충분한 한국어 사전학습 = 둘 다** 필요합니다.

그래서 토크나이저는 *품질을 보장* 하는 지표가 아니라 *품질의 상한과 효율을 예고* 하는 **선행 지표** 입니다. 가중치를 받기 전에 vocab 만으로 빠르게 거를 수 있다는 점이 실무에서 강력합니다.

---""")

# ----- 12. 체크포인트 + FAQ -----
md(r"""## 🎯 체크포인트 질문

1. **fertility 의 분모** — 한국어 fertility 를 `tokens/word` (어절 기준) 가 아니라 `tokens/char` (글자 기준) 로 잰 이유는 무엇인가요? 영어에서는 왜 `tokens/word` 가 흔히 쓰일까요?
2. **다국어 토크나이저의 위치** — 산점도에서 다국어 LLM (Qwen·mBERT·BLOOM) 이 한국어 특화와 영어 BPE *사이* 에 놓이는 이유를 vocab 예산 배분 관점에서 설명해 보세요.
3. **선행 지표의 한계** — 한국어 vocab 점유율이 높은데도 실제 한국어 품질이 낮은 모델이 있을 수 있습니다. 어떤 경우에 그렇고, 왜 토크나이저가 *충분조건* 이 아닌가요?""")

md(r"""## ❓ FAQ

**Q1. fertility 가 낮으면 추론 비용이 정확히 얼마나 줄어드나요?**
추론 비용·지연은 대체로 *처리하는 토큰 수* 에 비례합니다. 같은 한국어 문서를 fertility 0.7 인 토크나이저와 2.1 인 토크나이저로 넣으면 후자가 약 3배 토큰을 만들어, 같은 모델·같은 하드웨어에서 약 3배의 토큰 연산·시간·과금이 듭니다. context window 도 같은 비율로 빨리 찹니다. 그래서 한국어 서비스에서는 토크나이저 fertility 가 *직접적인 운영 비용* 입니다.

```python
# 한국어 문서 하나의 토큰 수를 토크나이저별로 비교
doc = open("some_korean_doc.txt", encoding="utf-8").read()
for label, tk in tokenizers.items():
    print(f"{label:16s} {len(tk.encode(doc)):,} tokens")
```

**Q2. 왜 한국어는 `tokens/char` 로 재나요? 영어처럼 `tokens/word` 가 아니라?**
영어는 띄어쓰기가 단어 경계와 거의 일치해 `tokens/word` 가 깔끔합니다. 한국어는 한 *어절* 안에 형태소가 여럿 (`한강에서` = `한강`+`에서`) 이고 띄어쓰기 규칙도 느슨해, 어절 단위가 불안정합니다. 글자(char) 는 어떤 토크나이저에도 흔들리지 않는 공정한 분모라 한국어 fertility 비교에서 표준으로 씁니다. 셀 때 공백을 빼는 이유는, 공백 처리 방식 (`▁` 표식 등) 이 토크나이저마다 달라 분모를 오염시키지 않기 위해서입니다.

**Q3. 다국어 모델은 한국어 토크나이저가 왜 어중간한가요? trade-off 가 무엇인가요?**
다국어 LLM 은 하나의 vocab 예산 (예: 15만 토큰) 을 100여 개 언어에 나눠 씁니다. 한국어에 배정되는 몫이 한국어 전용 모델보다 작을 수밖에 없어, 한국어 fertility 가 다소 높아집니다. 대신 *여러 언어를 한 모델로* 처리하고 코드 스위칭·번역에 강한 이점이 있습니다. GPT-4o 의 `o200k_base` 가 GPT-4 의 `cl100k_base` 보다 한국어 fertility 가 낮은 것도, 더 큰 vocab 으로 한국어 등 비영어에 예산을 더 배정한 결과입니다 — 다국어 안에서도 세대가 올라가며 비영어 효율이 개선됩니다.

**Q4. 토크나이저가 별로인 모델을 vocab 확장 (vocab extension) 으로 살릴 수 있나요?**
부분적으로 가능합니다. 한국어 토큰을 vocab 에 *추가* 하고 임베딩 행렬을 늘린 뒤, 추가 토큰 위주로 *이어서 사전학습* 하면 fertility 를 낮출 수 있습니다 (`model.resize_token_embeddings(len(tokenizer))`). 다만 새 토큰의 임베딩은 처음엔 무의미해, 충분한 한국어 데이터로 다시 학습해야 제값을 합니다. 또 기존 토큰 분포가 바뀌어 원래 능력이 흔들릴 위험도 있습니다. 그래서 *작은 vocab 확장 + 짧은 continual pretraining* 은 흔한 한국어화 전략이지만, 공짜는 아니고 Ch 27 의 continual pretraining 과 짝을 이뤄야 효과가 납니다.

```python
# vocab 확장 스케치 — 새 한국어 토큰 추가 후 임베딩 리사이즈
num_added = tokenizer.add_tokens(["새토큰1", "새토큰2"])
model.resize_token_embeddings(len(tokenizer))  # 이후 한국어 데이터로 continual pretraining 필요
```

**Q5. vocab 점유율만 높으면 한국어를 잘하는 모델인가요?**
아닙니다. vocab 점유율과 fertility 는 *상한과 효율* 을 예고하는 선행 지표일 뿐, 실제 품질은 *그 토크나이저로 얼마나 많은·다양한 한국어를 사전학습했는가* 가 채웁니다. 토크나이저가 좋아도 한국어 데이터가 빈약하면 한국어를 못하고 (Ch 22·26), 사전학습 도메인이 task 와 어긋나면 transfer 가 약합니다 (Ch 23 부록). 그래서 실무에서는 *토크나이저 vocab 으로 1차 스크리닝 → 가중치로 실제 한국어 평가* 의 2단계를 거칩니다.""")

# ----- 13. 다음 단계 -----
md(r"""## 다음 단계

이 부록에서 *토크나이저 vocab 만으로 한국어 LLM 품질의 상한·효율을 예측* 하는 직관을, fertility·vocab 한국어 점유율·자모/byte 분해 세 지표로 정량 확인했습니다. KoGPT2·polyglot-ko 같은 한국어 특화 토크나이저가 *낮은 fertility + 높은 한국어 vocab 점유율* 로 한국어에 유리하고, 영어 BPE (`gpt2`) 는 byte 깨짐으로 fertility 가 폭증한다는 점 — 메인 챕터의 *`AutoTokenizer` fallback 함정* 이 왜 치명적인지를 숫자로 본 셈입니다.

- **메인 챕터로 돌아가기**: [`27_ko_gpt2_continual_pretrain.ipynb`](./27_ko_gpt2_continual_pretrain.ipynb) — KoGPT2 를 한국어 TinyStories 로 continual pretraining.
- **다음 챕터 예고**: Chapter 28 — **SFT (Supervised Fine-Tuning)**. 사전학습된 한국어 본체에 *지시-응답 (instruction-response)* 데이터로 *행동 정렬* 을 입히는 단계. 좋은 토크나이저 + 충분한 사전학습 위에, 이제 *어떻게 행동하게 만들 것인가* 로 넘어갑니다.

> 부록의 핵심 메시지 한 줄 — *토크나이저는 한국어 품질의 선행 지표다*. vocab 의 한국어 점유율과 fertility 만 봐도 그 모델이 한국어를 효율적으로 다룰 *잠재력* 이 보입니다. 단, 그 잠재력을 실현하는 건 충분하고 다양한 한국어 사전학습 데이터 — *좋은 토크나이저와 좋은 데이터는 둘 다 필요* 합니다.""")


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
