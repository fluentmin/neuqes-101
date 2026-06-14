## 환경 준비

```python
!pip install -q datasets scikit-learn pandas matplotlib
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

plt.rcParams["axes.unicode_minus"] = False
```

`yelp_review_full`은 Yelp 식당 리뷰 65만 건에 1-5점 별점이 달린 데이터셋입니다 (라벨은 0-4로 저장됨). 학습 흐름을 가볍게 유지하기 위해 **5,000건만 무작위 샘플링** 합니다.

```python
dataset = load_dataset("Yelp/yelp_review_full")
print(dataset)
```

**▶ 실행 결과**

```text
DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 650000
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 50000
    })
})
```

```python
SAMPLE_SIZE = 5000
ds = dataset["train"].shuffle(seed=42).select(range(SAMPLE_SIZE))
df = ds.to_pandas()

print(f"Sample count: {len(df)}")
df.head(3)
```

**▶ 실행 결과**

```text
Sample count: 5000
   label                                               text
0      4  I stalk this truck.  I've been to industrial p...
1      2  who really knows if this is good pho or not, i...
2      4  I LOVE Bloom Salon... all of their stylist are...
```

**결과 해석**

65만 건 중 5,000건만 무작위 샘플링한 결과입니다. `label`이 0-4 정수로 저장돼 있어 실제 별점은 +1 한 1-5점에 해당하고, `text`에는 원문 리뷰가 그대로 담겨 있습니다.

```python
counts = df["label"].value_counts().sort_index()
labels = [f"{i+1} star" for i in counts.index]
plt.bar(labels, counts.values)
plt.title("Star rating distribution (sampled 5,000)")
plt.ylabel("Reviews")
plt.show()
print(counts)
```

**▶ 실행 결과**

![output](../assets/01-tfidf-out1.png)

```text
label
0    1017
1    1027
2     960
3    1021
4     975
Name: count, dtype: int64
```

**결과 해석**

다섯 별점이 각각 약 1,000건씩으로 고르게 분포합니다. 무작위 샘플링이라 원본의 균형 잡힌 별점 구성이 그대로 유지된 것으로, 특정 별점에 편향되지 않은 데이터입니다.

```python
df["len_words"] = df["text"].str.split().str.len()
df[["len_words"]].describe()
```

**▶ 실행 결과**

```text
         len_words
count  5000.000000
mean    133.811400
std     119.787704
min       1.000000
25%      53.000000
50%     100.000000
75%     177.000000
max     977.000000
```

**결과 해석**

리뷰 한 건의 평균 길이는 약 134단어이고 중앙값은 100단어로, 짧은 한 단어짜리부터 977단어까지 편차가 큽니다. 이렇게 길이가 제각각인 텍스트를 곧 고정 길이 벡터로 바꾸게 됩니다.
