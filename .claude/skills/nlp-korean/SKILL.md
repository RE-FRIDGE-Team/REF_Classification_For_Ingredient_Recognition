---
name: nlp-korean-food
description: >
  Korean NLP preprocessing for food/grocery product names.
  Use when cleaning product names, removing brand tokens, normalizing units,
  performing data augmentation (Brand Swap, Quantity Perturbation, Token Shuffle),
  or building Korean tokenization pipelines for the RE:FRIDGE dataset.
  Triggers on: "브랜드 제거", "용량 처리", "증강", "augment", "tokenize korean",
  "전처리", "brand swap", "한국어 NLP" mentions.
compatibility: claude-code, cursor
source: custom (RE:FRIDGE food domain NLP)
---

# Korean Food NLP — 전처리 & 증강 가이드

## 식재료 도메인 제품명 패턴

```
풀무원 국산콩 순두부 300g       → [브랜드: 풀무원] [수식어: 국산콩] [핵심: 순두부] [용량: 300g]
CJ 비비고 왕교자 만두 1kg 20개  → [브랜드: CJ비비고] [핵심: 왕교자만두] [용량: 1kg 20개]
대상 청정원 순창 고추장 500g     → [브랜드: 대상청정원] [수식어: 순창] [핵심: 고추장] [용량: 500g]
```

## 브랜드명 제거 패턴

```python
import re

BRAND_PATTERNS = [
    r'^(풀무원|CJ|대상|오뚜기|농심|삼양|해태|롯데|동원|청정원|비비고|하림)\s*',
    r'^[가-힣]{2,4}(식품|농산|수산|유업|제과)\s*',
]

def remove_brand(product_name: str) -> str:
    for pat in BRAND_PATTERNS:
        product_name = re.sub(pat, '', product_name)
    return product_name.strip()
```

## 용량/단위 토큰 정규화

```python
UNIT_PATTERN = re.compile(
    r'\b\d+(\.\d+)?\s*(g|kg|ml|L|개|봉|팩|인분|장|매|포|병|캔|box)\b',
    re.IGNORECASE
)

def normalize_units(text: str, replace_with: str = '[QTY]') -> str:
    return UNIT_PATTERN.sub(replace_with, text)
```

## Data Augmentation

```python
import random

def brand_swap(product_name: str, brand_pool: list[str]) -> str:
    """다른 브랜드로 교체 — 핵심 식재료명 불변"""
    cleaned = remove_brand(product_name)
    new_brand = random.choice(brand_pool)
    return f"{new_brand} {cleaned}"

def quantity_perturbation(product_name: str) -> str:
    """용량 숫자를 ±20% 범위에서 무작위 변환"""
    def perturb(m):
        val = float(m.group(1) or m.group())
        new_val = val * random.uniform(0.8, 1.2)
        return f"{new_val:.0f}{m.group(2) if m.lastindex else ''}"
    return re.sub(r'(\d+(?:\.\d+)?)(g|kg|ml|L|개)', perturb, product_name)

def token_shuffle(product_name: str, keep_first: bool = True) -> str:
    """토큰 순서 섞기 (첫 토큰 고정 옵션)"""
    tokens = product_name.split()
    if keep_first and len(tokens) > 1:
        rest = tokens[1:]
        random.shuffle(rest)
        return ' '.join([tokens[0]] + rest)
    random.shuffle(tokens)
    return ' '.join(tokens)
```

## GroupKFold 그룹 설정

```python
from sklearn.model_selection import GroupKFold

def make_groups(df, brand_col: str = 'brand') -> list:
    """브랜드 기준 그룹 — 같은 브랜드가 train/val에 분리되도록"""
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder().fit_transform(df[brand_col])
```

## 클래스당 최소 샘플 확보

```python
def augment_to_min(df, label_col: str, min_samples: int = 50,
                   brand_pool: list = None) -> pd.DataFrame:
    augmented = []
    for label, group in df.groupby(label_col):
        shortage = min_samples - len(group)
        if shortage <= 0:
            continue
        samples = group.sample(shortage, replace=True)
        samples['product_name'] = samples['product_name'].apply(
            lambda x: random.choice([
                brand_swap(x, brand_pool or []),
                quantity_perturbation(x),
                token_shuffle(x),
            ])
        )
        augmented.append(samples)
    return pd.concat([df] + augmented).reset_index(drop=True)
```
