"""
룰 레이어 — '모디파이어 vs 헤드' 반례를 처리하는 소규모 오버라이드.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[설계 배경]
  "풀무원자연건면 로스팅 짜장"
    → 마지막 명사가 '짜장'이라 head-noun 피처만으로는 조미료(짜장)로 오인.
    → 실제 헤드는 '건면'이고, '건면 + 짜장'이 같이 나오면
      스프가 든 완제품 = 간편식>라면 이다.
  이런 케이스는 TF-IDF를 과공학하는 것보다 명시적 룰이 정직하고 디버깅 쉽다.

[적용 방식]
  모델 예측 이후(post-prediction) 문자열 라벨 레벨에서 오버라이드한다.
  - 룰은 (필수 토큰 집합, 금지 토큰 집합) → (대분류, 중분류) 매핑.
  - refined_text 에 필수 토큰이 '모두' 포함되고 금지 토큰이 '하나도' 없으면 발동.
  - 대분류만 정의하면 중분류는 모델 예측을 유지한다.

[사람 개입 최소화 — 5순위 원칙]
  시드 룰은 코드에 내장. 확장은 resources/head_rules.yaml 파일이
  존재할 때 자동 병합한다 (형식은 아래 docstring 참고). 파일이 없어도 동작.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadRule:
    """
    단일 오버라이드 룰.

    Attributes:
        require: refined_text 에 '모두' 존재해야 하는 토큰들
        forbid:  하나라도 존재하면 발동하지 않는 토큰들
        large:   강제할 대분류 라벨 문자열 (None이면 유지)
        medium:  강제할 중분류 라벨 문자열 (None이면 유지)
        name:    로그용 룰 이름
    """
    require: tuple[str, ...]
    forbid:  tuple[str, ...]
    large:   str | None
    medium:  str | None
    name:    str


# ── 시드 룰 ────────────────────────────────────────────────────────
# 실측 오류 리포트에서 확인된 '모호 헤드어' 반례들.
# 룰은 최소한으로 유지 — 일반화는 모델이, 반례 교정만 룰이 담당한다.
_SEED_RULES: tuple[HeadRule, ...] = (
    # 건면/생면 + 소스명(짜장·짬뽕 등) 조합 = 스프 포함 완제품 라면
    HeadRule(require=("건면", "짜장"), forbid=(), large="간편식", medium="라면", name="dried_noodle_jjajang"),
    HeadRule(require=("건면", "짬뽕"), forbid=(), large="간편식", medium="라면", name="dried_noodle_jjamppong"),
    HeadRule(require=("건면", "비빔"), forbid=(), large="간편식", medium="라면", name="dried_noodle_bibim"),
    # '라면' 토큰이 명시된 경우 (묶음/세트 포함) — 간편식>라면으로 고정
    HeadRule(require=("라면",), forbid=("떡볶이", "스낵"), large="간편식", medium="라면", name="ramen_token"),
    # '밀키트' 토큰 — 구성 재료명(고기·채소)이 앞에 붙어도 간편식>밀키트
    HeadRule(require=("밀키트",), forbid=(), large="간편식", medium="밀키트", name="mealkit_token"),
    # '시리얼' — 견과/과일 수식어에 끌려가는 오류 방지 (시리얼바는 간식이므로 제외)
    HeadRule(require=("시리얼",), forbid=("시리얼바", "바",), large="상온식품", medium="시리얼", name="cereal_token"),
    # '그래놀라' — 견과류로 오인 방지
    HeadRule(require=("그래놀라",), forbid=("바",), large="상온식품", medium="시리얼", name="granola_token"),
)


def _load_extra_rules(path: str | Path) -> list[HeadRule]:
    """
    YAML 확장 룰 파일을 로드한다. 없으면 빈 리스트.

    파일 형식 (resources/head_rules.yaml):
        rules:
          - name: my_rule
            require: [건면, 우동]
            forbid: []
            large: 간편식
            medium: 라면
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rules = []
        for r in data.get("rules", []):
            rules.append(HeadRule(
                require=tuple(r.get("require", [])),
                forbid=tuple(r.get("forbid", [])),
                large=r.get("large"),
                medium=r.get("medium"),
                name=r.get("name", "extra_rule"),
            ))
        logger.info("확장 룰 로드: %s (%d개)", p, len(rules))
        return rules
    except Exception as e:                        # 파일이 깨져도 학습은 계속
        logger.warning("확장 룰 로드 실패(무시): %s", e)
        return []


class HeadRuleEngine:
    """
    예측 후처리 룰 엔진.

    모델의 predict 결과(정수 인코딩)를 받아, refined_text 조건에 맞는 행의
    대분류/중분류를 룰이 지정한 라벨로 교체한다.
    LabelEncoder 를 주입받아 문자열 룰 ↔ 정수 라벨을 변환한다.
    """

    def __init__(
        self,
        label_encoders: dict | None = None,
        extra_rules_path: str = "resources/head_rules.yaml",
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled and label_encoders is not None
        self._le = label_encoders or {}
        self._rules: list[HeadRule] = list(_SEED_RULES) + _load_extra_rules(extra_rules_path)

        # 라벨 문자열 → 정수 인코딩 룩업 테이블 사전 구축 (predict 마다 재계산 방지)
        self._large_to_int:  dict[str, int] = {}
        self._medium_to_int: dict[str, int] = {}
        if self._enabled:
            le_l, le_m = self._le.get("large"), self._le.get("medium")
            if le_l is not None:
                self._large_to_int = {c: i for i, c in enumerate(le_l.classes_)}
            if le_m is not None:
                self._medium_to_int = {c: i for i, c in enumerate(le_m.classes_)}

    def apply(
        self,
        texts: pd.Series,
        pred_large: np.ndarray,
        pred_medium: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        룰을 적용한 (pred_large, pred_medium, 오버라이드 건수)를 반환한다.

        Args:
            texts:       refined_text 시리즈 (예측 대상과 같은 순서)
            pred_large:  모델 대분류 예측 (int)
            pred_medium: 모델 중분류 예측 (int)
        """
        if not self._enabled or not self._rules:
            return pred_large, pred_medium, 0

        out_l = pred_large.copy()     # 원본 배열 보존 (부작용 방지)
        out_m = pred_medium.copy()
        n_hit = 0

        for i, text in enumerate(texts.fillna("").values):
            for rule in self._rules:
                # 필수 토큰이 모두 포함 & 금지 토큰이 전무해야 발동
                if all(t in text for t in rule.require) and not any(t in text for t in rule.forbid):
                    if rule.large and rule.large in self._large_to_int:
                        out_l[i] = self._large_to_int[rule.large]
                    if rule.medium and rule.medium in self._medium_to_int:
                        out_m[i] = self._medium_to_int[rule.medium]
                    n_hit += 1
                    break             # 한 행에는 첫 매칭 룰만 적용 (룰 순서 = 우선순위)

        if n_hit:
            logger.debug("HeadRuleEngine — %d건 오버라이드", n_hit)
        return out_l, out_m, n_hit
