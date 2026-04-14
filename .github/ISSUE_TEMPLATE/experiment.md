---
name: 🔬 실험 & 분석
about: 모델 비교 실험, EDA, 하이퍼파라미터 탐색, 성능 분석
title: "[EXP] "
labels: ["experiment"]
assignees: ''
---

## 📌 실험 목표
> 이 실험으로 검증하려는 가설 또는 질문

<!-- 예: KoELECTRA가 TF-IDF+LightGBM 베이스라인 대비 macro_F1 5%p 이상 향상되는가? -->

---

## 🔬 실험 설계

**비교 대상**
| 모델 / 설정 | 설명 |
|------------|------|
| Baseline | <!-- 예: TF-IDF + LightGBM --> |
| Experiment A | |
| Experiment B | |

**고정 조건**
- 데이터: <!-- 예: 증강 포함 X개 샘플 -->
- 평가: 5-Fold GroupKFold (brand 기준)
- 지표: macro_F1 (대분류 우선)

**변경 변수**
- <!-- 예: backbone 종류, freeze layer 수, learning rate -->

---

## 🎯 성공 기준

- [ ] 대분류 macro_F1 ≥ <!-- 목표값 -->
- [ ] 베이스라인 대비 <!-- X -->%p 향상

---

## 📋 실험 결과 (완료 후 작성)

| 실험 | 대분류 F1 | 중분류 F1 | 비고 |
|------|----------|----------|------|
| Baseline | | | |
| A | | | |
| B | | | |

**결론 & 다음 액션**
> 

---

## 🔗 참조
- 관련 이슈: #
- 실험 로그: `.claude/skills/experiment-log/`
- 관련 브랜치: `experiment/`
