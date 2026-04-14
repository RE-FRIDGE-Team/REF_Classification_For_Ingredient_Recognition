---
name: 🧠 ML Task
about: 모델 학습, 실험, 파이프라인 구현 관련 작업
title: "[ML] "
labels: ["ml", "in-progress"]
assignees: ''
---

## 📌 작업 개요
> 이 이슈가 해결하려는 문제 또는 목표를 한 문장으로

<!-- 예: KoELECTRA Multi-task Fine-tuning으로 대/중분류 분류기 구현 -->

---

## 🗂️ 작업 분류

**파이프라인 단계**
- [ ] Phase 0 — 데이터 준비 / 전처리 / 증강
- [ ] Phase 1 — Stage 1: 분류 모델 학습 & 평가
- [ ] Phase 2 — Stage 2: 식재료 추출 (임베딩 / FAISS / LLM Fallback)
- [ ] Phase 3 — Serving (ONNX / Spring Boot 연동 / Redis)

**작업 유형**
- [ ] 🔬 데이터 분석 / EDA
- [ ] ⚙️ 인프라 / 환경 설정
- [ ] 🧠 학습 코드 구현
- [ ] 📊 실험 & 평가
- [ ] 🚀 서빙 / 배포
- [ ] 🧹 리팩토링 / 코드 정리
- [ ] 📝 문서화

---

## 🛠️ 개발 환경 & 기술 스택

| 항목 | 내용 |
|------|------|
| Language | Python 3.11 |
| Framework | <!-- 예: HuggingFace Transformers, scikit-learn, PyTorch --> |
| Model | <!-- 예: monologg/koelectra-base-v3-discriminator --> |
| 주요 라이브러리 | <!-- 예: transformers, torch, sklearn, faiss-cpu --> |
| 실행 환경 | <!-- 예: Local CPU / Colab GPU / HuggingFace Jobs --> |

---

## ✅ 완료 조건 (Definition of Done)

> 이 이슈가 완료되면 최종적으로 확인할 수 있는 것

- [ ] <!-- 예: `python src/train_stage1.py` 실행 시 에러 없이 학습 완료 -->
- [ ] <!-- 예: 대분류 macro_F1 ≥ 0.85 달성 (5-Fold GroupKFold) -->
- [ ] <!-- 예: `pytest tests/test_pipeline.py` 전체 통과 -->
- [ ] <!-- 예: `models/stage1_best.pt` 체크포인트 저장 확인 -->

---

## 📐 구현 범위

**포함**
- 

**제외 (다음 이슈에서)**
- 

---

## 🔗 참조
- 관련 이슈: #
- 설계 문서: `docs/architecture.md`, `docs/phase1.md`
- 관련 브랜치: `feature/`
