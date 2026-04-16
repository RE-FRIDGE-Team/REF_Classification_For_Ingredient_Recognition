# Phase 1 구현 체크리스트

> Claude Code 작업 시 이 파일을 진행 상황 트래킹에 활용

## 현재 진행 상황

### [x] Step 0: 환경 설정 (Docker)
- [x] Docker 환경 설정 (`docker-compose up --build`)
- [x] requirements.txt 설치
- [x] configs/stage1.yaml 초기화

### [ ] Step 1: 데이터 준비 (`src/preprocess.py`)
- [x] CSV 로드 + 스키마 검증 (`src/data_utils.py::load_data()`)
- [ ] EDA (클래스 분포, 토큰 길이, 결측치)
- [x] 브랜드명 분리 + 용량/단위 토큰 처리 (`REFPreprocessor._refine_row()`)
- [ ] Data Augmentation (Brand Swap / Quantity Perturbation / Token Shuffle)
- [x] GroupKFold 그룹 정의 (brand 컬럼) (`src/data_utils.py::make_folds()`)
- [ ] 전처리 결과 저장 (`data/processed/`, `data/augmented/`)

### [x] Step 2: 베이스라인 (`src/models/tfidf_lgbm.py`)
- [x] TF-IDF + LightGBM 학습
- [x] GroupKFold CV 평가 → macro_F1 기록 (`src/evaluate.py::cv_evaluate()`)

### [x] Step 3: Stage 1 Fine-tuning (`src/models/koelectra_multitask.py`)
- [x] KoBERT/KoELECTRA 토크나이저 + 모델 로드
- [x] Multi-task 3헤드 모델 구현 (`_KoElectraMultiTask`)
- [x] Layer-wise LR Decay + 하위 레이어 Freeze (`_freeze_bottom_layers()`)
- [x] 학습 루프 (Loss = 0.4×대분류 + 0.4×중분류 + 0.2×카테고리태그)
- [x] 체크포인트 저장 (`models/stage1_best.pt`)

### [ ] Step 4: 평가 (`src/evaluate.py`)
- [x] GroupKFold CV macro_F1 계산
- [x] Accuracy Gate 판정 (대분류 ≥ 85%) (`src/compare.py::print_comparison_table()`)
- [ ] Confidence 분포 분석

### [ ] Step 5: 추론 파이프라인 (`src/pipeline.py`)
- [ ] 전처리 → 모델 → Confidence Gate 통합
- [ ] 단일 제품명 추론 테스트
- [ ] 배치 추론 지원

### [ ] Step 6: 테스트 (`tests/test_models.py`)
- [x] 전처리 단위 테스트 (`TestPreprocessor`, `TestDataUtils`)
- [ ] Confidence Gate 단위 테스트
- [ ] 파이프라인 통합 테스트

---

## 완료 기준

대분류 macro_F1 ≥ 0.85 달성 → Phase 2 (Stage 2: 식재료 추출) 진행
