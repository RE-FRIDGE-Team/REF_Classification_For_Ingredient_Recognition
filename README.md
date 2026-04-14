<div align="center">

```
██████╗ ███████╗    ███████╗██████╗ ██╗██████╗  ██████╗ ███████╗
██╔══██╗██╔════╝    ██╔════╝██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
██████╔╝█████╗      █████╗  ██████╔╝██║██║  ██║██║  ███╗█████╗
██╔══██╗██╔══╝      ██╔══╝  ██╔══██╗██║██║  ██║██║   ██║██╔══╝
██║  ██║███████╗    ██║     ██║  ██║██║██████╔╝╚██████╔╝███████╗
╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
```

# REF_Classification_For_Ingredient_Recognition

**제품명 → 식재료명** 추출을 위한 한국어 NLP · ML 파이프라인

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.40+-FFD21E?style=flat-square)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![KoELECTRA](https://img.shields.io/badge/KoELECTRA-v3-5A67D8?style=flat-square)](https://github.com/monologg/KoELECTRA)
[![Claude Code](https://img.shields.io/badge/Claude_Code-Vibe_Coding-CC785C?style=flat-square)](https://claude.ai/code)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat-square)](LICENSE)

[📐 전체 파이프라인 설계도](https://github.com/RE-FRIDGE-Team/REF_Classification_For_Ingredient_Recognition) · [🗂️ Issues](https://github.com/RE-FRIDGE-Team/REF_Classification_For_Ingredient_Recognition/issues) · [📋 Branches](https://github.com/RE-FRIDGE-Team/REF_Classification_For_Ingredient_Recognition/branches)

</div>

---

## 🥦 프로젝트 개요

> **"풀무원 국산콩 순두부 300g"** 이라는 제품명에서 **"순두부"** 라는 보편적 식재료명을 추출한다.

RE:FRIDGE 앱의 핵심 인식 파이프라인 중 마지막 단계를 담당합니다. 수만 개의 제품명으로부터 일관된 식재료명을 추출해 냉장고 속 재료를 자동으로 분류하고 요리 추천에 활용합니다.

닫힌 집합(기존 식재료 사전)과 열린 집합(신규 식재료) 모두에 대응하는 **하이브리드 파이프라인**을 목표로 합니다.

---

## 🗺️ 전체 파이프라인

```
제품명 입력
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Phase 0 · Data Foundation                          │
│  Raw DataFrame (1,000 × 6)                          │
│  → EDA + 품질 검증                                   │
│  → 전처리 + Data Augmentation (클래스당 최소 50개)    │
│  → DAPT 도메인 사전학습 (라벨 불필요 10만 corpus)     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1 · Stage 1 — 카테고리 분류                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │TF-IDF+LGBM   │  │ FastText/    │  │★ DAPT-   │ │
│  │Baseline      │  │ KoNLPy+ML   │  │KoELECTRA  │ │
│  │70~80% F1    │  │ 75~83% F1   │  │85~93% F1  │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                     │
│  Multi-task 3-Head Fine-tuning                      │
│  출력: 대분류(≥0.80) · 중분류(≥0.60) · 카테고리태그  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2 · Stage 2 — 식재료 추출                     │
│                                                     │
│  DAPT 임베딩 → FAISS Cosine Similarity               │
│  ├─ similarity ≥ 0.85 → 사전에서 직접 반환            │
│  └─ similarity < 0.85 → GPT-4o-mini Fine-tuned      │
│                          + Human-in-the-loop         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
         식재료명 확보 + Redis 캐싱 → Spring Boot
```

> 📐 인터랙티브 전체 설계도 → [pipeline-v2.html](https://github.com/RE-FRIDGE-Team/REF_Classification_For_Ingredient_Recognition)

---

## 🛠️ 기술 스택

### ML / NLP
| 범주 | 기술 |
|------|------|
| 핵심 모델 | `monologg/koelectra-base-v3-discriminator`, `monologg/kobert` |
| 학습 프레임워크 | PyTorch 2.3, HuggingFace Transformers 4.40 |
| 임베딩 검색 | FAISS (Cosine Similarity) |
| 한국어 NLP | KoNLPy, pyahocorasick (Aho-Corasick trie) |
| 베이스라인 | TF-IDF + LightGBM (scikit-learn) |
| LLM Fallback | GPT-4o-mini Fine-tuned |

### 데이터 & 실험
| 범주 | 기술 |
|------|------|
| 데이터 처리 | pandas, numpy |
| 실험 평가 | scikit-learn GroupKFold CV (macro_F1) |
| 시각화 | matplotlib, seaborn, plotly, wordcloud |

### 인프라
| 범주 | 기술 |
|------|------|
| 환경 | Python 3.11.9, conda |
| 컨테이너 | Docker, docker-compose |
| 캐싱 (연동) | Redis |
| 백엔드 (연동) | Spring Boot 4 / JDK 21 |

---

## 🤖 Claude Code · Vibe Coding & Agent Engineering

이 프로젝트는 **Claude Code**를 활용한 AI-assisted 개발 방식을 적극적으로 시도합니다.

### 적용 방식

**Vibe Coding**
자연어로 구현 의도를 설명하면 Claude Code가 코드를 작성합니다. 개발자는 설계와 검증에 집중합니다.

```
"KoELECTRA Multi-task 3헤드 모델 구현해줘.
Loss는 대분류 0.4, 중분류 0.4, 카테고리태그 0.2 가중치로 합산하고
Layer-wise LR Decay 적용해줘"
```

**Agent Orchestration**
병렬 실험이 필요한 시점에 멀티 에이전트를 활용합니다.

```
Orchestrator Claude
  ├── Agent A (worktree: baseline)   → TF-IDF+LightGBM 학습
  ├── Agent B (worktree: koelectra)  → KoELECTRA Fine-tuning
  └── 결과 집계 후 최적 모델 선정
```

**Harness Engineering**
실험 재현성과 컨텍스트 관리를 위한 구조:

```
.claude/
├── skills/          ← 도메인 특화 Skill 파일 (lazy-load)
│   ├── hf-transformers/SKILL.md
│   ├── nlp-korean/SKILL.md
│   └── experiment-log/    ← 실험 회고 자동 기록
├── agents/          ← 오케스트레이션 전략
└── rules/           ← 코드 품질 규칙 자동 적용
```

- **CLAUDE.md** — 28줄 slim core. 세부 참조 문서는 필요 시에만 lazy-load
- **Skills** — frontmatter ~100 토큰만 상시 로드, 본문은 관련 작업 감지 시만 로드
- **/compact** — 장시간 세션에서 컨텍스트 요약으로 토큰 절약
- **/retrospective** — 실험 완료 후 회고를 `experiment-log/`에 자동 저장

---

## 📁 프로젝트 구조

```
REF_Classification_For_Ingredient_Recognition/
├── CLAUDE.md                        ← Claude Code 컨텍스트
├── .python-version                  ← 3.11.9
├── environment.yml                  ← conda 환경 정의
├── requirements.txt                 ← 패키지 버전 고정
│
├── .claude/                         ← Agent Engineering
│   ├── skills/                      ← Lazy-load Skills
│   └── agents/                      ← 오케스트레이션 가이드
│
├── src/                             ← 학습/추론 Python 코드
│   ├── preprocess.py
│   ├── train_stage1.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── classification_ML/               ← EDA · 기존 분류 실험 notebooks
├── product_data_collection/         ← 데이터 수집 스크립트
├── product_name_parser/             ← 제품명 파서 (Java interop)
├── raw_dataset/                     ← 원본 데이터셋
│
├── configs/stage1.yaml              ← 하이퍼파라미터 (하드코딩 금지)
├── docs/                            ← 설계 문서
├── models/                          ← 체크포인트 (gitignore)
├── tests/                           ← pytest
└── logs/                            ← 학습 로그 (gitignore)
```

---

## 🚀 빠른 시작

```bash
# 1. 환경 설정
conda env create -f environment.yml
conda activate refridge-ml

# 2. 전처리
python src/preprocess.py --config configs/stage1.yaml

# 3. 학습
python src/train_stage1.py --config configs/stage1.yaml

# 4. 평가
python src/evaluate.py --model_path models/stage1_best.pt

# 5. 추론 테스트
python src/pipeline.py --input "풀무원 국산콩 순두부 300g"
# → 순두부

# 6. 테스트
pytest tests/ -v
```

---

## 🎯 최종 목표

| 단계 | 목표 | 상태 |
|------|------|------|
| Phase 0 | 전처리 + 증강 파이프라인 완성 | 🔄 진행 중 |
| Phase 1 | 대분류 macro_F1 ≥ **0.85** | ⏳ 예정 |
| Phase 2 | 식재료 추출 (FAISS + LLM Fallback) | ⏳ 예정 |
| Phase 3 | ONNX 서빙 + Spring Boot 연동 | ⏳ 예정 |

**궁극적 목표**: 임의의 한국어 식품 제품명 입력 시 보편적 식재료명을 자동 추출하는 프로덕션 수준의 하이브리드 파이프라인 완성

---

## 🤝 기여 가이드

Issues는 `.github/ISSUE_TEMPLATE/`의 3가지 템플릿을 사용합니다.

- `🧠 ML Task` — 모델 학습, 파이프라인 구현
- `🔬 실험 & 분석` — 모델 비교, 하이퍼파라미터 탐색
- `🗄️ Data / Infra` — 데이터 수집, 환경 설정

브랜치 전략: `feature/REF-XXX-description` · `experiment/model-name`

---

<div align="center">
<sub>RE:FRIDGE Team · Built with Claude Code</sub>
</div>
