# RE:FRIDGE ML — CLAUDE.md
> 제품명 → 대/중분류 + 카테고리태그 Multi-task 분류기 (Phase 1)

## 핵심 명령어
```bash
conda activate refridge-ml
python src/preprocess.py --config configs/stage1.yaml
python src/train_stage1.py --config configs/stage1.yaml
python src/evaluate.py --model_path models/stage1_best.pt
pytest tests/ -v
```

## 코드 규칙
- Python 3.11+, type hints 필수, `logging` (print 금지)
- 모든 상수: `configs/stage1.yaml` (하드코딩 금지)
- `.ipynb`: `notebooks/` 전용, 학습/추론 로직은 `.py`로

## Phase 1 합격 기준
대분류 macro_F1 ≥ 0.85 (5-Fold GroupKFold, brand 기준 그룹)

## 참조 문서 — 필요할 때만 읽을 것
| 목적 | 파일 |
|------|------|
| 전체 파이프라인 설계 | @docs/architecture.md |
| Phase 1 체크리스트 | @docs/phase1.md |
| Skills 목록 & 사용법 | @.claude/skills/README.md |
| 에이전트 오케스트레이션 | @.claude/agents/README.md |
| ML 실험 회고 기록 | @.claude/skills/experiment-log/ |

## 기존 폴더 (참조용)
- `classification_ML/` — 기존 EDA/분류 ipynb
- `product_data_collection/` — 데이터 수집 스크립트
- `product_name_parser/` — 제품명 파서
- `raw_dataset/` — 원본 데이터셋