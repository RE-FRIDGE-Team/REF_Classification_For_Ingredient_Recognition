# RE:FRIDGE ML Pipeline — 전체 설계 (상세 참조 문서)

> CLAUDE.md에서 `@docs/architecture.md`로 임포트됨  
> 이 파일은 상세 참조용. Claude Code가 필요 시에만 로드.

---

## 전체 파이프라인 (Phase 0 → 3)

```
Phase 0: Data Foundation
  Raw DataFrame (1,000 × 6)
  → EDA + 품질 검증
  → 전처리 + Data Augmentation
  → DAPT (도메인 사전학습, 선택사항)

Phase 1: Stage 1 — 분류 (현재 집중)
  비교: TF-IDF+LightGBM vs KoBERT vs DAPT-KoBERT/ELECTRA
  → 최고 성능 모델 선정
  → Multi-task Fine-tuning (3헤드)
  → Accuracy Gate (대분류 ≥ 85%)
  → Confidence-based 출력 필터링

Phase 2: Stage 2 — 식재료 추출 (Phase 1 완료 후)
  Stage 1 출력 + 제품명 → DAPT-KoBERT 임베딩
  → FAISS Cosine Similarity (≥ 0.85 → 사전 반환)
  → 미달 시 GPT-4o-mini Fine-tuned Fallback
  → Human-in-the-loop 검증 → 사전 확장

Phase 3: Output
  "풀무원 국산콩 순두부 300g" → "순두부"
  Redis 캐싱 → Spring Boot 연동
```

---

## 데이터 스키마

| 컬럼 | 타입 | 설명 |
|------|------|------|
| product_name | str | 원본 제품명 |
| large_category | str | 대분류 (Stage 1 타깃 1) |
| medium_category | str | 중분류 (Stage 1 타깃 2) |
| category_tag | str/None | 카테고리태그 (Stage 1 타깃 3) |
| ingredient_name | str | 식재료명 (Stage 2 최종 타깃) |
| brand | str | 브랜드명 (GroupKFold 그룹 키) |

---

## 모델 비교 전략

| 모델 | 예상 성능 | 비고 |
|------|----------|------|
| TF-IDF + LightGBM | 70~78% | 베이스라인 |
| KoBERT Fine-tune | 80~88% | 중간 단계 |
| DAPT-KoBERT/ELECTRA | 85~93% | 권장, 구현 난이도 높음 |

동일 5-Fold GroupKFold CV로 비교 → macro_F1 기준

---

## Serving Layer (Spring Boot 연동 참고)

- 동일 제품명 재요청 → Redis 캐싱으로 파이프라인 우회
- Stage 1 → ONNX 변환 후 경량 서빙
- LLM Fallback 비율 모니터링 → 일정 수준 이하 시 재Fine-tuning 트리거
