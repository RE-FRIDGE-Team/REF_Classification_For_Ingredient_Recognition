---
name: ml-orchestration
description: >
  Multi-agent orchestration for RE:FRIDGE ML pipeline tasks.
  Use when running parallel subtasks, coordinating baseline vs fine-tuning
  experiments, or managing long-running training workflows with Git worktrees.
  Triggers on: "parallel", "동시에", "agent", "orchestrat", "worktree",
  "baseline 비교", "실험 분리", "병렬" mentions.
compatibility: claude-code
source: custom (RE:FRIDGE ML orchestration)
---

# ML 오케스트레이션 — 멀티 에이전트 패턴

## RE:FRIDGE Phase 1 병렬 실험 전략

```
[Orchestrator Claude]
       ├── Agent A (worktree: baseline)
       │     └── TF-IDF + LightGBM 학습 & 평가
       ├── Agent B (worktree: kobert)
       │     └── KoBERT Fine-tuning & 평가
       └── Agent C (worktree: koelectra)
             └── KoELECTRA Fine-tuning & 평가
```

## Git Worktree 설정 (병렬 실험 격리)

```bash
# 각 실험을 독립된 브랜치로 분리
git worktree add ../refridge-baseline  experiment/baseline
git worktree add ../refridge-kobert    experiment/kobert
git worktree add ../refridge-electra   experiment/koelectra

# 각 worktree에서 독립 실행
cd ../refridge-baseline && python src/train_baseline.py &
cd ../refridge-kobert   && python src/train_stage1.py --backbone kobert &
cd ../refridge-electra  && python src/train_stage1.py --backbone koelectra &
```

## Orchestrator 프롬프트 패턴

Claude Code에서 서브에이전트 사용 시:

```
"다음 세 작업을 병렬로 실행해줘:
1. [worktree: experiment/baseline] TF-IDF+LightGBM 베이스라인 학습하고
   results/baseline.json에 macro_F1 저장
2. [worktree: experiment/koelectra] KoELECTRA fine-tuning하고
   results/koelectra.json에 macro_F1 저장
3. 두 결과가 나오면 비교해서 어떤 모델을 Stage 1으로 쓸지 추천해줘"
```

## 실험 결과 집계 패턴

```python
# results/ 디렉토리에 각 에이전트 결과 저장
# 형식: results/{experiment_name}.json
{
  "experiment": "koelectra-v1",
  "macro_f1_large": 0.887,
  "macro_f1_medium": 0.821,
  "epochs": 8,
  "backbone": "monologg/koelectra-base-v3-discriminator",
  "timestamp": "2026-04-15T10:30:00"
}
```

## /advise 패턴 (실험 전 과거 기록 조회)

```
"이전 실험 기록에서 KoELECTRA 관련된 것 찾아서
학습률이나 freeze 레이어 수 추천해줘"
→ Claude가 .claude/skills/experiment-log/ 스캔
```

## /retrospective 패턴 (실험 후 회고 저장)

```
"이번 실험 결과를 회고로 저장해줘:
- 무엇을 시도했는지
- 무엇이 잘 됐는지
- 무엇이 실패했는지
- 다음에 시도할 것"
→ .claude/skills/experiment-log/에 새 SKILL.md 생성
```

## 토큰 절약 오케스트레이션 규칙

1. 오케스트레이터는 **계획과 집계만** 담당 (직접 코드 작성 금지)
2. 서브에이전트는 **독립 컨텍스트**로 실행 (결과 요약만 반환)
3. 대용량 파일(`*.ipynb`, 모델 체크포인트)은 에이전트에 직접 전달 금지
4. 각 에이전트 완료 후 `/compact`
