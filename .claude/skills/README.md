# Skills 사용 가이드

> Claude Code는 세션 시작 시 각 SKILL.md의 frontmatter(~100 토큰)만 스캔합니다.
> 전체 내용은 관련 작업이 감지될 때만 로드됩니다 (Progressive Disclosure).
> **CLAUDE.md에서 이 파일을 항상 읽지 말 것 — 필요할 때만 `@.claude/skills/README.md`로 참조**

---

## 설치된 Skills (RE:FRIDGE ML 프로젝트)

| 폴더 | 출처 | 용도 | 트리거 조건 |
|------|------|------|------------|
| `hf-transformers/` | [huggingface/skills](https://github.com/huggingface/skills) | KoBERT/KoELECTRA Fine-tuning, ONNX 변환 | "transformers", "fine-tune", "kobert", "electra" 언급 시 |
| `hf-trl/` | [huggingface/skills](https://github.com/huggingface/skills) | SFT/DPO/GRPO 학습, GGUF 변환 | "trl", "rlhf", "sft", "reward model" 언급 시 |
| `python-ml/` | [mcpmarket - expert-python](https://mcpmarket.com/ko/tools/skills/expert-python-developer) | 고품질 Python ML 코드 작성 패턴 | Python 코드 작성 / 리팩토링 요청 시 |
| `nlp-korean/` | [mcpmarket - nlp-text](https://mcpmarket.com/ko/tools/skills/nlp-text-analysis) | 한국어 NLP, 토크나이저, 형태소 | "korean nlp", "tokenize", "morpheme" 언급 시 |
| `orchestration/` | 자체 작성 | 멀티 에이전트 워크플로 | "parallel", "agent", "orchestrate" 언급 시 |

---

## Skills 설치 방법

```bash
# HuggingFace 공식 Skills (transformers, trl, datasets, hub)
npx skills add huggingface/skills --skill hf-text-classification
npx skills add huggingface/skills --skill hf-trl-training

# 커뮤니티 Skills (awesome-claude-skills 목록 기반)
npx skills add travisvn/awesome-claude-skills --skill python-ml
npx skills add mcpmarket/nlp-text-analysis

# ML 실험 관리 Skills (Sionic AI 방식)
npx skills add sionic-ai/claude-code-skills --skill ml-experiment

# 수동 설치: GitHub에서 SKILL.md 폴더를 .claude/skills/에 복사
```

---

## 추천 추가 Skills (우선순위 순)

| 추천 Skill | GitHub | 이유 |
|-----------|--------|------|
| **huggingface/hf-text-classification** | [github.com/huggingface/skills](https://github.com/huggingface/skills) | KoBERT Multi-task 분류 핵심 |
| **huggingface/hf-trl-training** | 위 동일 | Phase 3 RL 학습용 |
| **huggingface/hf-datasets** | 위 동일 | 데이터 증강/로딩 |
| **K-Dense-AI/machine-learning** | [github.com/K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) | ML 파이프라인 베스트 프랙티스 |
| **davila7/senior-ml-engineer** | [agentskills.so](https://agentskills.so/skills/davila7-claude-code-templates-senior-ml-engineer) | ML 코드 설계 패턴 |
| **test-driven-development** | ComposioHQ/awesome-claude-skills | pytest 기반 테스트 자동화 |
| **using-git-worktrees** | ComposioHQ/awesome-claude-skills | 병렬 에이전트 격리 |

---

## 직접 작성한 Skills

- `nlp-korean/SKILL.md` — 한국어 식재료 도메인 특화 (직접 작성)
- `orchestration/SKILL.md` — RE:FRIDGE ML 멀티 에이전트 오케스트레이터 (직접 작성)
- `experiment-log/` — ML 실험 회고 템플릿 (Sionic AI 방식 적용)
