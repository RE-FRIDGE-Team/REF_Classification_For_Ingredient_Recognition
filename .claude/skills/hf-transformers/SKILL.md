---
name: hf-transformers-kobert
description: >
  KoBERT/KoELECTRA Fine-tuning with HuggingFace Transformers for Korean NLP classification.
  Use when implementing multi-task classification, ONNX export, or tokenizer setup with
  monologg/kobert or monologg/koelectra-base-v3-discriminator models.
  Triggers on: "kobert", "koelectra", "fine-tune", "multi-task", "transformers trainer",
  "onnx export", "classification head" mentions.
compatibility: claude-code, cursor, codex
source: huggingface/skills (adapted for RE:FRIDGE Korean NLP)
---

# HuggingFace Transformers — KoBERT/KoELECTRA 가이드

## 모델 로드 패턴

```python
from transformers import AutoTokenizer, AutoModel
import torch

# KoBERT
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# KoELECTRA (권장 — 더 빠르고 성능 유사)
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
```

## Multi-task 3헤드 모델 구조

```python
class Stage1MultiTaskModel(torch.nn.Module):
    """대분류 / 중분류 / 카테고리태그 동시 예측"""
    def __init__(self, backbone_name: str, n_large: int, n_medium: int, n_tag: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size  # 768
        self.head_large  = torch.nn.Linear(hidden, n_large)
        self.head_medium = torch.nn.Linear(hidden, n_medium)
        self.head_tag    = torch.nn.Linear(hidden, n_tag)

    def forward(self, input_ids, attention_mask):
        cls = self.backbone(input_ids, attention_mask).last_hidden_state[:, 0]
        return (self.head_large(cls),
                self.head_medium(cls),
                self.head_tag(cls))
```

## Layer-wise LR Decay

```python
def get_layerwise_params(model, base_lr: float, decay: float = 0.9):
    params = []
    layers = list(model.backbone.encoder.layer)
    for i, layer in enumerate(layers):
        lr = base_lr * (decay ** (len(layers) - i))
        params.append({"params": layer.parameters(), "lr": lr})
    params.append({"params": model.head_large.parameters(),  "lr": base_lr})
    params.append({"params": model.head_medium.parameters(), "lr": base_lr})
    params.append({"params": model.head_tag.parameters(),    "lr": base_lr})
    return params
```

## 합성 Loss (가중치)

```python
def multitask_loss(logits_large, logits_medium, logits_tag,
                   labels_large, labels_medium, labels_tag,
                   w=(0.4, 0.4, 0.2)):
    ce = torch.nn.CrossEntropyLoss()
    return w[0]*ce(logits_large, labels_large) + \
           w[1]*ce(logits_medium, labels_medium) + \
           w[2]*ce(logits_tag, labels_tag)
```

## ONNX 내보내기 (서빙용)

```python
torch.onnx.export(
    model, (input_ids, attention_mask),
    "models/stage1.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["large", "medium", "tag"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
    opset_version=17,
)
```

## Confidence Gate

```python
import torch.nn.functional as F

def apply_confidence_gate(logits_large, logits_medium, logits_tag,
                           th=(0.80, 0.60, 0.50)):
    prob_l = F.softmax(logits_large,  dim=-1)
    prob_m = F.softmax(logits_medium, dim=-1)
    prob_t = F.softmax(logits_tag,    dim=-1)

    conf_l, pred_l = prob_l.max(-1)
    conf_m, pred_m = prob_m.topk(2, dim=-1)  # top-2 반환
    conf_t, pred_t = prob_t.max(-1)

    return {
        "large":  pred_l if conf_l >= th[0] else None,
        "medium": pred_m if conf_m[:, 0] >= th[1] else pred_m,  # top-2 전달
        "tag":    pred_t if conf_t >= th[2] else None,
        "conf_large": conf_l.item(),
    }
```
