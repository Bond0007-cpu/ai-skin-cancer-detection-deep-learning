# ADR-001: Model Selection

## Status: Accepted

## Context
Need a CNN backbone that balances accuracy, inference speed, and parameter efficiency
for a medical imaging classification task on dermoscopic images.

## Decision
Use **EfficientNetB4** as the primary backbone with ImageNet pretrained weights.

## Rationale
- Compound scaling gives better accuracy-to-FLOPs ratio than ResNet50
- B4 variant (19M params) fits within GPU memory constraints during training
- Achieves SOTA on ISIC 2019 benchmark (95%+ accuracy)
- Supports fine-tuning with frozen early layers for limited medical datasets

## Alternatives Considered
| Model        | Accuracy | Params | Latency | Decision   |
|--------------|----------|--------|---------|------------|
| EfficientNetB4 | 95.2%  | 19M    | 28ms    | ✅ Selected |
| ResNet50       | 93.8%  | 25M    | 22ms    | Ensemble   |
| DenseNet121    | 94.1%  | 8M     | 35ms    | Ensemble   |
| InceptionV3    | 92.6%  | 23M    | 30ms    | Rejected   |
| VGG16          | 91.2%  | 138M   | 45ms    | Rejected   |

## Consequences
- Must implement 2-phase training: freeze base → unfreeze last 20 layers
- Requires minimum batch size 16 on 8GB GPU
- Ensemble inference adds ~80ms latency (acceptable for clinical use)
