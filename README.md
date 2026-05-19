# HelioMorph-Edge (HME)

**Prompt-Adaptive Tiny Vision-Language Model for Energy-Aware Solar Fault Diagnostics on Edge UAVs**

This repository provides a full, lightweight, edge-deployable framework for UAV-based solar panel fault detection with explainable semantic reasoning.

## Key Features
- **Tiny semantic vision encoder** (Mobile stem + Tiny transformer)
- **Token-pruned VLM adapter** for efficient inference
- **Prompt-adaptive semantic router**
- **Tiny memory-bank few-shot adapter**
- **Explainability engine** (Grad-CAM heatmaps + textual reasoning)
- **Edge optimizations** (ONNX + OpenVINO)

## Dataset
Default: **Kaggle Solar Panel Images Clean and Faulty**
Classes: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered

Update `configs/heliomorph.yaml` with your dataset path.

## Quickstart
```bash
bash setup.sh
python src/train.py --config configs/heliomorph.yaml
python src/infer.py --config configs/heliomorph.yaml --image path/to/image.jpg
```

## Explainability
```bash
python src/explain.py --config configs/heliomorph.yaml --image path/to/image.jpg --out outputs/heatmap.jpg
```

## Edge Export
```bash
python edge/export_onnx.py --config configs/heliomorph.yaml
python edge/optimize_openvino.py --onnx exports/heliomorph.onnx --outdir exports/openvino
```

## Repo Structure
See `docs/architecture.md` for details.
