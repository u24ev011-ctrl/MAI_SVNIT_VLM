import argparse
import os
import yaml
import torch
from models.heliomorph import HelioMorphEdge


def main(cfg):
    os.makedirs(os.path.dirname(cfg['export']['onnx_path']), exist_ok=True)
    model = HelioMorphEdge(num_classes=cfg['model']['num_classes'], dim=cfg['model']['tsve_dim'], k=cfg['model']['token_prune_k']).eval()
    dummy = torch.randn(1, 3, cfg['data']['img_size'], cfg['data']['img_size'])
    torch.onnx.export(model, dummy, cfg['export']['onnx_path'], opset_version=12)
    print("Exported ONNX to", cfg['export']['onnx_path'])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
