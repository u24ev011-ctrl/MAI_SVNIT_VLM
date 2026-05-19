import argparse
import yaml
import torch
from models.tsve import TSVE


def main(cfg):
    model = TSVE(dim=cfg['model']['tsve_dim'])
    dummy = torch.randn(1, 3, cfg['data']['img_size'], cfg['data']['img_size'])
    torch.onnx.export(model, dummy, "exports/heliomorph.onnx", opset_version=12)
    print("Exported ONNX")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
