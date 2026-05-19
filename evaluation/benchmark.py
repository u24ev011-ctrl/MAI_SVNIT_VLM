import argparse
import time
import yaml
import torch
import numpy as np
from models.heliomorph import HelioMorphEdge


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HelioMorphEdge(num_classes=cfg['model']['num_classes'], dim=cfg['model']['tsve_dim'], k=cfg['model']['token_prune_k']).to(device)
    model.eval()
    x = torch.randn(1, 3, cfg['data']['img_size'], cfg['data']['img_size']).to(device)

    # warmup
    for _ in range(10):
        _ = model(x)

    runs = 50
    start = time.time()
    for _ in range(runs):
        _ = model(x)
    end = time.time()
    avg_ms = (end - start) / runs * 1000
    print(f"Avg latency: {avg_ms:.2f} ms on {device}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
