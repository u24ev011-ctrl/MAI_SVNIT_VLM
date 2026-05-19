import argparse
import yaml
import torch
from thop import profile
from models.heliomorph import HelioMorphEdge


def main(cfg):
    model = HelioMorphEdge(num_classes=cfg['model']['num_classes'], dim=cfg['model']['tsve_dim'], k=cfg['model']['token_prune_k'])
    dummy = torch.randn(1, 3, cfg['data']['img_size'], cfg['data']['img_size'])
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    print(f"FLOPs: {flops/1e6:.2f} MFLOPs")
    print(f"Params: {params/1e6:.2f} M")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
