import argparse
import yaml
import torch
from torchvision import transforms
from PIL import Image
from models.tsve import TSVE


def main(cfg, image_path):
    model = TSVE(dim=cfg['model']['tsve_dim']).eval()
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(x)
    print("Inference done", feat.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.image)
