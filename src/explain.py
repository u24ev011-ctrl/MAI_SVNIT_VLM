import argparse
import yaml
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from models.heliomorph import HelioMorphEdge


def grad_cam(model, x, target_class):
    x.requires_grad_(True)
    logits, tokens = model(x)
    score = logits[:, target_class].sum()
    model.zero_grad()
    score.backward()
    # tokens gradient -> simple pooling
    grads = x.grad.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    heat = grads[0,0].detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return heat


def main(cfg, image_path, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HelioMorphEdge(num_classes=cfg['model']['num_classes'], dim=cfg['model']['tsve_dim'], k=cfg['model']['token_prune_k']).to(device)
    model.load_state_dict(torch.load("./outputs/best_model.pt", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x)
        pred = logits.argmax(dim=1).item()

    heat = grad_cam(model, x, pred)
    img_cv = cv2.resize(cv2.imread(image_path), (cfg['data']['img_size'], cfg['data']['img_size']))
    heatmap = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(out_path, overlay)
    print("Saved heatmap to", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.image, args.out)
