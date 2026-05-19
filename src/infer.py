import argparse
import yaml
import torch
from torchvision import transforms
from PIL import Image
from models.heliomorph import HelioMorphEdge


def main(cfg, image_path):
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
    print("Predicted class:", cfg['data']['classes'][pred])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.image)
