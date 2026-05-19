import argparse
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.heliomorph import HelioMorphEdge
from dataset.solar_dataset import SolarFaultDataset


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor()
    ])
    ds = SolarFaultDataset(cfg['data']['root'], transform=transform)
    n = len(ds)
    n_train = int(n * cfg['data']['split'][0])
    n_val = int(n * cfg['data']['split'][1])
    n_test = n - n_train - n_val
    _, _, test_ds = random_split(ds, [n_train, n_val, n_test])
    loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = HelioMorphEdge(num_classes=cfg['model']['num_classes'], dim=cfg['model']['tsve_dim'], k=cfg['model']['token_prune_k']).to(device)
    model.load_state_dict(torch.load("./outputs/best_model.pt", map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(pred)

    print(classification_report(y_true, y_pred, target_names=cfg['data']['classes']))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
