import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.heliomorph import HelioMorphEdge
from dataset.solar_dataset import SolarFaultDataset
from tqdm import tqdm


def build_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor()
    ])


def main(cfg):
    os.makedirs(cfg['output']['dir'], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_transforms(cfg['data']['img_size'])
    ds = SolarFaultDataset(cfg['data']['root'], transform=transform)
    n = len(ds)
    n_train = int(n * cfg['data']['split'][0])
    n_val = int(n * cfg['data']['split'][1])
    n_test = n - n_train - n_val
    train_ds, val_ds, _ = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    model = HelioMorphEdge(num_classes=cfg['model']['num_classes'], dim=cfg['model']['tsve_dim'], k=cfg['model']['token_prune_k']).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['training']['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])

    best_val = 0.0
    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch} | TrainLoss {train_loss/len(train_loader):.4f} | ValAcc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(cfg['output']['dir'], "best_model.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
