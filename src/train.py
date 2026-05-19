import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.tsve import TSVE
from dataset.solar_dataset import SolarFaultDataset


def main(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        transforms.ToTensor()
    ])
    ds = SolarFaultDataset(cfg['data']['root'], transform=transform)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    model = TSVE(dim=cfg['model']['tsve_dim']).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    model.train()
    for epoch in range(cfg['training']['epochs']):
        for x, y in loader:
            x = x.cuda()
            out = model(x)
            loss = out.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
