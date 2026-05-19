import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class SolarFaultDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(root))
        for cls in self.classes:
            for img in glob.glob(os.path.join(root, cls, "*.jpg")):
                self.samples.append((img, self.classes.index(cls)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
