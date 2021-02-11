import os
import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, mode='train', transform=None, unaligned=False):
        self.transform = transform
        self.unaligned = unaligned

        self.imgsX = sorted(glob.glob(os.path.join(root, f"{mode}A") + '/*.jpg'))
        self.imgsY = sorted(glob.glob(os.path.join(root, f"{mode}B") + '/*.jpg'))

    def __getitem__(self, ind):
        imgX = self.transform(Image.open(self.imgsX[ind % len(self.imgsX)]))

        if self.unaligned:
            imgY = self.transform(Image.open(self.imgsY[np.random.randint(0, len(self.imgsY)-1)]))
        else:
            imgY = self.transform(Image.open(self.imgsY[ind % len(self.imgsY)]))

        return {"A": imgX, "B": imgY}

    def __len__(self):
        return int(min(len(self.imgsX), len(self.imgsY)) * 0.75)
