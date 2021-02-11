import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

import modules
import utils
import dataset


def test(opt, Images, G, F):
    G.load_state_dict(torch.load(os.path.abspath(opt.G)))
    F.load_state_dict(torch.load(os.path.abspath(opt.F)))
    G.eval()
    F.eval()

    progress = tqdm(enumerate(Images), total=len(Images))
    for i, item in progress:
        real_X = item['A'].to(device)
        real_Y = item['B'].to(device)
        fake_X = 0.5 * (F(real_Y).data + 1.0)
        fake_Y = 0.5 * (G(real_X).data + 1.0)
        save_image(fake_X.detach(), f'{opt.outf}/{opt.dataset}/A/fake_{i + 1:04d}.png', normalize=True)
        save_image(fake_Y.detach(), f'{opt.outf}/{opt.dataset}/B/fake_{i + 1:04d}.png', normalize=True)
        save_image(real_X.detach(), f'{opt.outf}/{opt.dataset}/B/real_{i + 1:04d}.png', normalize=True)
        save_image(real_Y.detach(), f'{opt.outf}/{opt.dataset}/A/real_{i + 1:04d}.png', normalize=True)
        progress.set_description(f"Generating image {i + 1} of {len(Images)}")


parser = argparse.ArgumentParser(
    description='CycleGAN test images generation'
)
parser.add_argument('--dataroot', type=str, default='', help='path to images')
parser.add_argument('--dataset', type=str, default='monet2photo', help='name of dataset (as in cyclegan paper)')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='./results', help='path to generated images')
parser.add_argument('--G', type=str, default='', help='path to trained G-models weights')
parser.add_argument('--F', type=str, default='', help='path to trained F-models weights')
parser.add_argument('--in_ch', type=int, default=3)
parser.add_argument('--out_ch', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--image_size', type=int, default=256, help='size of images (as squares)')
parser.add_argument('--seed', type=int, help='seed for testing')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(os.path.join(opt.outf, opt.dataset))
except OSError:
    pass

if opt.seed is None:
    opt.seed = np.random.randint(1, 1000)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

try:
    os.makedirs(os.path.join(opt.outf, opt.dataset, "A"))
    os.makedirs(os.path.join(opt.outf, opt.dataset, "B"))
except OSError:
    pass


device = torch.device('cuda:0' if opt.cuda else 'cpu')


ImgDataset = dataset.ImageDataset(
    root=os.path.abspath(os.path.join(opt.dataroot, opt.dataset)),
    mode='test',
    transform=transforms.Compose([
        transforms.Resize(int(opt.image_size * 1.2)),
        transforms.RandomCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    unaligned=True
)

ImgDataLoader = DataLoader(ImgDataset, batch_size=opt.batch_size,
                           shuffle=False, pin_memory=True)

G_X2Y = modules.Generator(in_ch=opt.in_ch, out_ch=opt.out_ch).to(device)
F_Y2X = modules.Generator(in_ch=opt.in_ch, out_ch=opt.out_ch).to(device)
test(opt, ImgDataLoader, G_X2Y, F_Y2X)



