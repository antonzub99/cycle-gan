import argparse
import os
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

import modules
import utils
import dataset


def load_weights(opt, G, F, DY, DX):
    if opt.G != '':
        G.load_state_dict(torch.load(opt.G))
    if opt.F != '':
        F.load_state_dict(torch.load(opt.F))
    if opt.DY != '':
        DY.load_state_dict(torch.load(opt.DY))
    if opt.DX != '':
        DX.load_state_dict(torch.load(opt.DX))


def save_weights(opt, G, F, DY, DX, folder=None, epoch=None):
    if (epoch is None) and (folder is None):
        torch.save(G.state_dict(), f'weights/{opt.dataset}/G_X2Y.pth')
        torch.save(F.state_dict(), f'weights/{opt.dataset}/F_Y2X.pth')
        torch.save(DY.state_dict(), f'weights/{opt.dataset}/DY.pth')
        torch.save(DX.state_dict(), f'weights/{opt.dataset}/DX.pth')
    else:
        torch.save(G.state_dict(), f'{opt.outf}/{opt.dataset}/{folder}/G_X2Y_{epoch}.pth')
        torch.save(F.state_dict(), f'{opt.outf}/{opt.dataset}/{folder}/F_Y2X_{epoch}.pth')
        torch.save(DY.state_dict(), f'{opt.outf}/{opt.dataset}/{folder}/DY_{epoch}.pth')
        torch.save(DX.state_dict(), f'{opt.outf}/{opt.dataset}/{folder}/DX_{epoch}.pth')


def train(opt, Data, G, F, DY, DX, weights_folder):
    G.train()
    F.train()
    DY.train()
    DX.train()
    losses = {
        'gen': [],
        'disc': [],
        'idt': [],
        'cycle': [],
        'gan': []
    }
    gancrit = nn.MSELoss().to(device)
    cyclecrit = nn.L1Loss().to(device)
    idtcrit = nn.L1Loss().to(device)

    optimG = optim.Adam(itertools.chain(G.parameters(), F.parameters()),
                        lr=opt.lr, betas=(opt.beta, 0.999))
    optimDY = optim.Adam(DY.parameters(), lr=0.5 * opt.lr, betas=(opt.beta, 0.999))
    optimDX = optim.Adam(DX.parameters(), lr=0.5 * opt.lr, betas=(opt.beta, 0.999))

    schedulerG = utils.linear_decay_scheduler(optimG, opt)
    schedulerDY = utils.linear_decay_scheduler(optimDY, opt)
    schedulerDX = utils.linear_decay_scheduler(optimDX, opt)

    fake_X_pool = utils.ImagePool(opt.pool_size)
    fake_Y_pool = utils.ImagePool(opt.pool_size)

    for epoch in range(opt.epochs):
        progress = tqdm(enumerate(ImgDataLoader), total=len(ImgDataLoader))
        for i, item in progress:
            direct = opt.direction == 'XtoY'
            real_X = item['A' if direct else 'B'].to(device)
            real_Y = item['B' if direct else 'A'].to(device)
            batch_size = real_X.size(0)
            real_label = torch.tensor(1.0, dtype=torch.float32).to(device)
            fake_label = torch.tensor(0.0, dtype=torch.float32).to(device)

            optimG.zero_grad()

            # GAN loss
            fake_X = F(real_Y)
            fake_res_X = DX(fake_X)
            gan_loss_Y2X = gancrit(fake_res_X, real_label.expand_as(fake_res_X))
            fake_Y = G(real_X)
            fake_res_Y = DY(fake_Y)
            gan_loss_X2Y = gancrit(fake_res_Y, real_label.expand_as(fake_res_X))

            # cycle loss
            rec_X = F(fake_Y)
            cycle_loss_X = cyclecrit(rec_X, real_X) * opt.factor
            rec_Y = G(fake_X)
            cycle_loss_Y = cyclecrit(rec_Y, real_Y) * opt.factor

            # identity loss: F(X) should be aprrox X
            idt_X = F(real_X)
            idt_loss_X = idtcrit(idt_X, real_X) * opt.factor * 0.5
            idt_Y = G(real_Y)
            idt_loss_Y = idtcrit(idt_Y, real_Y) * opt.factor * 0.5

            loss_G = gan_loss_X2Y + gan_loss_Y2X + cycle_loss_X + cycle_loss_Y + idt_loss_X + idt_loss_Y
            loss_G.backward()
            optimG.step()

            optimDY.zero_grad()
            real_res_Y = DY(real_Y)
            fake_Y = fake_Y_pool.query(fake_Y)
            fake_res_Y = DY(fake_Y.detach())
            loss_DY = (gancrit(real_res_Y, real_label.expand_as(real_res_Y)) +
                   gancrit(fake_res_Y, fake_label.expand_as(fake_res_Y))) * 0.5
            loss_DY.backward()
            optimDY.step()

            optimDX.zero_grad()
            real_res_X = DX(real_X)
            fake_X = fake_X_pool.query(fake_X)
            fake_res_X = DX(fake_X.detach())
            loss_DX = (gancrit(real_res_X, real_label.expand_as(real_res_X)) +
                   gancrit(fake_res_X, fake_label.expand_as(fake_res_X))) * 0.5
            loss_DX.backward()
            optimDX.step()

            progress.set_description(
                f"Epoch: [{epoch}/{opt.epochs - 1}] "
                f"Image: [{i}/{len(ImgDataLoader) - 1}] "
                f"Learning rate: {optimG.param_groups[0]['lr']:.7f} "
                f"Generator Loss: {loss_G.item():.3f} "
                f"Discriminator Loss: {(loss_DY + loss_DX).item():.3f} "
                f"GAN Loss: {(gan_loss_X2Y + gan_loss_Y2X).item():.3f} "
                f"Cycle Consistency Loss: {(cycle_loss_X + cycle_loss_Y).item():.3f} "
                f"Identity Loss: {(idt_loss_X + idt_loss_Y).item():.3f} "
            )
            losses['gen'].append(loss_G.item())
            losses['disc'].append((loss_DY + loss_DX).item())
            losses['idt'].append((idt_loss_X + idt_loss_Y).item())
            losses['cycle'].append((cycle_loss_X + cycle_loss_Y).item())
            losses['gan'].append((gan_loss_X2Y + gan_loss_Y2X).item())

            if i % opt.print == 0:
                save_image(real_X, f'{opt.outf}/{opt.dataset}/A/real_images.png', normalize=True)
                save_image(real_Y, f'{opt.outf}/{opt.dataset}/B/real_images.png', normalize=True)
                fake_print_X = (F(real_Y).data + 1.0) * 0.5
                fake_print_Y = (G(real_X).data + 1.0) * 0.5
                save_image(fake_print_X.detach(), f'{opt.outf}/{opt.dataset}/A/fake_images_{epoch}.png', normalize=True)
                save_image(fake_print_Y.detach(), f'{opt.outf}/{opt.dataset}/B/fake_images_{epoch}.png', normalize=True)

        save_weights(opt, G, F, DY, DX, weights_folder, epoch=epoch)

        schedulerG.step()
        schedulerDY.step()
        schedulerDX.step()

    save_weights(opt, G, F, DY, DX)
    return losses


parser = argparse.ArgumentParser(
    description="CycleGAN implementation by Anton Zubekhin with help of many people on GitHub, version 0.0")
parser.add_argument('--dataroot', default='./cyclegan/bin/datasets',
                    help='path to images')
parser.add_argument('--dataset', type=str, default='monet2photo', help='name of dataset (as in paper)')
parser.add_argument('--epochs', type=int, default=200, help='amount of epochs to run')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to begin reducing learning rate')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--in_ch', type=int, default=3)
parser.add_argument('--out_ch', type=int, default=3)
parser.add_argument('--pool_size', type=int, default=50, help='size of image pool')
parser.add_argument('--print', type=int, default=100, help='save every #th generated image')
parser.add_argument('--factor', type=float, default=10.0,
                    help='lambda for consistency loss and half-lambda for identity loss')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--direction', type=str, default='XtoY', help='XtoY or YtoX')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--image_size', type=int, default=256, help='size of images (as squares)')
parser.add_argument('--outf', default='./train_outputs')
parser.add_argument('--seed', type=int, help='seed for training')
parser.add_argument('--G', default='', help='path to saved G weights')
parser.add_argument('--F', default='', help='path to saved F weights')
parser.add_argument('--DY', default='', help='path to saved DY weights')
parser.add_argument('--DX', default='', help='path to saved DX weights')

opt = parser.parse_args()


try:
    os.makedirs(os.path.join(opt.outf, opt.dataset))
except OSError:
    pass


if opt.seed is None:
    opt.seed = np.random.randint(1, 1000)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

ImgDataset = dataset.ImageDataset(
    root=os.path.abspath(os.path.join(opt.dataroot, opt.dataset)),
    transform=transforms.Compose([
        transforms.Resize(int(opt.image_size * 1.2)),
        transforms.RandomCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    unaligned=True
)

ImgDataLoader = DataLoader(ImgDataset, batch_size=opt.batch_size,
                           shuffle=True, pin_memory=True)

weights_folder = 'saved_weights'

try:
    os.makedirs(os.path.join(opt.outf, opt.dataset, "A"))
    os.makedirs(os.path.join(opt.outf, opt.dataset, "B"))
    os.makedirs(os.path.join(opt.outf, opt.dataset, weights_folder))
    os.makedirs(os.path.abspath('weights'))
except OSError:
    pass

device = torch.device('cuda:0' if opt.cuda else 'cpu')

G_X2Y = modules.Generator(in_ch=opt.in_ch, out_ch=opt.out_ch).to(device)
F_Y2X = modules.Generator(in_ch=opt.in_ch, out_ch=opt.out_ch).to(device)
DY = modules.Discriminator(in_ch=opt.in_ch).to(device)
DX = modules.Discriminator(in_ch=opt.in_ch).to(device)

G_X2Y.apply(modules.init_weights)
F_Y2X.apply(modules.init_weights)
DY.apply(modules.init_weights)
DX.apply(modules.init_weights)


load_weights(opt, G_X2Y, F_Y2X, DY, DX)

loss_dict = train(opt, ImgDataLoader, G_X2Y, F_Y2X, DY, DX, weights_folder)


