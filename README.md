# cycle-gan implementation 

## 1.Downloading dataset
Use script bin/download_cyclegan_dataset.sh, which was taken from original CycleGAN repository https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. As I use Windows, original script doesn't work well for me, so I slightly changed paths to directories. To run shell scripts I use Git Bash. While in project directory use
```
bash bin/download_cyclegan_dataset.sh monet2photo
```
My implementation was trained on monet2photo dataset and there are trained weights in ```src/weights/monet2photo ``` for both generators and discriminators.

## 2. Train model
Run this command
```
cd src
python train.py --cuda --dataroot C:\\...\\cyclegan\\bin\\datasets --dataset monet2photo 
--G weights\\monet2photo\\G_X2Y.pth --F weights\\monet2photo\\F_Y2X.pth 
--DX weights\\monet2photo\\DX.pth --DY weights\\monet2photo\\DY.pth
```
Option ```--cuda``` enables training on GPU, ```--dataroot``` is required to specify path to images and ```--dataset``` specifies dataset's name. 
Options ```--G, --F, --DX, --DY``` are optional if there pretrained weights and require paths to them. Unfortunately, I didn't make comfortable downloading script that downloads dataset in the same ```cyclegan\\src``` directory where ```train.py``` is located, so one needs (at least on Windows) to enter absolute path to dataroot in ```--dataroot``` option.   

## 3. Test model
```
python test.py --cuda --dataroot C:\\...\\cyclegan\\bin\\datasets --dataset monet2photo 
--G weights\\monet2photo\\G_X2Y.pth --F weights\\monet2photo\\F_Y2X.pth --outf outputs
```
Option ```--outf``` specifies directory, where generated images will be stored.
