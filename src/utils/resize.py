import os
from PIL import Image


def resize(path, size=300):
    imgs = os.listdir(path)
    for i, img in enumerate(imgs):
        print(os.path.join(path, img))
        im = Image.open(os.path.join(path, img))
        im_new = im.resize((size, size), resample=Image.BICUBIC)
        im_new.save(f'{path}/{i+1:03d}.jpg', format='JPEG')


pathA = 'C:\\Users\\User_Anton\\PycharmProjects\\cyclegan\\bin\\datasets\\young2old\\trainA'
resize(pathA)
pathB = 'C:\\Users\\User_Anton\\PycharmProjects\\cyclegan\\bin\\datasets\\young2old\\trainB'
resize(pathB)