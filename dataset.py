import glob
import numpy as np
import tensorflow as tf
import os
import time
import rawpy
from config import cfg
import utils
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import random
import pandas as pd

gt_images = [None] * 6000



def crop_normal(img):
    H = img.shape[0]
    W = img.shape[1]
    ps = cfg.image_size
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    patch = img[yy:yy + ps, xx:xx + ps, :]
    return patch


def augmentation(img):
    if random.random() > 0.5:
        img = np.flip(img, axis=0)
    if random.random() > 0.5:
        img = np.flip(img, axis=1)
    if random.random() > 0.5:
        img = np.transpose(img, (1, 0, 2))
    return img


def _parse_normal(filename):
    im = scipy.misc.imread(filename)
    if im.ndim == 2:
        im = im[:,:,np.newaxis]
        im = np.concatenate((im, im, im), axis=2)
    elif im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    #print('raw', im.shape)
    im = scipy.misc.imresize(im, (cfg.img_h, cfg.img_w))
    '''
    try:
        im = scipy.misc.imresize(im, (cfg.img_h, cfg.img_w))
    except:
        print ('failed resizing image %s' % (filename))
        raise '''
    gt_image = np.float32(im / 255.0)
    # gt_patch = crop_normal(gt_image)
    # gt_patch = augmentation(gt_patch)
    return gt_image

class Dataset:
    def __init__(self, data_dir, train=True):
        if train:
            self.files = glob.glob(data_dir + '/*.jpg')
            self.num_imgs = len(self.files)
            print(self.num_imgs)
            dataset = tf.data.Dataset.from_tensor_slices((self.files))
            dataset.shuffle(buffer_size=10000)
            dataset = dataset.map(
                lambda fn: tuple(tf.py_func(
                    _parse_normal, [fn], [tf.float32])), num_parallel_calls=1)
            dataset = dataset.batch(1).prefetch(1).repeat()
            iterator = dataset.make_one_shot_iterator()
            self.data = iterator.get_next()
            self.dataset = dataset

'''
if __name__ == '__main__':
    df = pd.read_csv(cfg.label_file)
    dark = df[df['lighting'] == 'Night']
    dark_list = dark['image_filename'].tolist()
    print(len(dark_list))

    from tqdm import tqdm

    pbar = tqdm(total=len(dark_list))
    for filename in dark_list:
        path = os.path.join(cfg.trainA_dir, filename)
        im = scipy.misc.imread(path)
        h, w = im.shape[:2]
        mean = im.mean()
        pbar.update(1)
    # ds = Dataset(cfg.trainA_dir, 'dark')
    # ds = Dataset(cfg.trainB_dir, 'normal')
    # ds = Dataset(cfg.train_ex_normal_dir, 'ex_normal')
     
    with tf.Session() as sess:
        for i in range(100):
            st = time.time()
            val = sess.run(ds.data)
            #val2 = sess.run(ds2.data)
            print(i, val[0].shape, val[0].dtype, time.time() - st)
    '''





