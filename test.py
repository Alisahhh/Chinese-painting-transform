import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from network import CycleGAN
from dataset import Dataset
from config import cfg
import os
import logging
import time
import scipy
from datetime import datetime

def test():
    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(img_h=cfg.img_h, img_w=cfg.img_w)
        cycle_gan.fake_y = cycle_gan.G(cycle_gan.x_image)
    with tf.Session(graph=graph) as sess:
        restore = tf.train.Saver()
        restore.restore(sess, tf.train.latest_checkpoint(cfg.model_dump_dir))
        print(tf.train.latest_checkpoint(cfg.model_dump_dir))
        
        testA = Dataset(cfg.trainA_dir, train=False)
        print(testA.num_imgs)
        for i in range(testA.num_imgs):
            img_data = sess.run(testA.data)
            dark_img = img_data[0]
            ratio = img_data[1]
            fake_normal = sess.run([cycle_gan.fake_y],
                feed_dict={cycle_gan.x_image: dark_img})
            #print(i, fake_normal[0].shape)
            temp = fake_normal[0][0]
            print(i)
            scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(
                cfg.result_dir + '/%d_%d.jpg' % (i, ratio))
        
def main(unused_argv):
    test()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()


