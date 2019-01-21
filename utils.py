import tensorflow as tf
import numpy as np
import scipy.misc
import random

try:
    from StringIO import StringIO # Python 2.7
except:
    from io import BytesIO # Python 3.x

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    
    def __init__(self, logdir):
        self.writer = tf.summary.FileWriter(logdir)

    def scalar_summary(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
            simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        
        img_summaries = []
        for i, img in enumerate(images):
            # write image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format='png')
            
            # create a image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                    height=img.shape[0], width=img.shape[1])

            img_summaries.append(tf.Summary.Value(
                    tag='%s/%d' % (tag, i), image=img_sum))
        
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)


def convert2int(image):
    # [0. 1] -> [0, 255]
    return tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

def convert2float(image):
    # [0, 255] -> [0, 1]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image/255.0)

def batch_convert2int(images):
    return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
    return tf.map_fn(convert2float, images, dtype=tf.float32)

class ImagePool:
    # history of generated images

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
