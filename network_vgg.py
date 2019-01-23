import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import utils
import cv2
import numpy as np
import math
import vgg

CONTENT_LAYERS = ('relu2_2')
network = "imagenet-vgg-verydeep-19.mat"

def gaussian_kernel(size: int, mean: float, std: float,):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',vals,vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def safe_log(x, eps=1e-12):
    return tf.log(x + eps)

def _norm(input, is_training, norm='instance', real_fake='0'):
    if norm == 'instance':
        with tf.variable_scope('instance_norm'):
            depth = input.get_shape()[3]
            scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(
                        mean=1.0, stddev=0.001, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=0.001, dtype=tf.float32))
            mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input-mean)*inv
            return scale*normalized + offset
    elif norm == 'batch':
        # tf.AUTO_REUSE to create new var if doesn't exist
        with tf.variable_scope("batch_norm_" + real_fake, reuse=tf.AUTO_REUSE):
            return tf.contrib.layers.batch_norm(input, decay=0.9, scale=True,
                        updates_collections=None, is_training=is_training)
    else:
        return input

def _leaky_relu(input):
    return tf.maximum(0.2*input, input)

def conv_norm_act(input, n_channel, k_size, stride, norm, reuse, is_training, pad='VALID', act=None, real_fake='0', name=None):
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(input, filters=n_channel, kernel_size=k_size, use_bias=True,
                        strides=stride, padding=pad, reuse=reuse,
                        kernel_initializer=tf.truncated_normal_initializer(
                            mean=0.0, stddev=cfg.D_init_weight, dtype=tf.float32))
        normed = _norm(conv, is_training, norm, real_fake)
        if act == None:
            return normed
        elif act == 'relu':
            return tf.nn.relu(normed)
        elif act == 'tanh':
            return tf.nn.tanh(normed)
        elif act == 'leaky_relu':
            return _leaky_relu(normed)

def upsample_and_concat(x1, x2, output_channels, name, reuse, norm, is_training, real_fake='0'):
    with tf.variable_scope(name, reuse=reuse):
        # ban norm in up_concat layer
        norm = None

        # conv + resize + concat
        conv1 = tf.layers.conv2d(x1, filters=output_channels, kernel_size=3, use_bias=True,
                    strides=(1, 1), padding='SAME', name='upconv', reuse=reuse,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=cfg.G_init_weight))
        uh = x2.get_shape()[1]
        uw = x2.get_shape()[2]
        upsampled = tf.image.resize_images(conv1, (uh, uw))
        up_concat = tf.concat([upsampled, x2], 3)
        normed = _norm(up_concat, is_training, norm, real_fake)
        acted = _leaky_relu(normed)
    
    return acted

def simple_discriminator(input, name, reuse, is_training, norm='batch'):
    with tf.variable_scope(name, reuse=reuse):
        # convolution layers
        conv1 = conv_norm_act(input, 32, 3, (1, 1), None, reuse, is_training,
                pad='SAME', act='leaky_relu', name='d_conv1') # (w, h, 16)
        conv2 = conv_norm_act(conv1, 64, 3, (2, 2), norm, reuse, is_training,
                pad='SAME', act='leaky_relu', name='d_conv2') # (w/2, h/2, 32)
        conv3 = conv_norm_act(conv2, 128, 3, (2, 2), norm, reuse, is_training,
                pad='SAME', act='leaky_relu', name='d_conv3') # (w/4, h/4, 64)
        conv4 = conv_norm_act(conv3, 128, 3, (2, 2), norm, reuse, is_training,
                pad='SAME', act='leaky_relu', name='d_conv4') # (w/8, h/8, 128)
        conv5 = conv_norm_act(conv4, 128, 3, (2, 2), norm, reuse, is_training,
                pad='SAME', act='leaky_relu', name='d_conv5') # (w/16, h/16, 128)
        
        conv6 = tf.layers.conv2d(conv5, filters=1, kernel_size=3, strides=(1, 1),
                kernel_initializer=tf.random_normal_initializer(stddev=cfg.D_init_weight),
                padding='VALID',
                reuse=reuse, name='d_fc')
        output = conv6
        return output

class Generator:
    # Dark to Normal
    def __init__(self, name, is_training):
        self.reuse = False
        self.name = name
        self.norm = 'batch'
        self.is_training = is_training

    def __call__(self, input, real_fake='0'):
        print(self.name)
        with tf.variable_scope(self.name):
            # u-net
            # downsample
            conv1 = conv_norm_act(input, 32, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv1_1') # (512, 512, 32)
            dn1 = conv_norm_act(conv1, 32, 3, (2, 2), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv1_2') # (256, 256, 32)

            conv2 = conv_norm_act(dn1, 64, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv2_1') # (256, 256, 64)
            dn2 = conv_norm_act(conv2, 64, 3, (2, 2), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv2_2') # (128, 128, 64)

            conv3 = conv_norm_act(dn2, 128, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv3_1') # (128, 128, 128)
            dn3 = conv_norm_act(conv3, 128, 3, (2, 2), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv3_2') # (64, 64, 128)

            conv4 = conv_norm_act(dn3, 256, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv4_1') # (64, 64, 256)
            dn4 = conv_norm_act(conv4, 256, 3, (2, 2), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv4_2') # (32, 32, 256)

            conv5 = conv_norm_act(dn4, 512, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='g_conv5_1') # (32, 32, 512)


            # upsample
            up6 = upsample_and_concat(conv5, conv4, 256, 'up_conv6', self.reuse, self.norm, self.is_training, real_fake) # (64, 64, 256)
            up6 = conv_norm_act(up6, 256, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='up_conv6_conv')

            up7 = upsample_and_concat(up6, conv3, 128, 'up_conv7', self.reuse, self.norm, self.is_training, real_fake) # (128, 128, 128)
            up7 = conv_norm_act(up7, 128, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='up_conv7_conv')

            up8 = upsample_and_concat(up7, conv2, 64, 'up_conv8', self.reuse, self.norm, self.is_training, real_fake) # (256, 256, 64)
            up8 = conv_norm_act(up8, 64, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='up_conv8_conv')

            up9 = upsample_and_concat(up8, conv1, 32, 'up_conv9', self.reuse, self.norm, self.is_training, real_fake) # (512, 512, 32)
            up9 = conv_norm_act(up9, 32, 3, (1, 1), self.norm, self.reuse, self.is_training,
                pad='SAME', act='leaky_relu', real_fake=real_fake, name='up_conv9_conv')
            
            conv10 = tf.layers.conv2d(up9, filters=3, kernel_size=1, strides=(1, 1),
                    kernel_initializer=tf.random_normal_initializer(stddev=cfg.G_init_weight),
                    reuse=self.reuse, name='g_conv10')

            output = tf.nn.tanh(conv10)
        
        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        return output

class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid
        self.grad_filter = tf.reshape(tf.constant([0,0,0,-1,0,1,0,0,0], dtype=tf.float32), [1, 3, 3, 1])


    def __call__(self, input):
        with tf.variable_scope(self.name):
            rgb_opt = simple_discriminator(input, 'rgb', self.reuse, self.is_training)
            output = rgb_opt 
            """
            gray = (0.299*input[:, :, :, 0:1] + 0.587*input[:, :, :, 1:2] + 0.114*input[:, :, :, 2:3])
            ray_opt = simple_discriminator(gray, 'gray', self.reuse, self.is_training)
            dx = tf.nn.conv2d(input, self.grad_filter, [1, 1, 1, 1], padding='SAME', name='grad_x')
            dy = tf.nn.conv2d(input, tf.transpose(self.grad_filter, perm=[0, 2, 1, 3]), 
                [1, 1, 1, 1], padding='SAME', name='grad_y')
            grad = tf.concat([dx, dy], axis=3)
            grad_opt = simple_discriminator(grad, 'grad', self.reuse, self.is_training)
            output = tf.concat([rgb_opt, gray_opt, grad_opt], axis=3)
            """
            
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

class CycleGAN:
    def __init__(self, img_h=cfg.img_h, img_w=cfg.img_w):
        self.use_sigmoid = False
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        
        self.x_image = tf.placeholder(tf.float32,
            shape=[cfg.batch_size, img_h, img_w, 3])
        self.y_image = tf.placeholder(tf.float32,
            shape=[cfg.batch_size, img_h, img_w, 3])
        self.fake_x = tf.placeholder(tf.float32,
            shape=[cfg.batch_size, img_h, img_w, 3])
        self.fake_y = tf.placeholder(tf.float32,
            shape=[cfg.batch_size, img_h, img_w, 3])

        self.vgg_weights, self.vgg_mean_pixel = vgg.load_net(network)
        
        self.G = Generator('G', self.is_training)
        self.D_Y = Discriminator('D_Y', self.is_training, norm=cfg.norm,
                            use_sigmoid=self.use_sigmoid)
        self.F = Generator('F', self.is_training)
        self.D_X = Discriminator('D_X', self.is_training, norm=cfg.norm,
                            use_sigmoid=self.use_sigmoid)
        

    def model(self):

        vgg_mean_pixel = tf.cast(self.vgg_mean_pixel, tf.float32)

        fake_y = self.G(self.x_image)
        fake_x = self.F(self.y_image)
        # cycle loss
        cycle_loss = self.cycle_consistency_loss(self.G, self.F, self.x_image, self.y_image, fake_x, fake_y)
        # ink_loss
        ink_loss = self.discriminator_loss(self.D_Y, self.y_image, fake_y, gan="ink_loss")

        # identity loss
        id_loss = self.cycle_consistency_loss(self.G, self.F, self.x_image, self.y_image, self.y_image, self.x_image)

        pre_x_image = tf.subtract(self.x_image, vgg_mean_pixel)
        vgg_x = vgg.net_preloaded(self.vgg_weights, pre_x_image, max)
        pre_fake_y_image = tf.subtract(self.fake_y, vgg_mean_pixel)
        vgg_fake_y = vgg.net_preloaded(self.vgg_weights, pre_fake_y_image, max)

        pre_y_image = tf.subtract(self.y_image, vgg_mean_pixel)
        vgg_y = vgg.net_preloaded(self.vgg_weights, pre_y_image, max)
        pre_fake_x_image = tf.subtract(self.fake_x, vgg_mean_pixel)
        vgg_fake_x = vgg.net_preloaded(self.vgg_weights, pre_fake_x_image, max)

        # content loss
        index = CONTENT_LAYERS
        content_loss = self.content_loss(vgg_x[index], vgg_y[index], vgg_fake_x[index], vgg_fake_y[index])

        # X -> Y
        G_gan_loss = self.generator_loss(self.D_Y, fake_y, gan=cfg.gan)
        G_loss = G_gan_loss + cycle_loss + id_loss + content_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, self.y_image, self.fake_y, gan=cfg.gan)+ink_loss


        # Y -> X
        F_gan_loss = self.generator_loss(self.D_X, fake_x, gan=cfg.gan)
        F_loss = F_gan_loss + cycle_loss + id_loss + content_loss
        D_X_loss = self.discriminator_loss(self.D_X, self.x_image, self.fake_x, gan=cfg.gan)



        # summary
        tf.summary.histogram('D_Y/true', tf.reduce_mean(self.D_Y(self.y_image)))
        tf.summary.histogram('D_Y/fake', tf.reduce_mean(self.D_Y(self.G(self.x_image))))
        tf.summary.histogram('D_X/true', tf.reduce_mean(self.D_X(self.x_image)))
        tf.summary.histogram('D_X/fake', tf.reduce_mean(self.D_X(self.F(self.y_image))))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        tf.summary.scalar('loss/ink', ink_loss)
        tf.summary.scalar('loss/id', id_loss)
        tf.summary.scalar('loss/id', content_loss)

        x_generate = fake_y
        x_reconstruct = self.F(fake_y)

        y_generate = fake_x
        y_reconstruct = self.G(fake_x)

        tf.summary.scalar('debug/real_x_mean', tf.reduce_mean(self.x_image))
        tf.summary.scalar('debug/fake_x_mean', tf.reduce_mean(y_generate))
        tf.summary.scalar('debug/real_y_mean', tf.reduce_mean(self.y_image))
        tf.summary.scalar('debug/fake_y_mean', tf.reduce_mean(x_generate))
        
        tf.summary.image('X/input', utils.batch_convert2int(self.x_image[:, :, :, :3]))
        tf.summary.image('X/generated', utils.batch_convert2int(x_generate))
        tf.summary.image('X/reconstruction', utils.batch_convert2int(x_reconstruct[:, :, :, :3]))
        tf.summary.image('Y/input', utils.batch_convert2int(self.y_image))
        tf.summary.image('Y/generated', utils.batch_convert2int(y_generate[:, :, :, :3]))
        tf.summary.image('Y/reconstruction', utils.batch_convert2int(y_reconstruct))

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x
     
    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss, gan='lsgan'):
        def make_optimizer(loss, variables, name='Adam', gan='lsgan'):
            global_step = tf.Variable(0, trainable=False)
            """
            # linear decay lr schedule
            
            starter_learning_rate = cfg.learning_rate
            end_learning_rate = cfg.end_learning_rate
            start_decay_step = cfg.start_decay_step
            decay_steps = cfg.decay_steps
            beta1 = cfg.beta1
            
            learning_rate = (tf.where(
                tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate,
                    global_step-start_decay_step, decay_steps,
                    end_learning_rate, power=1.0),
                starter_learning_rate)
            )
            """
            learning_rate = tf.train.piecewise_constant(global_step, cfg.boundaries, cfg.lr_values) # constant lr schedule
            
            tf.summary.scalar('learning_rate/{}'.format(name),learning_rate)

            learning_step = (tf.train.AdamOptimizer(learning_rate,
                beta1=cfg.beta1, name=name).minimize(loss,
                global_step=global_step, var_list=variables))

            return learning_step

        oname = 'Adam'

        G_optimizer = make_optimizer(G_loss, self.G.variables, name=oname+'_G', gan=gan)
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name=oname+'_D_Y', gan=gan)
        F_optimizer = make_optimizer(F_loss, self.F.variables, name=oname+'_F', gan=gan)
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name=oname+'_D_X', gan=gan)

        with tf.control_dependencies([G_optimizer, F_optimizer]):
            G_opts = tf.no_op(name='G_optimizers')
        with tf.control_dependencies([D_Y_optimizer, D_X_optimizer]):
            D_opts = tf.no_op(name='D_optimizers')
        return G_opts, D_opts

    def discriminator_loss(self, D, y, fake_y, gan='lsgan'):
        if gan == 'lsgan':
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(D(y), cfg.REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
            loss = (error_real + error_fake) / 2

        elif gan == "ink_loss":

            y_grey = tf.image.rgb_to_grayscale(y)
            fake_y_grey = tf.image.rgb_to_grayscale(fake_y)

            filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)
            filter = filter[:, np.newaxis]
            filter = tf.constant(filter.reshape(3, 3, 1))

            y_erode = tf.nn.erosion2d(y_grey, filter, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
            fake_y_erode = tf.nn.erosion2d(fake_y_grey, filter, strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")

            gauss_kernel = gaussian_kernel(5, 0.0, 5.0)
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

            y_blur = tf.nn.conv2d(y_erode, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
            fake_y_blur = tf.nn.conv2d(fake_y_erode, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

            y_blur = tf.image.grayscale_to_rgb(y_blur)
            fake_y_blur = tf.image.grayscale_to_rgb(fake_y_blur)
            error_real = tf.reduce_mean(tf.squared_difference(D(y_blur), cfg.REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y_blur)))
            loss = 0.05 * (error_real + error_fake) / 2

    
        return loss

    def generator_loss(self, D, fake_y, gan='lsgan'):
        if gan == 'lsgan':
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), cfg.REAL_LABEL))

        return loss

    def cycle_consistency_loss(self, G, F, x, y, fake_x, fake_y):
        print('consistency loss')
        forward_loss = tf.reduce_mean(tf.squared_difference(F(fake_y), x))
        backward_loss = tf.reduce_mean(tf.squared_difference(G(fake_x), y))
        loss = (forward_loss + backward_loss) * cfg.lambda1 * cfg.alpha
        print('end consistency loss')
        return loss

    def content_loss(self, x, y, fake_x, fake_y):
        print('content loss')
        forward_loss = tf.reduce_mean(tf.squared_difference(fake_y, x))
        backward_loss = tf.reduce_mean(tf.squared_difference(fake_x, y))
        loss = (forward_loss + backward_loss) / 2
        print('end content loss')
        return loss

