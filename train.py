from dataset import Dataset
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from network import CycleGAN
import os
import logging
import time
from config import cfg
from datetime import datetime
from utils import ImagePool


def train():
    if cfg.load_model is not None:
        checkpoints_dir = cfg.load_model
    
    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN()
        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.model()
        G_optimizers, D_optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss, gan=cfg.gan)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(cfg.tb_dir, graph)
        #for v in tf.global_variables():
        #    print(v.name)
        if cfg.new_pretrain is not None:
            var_to_restore = []
            for v in tf.global_variables():
                var_to_restore.append(v)
            saver = tf.train.Saver(var_to_restore)
            saver_dump = tf.train.Saver()
        else:
            saver = tf.train.Saver()
            saver_dump = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if cfg.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[1].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0
            print('--------------------------------------------------------------------------------')
            if cfg.new_pretrain is not None:
                saver.restore(sess, cfg.new_pretrain)
        
        ## TODO dataset
        trainA = Dataset(cfg.trainA_dir)
        trainB = Dataset(cfg.trainB_dir)

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        D_times = 0
        G_train_times = 0

        try:
            fake_Y_pool = ImagePool(cfg.pool_size)
            fake_X_pool = ImagePool(cfg.pool_size)

            while not coord.should_stop():
                st_t = time.time()
                # generate data
                x_image = sess.run(trainA.data)[0]
                #x_image = x_image + tf.random_normal(shape=tf.shape(x_image), mean=0.0, stddev=0.1, dtype=tf.float32)
                y_image = sess.run(trainB.data)[0]
               # y_image = y_image + tf.random_normal(shape=tf.shape(y_image), mean=0.0, stddev=0.1, dtype=tf.float32)


                data_time = time.time() - st_t
                st_t = time.time()
                # generate fake_x, fake_y
                fake_y_val, fake_x_val = sess.run([fake_y, fake_x],
                    feed_dict={cycle_gan.x_image: x_image,
                               cycle_gan.y_image: y_image})
                gen_fake_time = time.time() - st_t
                st_t = time.time()
                # train
                # Discrminator
                
                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = \
                        sess.run([D_optimizers, G_loss, D_Y_loss, F_loss, D_X_loss,
                        summary_op], feed_dict={
                            cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                            cycle_gan.fake_x: fake_X_pool.query(fake_x_val),
                            cycle_gan.x_image: x_image,
                            cycle_gan.y_image: y_image})
                
                if D_times > 0 and D_times % cfg.D_times == 0:
                    D_times = 0
                    G_train_times += 1
                    _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = \
                        sess.run([G_optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, 
                        summary_op], feed_dict={
                            cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                            cycle_gan.fake_x: fake_X_pool.query(fake_x_val),
                            cycle_gan.x_image: x_image,
                            cycle_gan.y_image: y_image})
                
                bp_time = time.time() - st_t
                train_writer.add_summary(summary, step)
                train_writer.flush()
                
                if step % 1 == 0:
                    logging.info('step {} | G_loss : {:.4f} | D_Y_loss : {:.4f} | F_loss : {:.4f} |'
                    'D_X_loss : {:.4f} | g_train_times: {} | data {:.3f}s | gen_fake {:.3f}s | bp {:.3f}s'.format(step, 
                    G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, G_train_times,
                    data_time, gen_fake_time, bp_time))

                if step % 100 == 0:
                    save_path = saver_dump.save(sess, cfg.model_dump_dir + '/model.ckpt', global_step=step)
                    logging.info('model saved in files: %s' % save_path)
                
                D_times += 1
                step += 1
        
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver_dump.save(sess, cfg.model_dump_dir + '/model.ckpt', global_step=step)
            logging.info('model saved in files: %s' % save_path)
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()

            

