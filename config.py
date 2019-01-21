import os
import os.path as osp
import sys
import numpy as np
import glob

class Config:

    trainA_dir = '../../ChinesePaintingDataset/content_images/main'
    trainB_dir = '../../ChinesePaintingDataset/style_images'
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    user_name = 'default'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..', '..')
    proj_name = this_dir_name

    # output path
    output_dir = os.path.join(root_dir, 'logs', user_name + '.' + this_dir_name)
    model_dump_dir = os.path.join(output_dir, 'model_dump')
    result_dir = os.path.join(output_dir, 'result')
    tb_dir = os.path.join(output_dir, 'tf_board')
    
    new_pretrain = None
    # hyper param
    REAL_LABEL = 0.9
    batch_size = 1
    img_h = 256
    img_w = 256
    use_lsgan = True
    norm = 'instance'
    lambda1 = 10
    lambda2 = 10
    alpha = 10
    gan = 'lsgan'
    D_init_weight = 2e-2
    G_init_weight = 2e-2

    # train
    D_times = 1

    load_model = None
    #load_model = model_dump_dir

    pool_size = 50
    save_freq = 500
    
    learning_rate = 1e-4

    """
    # linear lr decay schedule
    start_decay_step = 3000
    decay_steps = 6000
    end_learning_rate = 1e-5
    """

    boundaries = [100000, 200000]
    lr_values = [1e-4, 1e-5, 5e-6]

    beta1 = 0.5 # adam optimizer
    ngf = 64

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def make_link(dest_path, link_path):
    if os.path.islink(link_path):
        os.system('rm {}'.format(link_path))
    os.system('ln -s {} {}'.format(dest_path, link_path))

def make_dir(path):
    if os.path.exists(path) or os.path.islink(path):
        return
    os.makedirs(path)

def del_file(path, msg='{} deleted.'):
    if os.path.exists(path):
        os.remove(path)
        print(msg.format(path))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'lib'))
make_link(cfg.output_dir, './log')
make_dir(cfg.output_dir)
make_dir(cfg.model_dump_dir)
make_dir(cfg.result_dir)
make_dir(cfg.tb_dir)
