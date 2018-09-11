import os
import os.path as osp
import sys
import numpy as np

class Config:
    username = 'default'

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..','..')

    proj_name = this_dir_name

    # output path
    output_dir = os.path.join(root_dir, 'logs', username + '.' + this_dir_name)
    model_dump_dir = osp.join(output_dir, 'model_dump')

    display = 1

    lr = 5e-4
    lr_gamma = 0.5
    lr_dec_epoch = 60

    epoch_size = 60000 
    optimizer = 'adam'

    batch_size = 4 
    weight_decay = 1e-5

    step_size = epoch_size * lr_dec_epoch
    max_itr = epoch_size * 400
    double_bias = False

    dpflow_enable = True
    nr_dpflows = 10

    gpu_ids = '0'
    nr_gpus = 1
    continue_train = False

    def get_lr(self, itr):
        lr = self.lr * self.lr_gamma ** (itr // self.step_size)
        return lr

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.nr_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))

    bn_train = True
    init_model = osp.join(root_dir, 'inpu', 'imagenet_weights', 'res50.ckpt')

    nr_skeleton = 17
    img_path = os.path.join(root_dir, 'input', 'coco-train-2017')
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    imgExtXBorder = 0.1
    imgExtYBorder = 0.15
    min_kps = 1

    use_seg = False

    data_aug = True # has to be true
    nr_aug = 4

    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR
    pixel_norm = True
    data_shape = (256, 256) #height, width
    output_shape = (64, 64) #height, width
    gaussain_kernel = (7, 7)
    #
    gk15 = (15, 15)
    gk11 = (11, 11)
    gk9 = (9, 9)
    gk7 = (7, 7)

    gt_path = osp.join(root_dir, 'input', 'coco-train-val2017','person_keypoints_val2017.json')
    det_path = osp.join(root_dir, 'data', 'COCO', 'dets', 'person_detection_minival411_human553.json')

cfg = Config()


