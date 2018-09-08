import tensorflow as tf
import numpy as np
from config import cfg
import math
share_count = 0


def conv_bn_layer(input_fm, filters, kernel_size, stride, training, padding=1,sdd = None, reuse=False, name=None):
    conv = tf.layers.conv2d(
        inputs=input_fm, filters=filters, kernel_size=kernel_size, strides=stride,
        padding=('SAME' if stride == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=sdd) if sdd is not None else
        tf.variance_scaling_initializer(scale=4.0, mode='fan_avg'), reuse=reuse, name=name)
    bn = tf.layers.batch_normalization(inputs=conv, training=training, fused=True, reuse=reuse, name='bn'+name if name is not None else None)
    bn_relu = tf.nn.relu(bn)
    return bn_relu


def compute_standard_deviation(c_i, c_o, n_i, n_o):
    # initialize  standard deviation according equation 15
    return 2.0/math.sqrt(c_i*n_i + c_o*n_o)


def prm(input_fm, is_train, f_i=3, f_o=3):
    share_input = conv_bn_layer(input_fm, 28, 1, 1, is_train, sdd=compute_standard_deviation(f_i, 4, 256, 3*28))
    sz = share_input.shape
    out_sz = (sz[1], sz[2])
    ratio0 = conv_bn_layer(input_fm, 28, 1, 1, is_train, compute_standard_deviation(f_i, 4, 256, 3*28))
    ratio1, _, _ = tf.nn.fractional_max_pool(share_input, [1.0, 1.189, 1.189, 1.0])
    ratio2, _, _ = tf.nn.fractional_max_pool(share_input, [1.0, 1.414, 1.414, 1.0])
    ratio3, _, _ = tf.nn.fractional_max_pool(share_input, [1.0, 1.681, 1.681, 1.0])
    ratio4, _, _ = tf.nn.fractional_max_pool(share_input, [1.0, 2.000, 2.000, 1.0])
    #todo add share
    global share_count
    conv_share0 = conv_bn_layer(ratio0, 28, 3, 1, is_train, padding=0,name='conv_share_'+str(share_count))
    conv_share1 = conv_bn_layer(ratio1, 28, 3, 1, is_train, padding=0,reuse=True, name='conv_share_'+str(share_count))
    conv_share2 = conv_bn_layer(ratio2, 28, 3, 1, is_train, padding=0,reuse=True, name='conv_share_'+str(share_count))
    conv_share3 = conv_bn_layer(ratio3, 28, 3, 1, is_train, padding=0,reuse=True, name='conv_share_'+str(share_count))
    conv_share4 = conv_bn_layer(ratio4, 28, 3, 1, is_train, padding=0,reuse=True, name='conv_share_'+str(share_count))
    share_count += 1
    upsample1 = tf.image.resize_bilinear(conv_share1, out_sz)
    upsample2 = tf.image.resize_bilinear(conv_share2, out_sz)
    upsample3 = tf.image.resize_bilinear(conv_share3, out_sz)
    upsample4 = tf.image.resize_bilinear(conv_share4, out_sz)
    additon = upsample1+upsample2+upsample3+upsample4
    conv_out0 = conv_bn_layer(conv_share0, 256, 1, 1, is_train, compute_standard_deviation(1, f_o, 3*28, 256))
    conv_out1 = conv_bn_layer(additon, 256, 1, 1, is_train, compute_standard_deviation(4, f_o, 3*28, 256))
    identity_conv_map = conv_bn_layer(input_fm, 256, 1, 1, is_train, compute_standard_deviation(f_i, f_o, 256, 256))
    return conv_out0+conv_out1+identity_conv_map


def hourglass(input_fm, is_train):
    fm_down = []
    fm = input_fm
    fm_down.append(fm)
    for down_n in range(3):
        fm = tf.layers.max_pooling2d(fm, pool_size=2, strides=2)
        fm = prm(fm, is_train, 3, 6)
        fm_down.append(fm)
    # below is up block
    last_fm = None
    for block in reversed(fm_down):
        bridge = prm(block, is_train)
        if last_fm is not None:
            sz = bridge.shape
            upsample = prm(last_fm, is_train, 6, 3)
            upsample = tf.image.resize_bilinear(upsample, (sz[-2], sz[-1]))
            last_fm = bridge+upsample
        else:
            last_fm = bridge
    last_fm = conv_bn_layer(last_fm, cfg.nr_skeleton, 1, 1, is_train)
    score_maps = conv_bn_layer(last_fm, cfg.nr_skeleton, 1, 1, is_train)
    hourglass_out0 = conv_bn_layer(score_maps, 256, 1, 1, is_train)
    hourglass_out1 = conv_bn_layer(last_fm, 256, 1, 1, is_train)
    hourglass_out = hourglass_out0+hourglass_out1+input_fm
    return hourglass_out, score_maps


def stack_hourglass(input_im, is_train):
    conv0 = conv_bn_layer(input_im, 64, 7, 2, is_train)
    prm0 = prm(conv0, is_train, 1, 1)
    max_pool = tf.layers.max_pooling2d(prm0, 2, 2)
    prm1 = prm(max_pool, is_train, 1, 3)
    score_maps = []
    hourglass_out = prm1
    for stage in range(8):
        hourglass_out, score_mapout = hourglass(hourglass_out, is_train)
        score_maps.append(score_mapout)
    return score_maps


def make_data():
    from COCOAllJoints import COCOJoints
    from dataset import Preprocessing
    def make_generator():
        d = COCOJoints()
        train_data, _ = d.load_data(1)
        for data in train_data:
            data_train = Preprocessing(data)
            for i in range(4):
                yield [data_train[j][i] for j in range(3)]
    a = make_generator()
    while True:
        image = []
        heatmap = []
        valid = []
        for i in range(cfg.batch_size):
            try:
                 b=next(a)
            except:
                yield [image, heatmap, valid]
                break
            else:
                image.append(b[0])
                heatmap.append(b[1])
                valid.append(b[2])
        yield [image, heatmap, valid]


def train():
    image = tf.placeholder(tf.float32, shape=[None, *cfg.data_shape, 3])
    labels = tf.placeholder(tf.float32, shape=[None, *cfg.output_shape, cfg.nr_skeleton])
    valids = tf.placeholder(tf.float32, shape=[None, cfg.nr_skeleton])
    is_train = tf.placeholder(tf.bool)
    stack_out = stack_hourglass(image, is_train)
    global_loss = 0.
    for i, (global_out, label) in enumerate(zip(stack_out, labels)):
        global_label = label * tf.to_float(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.nr_skeleton)), 1.1))
        global_loss += tf.reduce_mean(tf.square(global_out - global_label))
    global_loss /= 2.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.0007, batch * 3, 10, 0.95, staircase=True)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(global_loss, global_step=batch)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for data in make_data():
            loss,_ = sess.run([global_loss, train_step], feed_dict={image:data[0], labels:data[1], valids:data[2], is_train:True})
            print('loss is {}'.format(loss))

if __name__ == '__main__':
    train()
