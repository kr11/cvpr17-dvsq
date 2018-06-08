##################################################################################
# Deep Visual-Semantic Quantization for Efficient Image Retrieval                #
# Authors: Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu                    #
# Contact: caoyue10@gmail.com                                                    #
##################################################################################

import os
import sys
import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
from math import ceil
import random
from util import ProgressBar, MAPs, MAPs_CQ
from sklearn.cluster import MiniBatchKMeans

from vae_data import Vector_Dataset


class DVSQ(object):
    def __init__(self, config):
        ### Initialize setting
        print("initializing")
        np.set_printoptions(precision=4)
        self.stage = config['stage']
        self.device = config['device']
        #########  rename from output_dim  #########
        self.vector_dim = config['vector_dim']
        # self.n_class = config['label_dim']

        self.subspace_num = config['n_subspace']
        self.subcenter_num = config['n_subcenter']
        # self.code_batch_size = config['code_batch_size']
        self.cq_lambda = config['cq_lambda']
        self.max_iter_update_Cb = config['max_iter_update_Cb']
        self.max_iter_update_b = config['max_iter_update_b']
        # self.centers_device = config['centers_device']

        self.batch_size = config['batch_size']
        self.max_iter = config['max_iter']
        # self.img_model = config['img_model']
        self.loss_type = config['loss_type']
        self.console_log = (config['console_log'] == 1)
        self.learning_rate = config['learning_rate']
        self.learning_rate_decay_factor = config['learning_rate_decay_factor']
        self.decay_step = config['decay_step']

        # self.finetune_all = config['finetune_all']

        # self.margin_param = config['margin_param']
        # self.wordvec_dict = config['wordvec_dict']
        # self.part_ids_dict = config['part_ids_dict']
        # self.partlabel = config['partlabel']
        ### Format as 'path/to/save/dir/lr_{$0}_output_dim{$1}_iter_{$2}'
        self.save_dir = config['save_dir'] + self.loss_type + '_lr_' + str(self.learning_rate) + '_cqlambda_' + str(
            self.cq_lambda) + '_subspace_' + str(self.subspace_num)  +\
                          '_iter_' + str(self.max_iter) + '_output_' + str(self.vector_dim) + '_'
                        # '_margin_' + str(self.margin_param) + '_partlabel_' + str(self.partlabel) +\

        ### Setup session
        print("launching session")
        configProto = tf.ConfigProto()
        configProto.gpu_options.allow_growth = True
        configProto.allow_soft_placement = True
        self.sess = tf.Session(config=configProto)

        ### Create variables and placeholders

        with tf.device(self.device):
            # self.img = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
            # self.img_label = tf.placeholder(tf.float32, [self.batch_size, self.n_class])

            # self.img_last_layer, self.img_output, self.C = \
            #     self.load_model(config['model_weights'])

            ### Centers shared in different modalities (image & text)
            ### Binary codes for different modalities (image & text)
            # vae network
            self.input_vectors = tf.placeholder(tf.float32, [None, self.vector_dim], name='input_vectors')
            self.num_layers = config['num_layers']
            self.encode_weights = []
            self.encode_biases = []
            self.decode_weights = []
            self.decode_biases = []
            self.hidden_size = config['n_hidien']

            self.encode_weights.append(tf.Variable(tf.random_normal([self.vector_dim, self.hidden_size[0]])))
            self.encode_biases.append(tf.Variable(tf.random_normal([self.hidden_size[0]])))
            for layer in range(1, self.num_layers):
                self.encode_weights.append(
                    tf.Variable(tf.random_normal([self.hidden_size[layer - 1], self.hidden_size[layer]])))
                self.encode_biases.append(tf.Variable(tf.random_normal([self.hidden_size[layer]])))
            for layer in range(self.num_layers - 1, 0, -1):
                self.decode_weights.append(
                    tf.Variable(tf.random_normal([self.hidden_size[layer], self.hidden_size[layer - 1]])))
                self.decode_biases.append(tf.Variable(tf.random_normal([self.hidden_size[layer - 1]])))
            self.decode_weights.append(tf.Variable(tf.random_normal([self.hidden_size[0], self.vector_dim])))
            self.decode_biases.append(tf.Variable(tf.random_normal([self.vector_dim])))

            self.output_dim = self.hidden_size[-1]
            # self.output_vectors = tf.Variable(tf.float32, [self.batch_size, self.output_dim])

            self.C = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.output_dim],
                                                   minval=-1, maxval=1, dtype=tf.float32, name='centers'))
            self.b_img_for_loss = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num],
                                                 name='b_img_for_loss')
            self.ICM_m = tf.placeholder(tf.int32, [], name='ICM_m')
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.subcenter_num], name='ICM_b_m')
            self.ICM_b_all_for_quan = tf.placeholder(tf.float32, [None, self.subcenter_num * self.subspace_num],
                                                     name='ICM_b_all_for_quan')
            self.ICM_X = tf.placeholder(tf.float32, [self.batch_size, self.output_dim], name='ICM_X')
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0], [self.subcenter_num, self.output_dim])

            self.ICM_X_residual = tf.add(tf.subtract(self.ICM_X, tf.matmul(self.ICM_b_all_for_quan, self.C)),
                                         tf.matmul(self.ICM_b_m, self.ICM_C_m))
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 1)  # batch, 1, D
            ICM_C_m_expand = tf.expand_dims(self.ICM_C_m, 0)  # 1, |C|, D
            # N*sc*D  *  D*n
            # word_dict = tf.constant(np.loadtxt(self.wordvec_dict), dtype=tf.float32)
            # ICM_word_dict = tf.reshape(tf.matmul(tf.reshape(tf.sub(ICM_X_expand, ICM_C_m_expand), [self.code_batch_size*self.subcenter_num, self.output_dim]), tf.transpose(word_dict)), [self.code_batch_size, self.subcenter_num, self.n_class])
            # ICM_sum_squares = tf.reduce_sum(tf.square(ICM_word_dict), reduction_indices = 2)
            ICM_sum_squares = tf.reduce_sum(tf.square(tf.squeeze(tf.subtract(ICM_X_expand, ICM_C_m_expand))),
                                            axis=2)  # 计算平方和，算距离
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)  # 求最小的idx
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, self.subcenter_num, dtype=tf.float32)

            self.global_step = tf.Variable(0, trainable=False)
            self.train_op = self.apply_loss_function(self.global_step)
            self.sess.run(tf.initialize_all_variables())
        return

    def load_model(self, img_model_weights):
        if self.img_model == 'alexnet':
            img_output = self.img_alexnet_layers(img_model_weights)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def img_alexnet_layers(self, model_weights):
        self.deep_param_img = {}
        self.train_layers = []
        self.train_last_layer = []
        print("loading img model")
        net_data = np.load(model_weights).item()

        # swap(2,1,0)
        reshaped_image = tf.cast(self.img, tf.float32)
        tm = tf.Variable([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        reshaped_image = tf.reshape(reshaped_image, [self.batch_size * 256 * 256, 3])
        reshaped_image = tf.matmul(reshaped_image, tm)
        reshaped_image = tf.reshape(reshaped_image, [self.batch_size, 256, 256, 3])

        IMAGE_SIZE = 227
        height = IMAGE_SIZE
        width = IMAGE_SIZE

        ### Randomly crop a [height, width] section of each image
        distorted_image = tf.pack(
            [tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3]) for each_image in
             tf.unpack(reshaped_image)])

        ### Zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
            distorted_image = distorted_image - mean

        ### Conv1
        ### Output 96, kernel 11, stride 4
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(net_data['conv1'][0], name='weights')
            conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(net_data['conv1'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv1'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool1')

        ### LRN1
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        self.lrn1 = tf.nn.local_response_normalization(self.pool1,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv2
        ### Output 256, pad 2, kernel 5, group 2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(net_data['conv2'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.lrn1)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)

            biases = tf.Variable(net_data['conv2'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv2'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool2')

        ### LRN2
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        self.lrn2 = tf.nn.local_response_normalization(self.pool2,
                                                       depth_radius=radius,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       bias=bias)

        ### Conv3
        ### Output 384, pad 1, kernel 3
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(net_data['conv3'][0], name='weights')
            conv = tf.nn.conv2d(self.lrn2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(net_data['conv3'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv3'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Conv4
        ### Output 384, pad 1, kernel 3, group 2
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(net_data['conv4'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv3)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv4'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv4'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Conv5
        ### Output 256, pad 1, kernel 3, group 2
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(net_data['conv5'][0], name='weights')
            group = 2
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
            input_groups = tf.split(3, group, self.conv4)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            ### Concatenate the groups
            conv = tf.concat(3, output_groups)
            biases = tf.Variable(net_data['conv5'][1], name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = tf.nn.relu(out, name=scope)
            self.deep_param_img['conv5'] = [kernel, biases]
            self.train_layers += [kernel, biases]

        ### Pool5
        self.pool5 = tf.nn.max_pool(self.conv5,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='VALID',
                                    name='pool5')

        ### FC6
        ### Output 4096
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(net_data['fc6'][0], name='weights')
            fc6b = tf.Variable(net_data['fc6'][1], name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            self.fc5 = pool5_flat
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
            self.fc6o = tf.nn.relu(fc6l)
            self.deep_param_img['fc6'] = [fc6w, fc6b]
            self.train_layers += [fc6w, fc6b]

        ### FC7
        ### Output 4096
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(net_data['fc7'][0], name='weights')
            fc7b = tf.Variable(net_data['fc7'][1], name='biases')
            fc7l = tf.nn.bias_add(tf.matmul(self.fc6, fc7w), fc7b)
            self.fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
            fc7lo = tf.nn.bias_add(tf.matmul(self.fc6o, fc7w), fc7b)
            self.fc7o = tf.nn.relu(fc7lo)
            self.deep_param_img['fc7'] = [fc7w, fc7b]
            self.train_layers += [fc7w, fc7b]

        ### FC8
        ### Output output_dim
        with tf.name_scope('fc8') as scope:
            ### Differ train and val stage by 'fc8' as key
            if 'fc8' in net_data:
                fc8w = tf.Variable(net_data['fc8'][0], name='weights')
                fc8b = tf.Variable(net_data['fc8'][1], name='biases')
            else:
                fc8w = tf.Variable(tf.random_normal([4096, self.vector_dim],
                                                    dtype=tf.float32,
                                                    stddev=1e-2), name='weights')
                fc8b = tf.Variable(tf.constant(0.0, shape=[self.vector_dim],
                                               dtype=tf.float32), name='biases')
            fc8l = tf.nn.bias_add(tf.matmul(self.fc7, fc8w), fc8b)
            self.fc8 = tf.nn.tanh(fc8l)
            fc8lo = tf.nn.bias_add(tf.matmul(self.fc7o, fc8w), fc8b)
            self.fc8o = tf.nn.tanh(fc8lo)
            self.deep_param_img['fc8'] = [fc8w, fc8b]
            self.train_last_layer += [fc8w, fc8b]

        ### load centers
        if 'C' in net_data:
            self.centers = tf.Variable(net_data['C'], name='weights')
        else:
            self.centers = tf.Variable(tf.random_uniform([self.subspace_num * self.subcenter_num, self.vector_dim],
                                                         minval=-1, maxval=1, dtype=tf.float32, name='centers'))

        self.deep_param_img['C'] = self.centers

        print("img modal loading finished")
        ### Return outputs
        return self.fc8, self.fc8o, self.centers

    def save_model(self, model_file=None):
        if model_file == None:
            model_file = self.save_dir
        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])
        print("saving model to %s" % model_file)
        np.save(model_file, np.array(model))
        return

    # Building the encoder
    def encoder(self, x):
        encoded = x
        for layer in range(0, self.num_layers):
            encoded = tf.nn.sigmoid(tf.add(tf.matmul(encoded, self.encode_weights[layer]),
                                           self.encode_biases[layer]))
        return encoded

    # Building the decoder
    def decoder(self, x):
        decoded = x
        for layer in range(0, self.num_layers):
            decoded = tf.nn.sigmoid(tf.add(tf.matmul(decoded, self.decode_weights[layer]),
                                           self.decode_biases[layer]))
        return decoded

    def apply_loss_function(self, global_step):
        # autoencoder loss
        self.ICM_X = self.encoder(self.input_vectors)
        decoder_op = self.decoder(self.ICM_X)
        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = self.input_vectors
        # Define loss and optimizer, minimize the squared error
        self.cos_loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        # total loss
        self.cq_loss_img = tf.reduce_mean(
            tf.reduce_sum(tf.square(tf.subtract(self.ICM_X, tf.matmul(self.b_img_for_loss, self.C))), 1))
        self.q_lambda = tf.Variable(self.cq_lambda, name='cq_lambda')
        self.cq_loss = tf.multiply(self.q_lambda, self.cq_loss_img)
        self.loss = tf.add(self.cos_loss, self.cq_loss)

        # Last layer has a 10 times learning rate
        self.lr = tf.train.exponential_decay(self.learning_rate, global_step, self.decay_step,
                                             self.learning_rate_decay_factor, staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        return opt.minimize(self.loss, global_step=self.global_step)

    def initial_centers(self, img_output):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, self.output_dim])
        print("#DVSQ train# initilizing Centers")
        all_output = img_output
        for i in range(self.subspace_num):
            output_dim_start = int(i * self.output_dim / self.subspace_num)
            output_dim_end = int((i + 1) * self.output_dim / self.subspace_num)
            subcenter_num_start = int(i * self.subcenter_num)
            subcenter_num_end = int((i+1) * self.subcenter_num)
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(all_output[:, output_dim_start: output_dim_end])
            C_init[subcenter_num_start: subcenter_num_end, output_dim_start:output_dim_end] = kmeans.cluster_centers_
            print("step: ", i, " finish")
        return C_init

    def update_centers(self, img_dataset):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        old_C_value = self.sess.run(self.C)

        h = self.img_b_all
        U = self.img_output_all
        smallResidual = tf.constant(np.eye(self.subcenter_num * self.subspace_num, dtype=np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)

        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict={
            self.img_output_all: img_dataset.output_vectors,
            self.img_b_all: img_dataset.codes,
        })

        C_sums = np.sum(np.square(C_value), axis=1)
        C_zeros_ids = np.where(C_sums < 1e-8)
        C_value[C_zeros_ids, :] = old_C_value[C_zeros_ids, :]
        self.sess.run(self.C.assign(C_value))

        print('updated C is:')
        print(C_value)
        print("non zeros:")
        print(len(np.where(np.sum(C_value, 1) != 0)[0]))

    def update_codes_ICM(self, output, code):
        '''
        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, n_output]
            self.C: [n_subspace * n_subcenter, n_output]
                [C_1, C_2, ... C_M]
            codes: [n_train, n_subspace * n_subcenter]
        '''

        code = np.zeros(code.shape)

        for iterate in range(self.max_iter_update_b):
            sub_list = [i for i in range(self.subspace_num)]
            random.shuffle(sub_list)
            for m in sub_list:
                temp = np.zeros([self.batch_size, self.output_dim], dtype=float)
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict={
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all_for_quan: code,
                    self.ICM_m: m,
                    self.ICM_X: output,

                })

                code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot_val
        return code

    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        total_batch = int(ceil(dataset.n_samples / batch_size))
        print("start update codes in batch size ", batch_size)

        dataset.finish_epoch()

        for i in range(total_batch):
            print("Iter ", i, "of ", total_batch)
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            print(output_val, code_val)
            codes_val = self.update_codes_ICM(output_val, code_val)
            print(np.sum(np.sum(codes_val, 0) != 0))
            dataset.feed_batch_codes(batch_size, codes_val)

        print("update_code wrong:")
        print(np.sum(np.sum(dataset.codes, 1) != 4))

        print("######### update codes done ##########")

    def train_dvsq(self, img_dataset):
        print("%s #train# start training" % datetime.now())
        epoch = 0
        epoch_iter = int(ceil(img_dataset.n_samples / self.batch_size))

        for train_iter in range(self.max_iter):
            # images, labels, codes = img_dataset.next_batch(self.batch_size)
            batch_X, codes = img_dataset.next_batch(self.batch_size)
            start_time = time.time()

            if epoch > 0:
                assign_lambda = self.q_lambda.assign(self.cq_lambda)
            else:
                assign_lambda = self.q_lambda.assign(0.0)
            self.sess.run([assign_lambda])

            _, cos_loss, cq_loss, lr, output = self.sess.run(
                [self.train_op, self.cos_loss, self.cq_loss, self.lr, self.img_last_layer],
                feed_dict={self.ICM_X: batch_X,
                           self.b_img_for_loss: codes})
            img_dataset.feed_batch_output(self.batch_size, output)
            duration = time.time() - start_time

            # every epoch: update codes and centers
            if train_iter % (2 * epoch_iter) == 0 and train_iter != 0:
                if epoch == 0:
                    with tf.device(self.device):
                        for i in range(self.max_iter_update_Cb):
                            print("#DVSQ Train# initialize centers in ", i, " iter")
                            self.sess.run(self.C.assign(self.initial_centers(img_dataset.output_vectors)))

                        print("#DVSQ Train# initialize centers done!!!")
                epoch = epoch + 1
                for i in range(self.max_iter_update_Cb):
                    print("#DVSQ Train# update codes and centers in ", i, " iter")
                    self.update_codes_batch(img_dataset, self.batch_size)
                    self.update_centers(img_dataset)

            print("%s #train# step %4d, lr %.8f, cosine margin loss = %.4f, cq loss = %.4f, %.1f sec/batch" % (
                datetime.now(), train_iter + 1, lr, cos_loss, cq_loss, duration))

        print("%s #traing# finish training" % datetime.now())
        self.save_model()
        print("model saved")

    def train_pq(self, img_dataset):
        assert type(img_dataset) is Vector_Dataset
        print("%s #train# start training" % datetime.now())
        epoch = 0
        epoch_iter = int(ceil(img_dataset.n_samples / self.batch_size))

        for train_iter in range(self.max_iter):
            input_batch, codes = img_dataset.next_batch(self.batch_size)
            start_time = time.time()

            if epoch > 0:
                assign_lambda = self.q_lambda.assign(self.cq_lambda)
            else:
                assign_lambda = self.q_lambda.assign(0.0)
            self.sess.run([assign_lambda])

            _, cos_loss, cq_loss, lr, ICM_X = self.sess.run(
                [self.train_op, self.cos_loss, self.cq_loss, self.lr, self.ICM_X],
                feed_dict={self.input_vectors: input_batch,
                           self.b_img_for_loss: codes})
            img_dataset.feed_batch_output(self.batch_size, ICM_X)
            duration = time.time() - start_time

            # every epoch: update codes and centers
            if train_iter % (2 * epoch_iter) == 0 and train_iter != 0:
                epoch = epoch + 1
                for i in range(self.max_iter_update_Cb):
                    print("#DVSQ Train# update codes and centers in ", i, " iter")
                    # use kmeans to generate centers for each subspace
                    # self.sess.run(self.C.assign(self.initial_centers(img_dataset.output_vectors)))
                    self.C.assign(self.initial_centers(img_dataset.output_vectors))
                    self.update_codes_batch(img_dataset, self.batch_size)
                    # self.update_centers(img_dataset)

            print("%s #train# step %4d, lr %.8f, cosine margin loss = %.4f, cq loss = %.4f, %.1f sec/batch" % (
                datetime.now(), train_iter + 1, lr, cos_loss, cq_loss, duration))

        print("%s #traing# finish training" % datetime.now())
        self.save_model()
        print("model saved")


def train(train_img, config):
    model = DVSQ(config)
    img_dataset = Dataset(train_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    model.train_dvsq(img_dataset)
    return model.save_dir


def train_pq(train_img, config):
    model = DVSQ(config)
    img_dataset = Dataset(train_img, config['output_dim'], config['n_subspace'] * config['n_subcenter'])
    model.train_pq(img_dataset)
    return model.save_dir
