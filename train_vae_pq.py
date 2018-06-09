#          lr output iter gpu console
# train.sh 0.01 300 5000 0.1 3 4 0.7 100 0 (1)
import multiprocessing

import numpy as np
import scipy.io as sio
import warnings

import vae.vae_data as vae_data
import vae.vae_net as vae_net
import sys
import util
import vae.vae_valid as vae_valid

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.argv = ["", "0.005", "300", "300", "0.0001", "4", "0.0", "81", "2"]
### Define input arguments
lr = float(sys.argv[1])
output_dim = int(sys.argv[2])
iter_num = int(sys.argv[3])
cq_lambda = float(sys.argv[4])
# update_b = int(sys.argv[5])
subspace_num = int(sys.argv[5])
margin_param = float(sys.argv[6])
part_label = int(sys.argv[7])
gpu = sys.argv[8]

console = len(sys.argv) > 10

config = {
    'device': '/gpu:' + gpu,
    'gpu_usage': 11,  # G
    'max_iter': iter_num,
    'batch_size': 256,
    'moving_average_decay': 0.9999,  # The decay to use for the moving average.
    'decay_step': 500,  # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,  # Learning rate decay factor.
    'learning_rate': lr,  # Initial learning rate img.
    'console_log': console,

    'output_dim': output_dim,

    'R': 5000,
    'model_weights': 'models/reference_pretrain.npy',

    'img_model': 'alexnet',
    'stage': 'train',
    'loss_type': 'cos_softmargin_multi_label',

    'margin_param': margin_param,
    'wordvec_dict': "./data/nuswide_81/nuswide_wordvec.txt",
    'part_ids_dict': "./data/nuswide_81/train_" + str(part_label) + "_ids.txt",
    'partlabel': part_label,

    # only finetune last layer
    'finetune_all': True,

    ## CQ params
    'max_iter_update_b': 1,
    'max_iter_update_Cb': 1,
    'cq_lambda': cq_lambda,
    'code_batch_size': 500,
    'n_subspace': subspace_num,
    'n_subcenter': 256,

    'n_train': 10000 * (part_label / 81),
    'n_database': 218491,
    'n_query': 5000,

    'label_dim': 81,
    'img_tr': "./data/nuswide_81/train.txt",
    'img_te': "./data/nuswide_81/test.txt",
    'img_db': "./data/nuswide_81/database.txt",
    'save_dir': "./",

    # vae
    'num_layers': 2,
    'n_hidien': [128, 64]
}

import time

# t = time.time()
# # train_img = dataset.import_train(config)
# print(time.time() - t)
dataset_dir = '/Users/kangrong/code/github/lopq/data'
dataset_name = vae_data.Vector_Dataset.SIFT_SMALL
is_train = True
config['vector_dim'] = 128
config['batch_size'] = 2000
config['max_iter'] = 20


data_train, data_base, data_query, data_gt = vae_data.load_dataset(dataset_dir, dataset_name, train_size=10000)

#prepare config
config['output_dim'] = data_train.shape[1]
config['n_hidien'][-1] = config['output_dim']
config['num_layers'] = len(config['n_hidien'])




# output_dim = 64
# code_dim = 64
total_start_time = time.time()
model = vae_net.DVSQ(config)
data_train = data_train.astype(np.float32)
data_base = data_base.astype(np.float32)
data_query = data_query.astype(np.float32)

train_dataset = vae_data.Vector_Dataset(data_train, is_train, config['output_dim'],
                                      config['n_subspace'] * config['n_subcenter'])
model.train_pq(train_dataset)
model_dq = model.save_dir

# output_weights, output_bias = model.load_vae_model()
print("finished train!")
train_time = time.time() - total_start_time

# transform base data
print("start transform")
trans_start_time = time.time()
base_output = model.transform_vector_to_output(data_base)
query_output = model.transform_vector_to_output(data_query)
centers = model.transform_centers_to_pq()
trans_end_time = time.time() - trans_start_time
print("done transform")

print("start add data")
add_data_start_time = time.time()
searcher = vae_valid.Searcher(centers)
searcher.add_data(base_output, num_procs=multiprocessing.cpu_count())
add_data_end_time = time.time() - add_data_start_time
print("done add data")

recall, _, cell = searcher.get_recall(query_output, data_gt, [1, 10, 100, 1000])
other_promote = ""
print(
    'Recall (V=%d, M=%d, subquants=%d): %s, train: %d s, total time: %d s, visited cell: %f %s' % (
        0, len(centers), len(centers[0]), str(recall), train_time, (time.time() - total_start_time), cell, other_promote))