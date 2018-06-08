from vae.vae_utils import *


def load_deep_data_1M(report, dataset_dir, train_size=None, base_size=None, query_size=None):
    report.append("dataset:deep1M")
    data_train = load_xvecs(dataset_dir + '/deep10M/deep1M_learn.fvecs', max_num=train_size)
    print("done load train")
    data_base = load_xvecs(dataset_dir + '/deep10M/deep1M_base.fvecs', max_num=base_size)
    print("done load base")
    data_query = load_xvecs(dataset_dir + '/deep10M/deep1M_query.fvecs', max_num=query_size)
    print("done load query")
    data_gt = load_xvecs(dataset_dir + '/deep10M/deep1M_groundtruth.ivecs', base_type='i', max_num=query_size)
    print("done load groundtruth")
    return data_train, data_base, data_query, data_gt


def load_sift_data(report, dataset_dir, train_size=None, base_size=None, query_size=None):
    report.append("dataset:sift")
    # train_size = None
    # base_size = None
    data_train = load_xvecs(dataset_dir + '/sift/sift_learn.fvecs', max_num=train_size)
    print("done load train")
    data_base = load_xvecs(dataset_dir + '/sift/sift_base.fvecs', max_num=base_size)
    print("done load base")
    data_query = load_xvecs(dataset_dir + '/sift/sift_query.fvecs', max_num=query_size)
    print("done load query")
    data_gt = load_xvecs(dataset_dir + '/sift/sift_groundtruth.ivecs', base_type='i')
    data_gt = data_gt[:, 0]
    print("done load groundtruth")
    return data_train, data_base, data_query, data_gt


def load_siftsmall_data(report, dataset_dir, train_size=None, base_size=None, query_size=None):
    report.append("dataset:sift_small")
    # train_size = 10000
    # base_size = None
    data_train = load_xvecs(dataset_dir + '/siftsmall/siftsmall_learn.fvecs', max_num=train_size)
    print("done load train")
    data_base = load_xvecs(dataset_dir + '/siftsmall/siftsmall_base.fvecs', max_num=base_size)
    print("done load base")
    data_query = load_xvecs(dataset_dir + '/siftsmall/siftsmall_query.fvecs', max_num=query_size)
    print("done load query")
    data_gt = load_xvecs(dataset_dir + '/siftsmall/siftsmall_groundtruth.ivecs', base_type='i')
    data_gt = data_gt[:, 0]
    print("done load groundtruth")
    return data_train, data_base, data_query, data_gt


def load_gist_data(report, dataset_dir, train_size=None, base_size=None, query_size=None):
    report.append("dataset:gist")
    # train_size = None
    # base_size = None
    data_train = load_xvecs(dataset_dir + '/gist/gist_learn.fvecs', max_num=train_size)
    print("done load train")
    data_base = load_xvecs(dataset_dir + '/gist/gist_base.fvecs', max_num=base_size)
    print("done load base")
    data_query = load_xvecs(dataset_dir + '/gist/gist_query.fvecs', max_num=query_size)
    print("done load query")
    data_gt = load_xvecs(dataset_dir + '/gist/gist_groundtruth.ivecs', base_type='i')
    data_gt = data_gt[:, 0]
    print("done load groundtruth")
    return data_train, data_base, data_query, data_gt


# def test_data():
#     vae_utils.test_utils()
#     print("test_data")

class Vector_Dataset(object):
    GIST = 'GIST'
    DEEP1M = 'DEEP1M'
    SIFT = 'SIFT'
    CONV = 'CONV'
    SIFT_SMALL = 'SIFT_SMALL'

    def __init__(self, dataset_dir, dataset_name, is_train, output_dim, code_dim, train_size=None, base_size=None, query_size=None):
        """
        Generally, output_dim == code_dim, even if for PQ, we let codebooks be one-block-hot vectors.
        :param dataset_dir:
        :param dataset_name:
        :param is_train:
        :param output_dim:
        :param code_dim:
        :param train_size:
        :param base_size:
        :param query_size:
        """
        print("Initializing Dataset")
        report = []
        if dataset_name == self.GIST:
            data_train, data_base, data_query, data_gt = load_gist_data(report, dataset_dir, train_size, base_size,
                                                                        query_size)
        elif dataset_name == self.DEEP1M:
            data_train, data_base, data_query, data_gt = load_deep_data_1M(report, dataset_dir, train_size, base_size,
                                                                           query_size)
        elif dataset_name == self.SIFT:
            data_train, data_base, data_query, data_gt = load_sift_data(report, dataset_dir, train_size, base_size,
                                                                           query_size)
        elif dataset_name == self.SIFT_SMALL:
            data_train, data_base, data_query, data_gt = load_siftsmall_data(report, dataset_dir, train_size, base_size,
                                                                           query_size)
        else:
            raise Exception("Unknown dataset type! %s" % dataset_name)

        # self._dataset = dataset
        self.n_samples, self.vector_dim = data_train.shape
        self.is_train = is_train
        self.data_train = data_train
        self.output_dim = output_dim
        self._output_vectors = np.zeros((self.n_samples, self.output_dim), dtype=np.float32)
        self._codes = np.zeros((self.n_samples, code_dim), dtype=np.float32)

        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        print("Dataset already")
        return

    def next_batch(self, batch_size):
        """
        Args:
          batch_size
        Returns:
          [batch_size, (n_inputs)]: next batch images
          [batch_size, n_class]: next batch labels
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self.is_train:
                # Training stage need repeating get batch
                self._epochs_complete += 1
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                raise NotImplementedError("nonono, train?")
                # start = self.n_samples - batch_size
                # self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        data = self.data_train[self._perm[start:end]]
        return (data, self.codes[self._perm[start: end], :])

    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Another epoch finish
        if self._index_in_epoch > self.n_samples:
            if self.is_train:
                # Shuffle the data
                np.random.shuffle(self._perm)
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Validation stage only process once
                raise NotImplementedError("nonono, train?")
                # start = self.n_samples - batch_size
                # self._index_in_epoch = self.n_samples
        end = self._index_in_epoch

        return (self.output_vectors[self._perm[start: end], :],
                self.codes[self._perm[start: end], :])

    def feed_batch_output(self, batch_size, output):
        """
        Args:
          batch_size
          [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output_vectors[self._perm[start:end], :] = output
        return

    def feed_batch_codes(self, batch_size, codes):
        """
        Args:
          batch_size
          [batch_size, n_output]
        """
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.codes[self._perm[start:end], :] = codes
        return

    @property
    def output_vectors(self):
        return self._output_vectors

    @property
    def codes(self):
        return self._codes

    # @property
    # def label(self):
    #     return self._dataset.get_labels()

    def finish_epoch(self):
        self._index_in_epoch = 0
        np.random.shuffle(self._perm)
