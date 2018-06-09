import multiprocessing
from collections import namedtuple

import numpy as np
import time
from functools import reduce
from itertools import chain, count

from vae.vae_utils import iterate_splits


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    Parallel map implementation adapted from http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
    """

    def func_wrap(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))

    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=func_wrap, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]
    [p.terminate() for p in proc]

    return [x for i, x in sorted(res)]


def get_chunk_ranges(N, num_procs):
    """
    A helper that given a number N representing the size of an iterable and the num_procs over which to
    divide the data return a list of (start_index, end_index) pairs that divide the data as evenly as possible
    into num_procs buckets.
    """
    per_thread = N / num_procs
    allocation = [per_thread] * num_procs
    allocation[0] += N - num_procs * per_thread
    data_ranges = [0] + reduce(lambda acc, num: acc + [num + (acc[-1] if len(acc) else 0)], allocation, [])
    data_ranges = [(data_ranges[i], data_ranges[i + 1]) for i in range(len(data_ranges) - 1)]
    return data_ranges


def predict_cluster(x, centroids):
    return ((x - centroids) ** 2).sum(axis=1).argmin(axis=0)


def predict_coarse(x, Cs):
    return tuple([predict_cluster(cx, Cs[split]) for cx, split in iterate_splits(x, len(Cs))])


def compute_codes_parallel(data, Cs, num_procs=4):
    """
    A helper function that parallelizes the computation of LOPQ codes in
    a configurable number of processes.

    :param ndarray data:
        an ndarray of data points
    :param LOPQModel model:
        a model instance to use to compute codes
    :param int num_procs:
        the number of processes to spawn

    :returns iterable:
        an iterable of computed codes in the input order
    """

    def compute_partition(data):
        return [predict_coarse(d, Cs) for d in data]

    N = len(data)
    partitions = [data[int(a):int(b)] for a, b in get_chunk_ranges(N, num_procs)]
    codes = parmap(compute_partition, partitions, num_procs)

    return chain(*codes)


# def kmeans_quan_compute_distances(x, Cs, items):
#     results = []
#     temp_dist_map = {}
#     for item in items:
#         code = item[1]
#         if code not in temp_dist_map:
#             temp_dist_map[code] = np.sum((x - Cs[code]) ** 2)
#         dist = temp_dist_map[code]
#         results.append((dist, item))
#     return results


def compute_distances(Cs, x, items):
    results = []
    for item in items:
        codes = item[1]
        split_iter = iterate_splits(x, len(Cs))
        dist = 0
        for fx, sub_split in split_iter:
            dist += ((fx - Cs[sub_split][codes[sub_split]]) ** 2).sum()
        results.append((dist, item))
    return results


class Searcher:
    def __init__(self, Cs):
        self.quantized_codes = []
        self.Cs = Cs

    def add_codes(self, codes, ids=None):
        """
        Add LOPQ codes into the search index.

        :param iterable codes:
            an iterable of LOPQ code tuples
        :param iterable ids:
            an optional iterable of ids for each code;
            defaults to the index of the code tuple if not provided
        """
        # If a list of ids is not provided, assume it is the index of the data
        if ids is None:
            ids = count()

        for item_id, code in zip(ids, codes):
            self.quantized_codes.append((item_id, code))

    def add_data(self, data, ids=None, num_procs=4):
        codes = compute_codes_parallel(data, self.Cs, num_procs)
        self.add_codes(codes, ids)

    def search(self, x, quota=10, limit=None, with_dists=False):
        """
        Return euclidean distance ranked results, along with the number of cells
        traversed to fill the quota.

        :param ndarray x:
            a query vector
        :param int quota:
            the number of desired results to rank
        :param int limit:
            the number of desired results to return - defaults to quota
        :param bool with_dists:
            boolean indicating whether result items should be returned with their distance

        :returns list results:
            the list of ranked results
        :returns int visited:
            the number of cells visited in the query
        """
        # Retrieve results with multi-index
        retrieved, visited = self.quantized_codes, 0
        results = compute_distances(self.Cs, x, retrieved)

        # Sort by distance
        results = sorted(results, key=lambda d: d[0])

        # Limit number returned
        if limit is None:
            limit = quota
        results = results[:limit]

        if with_dists:
            Result = namedtuple('Result', ['id', 'code', 'dist'])
            results = map(lambda d: Result(d[1][0], d[1][1], d[0]), results)
        else:
            Result = namedtuple('Result', ['id', 'code'])
            results = map(lambda d: Result(d[1][0], d[1]), results)

        return results, visited

    def get_recall(self, queries, nns, thresholds=[1, 10, 100, 1000], normalize=True, verbose=False):
        recall = np.zeros(len(thresholds))
        query_time = 0.0
        total_visited_cell_count = 0.0
        for i, d in enumerate(queries):

            nn = nns[i]

            start = time.clock()
            results, cells_visited = self.search(d, thresholds[-1])
            query_time += time.clock() - start
            total_visited_cell_count += cells_visited
            if verbose and i % 50 == 0:
                print('%d cells visitied for query %d' % (cells_visited, i))

            for j, res in enumerate(results):
                rid, code = res

                if rid == nn:
                    for k, t in enumerate(thresholds):
                        if j < t:
                            recall[k] += 1

        if normalize:
            N = queries.shape[0]
            return recall / N, query_time / N, total_visited_cell_count / N
        else:
            return recall, query_time, total_visited_cell_count


if __name__ == '__main__':
    # X = np.random.rand(10, 4)
    V = 0
    M = 2
    subquantizer_clusters = 3
    query_start_time = time.time()
    report = []
    X = np.arange(20 * 4).reshape(20, 4)
    queries = np.arange(5 * 4).reshape(5, 4) + 5
    Cs = [np.arange(3 * 2).reshape(3, 2), np.arange(3 * 2).reshape(3, 2)]
    nns = np.arange(len(queries))

    searcher = Searcher(Cs)
    searcher.add_data(X, num_procs=1)
    recall, _, cell = searcher.get_recall(queries, nns, [1, 10])
    other_promote = ""
    print(
        'Recall (V=%d, M=%d, subquants=%d): %s, query time: %d s, total time: %d s, visited cell: %f %s' % (
            V, M, subquantizer_clusters, str(recall), (time.time() - query_start_time),
            (time.time() - query_start_time), cell,
            other_promote))

# def process_query(vae_model, base_data_codes, query_vectors)
# def test_load():
#     weight1 = np.arange(12).reshape(4, 3)
#     weight2 = np.arange(12).reshape(4, 3) + 10
#     bias1 = np.arange(3)
#     bias2 = np.arange(3) + 10
#
#     output_weights = [weight1, weight2]
#     output_bias = [bias1, bias2]
#     output = [(weight1, bias1), (weight2, bias2)]
#     # output = zip(output_weights, output_bias)
#     # print(output)
#
#     # np.save(model_file, np.array(model))
#     file_name = 'resultasd'
#     np.save(file_name, np.array(output))
#
#     net_data = np.load(file_name + '.npy')
#     print(net_data[0][0])
#     print(net_data[0][1])
#     print(net_data[1][0])
#     print(net_data[1][1])
