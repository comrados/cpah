import torch
import numpy as np
import visdom
import pickle
from tqdm import tqdm
from PIL import Image


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current query sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from current query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    #mask = (P > 0).float().sum(dim=0)
    #mask = mask + (mask == 0).float() * 0.001
    #P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    #mask = (R > 0).float().sum(dim=0)
    #mask = mask + (mask == 0).float() * 0.001
    #R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P, R


def p_top_k(qB, rB, query_label, retrieval_label, K, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'AP@K ' + tqdm_label

    num_query = qB.shape[0]
    PK = torch.zeros(len(K)).to(qB.device)

    for i in tqdm(range(num_query), desc=tqdm_label):
        # ground_truth[j] == 1 if same class (if at least 1 same label), otherwise 0, ground truth
        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        hamm_dist = calc_hamming_dist(qB[i, :], rB).squeeze()

        # for each k in K
        for j, k in enumerate(K):
            k = min(k, retrieval_label.shape[0])
            _, sorted_indexes = torch.sort(hamm_dist)
            retrieved_indexes = sorted_indexes[:k]
            retrieved_samples = ground_truth[retrieved_indexes]
            PK[j] += retrieved_samples.sum() / k

    PK = PK / num_query

    """
    import matplotlib.pyplot as plt
    plt.semilogx(K, PK)
    plt.savefig('/home/george/Downloads/_fig.png')
    """

    return PK


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)


def pr_curve2(qB, rB, query_label, retrieval_label, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]
    num_bit = qB.shape[1] + 1  # range(0, qB.shape[1])

    P = torch.zeros(num_query, num_bit)  # precisions (for all samples, for each radius)
    R = torch.zeros(num_query, num_bit)  # recalls (for all samples, for each radius)

    # i - current sample num
    for i in tqdm(range(num_query), desc=tqdm_label):

        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        # hamming distances from current sample to all db samples
        hamm_dist = calc_hamming_dist(qB[i, :], rB)

        """replace from this point for alternative calculation"""

        # 1 if sample is retrieved for each hamming radius in range(0, num_bit), 0 otherwise
        retrieved_for_each_radius = hamm_dist <= torch.arange(0, num_bit).reshape(-1, 1).float().to(hamm_dist.device)
        retrieved_for_each_radius = retrieved_for_each_radius.float()

        # count of retrieved samples for each hamming radius in range(0, num_bit)
        tp_fp = retrieved_for_each_radius.sum(dim=-1)
        tp_fp = tp_fp + (tp_fp == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero

        # intersection of retrieved and ground truth for each hamming radius in range(0, num_bit)
        retrieved_true = ground_truth * retrieved_for_each_radius
        # count of TP samples for each hamming radius in range(0, num_bit)
        tp = retrieved_true.sum(dim=-1)

        Pi = tp / tp_fp  # TP / (TP + FP), precisions (for current sample, for each hamming radius)
        Ri = tp / tp_fn  # TP / (TP + FN), recalls (for current sample, for each hamming radius)

        P[i] = Pi
        R[i] = Ri

        """
        # alternative calculation (slower)
        
        Pi = torch.zeros(num_bit)  # precisions (for current sample, for each hamming radius)
        Ri = torch.zeros(num_bit)  # recalls (for current sample, for each hamming radius)
        
        # k - current hamming radius
        for k in range(num_bit):
            retrieved_samples = (hamm_dist <= k).float().squeeze()  # sample is retrieved if hamming dist < k (current hamming radius)
            tp_fp = retrieved_samples.sum()

            retrieved_positive_samples = retrieved_samples * ground_truth

            tp = retrieved_positive_samples.sum()

            Pik = precision(tp, tp_fp)  # precision (for current sample, for current radius)
            Rik = recall(tp, tp_fn)  # recall (for current sample, for current radius)

            Pi[k] = Pik
            Ri[k] = Rik
        
        P[i] = Pi
        R[i] = Ri
        """

    # get mean value for precision and recall for each hamming radius in range(0, num_bit)
    P = P.mean(dim=0)
    R = R.mean(dim=0)

    """
    import matplotlib.pyplot as plt
    plt.plot(R, P)
    plt.savefig('/home/george/Downloads/_fig.png')
    """

    return P, R


def recall(TP, TP_plus_FN):
    # (relevant_samples in retrieved_samples) / relevant_samples
    # TP / (TP + FN)
    return TP / (TP_plus_FN + 0.0001)


def precision(TP, TP_plus_FP):
    # (relevant_samples in retrieved_samples) / retrieved_samples
    # TP / (TP + FP)
    return TP / (TP_plus_FP + 0.0001)
