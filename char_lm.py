"""
character level language model
"""

import argparse
import os
from collections import *
import pickle
import numpy as np
import pdb


def KN_smoothing(order):
    """
    :param lm: dictionary where key is order-k, val is csr matrix of size len(char) x len(hist)

    interpolated KneserNey smoothing

    h : history; h', history for lower order (-1 lower)
    w : next char

    P_KN(w | h_k) = A + lambda(h) P_KN(w | h_(k-1))
        where,
        A = max{N_1+(*,h,w)-D , 0} / N_1+(*,h,*) ; if k ~= highest order
        A = max{C(h,w)-D, 0} / \sum_w' C(h,w') ; if k == highest order
        lambda(h) = D / N_1+(*,h,*) * N_1+(h,*)

    eq.(22) in https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf?sequence=1 for more details
    return dictionary where key is sequence, value is probability of sequence and backoff weight(\lambda)

    notation down here :
        pkn(n) = A / sum_wi + D / sum_wi * N_w_ * pkn(n-1)
    """

    # order = 1
    pkn_w = np.array([ len(lm_r[1][w]) for h in lm[1] for w in lm_r[1][h]])
    pkn_w = pkn_w / sum(pkn_w)
    ret = dict(zip([h for h in lm[1] for w in lm_r[1][h]], pkn_w))
#     o = 1
#     n1 = len([0 for h in lm[o] if sum(lm[o][h].values()) == 1])
#     n2 = len([0 for h in lm[o] if sum(lm[o][h].values()) == 2])
#     D = n1 / (n1 + 2 * n2 + .0)
    D = 0.1
    ret["≈"] = D / len(pkn_w)
    # order 2 ~ n
    for o in range(1, order):
        n1 = len([0 for h in lm[o] if sum(lm[o][h].values()) == 1])
        n2 = len([0 for h in lm[o] if sum(lm[o][h].values()) == 2])
        D = n1 / (n1 + 2 * n2 + .0)
        A_, pkn_lower_order, N_w_ = [],[],[]
        for h in lm[o]:
            N_w = len(lm[o][h])
            for w in lm[o][h]:
                N_w_.append(N_w)            # number of continuation
                if order == o:              # highest order : frequency
                    A_.append(lm[o][h][w])
                else:                       # lower order: continuation
                    A_.append(len(lm_r[o][h[1:]+w]))
                pkn_lower_order.append(ret[h[1:]+w])
        sum_wi = sum(A_)
        A = (np.array(A_) - D) / sum_wi
        A[A < 0] = 0
        pkn_lower_order = np.array(pkn_lower_order)
        N_w_ = np.array(N_w_)
        lambda_h = (D / sum_wi) * N_w_
        pkn_w = A + lambda_h * pkn_lower_order
        tmp = dict(zip([h + w for h in lm[o] for w in lm[o][h]], pkn_w))
        ret.update(tmp)
    pdb.set_trace()
    return ret


def train_ngramlm(args):
    if not os.path.exists(args.corpus_path):
        raise Exception('Path not exist!')
    global lm, lm_r
    lm = [defaultdict(Counter) for i in range(args.order)] # lm[h][w]
    lm_r = [defaultdict(Counter) for i in range(args.order)] #lm[h[1:]w][h[0]]
    corpus_path = args.corpus_path
    order = args.order
    for file in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, file)) as f:
            data = f.read()
        if args.strip_newline:
            data = data.replace('\n', ' ')
        # language model
        pre = args.order * '~' # add prefix to data
        post = args.order * '|'
        data = pre + data + post
        for i in range(len(data) - order +1):
            for j in range(order):  # lm[n-1,n] for order from 1, n-1 contains empty string
                lm[j][data[i:i + j]][data[i + j]] += 1
                lm_r[j][data[i+1:i+j+1]][data[i]] += 1
        # pdb.set_trace()
    '''
        To compute Kneser-Ney smoothing
            - N(w_i | w_{i-n+1}^{i-1}) ## n-gram, len(v) x len(v_ngram) for each order
    '''

    # smoothing
    if args.KneserNey: # Kneser-Ney smoothing
        # compute discounting factor (see the paper for how d is computed)
        ret = KN_smoothing(args.order)
        # dump lm model to output
        with open(args.model_path, "wb") as f:
            pickle.dump(ret, f)
        return

def score(args):

    # load model
    if not os.path.exists(args.model_path):
        raise Exception("model not exists")

    if not os.path.exists(args.corpus_path):
        raise Exception("path not exists")

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    files = os.listdir(args.corpus_path)
    order = args.order
    scores = []
    V = [k for k in model if len(k) == 1]
    for file in files:
        with open(os.path.join(args.corpus_path, file), "r") as f:
            data = f.read()
        if args.strip_newline:
            data = data.replace('\n', ' ')
        # start computing log probabilities, log 10 based
        prefix = args.order * '~'  # add prefix to data
        postfix = args.order * '|'
        data = prefix + data
        score = 0
        data = ''.join([c if c in V else "≈" for c in data])
        for i in range(len(data) - order + 1):
            for j in range(order, 0, -1):
                if data[i+order-j:i+order] in model:
                    score += np.log10(model[data[i+order-j:i+order]])
                    # pdb.set_trace()
                    break
        scores.append(score/(len(data) + args.order))
    with open(args.result_path, "w") as writer:
        for s, f in zip(scores, files):
            writer.write('{},{}\n'.format(s, f.strip('.txt')))


def main(args):
    if args.mode == "train":
        train_ngramlm(args)
    else:
        score(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, required=False, default="./train/", help="if mode is 'train', this path should contain files used to train char-ngram-lm, if 'score', this path should be the path to be tested")
    parser.add_argument('--model_path', type=str, default="./model.p", help="if mode is 'train', this should be output path of language model, if 'score', language model will be loaded from this path")
    parser.add_argument('--strip_newline', type=bool, required=False, default=True,
                        help='turn all \\n to space')
    parser.add_argument('--result_path', type=str, default="./result.txt", help='result path for saving scores')
    parser.add_argument('--order', type=int, required=True, default=5)
    parser.add_argument('--KneserNey', type=bool, required=False, default=True, help='interpolated KneserNey smoothing')
    parser.add_argument('--mode', type=str, required=True, default='train', help='train/score')
    args = parser.parse_args()
    main(args)
