import argparse
import functools
import gzip
import json
import pickle
from collections import defaultdict
from operator import mul

from tqdm import tqdm
from squad.utils import get_phrase, get_best_span, get_best_span2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-o', '--out', default='ensemble.json')
    parser.add_argument("--data_path", default="data/squad/data_test.json")
    parser.add_argument("--shared_path", default="data/squad/shared_test.json")
    args = parser.parse_args()
    weights[0]+=0.06
    return args


def ensemble(args):
    e_list = []
    for path in tqdm(args.paths):
        with gzip.open(path, 'r') as fh:
            e = pickle.load(fh)
            e_list.append(e)

    with open(args.data_path, 'r') as fh:
        data = json.load(fh)

    with open(args.shared_path, 'r') as fh:
        shared = json.load(fh)

    out = {}
    for idx, (id_, rx) in tqdm(enumerate(zip(data['ids'], data['*x'])), total=len(e['yp'])):
        if idx >= len(e['yp']):
            # for debugging purpose
            break
        context = shared['p'][rx[0]][rx[1]]
        wordss = shared['x'][rx[0]][rx[1]]
        yp_list = [e['yp'][idx] for e in e_list]
        yp2_list = [e['yp2'][idx] for e in e_list]
        answer = ensemble3(context, wordss, yp_list, yp2_list)
        out[id_] = answer

    with open(args.out, 'w') as fh:
        json.dump(out, fh)


def ensemble1(context, wordss, y1_list, y2_list):
    """

    :param context: Original context
    :param wordss: tokenized words (nested 2D list)
    :param y1_list: list of start index probs (each element corresponds to probs form single model)
    :param y2_list: list of stop index probs
    :return:
    """
    sum_y1 = combine_y_list(y1_list,'+')
    sum_y2 = combine_y_list(y2_list,'+')
    span, score = get_best_span(sum_y1, sum_y2)
    return get_phrase(context, wordss, span)


def ensemble2(context, wordss, y1_list, y2_list):
    start_dict = defaultdict(float)
    stop_dict = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        a0=span[0][0]
        a1=span[0][1]
        b0=span[1][0]
        b1=span[1][1]
        if(span[0][0]>=len(y1)): a0 = len(y1)-1
        if(span[0][1]>=len(y1[0])): a1 = len(y1[0])-1
        if(span[1][0]>=len(y2)): b0 = len(y2)-1
        if(span[1][1]>=len(y2[0])): b1 = len(y2[0])-1
        start_dict[(a0,a1)] += y1[a0][a1]
        stop_dict[(b0,b1)] += y2[b0][b1]
    start = max(start_dict.items(), key=lambda pair: pair[1])[0]
    stop = max(stop_dict.items(), key=lambda pair: pair[1])[0]
    best_span = (start, stop)
    return get_phrase(context, wordss, best_span)


def ensemble3(context, wordss, y1_list, y2_list):
    d = defaultdict(float)
    ct = 0
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        phrase = get_phrase(context, wordss, span)
        d[phrase] += weights[ct]*score
        ct += 1
    return max(d.items(), key=lambda pair: pair[1])[0]

def one():
    return 1.0

def ensemble4(context, wordss, y1_list, y2_list):
    d = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        phrase = get_phrase(context, wordss, span)
        d[phrase] += score
    return max(d.items(), key=lambda pair: pair[1])[0]

weights = [0.0890552, 0.06648684, 0.07300555, 0.07796152, 0.07443654, 0.07239726, 0.0886942, 0.07348367, 0.08823036, 0.0718033, 0.07231252, 0.06849918, 0.08268909]
def combine_y_list(y_list, op='+'):
    if op == '^':
        def func(l): return sum([x*x for x in l])
    elif op == '/':
        def func(l): return 1/sum([1.0/x for x in l])
    elif op == '+':
        func = sum
    elif op == '*':
        def func(l): return functools.reduce(mul, l)
    else:
        func = op
    return [[func(yij_list) for yij_list in zip(*yi_list)] for yi_list in zip(*y_list)]


def main():
    args = get_args()
    ensemble(args)

if __name__ == "__main__":
    main()


