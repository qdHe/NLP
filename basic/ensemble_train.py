import tensorflow as tf  
import numpy as np    
import argparse
import functools
import gzip
import json
import pickle
from collections import defaultdict
from operator import mul

from tqdm import tqdm
from squad.utils import get_phrase, get_best_span


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    #parser.add_argument('-o', '--out', default='ensemble.json')
    parser.add_argument("--data_path", default="inter_ensemble/data_single.json")
    parser.add_argument("--shared_path", default="inter_ensemble/shared_single.json")
    parser.add_argument('--train_data', default="/home/vera/data/squad/dev-v1.1.json")
    args = parser.parse_args()
    return args

def get_scores(context, wordss, y1_list, y2_list):
    scores = []
    phrases = []
    d = defaultdict(float)
    for y1, y2 in zip(y1_list, y2_list):
        span, score = get_best_span(y1, y2)
        phrase = get_phrase(context, wordss, span)
        scores += [score]
        phrases += [phrase]
    return phrases,scores

def prepare_data(args):
    e_list = []
    for path in tqdm(args.paths):
        with gzip.open(path, 'r') as fh:
            e = pickle.load(fh)
            e_list.append(e)

    with open(args.data_path, 'r') as fh:
        data = json.load(fh)

    with open(args.shared_path, 'r') as fh:
        shared = json.load(fh)

    predictions = {}
    for idx, (id_, rx) in tqdm(enumerate(zip(data['ids'], data['*x'])), total=len(e['yp'])):
        if idx >= len(e['yp']):
            break
        context = shared['p'][rx[0]][rx[1]]
        wordss = shared['x'][rx[0]][rx[1]]
        yp_list = [e['yp'][idx] for e in e_list]
        yp2_list = [e['yp2'][idx] for e in e_list]
        phrases, scores = get_scores(context, wordss, yp_list, yp2_list)
        predictions[id_] = (phrases, scores)

    with open(args.train_data) as dataset_file:
        dataset_json = json.load(dataset_file)
    dataset = dataset_json['data']
    train_x = []
    train_y = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                #total += 1
                if qa['id'] not in predictions:
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                phrases, scores = predictions[qa['id']]
                flag = False
                for truth in ground_truths:
                    if truth in phrases:
                        labels = [1 if ph==truth else 0 for ph in phrases]
                        flag = True
                        break
                if flag:
                    train_x += [scores]
                    train_y += [labels]
    return train_x, train_y

def train_weights(train_x,train_y,epoch=10000,rate = 0.001):  
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    n = train_x.shape[1]
    x = tf.placeholder("float")  
    y = tf.placeholder("float")  
    w = tf.Variable(tf.constant([1.0/n for i in range(n)]))
  
  
    pred = x*w
    pred = pred/tf.reduce_sum(pred)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred,y)
    loss = tf.reduce_sum(cross_entropy)  
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)  
  
    init = tf.initialize_all_variables()  
  
    sess = tf.Session() 
    sess.run(init)    
    for index in range(epoch):  
        sess.run(optimizer,{x:train_x,y:train_y})    
    loss = sess.run(loss,{x:train_x,y:train_y})
    w =  sess.run(w)
    print("loss:",loss)
    return w
  
def predictionTest(test_x,test_y,w,b):  
    W = tf.placeholder(tf.float32)  
    B = tf.placeholder(tf.float32)  
    X = tf.placeholder(tf.float32)  
    Y = tf.placeholder(tf.float32)  
    n = test_x.shape[0]  
    pred = tf.add(tf.mul(X,W),B)  
    loss = tf.reduce_mean(tf.pow(pred-Y,2))  
    sess = tf.Session()  
    loss = sess.run(loss,{X:test_x,Y:test_y,W:w,B:b})  
    return loss  
  
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
  
  
if __name__ == "__main__":
    args = get_args()
    train_x,train_y = prepare_data(args)  
    w = train_weights(train_x,train_y)  
    print ('weights:',w)
    print ('softmax(weights):',softmax(w))
