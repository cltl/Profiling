import lasagne
import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
import os
import sys
import json
import random

values_file="values.pkl"
ATTRIBUTES=config.get_columns()

def load_data(in_file, emb, max_example=None):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """
    inp = open(in_file, "r")
    q = []
    for i, st in enumerate(inp.readlines()):
        line = []
        for ss in st.split("\t"):
            line += [ss.strip()]
        if emb:
            q += [line[:-1]]
        else:
            q += [line]
        if max_example is not None and i > max_example:
            break
    return q


def load_data_with_emb(in_file, max_example=None):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """
    inp = open(in_file, "r")
    q = []
    em = []
    for i, st in enumerate(inp.readlines()):
        line = []
        for ss in st.split("\t"):
            line += [ss.strip()]
        q += [line[:-1]]
        em += [eval(line[-1])]
        if max_example is not None and i > max_example:
            break
    print(len(q), len(q[0]))
    print(len(em), len(em[0]))
    return q, np.array(em)

def build_dict(data, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    dicts = []
    inv_dicts = []
    for i in range(len(data[0])):
        word_count = Counter()
        for j in range(len(data)):
            word_count[data[j][i]] += 1
        ls = word_count.most_common(max_words)
        logging.info('r %d #Words: %d -> %d' % (i, len(word_count), len(ls)))
        for key in ls[:5]:
            logging.info(key)
        logging.info('...')
        for key in ls[-5:]:
            logging.info(key)
        ori_dict = {w[0]: index + 2 for (index, w) in enumerate(ls)}
        ori_dict[''] = 0
        inv_ori_dict={str(index+2): w[0] for (index, w) in enumerate(ls)}
        inv_ori_dict['0']=''
        if '1' not in inv_ori_dict:
            inv_ori_dict['1']=''
        dicts += [ori_dict]
        inv_dicts += [inv_ori_dict]
    return dicts, inv_dicts

def get_random_example_index(arr):
    indices=[]
    for index, item in enumerate(arr):
        if item:
            indices+=[index]
    return random.choice(indices)

def store_labels_to_pkl(inv_word_dicts):
    pickle.dump(inv_word_dicts, open(values_file, 'wb'))

def vectorize(examples, word_dicts, args, print_allowed=False, labels=[]):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    inxs = []
    masks = []
    if print_allowed:
        print_me="<h4>ENTITY: "
        for j in range(len(examples)):
            logging.info(len(examples[j]))
            for i in range(args.relations):
                if examples[j][i]!="":
                    print_me+="%s=<a href=\"%s\">%s</a>, " % (ATTRIBUTES[i], examples[j][i], labels[i][examples[j][i].split('/')[-1]])
            #print(examples[j])
        if len(examples):
            print print_me + '</h4>'
    for i in range(args.relations):
        in_data = []
        mask = []
        for j in range(len(examples)):
            if examples[j][i] in word_dicts[i]:
                in_data += [word_dicts[i][examples[j][i]]]
            else:
                in_data += [1] #unk
            if examples[j][i].strip() == "":
                mask += [0]
            else:
                mask += [1]
        inxs += [in_data]
        masks += [mask]
    return np.array(inxs).astype('int32'), np.array(masks).astype('float32')

def oversample(all_train_old, args):
    logging.info("start oversampling")
    all_train=[]
    all_train_old2=np.copy(all_train_old)
    for idx, joint_inps in enumerate(all_train_old):
        new_inps=np.copy(joint_inps)
        for i in range(args.relations):
            if joint_inps[args.relations + i].sum() < 1:
                np.random.shuffle(all_train_old2)
                for idx2, joint_inps2 in enumerate(all_train_old2):
                    if joint_inps2[args.relations + i].sum() >= 1:
                        random_index=get_random_example_index(joint_inps2[args.relations + i])
                        my_col = [joint_inps2[j][random_index] for j in range(len(joint_inps2))]
                        new_inps=np.insert(new_inps, len(new_inps[i]), my_col, axis=1)
                        break
        all_train+=[new_inps]
    logging.info("done oversampling")

    return all_train

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(config._floatX)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def load_embeddings(dim):

    #init=lasagne.init.Uniform((1000, 64))
    #dim=1000
    return np.random.rand(dim, 1000) # 64,1000)

    #total_emb += [embeddings]
    #return total_emb

def gen_embeddings(word_dicts, dim,
                   init=lasagne.init.Uniform()):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    total_emb = []
    for word_dict in word_dicts:
        num_words = max(word_dict.values()) + 1
        embeddings = init((num_words, dim))
        logging.info('Embeddings: %d x %d' % (num_words, dim))

        total_emb += [embeddings]
    return total_emb


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic

def compute_attr_entropy(cntr, n_size):
    v=0.0
    cnt=0
    #print(len(cntr.keys()))
    voc_size = len(cntr.keys())
    for k in cntr:
        if k!=0:
            p= (cntr[k]*1.0 / n_size)
            v-=p*np.log2(p)
            cnt+=cntr[k]
    return v/np.log2(voc_size)

def compute_avg_entropy(examples, word_dicts):
    density=[]
    #    density.append(5)
    #    print(len(facet))
    inxs = []
    masks = []
    unmasked = []
    for i in range(len(word_dicts)):
        rel_unmasked = 0
        in_data = []
        mask = []
        for j in range(len(examples)):
            if examples[j][i] in word_dicts[i]:
                in_data += [word_dicts[i][examples[j][i]]]
            else:
                in_data += [1] #unk
            if examples[j][i].strip() == "":
                mask += [0]
            else:
                mask += [1]
                rel_unmasked +=1
        inxs += [in_data]
        masks += [mask]
        unmasked += [rel_unmasked]
        attr_entropy=compute_attr_entropy(Counter(in_data), rel_unmasked)
        #print('attr %d' % (i+1), attr_entropy)
        density.append(attr_entropy)
    #return np.array(inxs).astype('int32'), np.array(masks).astype('float32')
    #return 
    print(density)
    return np.mean(np.nan_to_num(density))
