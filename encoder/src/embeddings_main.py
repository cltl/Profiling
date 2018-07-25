import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
from collections import defaultdict

import sys
import time
import utils
import config
import logging
import nn_layers
import lasagne.layers as L

ATTRIBUTES=config.get_columns()
MAX_NO_PROGRESS=10
labels_file='labels.pkl'
#['educated at', 'sex or gender', 'country of citizenship', 'native language', 'notable work', 'award received', "religion", 'participant of', 'member of political party', 'member of sports team']


def gen_examples(data, mask, embs, batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(data[0]), batch_size)
    all_ex = []
    all_em = []
    for minibatch in minibatches:
        dm = []
        for d in data:
            dm += [d[minibatch]]
        for m in mask:
            dm += [m[minibatch]]
        for e in embs:
            dm += [e[minibatch]]
        all_ex += [dm]
    return all_ex


def build_fn(args, embeddings, inv_word_dicts):
    """
        Build training and testing functions.
    """
    inputs=T.dmatrix('x')
    targets=T.imatrix('y')

    l_in = lasagne.layers.InputLayer((None, embeddings.shape[1]), inputs)
    representation = lasagne.layers.DenseLayer(l_in, num_units=args.hidden_size)
    outputs = []
    for i in range(args.relations):
        num_cat = len(inv_word_dicts[i].keys())
        l_out = lasagne.layers.DenseLayer(representation, num_units=num_cat, nonlinearity=T.nnet.softmax)
        outputs += [l_out]
    network = lasagne.layers.ConcatLayer(outputs)
    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        lasagne.layers.set_all_param_values(network, dic['params'])
        del dic['params']
        logging.info('Loaded pre-trained model: %s' % args.pre_trained)
        for dic_param in dic.iteritems():
            logging.info(dic_param)

    logging.info('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    logging.info('#fixed params: %d' % lasagne.layers.count_params(network, trainable=False))
    for layer in lasagne.layers.get_all_layers(network):
        logging.info(layer)

    # Test functions
    accs = []
    test_predictions = []
    for i in range(args.relations):
        test_prob = lasagne.layers.get_output(outputs[i], deterministic=True)
        test_prediction = T.argmax(test_prob, axis=-1)
        test_predictions += [test_prob]
        
        acc = (T.eq(test_prediction, targets[i]) * targets[i + args.relations]).sum()
        accs += [acc]
        hwop=theano.printing.Print('targets')
        printed_x = hwop(targets.shape)
    #test_fn = theano.function([inputs, targets], printed_x, on_unused_input='warn')
    test_fn = theano.function([inputs, targets], accs+test_predictions, on_unused_input='warn')

    # Train functions
    losses = 0
    for i in range(args.relations):
        train_prediction = lasagne.layers.get_output(outputs[i])
        train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
        masked_cost = lasagne.objectives.categorical_crossentropy(train_prediction, targets[i]) * targets[i + args.relations]
        losses = losses + masked_cost.sum() / targets[i+args.relations].sum()
    params = lasagne.layers.get_all_params(network)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(losses, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(losses, params)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(losses, params)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    train_fn = theano.function([inputs, targets], losses, updates=updates, on_unused_input='warn')

    return train_fn, test_fn, params


def eval_acc(test_fn, all_examples, inv_word_dicts, topk_acc=1, print_allowed=False, labels_data=[]):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = np.zeros((args.relations))
    n_examples = np.zeros((args.relations))
    batch_counter=0
    print(len(all_examples), 'num batches')
    for joint_inps in all_examples:
        inps=np.array(joint_inps[:2*args.relations]).astype("int32")
        my_embeddings=np.transpose(joint_inps[2*args.relations:])
        print(inps.shape)
        print(len(my_embeddings), len(my_embeddings[0]))
        if args.complete_AE:
            tot_acc = test_fn(my_embeddings, *inps)
            tot_acc = tot_acc[:args.relations]
        else:
            tot_acc = []
            print(inps.shape)
            tmp_acc_prob = test_fn(my_embeddings, inps)
            for k in range(args.relations):
                prob = tmp_acc_prob[k + args.relations]
                sort_index = prob.argsort(axis=1)
                real_acc = 0
                for j in range(prob.shape[0]):
                    if inps[k+args.relations][j] == 1 and prob[j][inps[k][j]] >= prob[j][sort_index[j][topk_acc*(-1)]]:
                        real_acc += 1
                tot_acc.append(real_acc)

        for k in range(args.relations):
            n_examples[k] += inps[k + args.relations].sum()
        acc += np.array(tot_acc)
        batch_counter+=1
        if batch_counter==3: 
            print(acc)
            break
    logging.info((acc * 100.0 / n_examples).tolist())
    return (acc * 100.0 / n_examples).tolist()

def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    question_belong = []
    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_attrs, train_embeddings = utils.load_data_with_emb(args.train_file, 100)
        logging.info('*' * 10 + ' Dev')
        dev_attrs, dev_embeddings = utils.load_data_with_emb(args.dev_file, 100)
        test_attrs, test_embeddings = dev_examples
    else:
        logging.info('*' * 10 + ' Train')
        train_attrs, train_embeddings = utils.load_data_with_emb(args.train_file)
        logging.info('*' * 10 + ' Dev')
        dev_attrs, dev_embeddings = utils.load_data_with_emb(args.dev_file)
        test_attrs, test_embeddings = utils.load_data_with_emb(args.test_file)
    args.num_train = len(train_attrs)
    args.num_dev = len(dev_attrs)

    logging.info('-' * 50)
    logging.info('Build dictionary..')

    word_dicts = pickle.load(open('%s/train_dicts.pickle' % args.data_dir, 'rb'))
    inv_word_dicts = pickle.load(open('%s/train_inv_dicts.pickle' % args.data_dir, 'rb'))

    default_value = []
    for word_dict in word_dicts:
        default_value.append(word_dict[''])
        num_words = max(word_dict.values()) + 1
        logging.info('num categories: %d ' % num_words)

    print(train_embeddings.shape)

    #utils.store_labels_to_pkl(inv_word_dicts)
    args.default_value = default_value
    #embeddings = utils.load_embeddings(args.batch_size)
    data=[]
    train_fn, test_fn, params = build_fn(args, train_embeddings, inv_word_dicts)

    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    topk_acc=args.topk_accuracy

    labels_data=[]
    if args.test_print_allowed:
        labels_data=pickle.load(open(labels_file, 'rb')) 

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_data, dev_mask = utils.vectorize(dev_attrs, word_dicts, args)
    dev_embeddings=np.transpose(dev_embeddings) 
    all_dev = gen_examples(dev_data, dev_mask, dev_embeddings, args.batch_size)
    dev_acc = eval_acc(test_fn, all_dev, inv_word_dicts, topk_acc)

    logging.info('Dev accuracy: %s %%' % str(dev_acc))
    test_data, test_mask = utils.vectorize(test_attrs, word_dicts, args, args.test_print_allowed, labels_data)
    test_embeddings=np.transpose(test_embeddings)
    all_test = gen_examples(test_data, test_mask, test_embeddings, args.batch_size)
    test_acc = eval_acc(test_fn, all_test, inv_word_dicts, topk_acc, args.test_print_allowed, labels_data)
    logging.info('Test accuracy: %s %%' % str(test_acc))
    best_acc = dev_acc
    if args.test_only:
        return
    utils.save_params(args.model_file, params, epoch=0, n_updates=0)

    #utils.store_labels_to_pkl(inv_word_dicts)
    # Training
    if args.num_epoches>0:
        logging.info('-' * 50)
        logging.info('Start training..')
        train_data, train_mask = utils.vectorize(train_attrs, word_dicts, args)
        start_time = time.time()
        n_updates = 0
        train_embeddings=np.transpose(train_embeddings)
        all_train_old = gen_examples(train_data, train_mask, train_embeddings, args.batch_size)

        all_train=utils.oversample(all_train_old, args)
        #all_train=all_train_old
        no_progress=0
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, joint_inps in enumerate(all_train):
            inps=np.int32(joint_inps[:2*args.relations])
            my_embeddings=np.array(joint_inps[2*args.relations:])
            #train_embeddings=utils.load_embeddings(inps.shape[1])
            train_loss = train_fn(np.transpose(my_embeddings), inps)
            if idx % 1000 == 0:
                #logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1
            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                train_data_sample = [train_data[j][samples] for j in range(args.relations)]
                train_mask_sample = [train_mask[j][samples] for j in range(args.relations)]
                train_emb_sample = [train_embeddings[j][samples] for j in range(args.embedding_size)]
                sample_train = gen_examples(train_data_sample, train_mask_sample, train_emb_sample, args.batch_size)
                #acc = eval_acc(test_fn, sample_train)
                #logging.info('Train accuracy: %s %%' % str(acc))
                dev_acc = eval_acc(test_fn, all_dev, inv_word_dicts, topk_acc)
                logging.info('Dev accuracy: %s %%' % str(dev_acc))
                #test_acc = eval_acc(test_fn, all_test)
                #logging.info('Test accuracy: %s %%' % str(test_acc))
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.info('Best dev accuracy!')
                    utils.save_params(args.model_file, params, epoch=epoch, n_updates=n_updates)
                    no_progress=0
                else:
                    no_progress+=1
                    logging.info('Dev accuracy has not improved in the past %d evaluations' % no_progress)
                    if no_progress>=MAX_NO_PROGRESS:
                        logging.info("Reached the limit of stagnation. Exiting now...")
                        sys.exit(0)
        #sys.exit(0)

if __name__ == '__main__':
    args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
