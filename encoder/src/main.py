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

import theano.gpuarray
theano.gpuarray.use("cuda")
print theano.config.blas.ldflags



MAX_NO_PROGRESS=10
labels_file='labels.pkl'

def gen_examples(data, mask, batch_size, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(data[0]), batch_size)
    all_ex = []
    for minibatch in minibatches:
        dm = []
        for d in data:
            dm += [d[minibatch]]
        for m in mask:
            dm += [m[minibatch]]
        all_ex += [dm]
    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    inputs = []
    in_xs = []
    in_masks = []
    print(args.relations)
    for i in range(args.relations):
        in_x1 = T.ivector('x%d' % i)
        in_mask = T.vector('mask%d' % i)
        in_xs += [in_x1]
        in_masks += [in_mask]
        l_in1 = lasagne.layers.InputLayer((None,), in_x1)
        l_in1 = lasagne.layers.DropoutLayer(l_in1, p=args.dropout_rate, rescale=False)
        l_in1 = nn_layers.IntLayer(l_in1)
        #print len(embeddings)
        l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, embeddings[i].shape[0],
                                               embeddings[i].shape[1], W=embeddings[i])
        inputs += [l_emb1]
    input = lasagne.layers.ConcatLayer(inputs)
    representation = lasagne.layers.DenseLayer(input, num_units=args.hidden_size)
    outputs = []
    for i in range(args.relations):
        num_cat = embeddings[i].shape[0]
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
        #if i == 1:
        #    print_func = theano.printing.Print("prediction")
        #    test_prediction = print_func(test_prediction)
        test_predictions += [test_prob]
        acc = (T.eq(test_prediction, in_xs[i]) * in_masks[i]).sum()
        accs += [acc]
    test_fn = theano.function(in_xs + in_masks, accs + test_predictions, on_unused_input='warn')

    # Train functions
    losses = 0
    for i in range(args.relations):
        train_prediction = lasagne.layers.get_output(outputs[i])
        train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
        masked_cost = lasagne.objectives.categorical_crossentropy(train_prediction, in_xs[i]) * in_masks[i]
        #print_func = theano.printing.Print("masked cost")
        #masked_cost = print_func(masked_cost)
        losses = losses + masked_cost.sum() / in_masks[i].sum()
    params = lasagne.layers.get_all_params(network)

    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(losses, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(losses, params)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(losses, params)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    train_fn = theano.function(in_xs + in_masks, losses, updates=updates, on_unused_input='warn')

    return train_fn, test_fn, params


def eval_acc(test_fn, all_examples, inv_word_dicts, topk_acc=1, print_allowed=False, labels_data=[]):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = np.zeros((args.relations))
    n_examples = np.zeros((args.relations))
    #if print_allowed:
    my_rel_num=10
    acc_per_num = {}
    attr_acc = {}
    influences = {}
    ATTRIBUTES=config.get_columns(args.experiment)
    count_per_total = defaultdict(int)
    all_predictions = defaultdict(list)
    for i in range(len(ATTRIBUTES)):
        acc_per_num[i]={'c':0, 'i':0}
        attr_acc[i]={'c':0, 'i':0}
        influences[i]={'c':0, 'i':0}
    print('SIZE', len(all_examples))
    for inps in all_examples:
        if args.complete_AE:
        #if True:
            tot_acc = test_fn(*inps)
            tot_acc = tot_acc[:args.relations]
        else:
            #logging.info(inps[3])
            tot_acc = []
            totals=defaultdict(int)
            corrects=defaultdict(int)
            attr_totals=defaultdict(int)
            attr_corrects=defaultdict(int)
            present=defaultdict(int)
            for i in range(args.relations):
                for index, val in enumerate(inps[i+args.relations]):
                    if val>0:
                        totals[index]+=1
            correct = 0
            for k in range(args.relations):
                new_inp = []
                #if print_allowed:
                for i in range(args.relations):
                    if i == k:
                        new_inp += [np.zeros((inps[0].shape[0],)).astype("int32")]
                    else:
                        #new_inp += [inps[i] * np.random.binomial(1, args.dropout_rate, inps[i].shape).astype("int32")]
                        new_inp += [inps[i]]
                #logging.info(inps[k])
                new_inp += inps[args.relations:]
                tmp_acc_prob = test_fn(*new_inp)
                prob = tmp_acc_prob[k + args.relations]
                sort_index = prob.argsort(axis=1)
                #logging.info(sort_index)
                real_acc = 0
                #logging.info(prob)
                for j in range(prob.shape[0]):
                    if print_allowed:
                        to_print="<ol>"
                        top_predictions=[]
                        print "System predictions for %s:" % ATTRIBUTES[k]
                        for prediction in range(1, min(topk_acc+1, len(sort_index[j]))):
                            my_label=''
                            choice=inv_word_dicts[k][str(sort_index[j][-prediction])]
                            choice_prob=prob[j][sort_index[j][-prediction]]
                            #if choice.strip()!='':
                            #    my_label=labels_data[k][choice.split('/')[-1]].encode('utf-8')
                            to_print += "%s (%s)" % (choice, choice_prob) #my_label)
                            top_predictions.append(tuple([choice, choice_prob]))
                        all_predictions[ATTRIBUTES[k]].append(top_predictions)
                        to_print+="</ol>"
                        try:
                            print to_print
                        except Exception as exc:
                            print exc
                    if k==my_rel_num and inps[k+args.relations][j] == 1:
                        s=0
                        present_attrs=set()
                        for kr in range(args.relations):
                            s += inps[kr + args.relations][j]
                            if inps[kr + args.relations][j]>0:
                                present_attrs.add(kr)
                        attr_totals[j]=s
                        attr_corrects[j]=0
                        present[j]=present_attrs
                    if inps[k+args.relations][j] == 1 and prob[j][inps[k][j]] >= prob[j][sort_index[j][(-1)*min(topk_acc, len(sort_index[j]))]]:
                        real_acc += 1
                        corrects[j] += 1
                        if k==my_rel_num:
                            attr_corrects[j]+=1
                            #logging.info(sort_index[j][-1])
                            #logging.info("System was correct for attribute 3 in row %d. Value: %d" % (j, inps[k][j]))
                        #logging.info(prob[j][inps[k][j]])
                        #logging.info(prob[j][sort_index[j][-3]])
                #if print_allowed:
                tot_acc.append(real_acc)
            for row, total in totals.items():
                acc_per_num[total-1]['c'] += corrects[row]
                acc_per_num[total-1]['i'] += total - corrects[row]
                count_per_total[total]+=1
            for row, total in attr_totals.items():
                attr_acc[total-1]['c'] += attr_corrects[row]
                attr_acc[total-1]['i'] += (1 - attr_corrects[row])
                c_or_i=''
                if attr_corrects[row]>0:
                    c_or_i='c'
                else:
                    c_or_i='i'
                for elem in present[row]:
                    if elem!=my_rel_num:
                        influences[elem][c_or_i]+=1
        for k in range(args.relations):
            n_examples[k] += inps[k + args.relations].sum()
        acc += np.array(tot_acc)
    #if print_allowed:
    logging.info(acc)
    logging.info(acc_per_num)
    logging.info(count_per_total)
    with open('predicted.pkl', 'wb') as p:
        pickle.dump(all_predictions, p)
    my_graph={}
    for k in attr_acc:
        if attr_acc[k]['c']+attr_acc[k]['i']>=50:
            my_graph[k]=attr_acc[k]['c']*1.0/(attr_acc[k]['c']+attr_acc[k]['i'])
    logging.info(attr_acc)
    logging.info(my_graph)
    #logging.info(influences)
    #logging.info('Attribute 3 stats. Total of %d examples, %d correct, %d incorrect' % (n_examples[3], attr_correct, attr_incorrect))
    return (acc * 100.0 / n_examples).tolist()


def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, False, 100)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, False, 100)
        test_examples = dev_examples
    else:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, False)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, False)
        test_examples = utils.load_data(args.test_file, False)
    args.num_train = len(train_examples)
    args.num_dev = len(dev_examples)
    #args.relations = len(train_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dicts = pickle.load(open('%s/train_dicts.pickle' % args.data_dir, 'rb'))
    inv_word_dicts = pickle.load(open('%s/train_inv_dicts.pickle' % args.data_dir, 'rb'))
    default_value = []
    for word_dict in word_dicts:
        default_value.append(word_dict[''])
    #logging.info(word_dicts[1])
    #logging.info(inv_word_dicts[1])

    #utils.store_labels_to_pkl(inv_word_dicts)
    #sys.exit(0)
    args.default_value = default_value
    embeddings = utils.gen_embeddings(word_dicts, args.embedding_size)
    train_fn, test_fn, params = build_fn(args, embeddings)
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    topk_acc=args.topk_accuracy
    #topk_acc=1

    labels_data=[]
    #if args.test_print_allowed:
    #    labels_data=pickle.load(open(labels_file, 'rb')) 

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_data, dev_mask = utils.vectorize(dev_examples, word_dicts, args)
    all_dev = gen_examples(dev_data, dev_mask, args.batch_size)
    dev_acc = eval_acc(test_fn, all_dev, inv_word_dicts, topk_acc)
    logging.info('Dev accuracy: %s %%' % str(dev_acc))
    test_data, test_mask = utils.vectorize(test_examples, word_dicts, args, args.test_print_allowed, labels_data)
    all_test = gen_examples(test_data, test_mask, args.batch_size)
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
        train_data, train_mask = utils.vectorize(train_examples, word_dicts, args)
        start_time = time.time()
        n_updates = 0
        all_train_old = gen_examples(train_data, train_mask, args.batch_size)

        logging.info("start oversampling")
        all_train=[]
        all_train_old2=np.copy(all_train_old)
        for idx, inps in enumerate(all_train_old):
            #tmp_inp=[] 
            new_inps=np.copy(inps)
            for i in range(args.relations):
                if inps[args.relations + i].sum() < 1:
                    np.random.shuffle(all_train_old2)
                    #inps[args.relations + i][0] = 1
                    for idx2, inps2 in enumerate(all_train_old2):
                        if inps2[args.relations + i].sum() >= 1:
                            random_index=utils.get_random_example_index(inps2[args.relations + i])
                            my_col = [inps2[j][random_index] for j in range(2*args.relations)]
                            new_inps=np.insert(new_inps, len(new_inps[i]), my_col, axis=1)
                            #new_inps[j+args.relations]=np.append(inps[j+args.relations], inps2[j+args.relations][random_index])
                            break
            all_train+=[new_inps.astype(np.int32)]
        logging.info("done oversampling")
        #sys.exit(0)
        no_progress=0
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, inps in enumerate(all_train):
            train_loss = train_fn(*inps)
            if idx % 1000 == 0:
                #logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1
            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                train_data_sample = [train_data[j][samples] for j in range(args.relations)]
                train_mask_sample = [train_mask[j][samples] for j in range(args.relations)]
                sample_train = gen_examples(train_data_sample, train_mask_sample, args.batch_size)
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
