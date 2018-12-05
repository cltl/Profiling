import config

import operator
import sys
import csv
from collections import defaultdict
import pickle

def get_most_frequent_values_train(data_path, num_attr, word_dicts):

    training_path='%s/train.txt' % data_path

    vals=defaultdict()
    max_vals={}

    with open(training_path, 'rb') as data:
        spamreader = csv.reader(data, delimiter='\t', quotechar='"')
        #headers = spamreader.next()
        for row in spamreader:
            for i in xrange(num_attr):
                if row[i] and row[i] in word_dicts[i]:
                    try:
                        vals[i][row[i]]+=1
                    except:
                        vals[i]=defaultdict(int)
                        vals[i][row[i]]+=1
    for attribute in vals:
        attribute_values=vals[attribute]
        sorted_av=sorted(attribute_values.items(), key=operator.itemgetter(1))    
        max_vals[attribute]={sorted_av[-1][0]:0}

    print(max_vals)

    return max_vals


if __name__=="__main__":
    entity_type = sys.argv[1]
    data_path='../data/%s' % entity_type
    word_dicts = pickle.load(open('%s/train_dicts.pickle' % data_path, 'rb'))
    if entity_type=='american':
        cols=config.get_columns(True)
    else:
        cols=config.get_columns()[:-1]
    num_attr=len(cols) # -1 for the embeddings thingie
    max_vals=get_most_frequent_values_train(data_path, num_attr, word_dicts)

    test_path='%s/test.txt' % data_path
    filled=defaultdict(int)
    acc={}
    with open(test_path, 'rb') as data:
        spamreader = csv.reader(data, delimiter='\t', quotechar='"')
        for row in spamreader:
            for i in xrange(num_attr):
                if row[i]:
                    filled[i]+=1
                    if row[i] in max_vals[i]:
                        max_vals[i][row[i]]+=1
    for attribute in filled:
        for max_val in max_vals[attribute]:
            acc[attribute]=max_vals[attribute][max_val]*100.0/filled[attribute]
    print acc

    print max_vals

    all_givens=[]
    all_predicted=[]
    crowd_test_path='%s/profiler_input.tsv' % data_path
    with open(crowd_test_path, 'rb') as data:
        spamreader = csv.reader(data, delimiter='\t', quotechar='"')
        for row in spamreader:
            givens_row={}
            preds_row={}
            for i in xrange(num_attr):
                if row[i]:
                    givens_row[cols[i]]=row[i]
                else:
                    preds_row[cols[i]]=max_vals[i]
            all_givens.append(givens_row)
            all_predicted.append(preds_row)

with open('mfv_givens.pkl', 'wb') as g:
    pickle.dump(all_givens, g)
    
with open('mfv_predicted.pkl', 'wb') as p:
    pickle.dump(all_predicted, p)
#print vals
