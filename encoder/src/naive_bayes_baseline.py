from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import sys
from collections import Counter

import config

def preprocess(data_path, mapping):
    all_data=[]

    all_classes={}
    for i in xrange(N):
        all_classes[i]=set()

    with open(data_path, 'rb') as data:
        spamreader = csv.reader(data, delimiter='\t', quotechar='"')
        headers = spamreader.next()

        for row in spamreader:
            tmp=[]
            for i in xrange(N):
                if row[i]=='': # empty value
                    val=0
                elif row[i] in mapping[i]: # any other
                    val=int(mapping[i][row[i]])
                else: # UNK
                    val=1
                tmp+=[val]
    
                all_classes[i].add(val)
            all_data +=[tmp]
    list_classes={}
    for i in xrange(N):
        all_classes[i].add(0)
        all_classes[i].add(1)
        list_classes[i] = list(all_classes[i])
    #sys.exit(0)
    return np.array(all_data), list_classes

def get_test_data(data, i, mapping):
    x_batch = []
    y_batch = []

    cnt=0


    for index,row in enumerate(data):
        if row[i]!=mapping[i]['']:
#            i_size = 1+max(mapping[i].values())
#            i_vector=[0]*i_size
#            i_vector[row[i]]=1
#            yrow=i_vector
            yrow=row[i]
            xrow=[]
            for k in xrange(len(row)):
               kx_size = 1+max(mapping[k].values()) # (+1 because of the 0 value)
               k_vector=[0]*kx_size # initialize zero vector
               if k!=i and row[k]!=0:
                   k_vector[row[k]]=1 # put the one-hot value in
               xrow+=k_vector
            x+=[xrow]
            y+=[yrow]
    print(len(x))
    return np.array(x), np.array(y)

def split_data_in_batches(data, i, mapping, batch_size=128):
    x_batch = []
    y_batch = []

    cnt=0

    for index,row in enumerate(data):
        if cnt>0 and cnt%batch_size==0:
            #x+=[x_batch]
            #y+=[y_batch]
            yield np.array(x_batch), np.array(y_batch)
            x_batch = []
            y_batch = []
        if row[i]!=mapping[i]['']:
            yrow=row[i]
            xrow=[]
            for k in xrange(len(row)):
               kx_size = 1+max(mapping[k].values()) # (+1 because of the 0 value)
               k_vector=[0]*kx_size # initialize zero vector
               if k!=i and row[k]!=0:
                   k_vector[row[k]]=1 # put the one-hot value in
               xrow+=k_vector
            x_batch+=[xrow]
            y_batch+=[yrow]
            cnt+=1
    if cnt%batch_size:
        print('remaining %d' % (cnt%batch_size))
        yield np.array(x_batch), np.array(y_batch)
    
        #x+=[x_batch]
        #y+=[y_batch]
    #return len(x), list(all_classes)

if __name__=="__main__": 
    attributes = config.get_columns()
    N=len(attributes)-1

    print("%d attributes" % N)

    batch_size=32768

    entity_type = sys.argv[1]
    data_path='../data/%s' % entity_type
    print("Data path: %s" % data_path)
    mapping = pickle.load(open('%s/train_dicts.pickle' % data_path, 'rb'))

    train_path='%s/train.txt' % data_path
    test_path='%s/test.txt' % data_path

    train_data, all_classes = preprocess(train_path, mapping)
    test_data, test_classes = preprocess(test_path, mapping)

    #gnb = GaussianNB()
    acc = []
    for i in xrange(N):
        #i=N-ix-1
        #if i in [0,1,13,3,6,8,11]: continue
        if i not in [2, 10]: continue
        gnb = MultinomialNB()
        print("Attribute %d" % i)
        train_iterator = split_data_in_batches(train_data, i, mapping, batch_size)
#        all_classes = np.unique(Ytrain)

        #if not len(g_classes):
        #    print('No classes. I am outta here')
        #    sys.exit()

        
        c=0
        for x_chunk, y_chunk in train_iterator:
            if len(x_chunk):
                gnb.partial_fit(x_chunk,y_chunk, classes=all_classes[i])
                print("Batch %d" % c)
                c+=1
        

        #for minibatch in range(num_batches): 
        #    print("Batch %d out of %d" % (minibatch, num_batches))
        #    x=Xtrain[minibatch]
        #    y=Ytrain[minibatch]
        #    if len(x)>0:
        #        gnb.partial_fit(x,y, classes=all_classes)
        #print("Training done!")

        test_iterator = split_data_in_batches(test_data, i, mapping, batch_size)
        c_test=0
        corrects=0
        totals=0
        for x_chunk, y_chunk in test_iterator:
            if len(x_chunk):
                y_pred = gnb.predict(x_chunk)
                print("Batch %d" % c_test)
                c_test+=1
                corrects += (y_chunk==y_pred).sum()
                totals += y_chunk.shape[0]
        print(corrects, totals)
        accuracy = corrects*100.0/totals
        print(accuracy)

        #Xtest, Ytest = get_test_data(test_data,i, mapping)
        #Ypred = gnb.predict(Xtest)
        #print "Number of mislabeled points %d out of %d points. Accuracy for attribute %s: %f" % ((Ytest != Ypred).sum(), Ytest.shape[0], attributes[i], (Ytest == Ypred).sum()*100.0/Ytest.shape[0])
        acc += [accuracy]

    print acc

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])

#gnb = GaussianNB()
#gnb.fit(X, Y)
#print gnb.predict([[-0.8, -1]])
#print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))
