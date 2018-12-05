from decimal import Decimal
import sys
import pickle
import random
import logging
import utils
import config

def get_row(sts, meta):
    q = []
    for st in sts:
        for i in range(len(meta)):
            if meta[i] == st:
                print(st, i)
                q.append(i)
    return q

time_slicing_factor=100 # group by this many years for year of birth
lifespan_factor=20  # group by this many years for lifespan

def create_data(entity_type, embeddings, experiment):
    if embeddings:
        fn="../data/%s_emb_data.tsv" % entity_type
    else:
        fn="../data/tabular_%s_data.tsv" % entity_type

    inp = open(fn, "r")
    meta = inp.readline().strip()
    meta = meta.split("\t")
#    if meta[0]=='instance uri':
    meta = ['unnamed'] + meta
   
    #if meta[0]=='instance uri': 
    #    meta=meta[1:]
    #dob_index=meta.index("date of birth")

    # NEEDED FOR MOST EXPERIMENTS MINUS THE CURRENT ONE
    if experiment!='gvamerican':
        try:
            dob_index=meta.index('century')
        except:
            dob_index=meta.index('date of birth')
        ls_index=meta.index("lifespan")
        print(dob_index, ls_index)

    cols = get_row(config.get_columns(experiment), meta)
    data = []
    print meta
    print cols
    for i, st in enumerate(inp.readlines()):
        if st.strip()=='':
            continue
        st=st.strip('\n')
        info = st.split("\t")
        #print(i, meta.index("embeddings"), len(info))
        if embeddings and (len(info)<=meta.index("embeddings") or info[meta.index('embeddings')].strip()==''):
            continue
        d = []
        for col in cols:
            try:
                st=info[col]
            except:
                st=''
                print(col, 'yo')
            if st!='':
                # Enable the next 6 lines for the experiment #1
#                if col==dob_index: # date of birth
#                    period_group=int(float(st[:4]))/time_slicing_factor
#                    st='%d-%d' % (period_group*time_slicing_factor, (period_group+1)*time_slicing_factor)
#                elif col==ls_index: # lifespan
#                    lifespan_group=int(float(st))/lifespan_factor
#                    st='%d-%d' % (lifespan_group*lifespan_factor, (lifespan_group+1)*lifespan_factor)
                if st.find("}") != -1: # if there are multiple values, get the one with the lowest id
                    #pos = st.rfind(",")
                    #rpos = st.rfind("\'")
                    #st = st[pos + 3: rpos].strip()
                    line = st[1:-1].split(", ")
                    min_pos = 0
                    for j in range(len(line)):
                        if len(line[j]) < len(line[min_pos]) or len(line[j]) == len(line[min_pos]) and line[j] < line[min_pos]:
                            min_pos = j
                    st = line[min_pos][1:-1]
            d += [st]
        data += [d]
    random.shuffle(data)
    if embeddings:
        dir_path = "../data/%smini/" % entity_type
    else:
        dir_path = "../data/%s/" % entity_type
    sets = ["train", "dev", "test"]
    outdirs = [dir_path + sets[i] + ".txt" for i in range(len(sets))]
    ratio = [0.8, 0.1, 0.1]
    cnt = 0
    for i in range(len(sets)):
        oup = open(outdirs[i], "w")
        for j in range(int(len(data) * ratio[i])):
            oup.write("\t".join(data[cnt]) + "\n")
            cnt += 1
        oup.close()
    print(outdirs)
    return outdirs[0], dir_path

def main(args):
    args = config.get_args()
    print(args)
    entity_type=args.entity_type
    embeddings=args.embeddings
    train_file, outdir = create_data(entity_type, embeddings, args.experiment)
    #train_file='../data/%s/train.txt' % entity_type
    #outdir='../data/%s/' % entity_type

    logging.info('-' * 50)
    logging.info('Load data files..')
    logging.info('*' * 10 + ' Train')
    train_examples = utils.load_data(train_file, embeddings)

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dicts, inv_word_dicts = utils.build_dict(train_examples, 3000)

    num_attr = len(inv_word_dicts)
    d_abs=1
    for i in inv_word_dicts:
        print(len(i))
        d_abs*=len(i)
    print("d_abs = %s" % "{:.2E}".format(Decimal(d_abs)))
    print("n_ex = %d" % len(train_examples))
    print("d_avgd = %s" % "{:.2E}".format(Decimal(d_abs/len(train_examples))))
    entropy = utils.compute_avg_entropy(train_examples, word_dicts)
    print("Entropy = %f" % entropy) 

    pickle.dump(word_dicts, open('%s/train_dicts.pickle' % outdir, 'wb'))
    pickle.dump(inv_word_dicts, open('%s/train_inv_dicts.pickle' % outdir, 'wb'))

if __name__ == "__main__":
    args = config.get_args()

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))

    main(args)
