
import theano
import argparse


_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_columns(experiment='crowd'):
    if experiment=='crowd':
        return ['lifespan', 'century', 'sex or gender', 'occupation', 'work location', 'educated at', 'religion', 'member of political party', 'place of death', 'place of birth']
    elif experiment=='gvamerican':
        return ['native language' , 'ethnic group', 'cause of death', 'sex or gender', 'religion', 'member of political party', 'occupation', 'age group']
    else:
        return ['educated at', 'sex or gender', 'country of citizenship', 'native language', 'position held', 'award received', 'religion', 'member of political party', 'work location', 'place of death', 'place of birth', 'cause of death', 'lifespan', 'date of birth', 'embeddings']

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # Basics
    parser.add_argument('-debug',
                        type='bool',
                        default=False,
                        help='whether it is debug mode')

    parser.add_argument('-test_only',
                        type='bool',
                        default=False,
                        help='test_only: no need to run training process')

    parser.add_argument('-random_seed',
                        type=int,
                        default=1013,
                        help='Random seed')

    parser.add_argument('-test_print_allowed',
                        type='bool',
                        default=False,
                        help='whether to allow printing of individual examples on the test data')

    parser.add_argument('-max_cat',
                        type=int,
                        default=3000,
                        help='Max categories per attribute')
    # Data file
    parser.add_argument('-data_dir',
                        type=str,
                        default="../data/politician/",
                        help='Data directory')
    # Data file
    parser.add_argument('-train_file',
                        type=str,
                        default="../data/politician/train.txt",
                        help='Training file')

    parser.add_argument('-dev_file',
                        type=str,
                        default="../data/politician/dev.txt",
                        help='Development file')
    parser.add_argument('-test_file',
                        type=str,
                        default="../data/politician/test.txt",
                        help='Test file')
    #We use test file here since it's just preliminary experiments.
    parser.add_argument('-relations',
                        type=int,
                        default=14, help="Number of relations to predict")
    parser.add_argument('-pre_trained',
                        type=str,
                        default=None,
                        help='Pre-trained model.')

    parser.add_argument('-model_file',
                        type=str,
                        default='../data/politician/model.pkl.gz',
                        help='Model file to save')

    parser.add_argument('-log_file',
                        type=str,
                        default=None,
                        help='Log file')

    # Model details
    parser.add_argument('-embedding_size',
                        type=int,
                        default=50,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('-hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')
    parser.add_argument('-optimizer',
                        type=str,
                        default='adam',
                        help='Optimizer: sgd (default) or adam or rmsprop')
    parser.add_argument('-complete_AE',
                        type=bool,
                        default=False, help="to do sanity test")

    parser.add_argument('-learning_rate', '-lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for SGD')


    # Optimization details
    parser.add_argument('-batch_size',
                        type=int,
                        default=64,
                        help='Batch size')

    parser.add_argument('-num_epoches',
                        type=int,
                        default=300,
                        help='Number of epoches')

    parser.add_argument('-eval_iter',
                        type=int,
                        default=5000,
                        help='Evaluation on dev set after K updates')

    parser.add_argument('-dropout_rate',
                        type=float,
                        default=0.5,
                        help='Dropout rate')
    # Evaluation setting
    parser.add_argument('-topk_accuracy',
                        type=int,
                        default=1,
                        help='Accuracy among the top K')

    # Which experiment 
    parser.add_argument('-experiment',
			type=str,
                        default='wikidata',
                        help="Which experiment: on Wikidata or on crowd data")

    # Which entity type
    parser.add_argument('-entity_type',
                        type=str,
                        default='politician',
                        help='Entity type')

    # Include embeddings or not?
    parser.add_argument('--embeddings',
                        action='store_true',
                        default=False,
                        help="Include embeddings or not")

    return parser.parse_args()
