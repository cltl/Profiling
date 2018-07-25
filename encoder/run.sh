mode="gpu"
data="american"
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -data_dir "../data/$data" -model_file "../data/$data/model.pkl.gz"
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,floatX=float64, force_device=True" stdbuf -i0 -e0 -o0 python2.7 main.py --crowd_experiment -dropout_rate 0.5 -embedding_size 30 -train_file "../data/$data/train.txt" -test_file "../data/$data/test.txt" -dev_file "../data/$data/dev.txt" -eval_iter 7500 -data_dir "../data/$data/" -model_file "../data/$data/model.pkl.gz" -num_epoches 300
fi
cd ..
