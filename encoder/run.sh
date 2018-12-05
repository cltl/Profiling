mode="gpu"
#data="american"
data="politician"
data="gvamerican"
exp="gvamerican"
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -data_dir "../data/$data" -model_file "../data/$data/model.pkl.gz"
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,floatX=float64, force_device=True" stdbuf -i0 -e0 -o0 python2.7 main.py -experiment $exp -relations 8 -dropout_rate 0.5 -embedding_size 30 -train_file "../data/$data/train.txt" -test_file "../data/$data/test.txt" -dev_file "../data/$data/dev.txt" -eval_iter 15000 -data_dir "../data/$data/" -model_file "../data/$data/model.pkl.gz" -num_epoches 300 -topk_accuracy 3
fi
cd ..
