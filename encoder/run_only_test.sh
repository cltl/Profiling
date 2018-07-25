mode="gpu"
data="actor"
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 main.py -topk_accuracy 1 -dropout_rate 0.5 -embedding_size 30 -pre_trained "../data/$data/model.pkl.gz" -num_epoches 0 -train_file "../data/$data/train.txt" -test_file "../data/$data/test.txt" -dev_file "../data/$data/dev.txt" -eval_iter 750 -data_dir "../data/$data/" -model_file "../data/$data/model.pkl.gz" > out.txt
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float64, force_device=True" stdbuf -i0 -e0 -o0 python2.7 main.py -topk_accuracy 1 -dropout_rate 0.5 -embedding_size 30 -pre_trained "../data/$data/model.pkl.gz" -num_epoches 0 -train_file "../data/$data/train.txt" -test_file "../data/$data/test.txt" -dev_file "../data/$data/dev.txt" -eval_iter 750 -data_dir "../data/$data/" -model_file "../data/$data/model.pkl.gz" > out.txt
fi
cd ..
