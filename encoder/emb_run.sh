mode="gpu"
data="personmini"
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 embeddings_main.py -dropout_rate 0.5 -embedding_size 30
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float64, force_device=True, exception_verbosity=high" stdbuf -i0 -e0 -o0 python2.7 embeddings_main.py -embedding_size 1000 -train_file "../data/$data/train.txt" -test_file "../data/$data/test.txt" -dev_file "../data/$data/dev.txt" -eval_iter 2000 -num_epoches 300 -model_file "../data/$data/emb_model.pkl.gz" -hidden_size 128
fi
cd ..
