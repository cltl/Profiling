mode="gpu"
data="personmini"
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 embeddings_main.py -dropout_rate 0.5 -embedding_size 30
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float64, force_device=True, exception_verbosity=high" stdbuf -i0 -e0 -o0 python2.7 embeddings_main.py  -topk_accuracy 1 -embedding_size 1000 -num_epoches 0 -train_file "../data/$data/train.txt" -test_file "../data/$data/test.txt" -dev_file "../data/$data/dev.txt" -hidden_size 128 -pre_trained "../data/$data/emb_model.pkl.gz" -model_file "../data/$data/emb_model.pkl.gz"
fi
cd ..
