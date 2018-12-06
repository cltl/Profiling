mode="gpu"
data="gvamerican"
filename="data/$data/test.txt"
exp='gvamerican'

chmod 775 $filename
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -test_print_allowed True -pre_trained saved_model.pkl.gz -num_epoches 0 -test_file "../$filename" -dev_file "../data/politician/empty_dev.txt"
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,floatX=float64, force_device=True" stdbuf -i0 -e0 -o0 python2.7 main.py -dropout_rate 0.5 -relations 8 -embedding_size 30 -experiment $exp -topk_accuracy 2 -model_file "../data/$data/model.pkl.gz" -num_epoches 0 -test_file "../$filename" -dev_file "../data/$data/dev.txt" -data_dir "../data/$data/"
fi
cd ..
#rm $filename
