mode="gpu"
data="american"
filename="data/$data/profiler_input.tsv"
chmod 775 $filename
touch data/$data/empty_dev.txt
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -test_print_allowed True -pre_trained saved_model.pkl.gz -num_epoches 0 -test_file "../$filename" -dev_file "../data/politician/empty_dev.txt"
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,floatX=float64, force_device=True" stdbuf -i0 -e0 -o0 python2.7 main.py -dropout_rate 0.5 -relations 10 -embedding_size 30 --crowd_experiment -topk_accuracy 10 -test_print_allowed True -pre_trained "../data/$data/model.pkl.gz" -num_epoches 0 -test_file "../$filename" -dev_file "../data/$data/empty_dev.txt" -data_dir "../data/$data/"
fi
cd ..
#rm $filename
