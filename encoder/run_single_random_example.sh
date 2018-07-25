mode="gpu"
filename="data/politician/test$$.txt"
touch $filename
chmod 775 $filename
shuf -n 1 "data/politician/test.txt" > $filename
touch data/politician/empty_dev.txt
cd src
if [ "$mode" = "cpu" ]; then
    echo "Running cpu"
    python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -test_print_allowed True -pre_trained saved_model.pkl.gz -num_epoches 0 -test_file "../$filename" -dev_file "../data/politician/empty_dev.txt"
else
    echo "Running gpu"
    THEANO_FLAGS="mode=FAST_RUN,device=cuda0,floatX=float64, force_device=True" stdbuf -i0 -e0 -o0 python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -test_print_allowed True -pre_trained saved_model.pkl.gz -num_epoches 0 -test_file "../$filename" -dev_file "../data/politician/empty_dev.txt"
fi
cd ..
rm $filename
