cd src
python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -test_print_allowed True -pre_trained saved_model.pkl.gz -num_epoches 0 -test_file "../data/politician/single_test_trans.txt" -dev_file "../data/politician/single_dev.txt" > out.txt
cd ..
