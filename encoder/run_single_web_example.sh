filename="demo_data/politician/test$1.txt"
chmod 775 $filename
touch data/politician/empty_dev.txt
cd src
python2.7 main.py -dropout_rate 0.5 -embedding_size 30 -test_print_allowed True -pre_trained saved_model.pkl.gz -num_epoches 0 -test_file "../$filename" -dev_file "../demo_data/politician/empty_dev.txt"
cd ..
#rm $filename
