## Profiling

`encoder/` contains the code of our two developed methods, as well as the data they use. This data has been extracted within the `data_extraction/` directory.

#### `encoder/`

In general, everything should be runable with the shell scripts in this folder. They then invoke python scripts found in the subfolder `src/`, and deal with the data found in the subfolder `data/`.

`prepare_all.sh` -> this script prepares the data to be in the right format as is needed by the neural nets. Internally, it runs the script `src/preprocess.py`. Make sure to set up the right parameters that correspond to the right experiment and method.

The autoencoder can be run with the file `run.sh`. Make sure you setup the right experiment (crowd experiment or not). Alternatively, you can only run the test cases with the file `run_only_test.sh`.

To run the EMB model, run the file `emb_run.sh`, or `emb_run_only_test.sh` for only evaluating on the test data without training.

In the subfolder `src/`, there are the implementation of the two baselines that we used in this paper: `mfv_baseline.py` is the most-frequent-value baseline, whereas `naive_bayes_baseline.py` contains the implementation of the Naive Bayes method.
