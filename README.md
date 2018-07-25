# Profiling
Extraction and categorization of world knowledge about people from Wikidata for the sake of creating profiles

### Brief guide to the directory structure

`encoder/` contains the code of our two developed methods, as well as the data they use. This data has been extracted within the `data_extraction/` directory.

#### `encoder/`

In general, everything should be runable with the shell scripts in this folder. They then invoke python scripts found in the subfolder `src/`, and deal with the data found in the subfolder `data/`.

`prepare_all.sh` -> this script prepares the data to be in the right format as is needed by the neural nets. Internally, it runs the script `src/preprocess.py`. Make sure to set up the right parameters that correspond to the right experiment and method.

The autoencoder can be run with the file `run.sh`. Make sure you setup the right experiment (crowd experiment or not). Alternatively, you can only run the test cases with the file `run_only_test.sh`.

To run the EMB model, run the file `emb_run.sh`, or `emb_run_only_test.sh` for only evaluating on the test data without training.

In the subfolder `src/`, there are the implementation of the two baselines that we used in this paper: `mfv_baseline.py` is the most-frequent-value baseline, whereas `naive_bayes_baseline.py` contains the implementation of the Naive Bayes method.

#### `data_extraction/`



#### Data description (needs an update)

1. **Which KB** This is [important](http://www.semantic-web-journal.net/system/files/swj1141.pdf), since DBpedia is much worse than wikidata in terms of both quantity (~5 times more people in Wikidata) and in terms of quality (Wikidata is manually curated all the time). Hence, I decided to use Wikidata (I also considered Freebase and BabelNet).

2. **Attributes** I started with the most common 100 attributes for people: after automatic removal of the ones that are instances of the class ID, we remain with around 60 attributes. These include also my inferred time attributes, like lifespan.

3. **Data sparsity** Most fields contain non-NILs for around 1% - 10% of all instances. This sounds quite sparse, but in practice this means between 20-30k and 300k instances have a value for an attribute (since we are dealing with 3 million people). Exact numbers can be found [here](https://docs.google.com/document/d/1qiKPNqBda1h17VoCVpS3pgGkMlDymKMjn3fyGdKZnSU/edit#).

4. **Data normalization** I performed all sorts of normalizations (removal of empty columns, parsing of integers and dates, removal of invalid dates, etc.). Still the data might need further normalization. For example, some attributes (e.g. occupation) have often sets of values (Bush is both an ex-president and a painter :) ).

5. **Embeddings** For ~200k entities, I included pre-trained entity embeddings for Freebase from the [word2vec page](https://code.google.com/archive/p/word2vec/).

6. **Data slicing** To give more focus to the initial experiments, yesterday we decided with Qizhe and Varun to create slices of data for some of the most common occupations. Based on a [list](https://docs.google.com/document/d/1qiKPNqBda1h17VoCVpS3pgGkMlDymKMjn3fyGdKZnSU/edit#) of the 10 most frequent occupations for people, i generated some slices of this data by keeping only the rows that contain the occupation specified.

And the actual data can be found at:
[tabular_person_data.tsv (all people data)](http://cm.fii800.lod.labs.vu.nl/tabular_person_data.tsv)
[tabular_actor_data.tsv (slice of people with an occupation ‘actor’)](http://cm.fii800.lod.labs.vu.nl/tabular_actor_data.tsv)
[tabular_politician_data.tsv (slice of people with an occupation ‘politician’)](http://cm.fii800.lod.labs.vu.nl/tabular_politician_data.tsv)
