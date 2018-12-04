# Data extraction
Extraction and categorization of world knowledge about people from Wikidata for the sake of creating profiles.

### Brief guide to the directory structure

#### Some background information

1. **Which KB** This is [important](http://www.semantic-web-journal.net/system/files/swj1141.pdf), since DBpedia is much worse than wikidata in terms of both quantity (~5 times more people in Wikidata) and in terms of quality (Wikidata is manually curated all the time). Hence, I decided to use Wikidata (I also considered Freebase and BabelNet).

2. **Attributes** I started with the most common 100 attributes for people: after automatic removal of the ones that are instances of the class ID, we remain with around 60 attributes. These include also my inferred time attributes, like lifespan.

3. **Data sparsity** Most properties contain non-empty values for around 1% - 10% of all instances. This sounds quite sparse, but in practice this means between 20-30k and 300k instances have a value for an attribute (since we are dealing with 3 million people). Exact numbers can be found [here](https://docs.google.com/document/d/1qiKPNqBda1h17VoCVpS3pgGkMlDymKMjn3fyGdKZnSU/edit#).

4. **Data normalization** I performed all sorts of normalizations (removal of empty columns, parsing of integers and dates, removal of invalid dates, etc.). Still the data might need further normalization. For example, some attributes (e.g. occupation) have often sets of values (Bush is both an ex-president and a painter :) ).

5. **Embeddings** For ~200k entities, I included pre-trained entity embeddings for Freebase from the [word2vec page](https://code.google.com/archive/p/word2vec/). Unfortunately, this mapping can not be done for all Wikidata entities, as most of them don't have an embedding in this project.

6. **Data slicing** To give more focus to the initial experiments, we decided to create slices of data for some of the most common occupations. Based on a [list](https://docs.google.com/document/d/1qiKPNqBda1h17VoCVpS3pgGkMlDymKMjn3fyGdKZnSU/edit#) of the 10 most frequent occupations for people, i generated some slices of this data by keeping only the rows that contain the occupation specified.


#### Data downloads

The actual data can be found at (contact me if you need the actual train-dev-test splits we used):
Full datasets:
[tabular_person_data.tsv (all people data)](http://cm.fii800.lod.labs.vu.nl/tabular_person_data.tsv)

[tabular_actor_data.tsv (slice of people with an occupation ‘actor’)](http://cm.fii800.lod.labs.vu.nl/tabular_actor_data.tsv)

[tabular_politician_data.tsv (slice of people with an occupation ‘politician’)](http://cm.fii800.lod.labs.vu.nl/tabular_politician_data.tsv)

[tabular_lawyer_data.tsv (slice of people with an occupation 'lawyer')](http://cm.fii800.lod.labs.vu.nl/tabular_lawyer_data.tsv)

Smaller datasets (with embeddings):
[person_emb_data.tsv](http://cm.fii800.lod.labs.vu.nl/person_emb_data.tsv)
[politician_emb_data.tsv](http://cm.fii800.lod.labs.vu.nl/politician_emb_data.tsv)
[actor_emb_data.tsv](http://cm.fii800.lod.labs.vu.nl/actor_emb_data.tsv)

#### More information

For more information, please consult our [paper on arXiv](https://arxiv.org/abs/1810.00782) - to be published officially soon, or contact Filip Ilievski: filip@ilievski.nl.
