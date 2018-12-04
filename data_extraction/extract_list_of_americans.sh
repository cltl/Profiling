#!/bin/bash

cd ../data/raw_instances
zgrep '> <http://www.wikidata.org/entity/P27c> <http://www.wikidata.org/entity/Q30>' wikidata-simple-statements.nt.gz > us_nationals.nt
#zgrep '> <http://www.wikidata.org/entity/P27c> <http://www.wikidata.org/entity/Q30>' wikidata-simple-statements.nt.gz | awk '{print $1}' > us_nationals.nt
cd ../../data_extraction
