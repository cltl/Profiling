#!/bin/bash

# properties="69 21 103 140 102 20 19 509 569 570 26 172 106 1050" # main data properties
#properties="361"
#properties="569"
properties="1399"
cd ../data/raw_instances
for i in $properties; do
	echo "P${i} starting..."
	fn="P${i}c.nt"
	zgrep "> <http://www.wikidata.org/entity/P${i}c> <" wikidata-simple-statements.nt.gz > $fn
	echo "${i} done"
done
cd ../../data_extraction
