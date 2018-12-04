#!/bin/bash

infile='../data/raw_instances/wikidata-simple-statements.nt'
outfile='../data/tmp/us_nationals.nt'

grep '> <http://www.wikidata.org/entity/P27c> <http://www.wikidata.org/entity/Q30>' $infile > $outfile
# ../data/raw_instances/wikidata-simple-statements.nt > us_nationals.nt
#zgrep '> <http://www.wikidata.org/entity/P27c> <http://www.wikidata.org/entity/Q30>' wikidata-simple-statements.nt.gz | awk '{print $1}' > us_nationals.nt
