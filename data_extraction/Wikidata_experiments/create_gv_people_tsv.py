import os
import pickle
import sys
sys.path.insert(0,'..')

import queries
import utils
import pickle_utils
import pandas as pd

# settings
person_ontology_uri="http://www.wikidata.org/entity/Q5"

# input directory and files
INDIR='../../data/raw_instances'
statements_file="%s/wikidata-simple-statements.nt" % INDIR

# intermediate 'tmp' directory and files
TMPDIR='../../data/tmp'
people_file="%s/list_of_persons.p" % TMPDIR
pickle_with_all_data='%s/gv_people.p' % TMPDIR

# output directory and files
OUTDIR='../../data/extracted_instances'
OUTFILE='%s/tabular_gvperson_data.tsv' % OUTDIR

# Define the set of attributes to use
clean_attributes = {'http://www.wikidata.org/entity/P27c': 'country of citizenship', 'http://www.wikidata.org/entity/P172c': 'ethnic group', 'http://www.wikidata.org/entity/P103c': 'native language', 'http://www.wikidata.org/entity/P509c': 'cause of death', 'http://www.wikidata.org/entity/P21c': 'sex or gender', 'http://www.wikidata.org/entity/P106c': 'occupation', 'http://www.wikidata.org/entity/P937c': 'work location', 'http://www.wikidata.org/entity/P551c': 'residence', 'http://www.wikidata.org/entity/P69c': 'educated at', 'http://www.wikidata.org/entity/P140c': 'religion', 'http://www.wikidata.org/entity/P102c': 'member of political party', 'http://www.wikidata.org/entity/P20c': 'place of death', 'http://www.wikidata.org/entity/P19c': 'place of birth', 'http://www.wikidata.org/entity/P570c': 'date of death', 'http://www.wikidata.org/entity/P569c': 'date of birth', 'http://www.wikidata.org/entity/P1399c': 'convicted of'}

all_people=set(pickle.load(open(people_file, 'rb')))

print('People data loaded. %d entries' % len(all_people))

# DEFINE HEADERS
header=['instance uri'] #, 'lifespan', 'century']
header+=clean_attributes.values()
print('HEADER:', header)

# EXTRACT PEOPLE DATA FROM WIKIDATA TO A PICKLE
people_data=pickle_utils.wikidata_people_to_pickle([statements_file], all_people, clean_attributes.keys(), TMPDIR, pickle_with_all_data)
print('people data loaded')

sys.exit()

people_for_pandas=[]
for person_uri, person_from_json in people_data.items():
    person_from_json=utils.sets_to_dates(person_from_json)
    person_for_pandas=[person_uri]

    for attruri, attrlabel in clean_attributes.items():
        if attruri in person_from_json.keys():
            person_for_pandas.append(person_from_json[attruri])
        else:
            person_for_pandas.append("")
    people_for_pandas.append(person_for_pandas)

frame=pd.DataFrame(people_for_pandas)
frame.columns=header

print('%d columns before removing NIL columns' % len(frame.columns))
frame=frame.dropna(axis=1, how='all')
print('%d columns after removing NIL columns' % len(frame.columns))

frame.to_csv('%s/%s' % (INSTANCEDIR, TSVFILENAME), '\t')
