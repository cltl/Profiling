import os
import pickle
import sys
sys.path.insert(0,'..')

import queries
import utils
import pandas as pd

def extract_americans(the_tsv):
    identifiers=set()
    uris=set()
    with open(the_tsv, 'r') as f:
        for line in f:
            uri=line.split()[0]
            uri=uri.rstrip('>').lstrip('<')
            identifier=uri.split('/')[-1]
            uris.add(uri)
            identifiers.add(identifier)
    return uris,identifiers

person_ontology_uri="http://www.wikidata.org/entity/Q5"
NUMATTR=100

INDIR='../../data/raw_instances'
INSTANCEDIR='../../data/instance_data'
TSVFILENAME='%s/tabular_americans_data.tsv' % INSTANCEDIR

statements_file="%s/wikidata-simple-statements.nt" % INSTANCEDIR

pickle_with_all_data='%s/us_crowd.p' % INSTANCEDIR

americans_tsv="%s/us_nationals.nt" % INSTANCEDIR
american_uris, american_ids=extract_americans(americans_tsv)

clean_attributes = {'http://www.wikidata.org/entity/P21c': 'sex or gender', 'http://www.wikidata.org/entity/P106c': 'occupation', 'http://www.wikidata.org/entity/P937c': 'work location', 'http://www.wikidata.org/entity/P69c': 'educated at', 'http://www.wikidata.org/entity/P140c': 'religion', 'http://www.wikidata.org/entity/P102c': 'member of political party', 'http://www.wikidata.org/entity/P20c': 'place of death', 'http://www.wikidata.org/entity/P19c': 'place of birth', 'http://www.wikidata.org/entity/P570c': 'date of death', 'http://www.wikidata.org/entity/P569c': 'date of birth'}


print('Extracting data for %d Americans..' % len(american_ids))

# DEFINE HEADERS
tmp_header=['instance uri', 'lifespan', 'century']
tmp_header+=clean_attributes.values()
header=[]
for col in tmp_header:
    if not col.startswith('date of'):
        header.append(col)
print(header)

# EXTRACT PEOPLE DATA FROM WIKIDATA TO A PICKLE
people_data=utils.wikidata_people_to_pickle([statements_file], american_uris, clean_attributes.keys(), INSTANCEDIR, pickle_with_all_data)

print('people data loaded')
print(people_data)

people_for_pandas=[]
for person_uri, person_from_json in people_data.items():
    person_from_json=utils.sets_to_dates(person_from_json)
    lifespan=utils.infer_lifespan(person_from_json, person_uri)
    century=utils.infer_century(person_from_json, person_uri)
    person_for_pandas=[person_uri, lifespan, century]

    for attruri, attrlabel in clean_attributes.items():
        if attrlabel.startswith('date of'):
            continue
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

