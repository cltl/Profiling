import os
import pickle
import sys
sys.path.insert(0,'..')

import queries
import utils
import pickle_utils
import pandas as pd

person_ontology_uri="http://www.wikidata.org/entity/Q5"
NUMATTR=100

INDIR='data'
INSTANCEDIR='instance_data'
TSVFILENAME='tabular_person_data.tsv'

statements_file="%s/wikidata-simple-statements.nt" % INDIR

people_file="%s/list_of_persons.p" % INSTANCEDIR
all_people=set(pickle.load(open(people_file, 'rb')))

person_common_attributes=queries.get_most_frequent_attributes(person_ontology_uri, NUMATTR)
clean_attributes=utils.clean_and_label_relations(person_common_attributes)

# DEFINE HEADERS
header=['instance uri', 'lifespan', 'active years', 'first activity', 'last activity']
header+=clean_attributes.values()

# EXTRACT PEOPLE DATA FROM WIKIDATA TO A PICKLE
people_data=pickle_utils.wikidata_people_to_pickle([statements_file], all_people, clean_attributes.keys(), INSTANCEDIR)

people_for_pandas=[]
#firstN=list(people_data.keys())[:10]
for person_uri in people_data:
    person_from_json=people_data[person_uri]
    person_for_pandas=[]
    person_for_pandas.append(person_uri)
    person_from_json=utils.sets_to_dates(person_from_json)
    person_for_pandas+=utils.infer_properties(person_from_json, person_uri)

    for attruri, attrlabel in clean_attributes.items():
        if attruri in person_from_json:
            person_for_pandas.append(person_from_json[attruri])
        else:
            person_for_pandas.append("")
    people_for_pandas.append(person_for_pandas)

frame=pd.DataFrame(people_for_pandas)
frame.columns=header

fields_to_fix=['height', 'sport number']
frame[fields_to_fix] = frame[fields_to_fix].apply(pd.to_numeric, errors='coerce')

print('%d columns before removing NIL columns' % len(frame.columns))
frame=frame.dropna(axis=1, how='all')
print('%d columns after removing NIL columns' % len(frame.columns))


"""
for i, row in frame.iterrows():
    print(row['instance uri'])
    print(row['occupation'])
    print(row['religion'])

    print(row['lifespan'])
    print(row['active years'])
    print(row['first activity'])
    print(row['last activity'])
"""
frame.to_csv('%s/%s' % (INSTANCEDIR, TSVFILENAME), '\t')

