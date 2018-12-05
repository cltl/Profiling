import os
import pickle
from rdflib.namespace import Namespace
import pandas as pd

import queries
import utils

# settings
WKD = Namespace('http://www.wikidata.org/entity/')
PERSON=WKD.Q5
person_ontology_uri="http://www.wikidata.org/entity/Q5"
NUMATTR=100

# input files
INDIR='../data/raw_instances'
statements_file="%s/wikidata-simple-statements.nt" % INDIR
ontology_file="%s/wikidata-instances.nt" % INDIR

# tmp files
TMPDIR='../data/tmp'
people_file="%s/list_of_persons.p" % TMPDIR
pickle_with_all_data='%s/gv_people.p' % TMPDIR

# output files
OUTDIR='../data/extracted_instances'
TSVFILENAME='%s/tabular_person_data.tsv' % OUTDIR

# load the list of all people URIs
if not os.path.exists(people_file):
    utils.extract_all_dudes(PERSON, ontology_file, people_file)
all_people=set(pickle.load(open(people_file, 'rb')))

# create attribute list
person_common_attributes=queries.get_most_frequent_attributes(person_ontology_uri, NUMATTR)
clean_attributes=utils.clean_and_label_relations(person_common_attributes)

# DEFINE HEADERS
header=['instance uri', 'lifespan', 'active years', 'first activity', 'last activity']
header+=clean_attributes.values()

# EXTRACT PEOPLE DATA FROM WIKIDATA TO A PICKLE
people_data=utils.wikidata_people_to_pickle([statements_file], all_people, clean_attributes.keys(), TMPDIR, pickle_with_all_data)

people_for_pandas=[]
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

frame.to_csv(TSVFILENAME, '\t')

