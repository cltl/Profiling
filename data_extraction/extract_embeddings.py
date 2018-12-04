import sys
import os
import pickle
import utils
import pandas as pd

INSTANCEDIR='instance_data'
freebase_txt="%s/freebase/freebase-skipgram1000.txt" % INSTANCEDIR
freebase_attr="http://www.wikidata.org/entity/P646-freebase"

#filename="%s/tabular_politician_data.tsv" % INSTANCEDIR
filename="%s/tabular_person_data.tsv" % INSTANCEDIR
df=pd.read_csv(filename, '\t')

freebase_vectors_pickle='%s/freebase_vectors.p' % INSTANCEDIR

freebase_people_uris=set()
for index, person_data in df.iterrows():
    if freebase_attr in person_data and person_data[freebase_attr]!="":
        if type(person_data[freebase_attr]) is str:
            freebase_people_uris.add(utils.normalize_freebase(person_data[freebase_attr]))
        elif type(person_data[freebase_attr]) is set:
            freebase_people_uris|=utils.normalize_freebase(person_data[freebase_attr])
print("Counted %d people in wikidata with freebase ids" % len(freebase_people_uris))
vector_json={}
with open(freebase_txt , 'r') as freebase_raw_file:
    for line in freebase_raw_file:
        fid, *numbers=line.split()
        if fid in freebase_people_uris:
            numbers=list(map(float, numbers))
            vector_json[fid]=numbers
print(len(vector_json.keys()))
