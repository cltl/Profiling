import os
import pickle
import sys
sys.path.insert(0,'..')

import queries
import utils
import pandas as pd
from collections import Counter, defaultdict
import csv

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

def map_values(tuples, mappings):
    new_tuples=defaultdict(int)
    for t, cnt in tuples:
        if str(t) not in mappings: continue
        new_t=mappings[str(t)]
        new_tuples[new_t]+=cnt
    sorted_new_tuples=[(k, new_tuples[k]) for k in sorted(new_tuples, key=new_tuples.get, reverse=True)]
    return sorted_new_tuples

def generate_rows(data, entropy_ordering, reverse=False):
    rows=[]
    if reverse: this_entropy_ordering=list(entropy_ordering[::-1])
    else: this_entropy_ordering=list(entropy_ordering)
    for entity_uri, prop_values in data.items():
        the_row=[]
        c=0
        for proppy in this_entropy_ordering:
            if proppy not in prop_values.keys():
                break
            the_row.append(prop_values[proppy])
            c+=1
            if c>=3 and c<10:
                if reverse:
                    rows.append([entity_uri] + ['']*(10-c) + the_row[::-1])
                else:
                    rows.append([entity_uri] + the_row + ['']*(10-c))
                the_row=list(the_row)
    return rows

def deduplicate(k):
    new_k = []
    for elem in k:
        if elem[1:] not in new_k:
            new_k.append(elem[1:])
    return new_k

def dump_to_file(header, data, fn):
    with open(fn, 'w') as w:
        w.write('\t'.join(header) + '\n')
        for row in data:
            w.write('\t'.join(row) + '\n')

INDIR='../../data/raw_instances'
INSTANCEDIR='../../data/instance_data'

statements_file="%s/wikidata-simple-statements.nt" % INSTANCEDIR

pickle_with_all_data='%s/us_crowd.p' % INSTANCEDIR

americans_tsv="%s/us_nationals.nt" % INSTANCEDIR
american_uris, american_ids=extract_americans(americans_tsv)

clean_attributes = {'http://www.wikidata.org/entity/P21c': 'sex or gender', 'http://www.wikidata.org/entity/P106c': 'occupation', 'http://www.wikidata.org/entity/P937c': 'work location', 'http://www.wikidata.org/entity/P69c': 'educated at', 'http://www.wikidata.org/entity/P140c': 'religion', 'http://www.wikidata.org/entity/P102c': 'member of political party', 'http://www.wikidata.org/entity/P20c': 'place of death', 'http://www.wikidata.org/entity/P19c': 'place of birth', 'http://www.wikidata.org/entity/P570c': 'date of death', 'http://www.wikidata.org/entity/P569c': 'date of birth'}

import pickle_utils

# DEFINE HEADERS
tmp_header=['instance uri', 'lifespan', 'century']
tmp_header+=clean_attributes.values()
header=[]
for col in tmp_header:
    if not col.startswith('date of'):
        header.append(col)
print(header)

# EXTRACT PEOPLE DATA FROM WIKIDATA TO A PICKLE
people_data=pickle_utils.wikidata_people_to_pickle([statements_file], american_uris, clean_attributes.keys(), INSTANCEDIR, pickle_with_all_data)

mappings={}
with open('mappings.tsv', 'r') as mapfile:
    rdr = csv.reader(mapfile, delimiter='\t', quotechar='"')
    hdr=next(rdr)
    for row in rdr:
        mappings[row[0]]=row[2]
print(mappings)

counts=defaultdict(list)

known_attributes_num=defaultdict(int)
the_head={}
the_reverse_head={}
locations_top_10={'place of death': {'New York City (NY)', 'Los Angeles (CA)', 'Washington D.C.', 'Chicago (IL)', 'Philadelphia (PA)', 'Boston (MA)', 'San Francisco (CA)', 'Santa Monica (CA)'}, 'place of birth': {'New York City (NY)', 'Chicago (IL)', 'Los Angeles (CA)', 'Philadelphia (PA)', 'Boston (MA)', 'Washington D.C.', 'San Francisco (CA)', 'Detroit (MI)'}, 'work location': {'Washington D.C.', 'Harrisburg (PA)', 'Sacramento (CA)', 'Austin (TX)', 'Springfield (IL)', 'Tallahassee (FL)', 'Baton Rouge (LA)', 'Montpelier (VT)', 'New York City (NY)', 'Phoenix (AZ)'}}
for person_uri, person_from_json in people_data.items():
    person_from_json=utils.sets_to_dates(person_from_json)
    lifespan=utils.infer_lifespan(person_from_json, person_uri)
    century=utils.infer_century(person_from_json, person_uri)
    if lifespan:
        counts['lifespan'].append(lifespan)
    if century:
        counts['century'].append(century)
    for att, att_val in person_from_json.items():
        att_lbl=clean_attributes[att]
        if att_lbl.startswith('date of'):
            continue
        if att_val:
            if isinstance(att_val, set):
                for v in att_val:
                    counts[att_lbl].append(v)
            else:
                counts[att_lbl].append(att_val)
    known_attrs=len(person_from_json.keys())
    known_attributes_num[known_attrs]+=1
    if known_attrs>=6:
        attr_values={}
        if str(century) in mappings.keys(): attr_values['century']=str(century)
        if lifespan in mappings.keys(): attr_values['lifespan']=lifespan
        for att, att_val in person_from_json.items():
            att_lbl=clean_attributes[att]
            if att_lbl.startswith('date of'):
                continue
            if isinstance(att_val, set):
                for v in att_val:
                    if v in mappings.keys():
                        mapped_value=mappings[v]
                        attr_values[att_lbl]=mapped_value
                        break
            else:
                if att_val in mappings.keys():
                    mapped_value=mappings[att_val]
                    attr_values[att_lbl]=mapped_value
        if len(attr_values)>=6:
            if 'century' in attr_values.keys() and 'religion' in attr_values.keys() and 'sex or gender' in attr_values.keys() and 'place of death' in attr_values.keys() and attr_values['place of death'] in locations_top_10['place of death'] and 'lifespan' in attr_values.keys() and 'place of birth' in attr_values.keys() and attr_values['place of birth'] in locations_top_10['place of birth']:# and 'work location' in attr_values.keys() and 'occupation' in attr_values.keys() and 'educated at' in attr_values.keys():
                the_head[person_uri]=attr_values
            if 'member of political party' in attr_values.keys() and 'educated at' in attr_values.keys() and 'occupation' in attr_values.keys() and 'work location' in attr_values.keys() and attr_values['work location'] in locations_top_10['work location'] and 'place of birth' in attr_values.keys() and attr_values['place of birth'] in locations_top_10['place of birth'] and 'lifespan' in attr_values.keys():# and 'place of death' in attr_values.keys() and 'sex or gender' in attr_values.keys() and 'religion' in attr_values.keys():
                the_reverse_head[person_uri]=attr_values

print(len(the_head))
print(len(the_reverse_head))
#print(the_head)


entropy_ordering=['century', 'religion', 'sex or gender', 'place of death', 'lifespan', 'place of birth', 'work location', 'occupation', 'educated at', 'member of political party']
header = ['Century', 'Religion', 'Gender', 'DeathPlace', 'Lifedur', 'BirthPlace', 'WorkLocation', 'Occupation', 'EducatedAt', 'PoliticalParty']

rows_inc_entropy=generate_rows(the_head, entropy_ordering, reverse=False)
rows_dec_entropy=generate_rows(the_reverse_head, entropy_ordering, reverse=True)

print(rows_inc_entropy)
print(len(rows_inc_entropy))

new_inc_entropy=deduplicate(rows_inc_entropy)
print(len(new_inc_entropy))

print('*****************')
print(rows_dec_entropy)
print(len(rows_dec_entropy))

new_dec_entropy=deduplicate(rows_dec_entropy)
print(len(new_dec_entropy))

dump_to_file(['instance_uri'] + header, rows_inc_entropy, 'increasing_entropy.tsv')
dump_to_file(['instance_uri'] + header, rows_dec_entropy, 'decreasing_entropy.tsv')

dump_to_file(header, new_inc_entropy, 'dedup_inc_entropy.tsv')
dump_to_file(header, new_dec_entropy, 'dedup_dec_entropy.tsv')

sys.exit()

for k, l in counts.items():
#    print(k, Counter(l))
    top10=Counter(l).most_common(10)
    print(k)
    mapped_top10=map_values(top10, mappings)
    for lbl, val in mapped_top10:
        print('%s\t%d' % (lbl, val))


sys.exit()

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

