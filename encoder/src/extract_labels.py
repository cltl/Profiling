import config

import sys
import pickle
import requests
import json

values_file='values.pkl'
labels_file='labels.pkl'
labels_json='labels.json'
ATTRIBUTES=config.get_columns()

output_data=[]

wikidata_api_link="https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&format=json&ids="

data=pickle.load(open(values_file, 'rb'))
num_attr=0
for attribute in data:
    attr_json={}
    print("%d IDs for attribute %d" % (len(attribute), num_attr))
    for num, wikidata_uri in attribute.items():
        if not wikidata_uri:
            attr_json[num]=''
            continue
        wiki_id=wikidata_uri.split('/')[-1]
        req_link=wikidata_api_link + wiki_id
        r = requests.get(req_link)
        result_json=r.json()
        try:
            if 'en' in result_json['entities'][wiki_id]['labels']:
                label=result_json['entities'][wiki_id]['labels']['en']['value']
            else:
                label=list(result_json['entities'][wiki_id]['labels'].values())[0]['value']
        except:
            label=wiki_id
        attr_json[wiki_id]=label
        #except:
        #    print("error for %s" % wiki_id)
    print("Attribute number %d done!" % num_attr)
    output_data.append(attr_json)
    num_attr+=1

pickle.dump(output_data, open(labels_file, 'wb'), protocol=2)

with open(labels_json, 'w') as fp:
    json.dump(output_data, fp)
