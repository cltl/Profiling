import os
import sys
import pickle
sys.path.insert(0,'..')
import utils

freebase_attr="http://www.wikidata.org/entity/P646-freebase"

def wikidata_people_to_pickle(files, all_people, attr_keys, idir, filename):
    try:
        people_data=pickle.load(open(filename, 'rb'))
        print("People data file found and loaded.")
    except:
        print("People data file not found. Extracting now...")
        people_datas=utils.extract_relations_from_files(files, all_people, attr_keys, idir)
        people_data=people_datas[0]
        print("People data extracted, to file %s." % filename)
        with open(filename, 'wb') as w:
            pickle.dump(filename,w)
    return people_data

def get_relevant_vectors(freebase_vectors_pickle, people_data, freebase_txt):
    if os.path.exists(freebase_vectors_pickle):
        vector_json=pickle.load(open(freebase_vectors_pickle, 'rb'))
    else:
        print("FB file does not exist. Creating it now...")
        freebase_people_uris=[utils.normalize_freebase(person_data[freebase_attr]) for person_uri, person_data in people_data.items() if freebase_attr in person_data and person_data[freebase_attr]!=""]
        print("Counted %d people with freebase ids" % len(freebase_people_uris))
        vector_json={}
        with open(freebase_txt , 'r') as freebase_raw_file:
            for line in freebase_raw_file:
                fid, *numbers=line.split()
                if len(numbers)>10 and fid in freebase_people_uris:
                    numbers=list(map(float, numbers))
                    vector_json[fid]=numbers

        pickle.dump(vector_json, open(freebase_vectors_pickle, 'wb'))
    return vector_json
