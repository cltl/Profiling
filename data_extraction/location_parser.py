import os
from geopy.geocoders import Nominatim
from SPARQLWrapper import SPARQLWrapper, JSON
from geopy.exc import GeocoderTimedOut
import time
import pickle

def sparql_get_latlong(query):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    print(results)
    if not len(results) or not(len(results["results"]["bindings"])):
        return None, None
    try:
        for result in results["results"]["bindings"]:
            return result["lat"]["value"], result["long"]["value"]
    except TypeError:
        return None, None

def isNaN(num):
    return num != num


sparql = SPARQLWrapper("http://sparql.fii800.lod.labs.vu.nl/sparql")

def get_states_for_resources(resources, data_pickle):
    if os.path.exists(data_pickle):
        data=pickle.load(open(data_pickle, 'rb'))
    else:
        data={}
    for resource in resources:
        if resource in data:
            continue

        geolocator = Nominatim()
        query = """
            select ?lat ?long where {
                <%s> <http://www.wikidata.org/entity/P625c> ?blank .
                ?blank <http://www.wikidata.org/ontology#latitude> ?lat ;
                <http://www.wikidata.org/ontology#longitude> ?long
            } limit 1
        """ % resource
        print(resource)
        latitude, longitude = sparql_get_latlong(query)
        if not latitude or not longitude: continue
        if isNaN(latitude) or isNaN(longitude): continue
        #try:
        location = geolocator.reverse("%s, %s" % (latitude, longitude), timeout=3)
        #except HTTPError as err:
        #    print(resource, err.code)  
        #    continue
        try:
            state = location.raw['address']['state']
            country_code = location.raw['address']['country_code']
            print(state, country_code)
            data[resource]={'lat': latitude, 'long': longitude, 'country': country_code, 'state': state}
        except KeyError:
            print(location.raw)
        pickle.dump(data, open(data_pickle, 'wb'))
        time.sleep(1)
    return data     
