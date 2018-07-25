import glob
from rdflib import Graph
import location_parser as loc
import os
import sys
import time

def refine_political_party(gr, data):
    c=0
    new_data=set()
    for o in data:
        part_of_result=gr.query("""
        SELECT ?part {
            <%s> <http://www.wikidata.org/entity/P361c> ?part
        } LIMIT 1
        """ % o)
#        print(o, len(part_of_result))
#        for v in part_of_result:
#            print(v)
        if len(part_of_result):
            for v in part_of_result:
                new_data.add(v['part'])
        else:
            new_data.add(o)

        c+=len(part_of_result)
    print(c)
    return new_data

def obtain_counts(gr, mid_query):
    q="""SELECT ?x (COUNT(DISTINCT(?a)) AS ?count) 
         WHERE { %s } 
         GROUP BY ?x """ % mid_query
    counts=gr.query(q)
    count_nums=[]
    for row in counts:
        count_nums.append(tuple([int(row['count']), str(row['x'])]))
        #count_nums.append(int(row['count']))
    print(sorted(count_nums))

in_dir = "../data/raw_instances"

us_nationals = "%s/us_nationals.nt" % in_dir

mapping={'http://www.wikidata.org/entity/P69c': 'educated at', 'http://www.wikidata.org/entity/P21c': 'sex or gender', 'http://www.wikidata.org/entity/P103c': 'native language', 'http://www.wikidata.org/entity/P140c': 'religion', 'http://www.wikidata.org/entity/P102c': 'member of political party', 'http://www.wikidata.org/entity/P20c': 'place of death', 'http://www.wikidata.org/entity/P19c': 'place of birth', 'http://www.wikidata.org/entity/P509c': 'cause of death', 'http://www.wikidata.org/entity/P569c': 'date of birth', 'http://www.wikidata.org/entity/P570c': 'date of death', 'http://www.wikidata.org/entity/P26c': 'spouse', 'http://www.wikidata.org/entity/P172c': 'ethnic group', 'http://www.wikidata.org/entity/P106c': 'occupation', 'http://www.wikidata.org/entity/P1050c': 'medical condition', 'http://www.wikidata.org/entity/P1399c': 'convicted of'}

aux_data = {'P102c': 'P361c'}


loc_pickle_file='%s/loc.json' % in_dir


def inspect(in_dir):
    for f in glob.glob("%s/P*.nt" % in_dir):
        print(f)
        fn = (f.split('/')[-1]).rstrip('.nt')
        try:
            label = mapping['http://www.wikidata.org/entity/%s' % fn]
        except KeyError:
            continue
        if 'place of' not in label:
            continue
#        if label!='convicted of': continue
        g = Graph().parse(f, format='nt').parse(us_nationals, format='nt')
        if fn in aux_data:
            g.parse('%s/%s.nt' % (in_dir, aux_data[fn]), format='nt')

        mid_query="""
            ?a <http://www.wikidata.org/entity/P27c> <http://www.wikidata.org/entity/Q30> ;
            <http://www.wikidata.org/entity/%s> ?x
        """ % fn

        unique_objects = g.query("""
        SELECT DISTINCT ?x {
            %s
        }
        """ % mid_query)
        unique_subjects = g.query("""
        SELECT DISTINCT ?a {
            %s
        }
        """ % mid_query)
        print("Property: %s (%s). Unique entities: %d. Unique values: %d" % (fn, label, len(unique_subjects), len(unique_objects)))
        if not len(unique_subjects): continue
        printable_objects=[]
        for o in unique_objects:
            printable_objects.append(str(o['x']))
        print(printable_objects)

        #obtain_counts(g, mid_query)
        state_data=loc.get_states_for_resources(printable_objects, loc_pickle_file)

if __name__ == '__main__':
    inspect(in_dir)
#    os.execv(__file__, sys.argv)
#    printable_objects=refine_political_party(g, printable_objects)
#    print(len(printable_objects))

