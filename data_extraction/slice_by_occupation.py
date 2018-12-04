import sys
sys.path.insert(0,'..')
import utils
import pandas as pd

top10={"politician": "http://www.wikidata.org/entity/Q82955", "association football player": "http://www.wikidata.org/entity/Q937857", "actor": "http://www.wikidata.org/entity/Q33999", "writer" : "http://www.wikidata.org/entity/Q36180", "painter": "http://www.wikidata.org/entity/Q1028181", "journalist": "http://www.wikidata.org/entity/Q1930187", "university teacher": "http://www.wikidata.org/entity/Q1622272", "singer": "http://www.wikidata.org/entity/Q177220", "lawyer": "http://www.wikidata.org/entity/Q40348", "composer": "http://www.wikidata.org/entity/Q36834"}

chosen={"politician": "http://www.wikidata.org/entity/Q82955", "lawyer": "http://www.wikidata.org/entity/Q40348"}

human=False
if len(sys.argv)<2:
    occupation='http://www.wikidata.org/entity/Q82955'
    occupation_tsv='tabular_politician_data.tsv'
else:
    occupation_tsv='tabular_%s_data.tsv' % sys.argv[1]
    if sys.argv[1]!='human' and sys.argv[1] in chosen:
        occupation=chosen[sys.argv[1]]
    else:
        occupation=''
        human=True

OUTDIR='../data/extracted_instances'

filename="%s/tabular_person_data.tsv" % OUTDIR
df=pd.read_csv(filename, '\t')

new_rows=[]
for index, row in df.iterrows():
    if human or utils.right_occupation(row, occupation):
        new_rows.append(row)

print(len(new_rows))
frame=pd.DataFrame(new_rows)
frame.columns=df.columns

print('%d columns before removing NIL columns' % len(frame.columns))
frame=frame.dropna(axis=1, how='all')
print('%d columns after removing NIL columns' % len(frame.columns))

frame.to_csv('%s/%s' % (OUTDIR, occupation_tsv), '\t')
#print(df['occupation'].str.contains('http://www.wikidata.org/entity/Q82955'))
