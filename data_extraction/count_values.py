import argparse
import pandas as pd
import sys
if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description='Inspect a given TSV file for the amount of non-empty values, and more.')
	parser.add_argument('-f', '--filename', required=True, type=str,
		help='File path')
    
	args = vars(parser.parse_args())
	filename=args['filename']

	mappings={}
	with open('../data/tmp/mappings.tsv', 'r') as f:
		for line in f:
			uri, original_lbl, new_lbl=line.strip().split('\t')
			mappings[uri]=new_lbl

	df=pd.read_csv(filename, sep='\t', header=0)
	print(df.count())
	for c in df.columns:
		total=df[c].count()
		one_pc=total*0.01
		print(df[c].nunique())
		more_than_1pc=set()
		mapped=0
		for val, cnt in df[c].value_counts().iteritems():
			if str(val).strip() in mappings.keys():
				mapped+=cnt
			if cnt>one_pc:
				more_than_1pc.add(val)
				print(val, cnt)
		print('mapped', mapped, mapped/total)
#		print(more_than_1pc, len(more_than_1pc))
