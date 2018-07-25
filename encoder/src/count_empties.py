import sys
import csv
from collections import defaultdict

data_path=sys.argv[1]

with open(data_path, 'rb') as data:
    spamreader = csv.reader(data, delimiter='\t', quotechar='"')
    headers = spamreader.next()

    filled=defaultdict(int)

    for row in spamreader:
        for i in xrange(len(row)):
            if row[i]:
                filled[headers[i]]+=1
    print filled
