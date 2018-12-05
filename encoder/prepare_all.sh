#mkdir data
#mkdir data/politician

#echo "Downloading politician data..."
#cd data
#wget http://cm.fii800.lod.labs.vu.nl/tabular_politician_data.tsv 
#cd ..
#echo "Politician data downloaded and stored."


entitytype="actor"
entitytype="gvamerican"
exp="gvamerican"
cd src
python2.7 preprocess.py -experiment $exp -entity_type $entitytype
cd ..
