#mkdir data
#mkdir data/politician

#echo "Downloading politician data..."
#cd data
#wget http://cm.fii800.lod.labs.vu.nl/tabular_politician_data.tsv 
#cd ..
#echo "Politician data downloaded and stored."


entitytype="american"
cd src
python2.7 preprocess.py --crowd_experiment -entity_type "$entitytype"
cd ..

