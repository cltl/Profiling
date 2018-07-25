import pickle

def get_states():
    loc_filename="../data/raw_instances/loc.json"
    locs=pickle.load(open(loc_filename, 'rb'))
    states=set()
    for entity, ldata in locs.items():
        if ldata['country']=='us':
            states.add(ldata['state'])
    states.add('other')
    return sorted(list(states))

def get_age_groups():
    return ['Child (0-11)', 'Teen (12-17)', 'Adult (18-64)', 'Senior (65+)']

def get_decades(earliest_year=1900, latest_year=2019):
    decades=[]
    c=earliest_year
    while c<latest_year:
        decades.append('%d-%d' % (c, c+9))
        c+=10
    decades.append('other')
    return decades

def get_causes_of_death():
    return ['Intentional', 'Accidental', 'Suicide', 'other']

def get_past_convictions():
    return ['Yes', 'No']
#    return ['Murder', 'Assault', 'Bribery', 'Robbery', 'other']

def get_ethnic_groups():
    return ['African American', 'White/Caucascian', 'Asian', 'Native American', 'Hispanic/Latin', 'other']

def get_languages():
    return ['English', 'Spanish', 'Chinese', 'other']

def get_political_parties():
    return ['Democratic Party', 'Republican Party', 'other']

def get_religions():
    return ['Christian', 'Islam', 'Judaism', 'Buddhism/Hinduism', 'Atheist/Agnostic', 'other']

def get_education_levels():
    return ['Less than high school', 'High school graduate', 'Higher degree', 'other']

def get_conditions():
    return ['Mental', 'Organic', 'other']
#    return ['Alcoholism', 'Paralysis', 'Neurological disorder', 'Visual impairment', 'Hearing disorder', 'Tuberculosis', 'other']

prop_values={}

#prop_values['age_group']=get_age_groups()
#prop_values['BirthDate']=get_decades()
#prop_values['DeathDate']=get_decades()
prop_values['BirthPlace']=get_states()
prop_values['Residence']=get_states()
prop_values['DeathPlace']=get_states()
prop_values['CauseOfDeath']=get_causes_of_death()
prop_values['PastConviction']=get_past_convictions()
prop_values['Ethnicity']=get_ethnic_groups()
prop_values['MedicalCondition']=get_conditions()
prop_values['NativeLanguage']=get_languages()
prop_values['EducationLevel']=get_education_levels()
prop_values['PoliticalParty']=get_political_parties()
prop_values['Religion']=get_religions()
print(prop_values)

import json
with open('values.json', 'w') as j:
    json.dump(prop_values, j)
