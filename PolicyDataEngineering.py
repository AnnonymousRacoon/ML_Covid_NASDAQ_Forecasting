import pandas as pd
import re
from datetime import datetime,timedelta
from matplotlib import pyplot as plt



DATA1 = pd.read_csv("COVID_DATA/coronanet_release_allvars.csv")
DATA2 = pd.read_csv("COVID_DATA/coronanet_release_new.csv")
DATA3 = pd.read_csv("COVID_DATA/coronanet_release.csv")

DATA = pd.concat([DATA1,DATA2,DATA3]).drop_duplicates().dropna(subset = ['date_announced']).reset_index(drop = True)


print(DATA.describe())

cols = ['entry_type', 'correct_type', 'update_type',
       'update_level','init_country_level']
DATA = DATA[['entry_type', 'correct_type', 'update_type',
       'update_level','date_announced', 'date_start',
       'date_end', 'ISO_A3', 'init_country_level',
       'domestic_policy', 'type', 'compliance']]

DATA = DATA[DATA['ISO_A3'].isin(['USA',"GBR","DEU","ESP","ITA","CHN","AUS","BRA","NZL","RUS"])]

for column in DATA.columns:
    print(column)
    print(DATA[column].unique())


MAP = {
    "entry_type":{'new_entry':2 ,'update':1},
    "correct_type":{'original':2, 'correction':1},
    "update_type":{'End of Policy':2,'Change of Policy':1},
    "update_level":{'Strengthening':2,'Relaxing':-1 ,'Both Strengthening and Relaxing':1},
    "init_country_level":{'National':3,'Municipal':1, 'Provincial':2,'Other (e.g., county)':1,'No, it is at the national level':3},
}

ENG_DATA = DATA.date_announced.apply(lambda d: datetime.strptime(d, '%Y-%m-%d').date()).reset_index()
for key in MAP.keys():
    ENG_DATA[key] = DATA[key].apply(lambda entry: MAP[key].get(entry,0))

for type in ['Border','Regulation','Lockdown','Curfew','Social Distancing']:
    ENG_DATA[type] = DATA.type.apply(lambda entry: 1 if re.search(type,entry) else 0)

ENG_DATA['compliance'] = DATA.compliance.fillna('Empty').apply(lambda entry: 1 if re.search('Mandatory',entry) else 0)
ENG_DATA.drop(columns = ['index'],inplace = True)
ENG_DATA.fillna(0,inplace = True)

print


ENG_DATA_NEW = ENG_DATA.copy(deep = True)
for  delta in range(1,8):
    DATACOPY = ENG_DATA.copy(deep = True)
    DATACOPY.date_announced = DATACOPY.date_announced + timedelta(days=delta)
    for col in cols:
        DATACOPY[col] = DATACOPY[col]/float(delta)
    ENG_DATA_NEW = pd.concat([ENG_DATA_NEW,DATACOPY])


ENG_DATA = ENG_DATA_NEW.groupby(['date_announced']).sum()

print(ENG_DATA.head())
ENG_DATA.to_csv('COVID_DATA/covid_policies_engineered.csv')

ENG_DATA.plot()

plt.show()

