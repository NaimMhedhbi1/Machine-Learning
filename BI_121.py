import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

#df = pd.read_csv("C:/Users/121/.spyder-py3/EDA projects/pg_maroc_364.csv", sep=';', error_bad_lines=False, index_col=False, dtype='unicode' , encoding = 'UTF-8')
df= pd.read_excel("C:/Users/121/.spyder-py3/EDA projects/ifes_16_12.xlsx.xlsx")
df1= pd.read_excel("C:/Users/121/.spyder-py3/EDA projects/qualité_c2.xlsx")
list_columns = df.columns.tolist() 
#fraud_detection
df['END_TIME'] = pd.to_datetime(df['END_TIME'] , errors = 'coerce')
df['START_TIME']= pd.to_datetime(df['START_TIME'], errors = 'coerce')
df['diff_hours'] = df['END_TIME'] - df['START_TIME']
df['difference_hours'] = df['diff_hours']/np.timedelta64(1,'h')
h_median = ( df['difference_hours'].median() ) * 0.6 ; 
df['diff_hours']= pd.to_datetime(df['diff_hours'] , errors = 'coerce')
df ['START_TIME_1'] =  df ['START_TIME'].dt.time
df['END_TIME_1'] =  df['END_TIME'].dt.time 
df['diff_hours_1'] = df ['diff_hours'].dt.time 

evaluation_qualité = [] 
for row in df['difference_hours'] : 
    if row <= df['difference_hours'].quantile(0.25):
        evaluation_qualité.append('courte_durée')
    elif df['difference_hours'].quantile(0.25) < row < df['difference_hours'].quantile(0.75):
        evaluation_qualité.append('durée_moyenne')
    elif row >= df['difference_hours'].quantile(0.25):
         evaluation_qualité.append('durée_bonne')
    else :
        evaluation_qualité.append('not allowed')

df['evaluation_qualité'] = evaluation_qualité 

occur = df['NOM_ENQ'].value_counts()
evaluation = df.groupby(['NOM_ENQ'])['evaluation_qualité'].value_counts()
evaluation_1 = df.groupby(['evaluation_qualité'])['NOM_ENQ'].value_counts()





"""y = df1.loc[df1['NOM_ENQ'], 'Remark'].sum()
x = df1.groupby('NOM_ENQ')["Remark"].sum() 
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(df1[['Remark']]).toarray())
# merge with main df bridge_df on key values
df1 = df1.join(enc_df)

# generate binary values using get_dummies
dum_df = pd.get_dummies(df1, columns=["Remark"], prefix=["Remark_is"] )
df2 = df1.join(dum_df)
"""

duplicateRowsDF2 = df[df.duplicated(['TEL'])]

duplicateRowsDF2_info =duplicateRowsDF2[['NOM_SUP','NOM_ENQ','TEL']] 
duplicate_counts= duplicateRowsDF2_info.groupby(['NOM_SUP','NOM_ENQ'])['TEL'].count()
duplicate_counts_1= duplicateRowsDF2_info.groupby(['NOM_ENQ'])['TEL'].count()
occur = df['NOM_ENQ'].value_counts()
dDF2_info = pd.DataFrame.from_items([('occur', occur), ('duplicate_counts_1',duplicate_counts_1)])
dDF2_info['duplicate_counts_1'].fillna(0, inplace=True)
dDF2_info['duplicate_counts_1'] = dDF2_info['duplicate_counts_1'].apply(np.int64)



df3 = pd.read_excel('C:/Users/121/.spyder-py3/EDA projects/score_qualité.xlsx')





























"""
## Select duplicate rows except first occurrence based on all columns
duplicateRowsDF = df[df.duplicated()] 
#Find Duplicate Rows based on selected columns
duplicateRowsDF1= df[df.duplicated(['QID'])]
duplicateRowsDF2 = df[df.duplicated(['D17'])]
duplicateRowsDF1_info =duplicateRowsDF1[['NOM_SUPERVISEUR','NOM_ENQUETEUR','D17']] 
duplicateRowsDF2_info =duplicateRowsDF2[['NOM_SUPERVISEUR','NOM_ENQUETEUR','D17']] 

age = df [['QID','Q62', 'evaluation_qualité']]       
#sns.distplot(age['Q62_new'], color='g', bins=10, hist_kws={'alpha': 0.9});

#age['age_pct_intervals'] = age.Q62_new.apply(lambda x : pd.age.Q62_new.value_counts()/age['Q62_new'].sum()) 
age_control = pd.crosstab(index=age["Q62_new"],  # Make a crosstab
                      columns="count")
age_control['count_prct'] = (age_control / age_control.sum()) * 100 
weights = [0.5 , 0.25 ,0 , 0.25]
age_control['compare_weights'] = age_control.count_prct.isin(weights).astype(int)
age_control['weights'] = np.array(weights)*100
age_control['weights_difference'] =  age_control['count_prct'] - age_control['weights'] 
age_controle_notes = [] 
for row in age_control['weights_difference'] : 
    if row > 0  :
        age_controle_notes.append('Alert : we should respect the weights ')        
    else :
        age_controle_notes.append('Alert : we should respect the weights')
        
age_control['age_controle_notes'] = age_controle_notes

        
#age_control_1['compare_diff'] = age_control_1.apply(lambda row , i : (compare - firsts) for row , i in (age.compare, firsts)  , axis = 1)
       
#overlap
duplicate_counts= duplicateRowsDF2_info.groupby(['NOM_SUPERVISEUR','NOM_ENQUETEUR'])['D17'].count()
duplicate_counts_1= duplicateRowsDF2_info.groupby(['NOM_ENQUETEUR'])['D17'].count()
occur = df['NOM_ENQUETEUR'].value_counts()


dDF2_info = pd.DataFrame.from_items([('occur', occur), ('duplicate_counts_1',duplicate_counts_1)])
periods = df[['START_TIME', 'TIMER_END']].apply(lambda x: (pd.date_range(x['START_TIME'], x['TIMER_END']),), axis=1)

overlap = periods.apply(lambda col: periods.apply(lambda col: col[0].isin(col[0]).any()))
df['overlap_count'] = overlap[overlap].apply(lambda x: x.count() - 1, axis=1)
#overlap_times= overlap_times.loc[~overlap_times.index.duplicated(keep='first')]
overlap_times.set_index(overlap_times['DATE'])
overlap_times = df[['QID','DATE','NOM_ENQUETEUR','D10A','overlap_count','START_TIME_1','END_TIME_1','NOM_SUPERVISEUR']]

#overlap_times.reset_index(inplace=True, drop=True)
ovppp = overlap_times.loc[(overlap_times.overlap_count == 1 )&(overlap_times.NOM_ENQUETEUR == overlap_times.D10A)]
#export_output_to_excel

duplicateRowsDF1_info.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/duplicateRowsDF1_info.xlsx')
frd_info.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/frd_info.xlsx')
frd_before_7_am_info.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/frd_before_7_am_info.xlsx')
frd_after_9pm_info.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/frd_after_9pm_info.xlsx')
duplicate_counts.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/duplicate_counts.xlsx', index = True)
dDF2_info.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/dDF2_info.xlsx', index = True)
overlap_times.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/overlap_times.xlsx') 
ovppp.to_excel('C:/Users/user/.spyder-py3/pg30_marroc/ovppp.xlsx')      
"""