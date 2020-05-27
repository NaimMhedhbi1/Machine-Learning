
import pandas as pd 
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn import preprocessing 
sns.set(color_codes=True)
 
df= pd.read_excel('qualité_c1.xlsx')

df['score'] = 1/2 * df['Courte duree'] + 1/6 * df['chev_temp'] + 1/6 * df['injoignable'] + 1/6 * df['conformité']


min_max_scaler = preprocessing.MinMaxScaler()
df_minmax = min_max_scaler.fit_transform(df)
df1 = pd.DataFrame(df_minmax)
df1.set_axis(['Enq','courte durée', 'chev_temp', 'injoignable', 'conformité', 'score_scaled'], axis=1, inplace=True)
df['score_scaled'] = df1['score_scaled']

df = df.sort_values(['score_scaled'] , ascending = False )





