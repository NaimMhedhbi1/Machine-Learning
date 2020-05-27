import pandas as pd 
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn.preprocessing import StandardScaler
sns.set(color_codes=True)
 
df= pd.read_excel('C:/Users/121/.spyder-py3/EDA projects/AI_2_12.xlsx')
#df_object = df_initial.select_dtypes(include = ['object'])
#df = df_initial.select_dtypes(include = ['float64','int64'])

#df.sort_values(by='Date', ascending=False).groupby(level=0).first()





#df1 = df.sort_values(by='Date_df').drop_duplicates('Date_df', keep='last')

df2 =df.sort_values(['QID', 'Date_df']).drop_duplicates('Date_df', keep='first') 
