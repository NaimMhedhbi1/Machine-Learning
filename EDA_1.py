import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

house = pd.read_csv('file:///C:/Users/121/.spyder-py3/kc_house_data.csv/kc_house_data.csv')
"""find out correaltion between columns  """
corr = house.corr() 
plt.figure(figsize=(16,8))
sns.heatmap(corr, annot = True ,cmap="RdBu")

'''highlight only the variables that are highly correlated ''' 
# corr[corr>=.5]
plt.figure(figsize=(10,10))
mask = np.zeros_like(corr[corr>=.5],dtype=np.bool)

# Create a msk to draw only lower diagonal corr map
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr[corr>=.5],annot=True,mask=mask,cbar=False)

date_house = house.sort_values(by="date")
plt.figure(figsize=(16,8))
sns.scatterplot(date_house.date,date_house.price,hue=date_house.floors,alpha=.9,size=date_house.grade,palette="winter_r")
plt.xticks([])
plt.show()

house.dtypes

#lets scatter plot all var against price to locate some outliers if possible
def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()


categorical_summarized(house, x='bedrooms', hue='floors', palette='Set1', verbose=True)
df = pd.read_csv('C:/Users/121/Downloads/train.csv')