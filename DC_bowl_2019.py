import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')
import matplotlib.pylab as plt
import calendar
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('C:/Users/121/.spyder-py3/data-science-bowl-2019/train.csv')

print(train.isnull().sum())
""" there is no missing values """ 
sns.boxplot(x = train['type']) 
