import pandas as pd 
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn.preprocessing import StandardScaler
sns.set(color_codes=True)
 
df_initial = pd.read_excel('AI_2_12.xlsx')
df_object = df_initial.select_dtypes(include = ['object'])
df = df_initial.select_dtypes(include = ['float64','int64'])

"display the top 10 rows "
head_1 = df.head(10) 
"display the bottom 10 rows "
tail_1 = df.tail(10) 
"checking the data types of all variables " 
type_of_variables = df.dtypes

"rows containing duplicated rows" 
duplicated_df_rows = df[df.duplicated()] 

"# Used to count the number of rows before removing the data"
count_df = df.count()

"drop the column if zeros make up more than 25% of the column"""       
drop_cols = df.columns[(df == 0).sum() > 0.25*df.shape[0]]
df.drop(drop_cols, axis = 1, inplace = True) 

missing_values = df.isnull().sum()

"drop columns with ( count = 0 ) , nan values "
df = df.dropna(axis = 1, how = 'all')
count_df_1 = df.count() 

"calculate the quantiles Q1 & Q3"""
Quantile_1 = df.quantile(0.25)
Quantile_3 = df.quantile(0.75)
IQR = (Quantile_3 - Quantile_1 )

"""describe the dataframe"""
describe_df = df.describe(include='all')
df_information = df.info() 

    
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

"Univariate Analysis"
"""# Target Variable: Survival
c_palette = ['tab:blue', 'tab:orange']
categorical_summarized(train_df, y = 'Survived', palette=c_palette)
Bivariate Analysis
# Feature Variable: Gender
categorical_summarized(train_df, y = 'Sex', hue='Survived', palette=c_palette)
"""


"""def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed
    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    plt.show()
# univariate analysis
quantitative_summarized(dataframe= train_df, y = 'Age', palette=c_palette, verbose=False, swarm=True)
# bivariate analysis with target variable
quantitative_summarized(dataframe= train_df, y = 'Age', x = 'Survived', palette=c_palette, verbose=False, swarm=True)

Multivariate Analysis
# multivariate analysis with Embarked variable and Pclass variable
quantitative_summarized(dataframe= train_df, y = 'Age', x = 'Embarked', hue = 'Pclass', palette=c_palette3, verbose=False, swarm=False)
"""

"""get value counts"""
cu = []
i = []
for cn in df.columns:
    cu.append(df[cn].value_counts())
    i.append(cn)

cuu = pd.DataFrame(cu, index=i).T


"EXECUTE in console "
"""boxplot of a column """
sns.boxplot(data = df, x='Q1011_1')
#df.boxplot(column = "Q1011_1")  


"count plot of any column"
sns.countplot(data = df, x = 'Q1011_1')
my_df = pd.crosstab(index = df["Q1011_2"],  # Make a crosstab
                              columns="sum")      # Name the count column
my_df.plot.bar()
my_df_1 = my_df/my_df.sum()
"""        
df.boxplot(column="Q1011_1",        # Column to plot
                 by= "Q1011_2",         # Column to split upon
                 figsize= (8,8)) """      
           
"""cross of a 2 column '''''croisement'''  """     
df_cross_table = pd.crosstab(index= df["Q1011_2"], 
                          columns=df["Q1011_1"])

df_cross_table.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)  

"""Correlation of variables """

x = df.iloc[:,5:11]  #independent columns
y = df.iloc[:,12]    #target column i.e price range
#get correlations of each features in dataset
corrmat = x.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(7,7))
#plot heat map
g=sns.heatmap(x[top_corr_features].corr(),annot=True,cmap="RdYlGn")


"to find the features that have a strong correlation with (a column ) "
df_corr = df.iloc[:,3:11].corr()['Q13']
golden_feature_list = df_corr[abs(df_corr)>0.1].sort_values(ascending = False)

"""scaling the data"""
from sklearn import preprocessing 
  
""" MIN MAX SCALE""" 
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
  
# Scaled feature 
df_minmax = min_max_scaler.fit_transform(df) 
    
""" Standardisation """
  
Standardisation = preprocessing.StandardScaler() 
  
# Scaled feature 
df_standardation = Standardisation.fit_transform(df) 

# SibSp vs Survived
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiancés were ignored)
print(train_dataset[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))


#train_dataset['IsMinor'] = 0
#train_dataset.loc[(train_dataset['Age'] < 14) & ((train_dataset['Pclass'] == 1) | (train_dataset['Pclass'] == 2) ), 'IsMinor'] = 1

#test_dataset['IsMinor'] = 0
#test_dataset.loc[(test_dataset['Age'] < 14) & ((test_dataset['Pclass'] == 1 ) | (test_dataset['Pclass'] == 2 )), 'IsMinor'] = 1

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)







"""
"https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8"
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final.columns.values
"""
# Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# Iterating over all the common columns in train and test
for col in df.columns.values:
       # Encoding only categorical variables
       if df[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
           data=df[col].append(df[col])
           le.fit(data.values)
           df[col]=le.transform(df[col])
                        
"""

"""
# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)
# display the relative importance of each attribute
print(model.feature_importances_)
"""


"""
import pandas as pd
import numpy as np
data = pd.read_csv("D://Blogs//train.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
""" 








