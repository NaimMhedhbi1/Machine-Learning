import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
df = pd.read_excel('C://Users/121//.spyder-py3//EDA projects//121_field_track_logistic_regression.xlsx')
X = df[[' qst sans num de TEL ','score']]
y = df['score_labeled']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
Accuracy = metrics.accuracy_score(y_test, y_pred)

df2 = pd.read_excel('C:/Users/121/.spyder-py3/EDA projects/121_field_track_logistic_regression_test.xlsx')

y_pred = logistic_regression.predict(df2)

