#Load the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
#Plot styling
import seaborn as sns; sns.set()  # for plot styling
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
df = pd.read_csv('C:/Users/121/Downloads/CLV.csv')

#using the elbow method to determine the optimal nbr of clusters  
from sklearn.cluster import KMeans 
wcss = [] 
for i in range(1,11): 
    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(df) 
    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)  
plt.title('elbow method')  
plt.xlabel('number of clusters') 
plt.ylabel('wcss') 
plt.show() 


##Fitting kmeans to the dataset with k=4
km4=KMeans(n_clusters=4,init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km4.fit_predict(df)

plt.scatter(df[y_means==0,0],df[y_means==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(df[y_means==1,0],df[y_means==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(df[y_means==2,0],df[y_means==2,1],s=50, c='green',label='Cluster3')
plt.scatter(df[y_means==3,0],df[y_means==3,1],s=50, c='cyan',label='Cluster4')

plt.scatter(km4.cluster_centers_[:,0], km4.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site')
plt.legend()
plt.show()    