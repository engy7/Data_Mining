from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd


dataFrame = pd.read_csv('/Users/ENGY/Desktop/ASU/Senior 2/Data Mining/Dataset/weather_nominal.csv', index_col=0)
print(dataFrame)

dataObjList = dataFrame.select_dtypes(include=['object' , 'boolean']).columns.tolist()
print (dataObjList)

labelEncoder= LabelEncoder()
for col in dataObjList:
    print(str(col))
    labelEncoder.fit(dataFrame[str(col)])
    dataFrame[str(col)]=labelEncoder.transform(dataFrame[str(col)])
print (dataFrame)

kmeans= KMeans(n_clusters=2).fit(dataFrame.iloc[:,1:])
print(kmeans.labels_)
print(kmeans.cluster_centers_)