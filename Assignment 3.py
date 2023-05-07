import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

dataFrame = pd.read_csv('/Users/ENGY/Desktop/ASU/Senior 2/Data Mining/Dataset/bank_dataset.csv')
print(dataFrame)

## Label Encoding ##
labelEncoder = LabelEncoder()
labelEncoder.fit(dataFrame["sex"])
dataFrame["sex"] = labelEncoder.transform(dataFrame["sex"])
labelEncoder.fit(dataFrame["region"])
dataFrame["region"] = labelEncoder.transform(dataFrame["region"])
labelEncoder.fit(dataFrame["married"])
labelEncoder.fit(dataFrame["car"])
dataFrame["car"] = labelEncoder.transform(dataFrame["car"])
labelEncoder.fit(dataFrame["married"])
dataFrame["married"] = labelEncoder.transform(dataFrame["married"])
labelEncoder.fit(dataFrame["save_act"])
dataFrame["save_act"] = labelEncoder.transform(dataFrame["save_act"])
labelEncoder.fit(dataFrame["current_act"])
dataFrame["current_act"] = labelEncoder.transform(dataFrame["current_act"])

## DB Scan before Normalization ##
dbscan = DBSCAN(eps=1.2 , min_samples=3).fit(dataFrame.iloc[:,1:])
print (dbscan.labels_)

## Normalization ##
normalizer = MinMaxScaler(feature_range=(0,1))
normalized_data = normalizer.fit_transform(dataFrame.iloc[:,1:])
print(normalized_data)

## DB Scan after Normalization ##
dbscan = DBSCAN(eps=1.2 , min_samples=3).fit(normalized_data)
print (dbscan.labels_)

## Hierarchial Clustering before Normalization ##
Agglomerative = AgglomerativeClustering().fit(dataFrame.iloc[:,1:])
print (Agglomerative.labels_)

## Hierarchial Clustering after Normalization ##
Agglomerative = AgglomerativeClustering().fit(normalized_data)
print(Agglomerative.labels_)
