import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler


### getting dataframe values from csv file ###
dataFrame = pd.read_csv('/Users/ENGY/Desktop/ASU/Senior 2/Data Mining/Lab 2/Handson_Data.csv',delimiter='\t')
##C:\Users\ENGY\Desktop\ASU\Senior 2\Data Mining\Lab 2
print(dataFrame)

### Data Preprocessing ###

#################################### Step 1: Filling empty tuples ##################################

### Determining which attributes are numeric and which are noiminal ###
dataNumList = dataFrame.select_dtypes(exclude= ['object']).columns.tolist()
dataObjList = dataFrame.select_dtypes(include= ['object']).columns.tolist()

print(dataNumList)
print(dataObjList)

### Separating numerical attributes -with their data- and nominal ones ###
dataNum = dataFrame[dataNumList]
dataObj = dataFrame[dataObjList]

print(dataNum)
print(dataObj)

### Filling empty tuples of both types ###
impNum = SimpleImputer(missing_values=np.nan,strategy='mean')
impNum.fit(dataNum)
dataNum=impNum.transform(dataNum)

impObj=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
impObj.fit_transform(dataObj)

dataFrame[dataNumList]=dataNum
dataFrame[dataObjList]=dataObj
print(dataFrame)

#################################### Data Transformation ######################################
############### Question 1: Data Discretization ##################

discretizer = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
dataNumDiscrete = discretizer.fit_transform(dataNum)

dataFrame[dataNumList]=dataNumDiscrete
dataFrame[dataObjList]=dataObj
print(dataFrame)

############### Question 2: Data Normalization ######################

normalizer=MinMaxScaler(feature_range=(0,1))
dataNumNormalized = normalizer.fit_transform(dataNum)

dataFrame[dataNumList]=dataNumNormalized
dataFrame[dataObjList]=dataObj
print(dataFrame)