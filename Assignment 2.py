import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


## Function for outliers detection ##
def OutliersDetector(actualData, predictedData):
    outliers=[]
    std=np.abs(np.std(np.subtract(actualData,predictedData)))
    for i,j in zip(actualData,predictedData):
        std_residual=(i-j)/std
        if np.abs(std_residual)>2:
            outliers.append(i)
    return outliers


dataFrame = pd.read_csv('/Users/ENGY/Desktop/ASU/Senior 2/Data Mining/Dataset/house_price.csv')
print(dataFrame)

## Plotting Data ##

x1 = dataFrame['area'].values.tolist()
x2= dataFrame['rooms'].values.tolist()
y=dataFrame['price'].values.tolist()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='r', marker='o')
ax.set_xlabel('Area')
ax.set_ylabel('Rooms')
ax.set_zlabel('Price')
plt.show()


## Representing the attributes as a list of lists ##
x = dataFrame.iloc[0:,0:2]
x= x.values.tolist()

## Feeding the model ##
model = LinearRegression().fit(x, y)
y_predict= model.predict(x)

## Regression Outliers ##
outliers=OutliersDetector(y,y_predict)
print("Here are the outliers: ",outliers)

## Testing ##
x_test=[[2567,5],[1200,2],[852,5],[1852,2],[1203,3]]
y_test=model.predict(x_test)
print(y_test)

