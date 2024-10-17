# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## DATE:
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by:Poovarasu V 
RegisterNumber:2305002017  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,pred)
print(f'mean square error (mse): {mse}')
```

## Output:

![Screenshot (25)](https://github.com/user-attachments/assets/4315b362-02d0-4252-9cf6-07fec076bd76)
![Screenshot (23)](https://github.com/user-attachments/assets/10bbb60e-d467-42b5-ae45-59711dbf83e2)

![Screenshot (24)](https://github.com/user-attachments/assets/9ba4dd9e-a28e-4781-a092-6fc807ae43f6)







## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
