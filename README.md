# EX 5 Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Sets the learning rate and the number of iterations.
 
2. Defines the logistic function for outputting probabilities.
 
3. Adjusts weights using gradient descent based on the difference between predicted and actual values.
  
4. Returns class predictions based on the learned weights. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: pranav k
RegisterNumber: 2305001026
```
```
import pandas as pd
import numpy as np
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
X=data1.iloc[:,:-1]
Y=data1["status"]
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
```

## Output:
![image](https://github.com/user-attachments/assets/ba401cd4-b337-4132-b271-109cb0b328d4)
![image](https://github.com/user-attachments/assets/cc564fb8-3801-4e8a-a433-94edf8dfd84c)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

# EX 5 Implementation of Logistic Regression Using Gradient Descent
## DATE:
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
```

## Output:



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

