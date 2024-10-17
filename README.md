# EX 5 Implementation of Logistic Regression Using Gradient Descent
## DATE:
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialization: Sets the learning rate and the number of iterations.
2.Sigmoid Function: Defines the logistic function for outputting probabilities.
3.Fit Method: Adjusts weights using gradient descent based on the difference between predicted and actual values.
4.Predict Method: Returns class predictions based on the learned weights. 


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: pranav k
RegisterNumber:2305001026  
*/
```
```
import pandas as pd
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

x=data1.iloc[:,:-1]
x

y=data1.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred,x_test

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy_score)
print("Confusion matrix:\n",confusion)
print("\nClassification_report:\n",cr)

from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion,display_labels=[True,False])
cm_display.plot()

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification_Report:\n",cr)

X=data1.iloc[:,:-1]
Y=data1["status"]

theta=np.random.randn(X.shape[1])
y=y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,x,y)
```

## Output:
![WhatsApp Image 2024-10-04 at 21 13 37_34158a07](https://github.com/user-attachments/assets/ea27a170-2620-4507-b63f-2a60cca6530f)
![WhatsApp Image 2024-10-04 at 21 14 10_df3d2f10](https://github.com/user-attachments/assets/8e8186c6-fcc0-4084-b309-43293c78c2a1)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

