#test
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#Create a function within many Machine Learning Models
def models_predict(X_train,Y_train):
  
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest



x=pd.read_csv("train_titanic.csv")
y=x.pop("Survived")
#Selecting Numerical values alone
x=x.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x['Sex']=labelencoder.fit_transform(x['Sex'])
#passenger ID 6 has actually no age but now we applied with mean value
x['Age'].fillna(x.Age.mean(), inplace=True)

#Calling Function
models_predict(x,y)

model = LogisticRegression(random_state = 0)
model.fit(x,y)
print ('Accuracy is :' , accuracy_score(y,model.predict(x)))
test=pd.read_csv ('test_titanic.csv')
test=test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test['Sex']=labelencoder.fit_transform(test['Sex'])
test["Age"].fillna(test.Age.mean(), inplace=True)
test["Fare"].fillna(test.Fare.mean(), inplace=True)
y_pred = model.predict(test)


# This method suits kaggle submission

submission = pd.DataFrame({
"PassengerId":test[
"PassengerId"], "Survived" : y_pred})
submission.to_csv('Titanic KNN.csv', index=False)
submission.head()

