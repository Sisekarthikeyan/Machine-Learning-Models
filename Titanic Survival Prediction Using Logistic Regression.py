
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
x=pd.read_csv("train_titanic.csv")
y=x.pop("Survived")
#Selecting Numerical values alone
x=x.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x['Sex']=labelencoder.fit_transform(x['Sex'])
#passenger ID 6 has actually no age but now we applied with mean value
x['Age'].fillna(x.Age.mean(), inplace=True)
model = LogisticRegression(random_state = 0)
model.fit(x,y)
print ('Accuracy is :' , accuracy_score(y,model.predict(x)))
test=pd.read_csv ('test_titanic.csv')
test=test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test['Sex']=labelencoder.fit_transform(test['Sex'])
test["Age"].fillna(test.Age.mean(), inplace=True)
test["Fare"].fillna(test.Fare.mean(), inplace=True)
y_pred = model.predict(test)


# In[ ]:


submission = pd.DataFrame({
"PassengerId":test[
"PassengerId"], "Survived" : y_pred})
submission.to_csv('Titanic KNN.csv', index=False)
submission.head()

