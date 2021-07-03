# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:49:35 2021

@author: ADMIN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes=pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/diabetes.csv")

#EDA
diabetes.columns
diabetes.head()
diabetes.info()
diabetes.describe()
diabetes.drop('Outcome', axis=1)

#checking missing values
diabetes.isna().sum()

#checking for outliers
plt.boxplot(diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']])

#Data Visualization
plt.figure(figsize = (12,10))
sns.heatmap(diabetes.corr(),annot = True)

is_diabetic = {0: 'No', 1: 'Yes'}
diabetes['is_diabetic'] = diabetes.Outcome.map(is_diabetic)

#histogram for each column
fig=diabetes.hist(figsize = (20,20), color='blue',alpha=0.7, rwidth=0.85)

#pie chart for non-diabetic vs non - diabetic
sns.set(style="whitegrid")
labels = ['Non-Diabetic', 'Diabetic']
sizes = diabetes['Outcome'].value_counts(sort = True)

colors = ["blue","red"]
explode = (0.05,0) 
 
plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)

plt.title('Non-Diabetic vs Diabetic')
plt.show()


#model making
X=diabetes.drop('Outcome', axis=1)
y=diabetes['Outcome']

#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#knn classification
from sklearn.neighbors import KNeighborsClassifier 
knnmodel=KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(X_train,y_train)
pred_y=knnmodel.predict(X_test)
actualy=y_test

#accuracy
knnmodel.score(X_test, y_test)
#accuracy=0.7207792207792207

#DecisionTre
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import tree
tree.plot_tree(dt)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt.score(X_test, y_test)
    
#accuracy=0.7467532467532467

#Random Forest
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)

#training the model
clf.fit(X_train, y_train)

#prediction
pred=clf.predict(X_test)
actual_y=y_test

#accuracy
from sklearn import metrics
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(actual_y, pred))
#accuracy = 0.8246753246753247

A = (clf.score(X_train, y_train),knnmodel.score(X_train,y_train),dt.score(X_train,y_train))
B = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
model_scores =(A)
model_compare = pd.DataFrame(model_scores)
plt.figure(figsize=(20, 18))
model_compare.T.plot.bar();
plt.xlabel("Model Predictions", size=15)
plt.ylabel("Percentage", size=15)
plt.title("Model Comparison for Prediction", size=18)
plt.legend(["RandomForest","KNN Classification","Decision Tree"]);


#Conclusion - Randomn Forest and Descion Tree have the higehest accuracy for the current data.