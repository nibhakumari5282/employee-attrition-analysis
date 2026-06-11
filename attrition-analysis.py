import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(data.head())
print(data.info())
data['Attrition'] = data['Attrition'].map({'Yes':1,'No':0})
sns.countplot(x='Attrition', data=data)
plt.title("Employee Attrition Count")
plt.show()
sns.boxplot(x='Attrition', y='Age', data=data)
plt.title("Age vs Attrition")
plt.show()
sns.boxplot(x='Attrition', y='MonthlyIncome', data=data)
plt.title("Salary vs Attrition")
plt.show()
sns.countplot(x='Department', hue='Attrition', data=data)
plt.xticks(rotation=45)
plt.show()
data = pd.get_dummies(data, drop_first=True)
X = data.drop('Attrition', axis=1)
y = data['Attrition']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)

print("Model Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.show()
