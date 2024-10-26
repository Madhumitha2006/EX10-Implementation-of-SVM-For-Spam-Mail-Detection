# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Collect and preprocess the email dataset, converting text data into numerical features, often using techniques like TF-IDF.

2.Data Splitting: Split the dataset into training and testing sets to train and evaluate the model.

3.Model Training: Use a Support Vector Machine (SVM) classifier on the training set to learn to distinguish between spam and non-spam emails.

4.Model Evaluation: Test the trained model on the testing set and assess its accuracy in predicting spam versus non-spam emails.

## Program:
```
Program to implement the SVM For Spam Mail Detection.
Developed by: Madhu Mitha V
RegisterNumber: 2305002013

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))

def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

new_message="Free prixze money winner"
result=predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")  
```

## Output:
![image](https://github.com/user-attachments/assets/d9cb46b5-6bf5-4afe-aa68-65b7717c012e)

![image](https://github.com/user-attachments/assets/127777ea-f6c3-4102-b5ce-eedca2582731)
![image](https://github.com/user-attachments/assets/35c900cf-2f87-4d56-a9d0-911944f58a0d)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
