#Import the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#Reading the csv files with pandas
train_data = pd.read_csv("Predict Career Longevity for NBA Rookies/Train_data.csv")
print(train_data.head())

test_data = pd.read_csv("Predict Career Longevity for NBA Rookies/Test_data.csv")
print(test_data.head())

#For the Games Played column we want to take these values as a integer so we print the type of this column for train_data and test_data
print(train_data["GP"].info())
print(test_data["GP"].info())

#For the two database GP columns have the type of float
# We're converting float to integer 
train_data["GP"] = train_data["GP"].astype("int")
test_data["GP"] = test_data["GP"].astype("int")

print(train_data.head())
print(test_data.head())

#Insert the "Target" column to test_data for the predictions
test_data.insert(19,"Target" , int)

#Splitting the data for training and testing 
X_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]

#we're using Random Forest Classifier for classify our data 
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
#make prediction for the test data
preds = rf.predict(X_test)

y_test = pd.Series(preds)

#create a csv file for the predictions
prediction = pd.Series(y_test, name = "prediction")
prediction.to_csv("submission.csv", index= False)




