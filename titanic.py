import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women) / len(women)

men = train_data[train_data.Sex=="male"].Survived
rate_men = sum(men) / len(men)

target = train_data["Survived"]
features = ["Pclass","Sex","SibSp","Parch"]
x_train = pd.get_dummies(train_data[features])#turn strings into integers 
x_test = pd.get_dummies(test_data[features])#turn strings into integers 

model = RandomForestClassifier(n_estimators = 100 , max_depth = 5 , random_state = 1)
model.fit(x_train,target)
predictions = model.predict(x_test)

output = pd.DataFrame({'PassengerID ': test_data.PassengerId, 'Survived' : predictions})
output.to_csv('my_submissions.csv',index = False)

print(output.head())