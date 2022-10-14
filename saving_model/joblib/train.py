from unittest import result
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib # Used to save and load the model

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'predi', 'age', 'class']

dataframe = pd.read_csv(url, names=names)

#print(dataframe)

# seperate iondependant and target variable
array = dataframe.values

x = array[:, 0:8]
y = array[:,8]

# train-test split
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=101)

# fit the model
model = LogisticRegression()
model.fit(x_train, y_train)

# accuracy
result = model.score(x_test, y_test)
print(result)

# saving the model
# naming convention : dataset_accuracy.extention
# extention for model file is pickle
filename = "diabetese_79.pkl"
joblib.dump(model, filename)



