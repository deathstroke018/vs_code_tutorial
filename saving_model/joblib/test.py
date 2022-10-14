from unittest import result
import joblib

# load the model
model = joblib.load('diabetese_79.pkl')

result = model.predict([[1,1,1,1,1,1,1,1]]) # 2D because the function expects 2D

if result[0]==1:
    print("\ndiabetic")
else:
    print('\nnot diabetic')
