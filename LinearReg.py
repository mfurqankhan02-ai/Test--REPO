import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#using Panda Library to retrieve Salary dataset
mydata = pd.read_csv(r"C:\Users\Furqan Khan\Desktop\Salary_Data.csv")

#assigning x and y variables
x = mydata[["YearsExperience"]]
y =  mydata["Salary"]

#split test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 42)

#applying Linear Regression
model = LinearRegression()

#model.fit adjust Weight and Bias
model.fit(x_train, y_train)

#Printing Weight and Bias
print("Learned W: ", model.coef_[0])
print("Learned B: ", model.intercept_)

#Predicting already tested dataset to check its accuracy
y_predict = model.predict(x_test)

#looping to know that Actual and Predicted values matches or not
for actual, predicted in zip(y_test, y_predict):
    print(f"Actual: {actual}, Predicted: {round(predicted,2)} ")

#Visualizing the Linear Regression
plt.plot(np.arange(len(y_test)), y_test.values, label="Actual", marker="o")
plt.plot(np.arange(len(y_test)), y_predict, label="Predicted", marker="x")
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Target Value")
plt.legend()
plt.show()

#Predicting Salary of random Years of Experience
new_data = [[22]]
new_pred= model.predict(new_data)

#Printing the predicted value
print(f"Prediction at the experience of {new_data} ", new_pred[0])