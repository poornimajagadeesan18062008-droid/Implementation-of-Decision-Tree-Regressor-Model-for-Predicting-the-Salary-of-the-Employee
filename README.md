# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and create the dataset containing employee levels and salaries.

2.Separate the input feature (Level) and output value (Salary) from the dataset.

3.Create and train the Decision Tree Regressor model using the training data.

4.Predict the salary values using the trained regression model for existing and new employee levels.

5.Display the predicted results and plot the graph showing actual salaries and decision tree regression predictions.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Poornima J
RegisterNumber:  212225040303
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}
df = pd.DataFrame(data)
X = df[['Level']]     
y = df['Salary']      
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01) 
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

```

## Output:
<img width="995" height="72" alt="589951942-16a84d7a-dc79-455f-bb5a-8993fa6aded5" src="https://github.com/user-attachments/assets/72a9f972-0bf0-43e2-883e-3d3059feb84a" />
<img width="995" height="566" alt="589952036-af6275fc-89d2-40ac-a988-dcd49bf1c3a4" src="https://github.com/user-attachments/assets/d01b248f-247b-43b7-8878-84fcb59fe60c" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
