# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Prepare the dataset using python lists
# (Using sample values for hours studied vs. examination scores)
hours_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores_list = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

# Separate the input features (x) and target values (y)
x = [[hours] for hours in hours_list]
y = scores_list

# Create a Linear Regression model
model = LinearRegression()

# Train the model using fit()
model.fit(x, y)

# Allow the user to enter the number of hours studied using input()
user_hours = float(input("Enter the number of hours studied:"))

# Use predict() to estimate the examination score
predicted_score = model.predict([[user_hours]])[0]

# Display the prediction rounded to two decimal places
print(f"Estimated examination score: {predicted_score:.2f}")


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
scores = np.array([30, 40, 45, 55, 65, 70, 80, 85, 90, 95])

# Fit model for the regression line
model = LinearRegression()
model.fit(hours, scores)
hours_pred = np.linspace(1, 10, 100).reshape(-1, 1)
scores_pred = model.predict(hours_pred)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(hours, scores, color='blue', label='Actual Data (Students)')
plt.plot(hours_pred, scores_pred, color='red', linewidth=2, label='Regression Line (Trend)')
plt.title('Relationship Between Study Hours and Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Examination Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

