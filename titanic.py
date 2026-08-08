import os
import pandas as pd
os.getcwd()
os.chdir("C://Users//rshan")

# Set working directory (change path as per your system)
# Load dataset
titanic = pd.read_csv("titanic.csv")

# Check data types
print(titanic.dtypes)

# Drop unnecessary columns
finaldata = titanic.drop(
    ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"],
    axis=1
)

# Fill missing Age with median
print(finaldata["Age"].median())
finaldata["Age"].fillna(finaldata["Age"].median(), inplace=True)

# Check Embarked mode
print(finaldata["Embarked"].value_counts())

# Fill missing Embarked with mode ('S')
finaldata["Embarked"].fillna("S", inplace=True)

# Check final data types
print(finaldata.dtypes)

# Separate target and features
y = finaldata["Survived"]
x = finaldata.drop(["Survived"], axis=1)

# Convert categorical variables into dummy variables
x = pd.get_dummies(x)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Build Logistic Regression model
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression(max_iter=1000)
lm.fit(xtrain, ytrain)

# Predictions
predicted_value = lm.predict(xtest)

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(ytest, predicted_value))

# Classification Report
from sklearn.metrics import classification_report

print(classification_report(ytest, predicted_value))

# Accuracy Score
from sklearn.metrics import accuracy_score

a1 = accuracy_score(ytest, predicted_value)
print("Accuracy:", a1)



