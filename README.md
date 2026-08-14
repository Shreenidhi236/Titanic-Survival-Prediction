Titanic Survival Prediction

A machine learning classification project that predicts passenger survival using the Titanic dataset.

📌 Project Overview

This project uses passenger information from the Titanic dataset to build a Logistic Regression model for predicting whether a passenger survived.

The project demonstrates data preprocessing, handling missing values, categorical variable encoding, model training, prediction, and evaluation.

🛠️ Technologies Used

Python
Pandas
Scikit-learn
Spyder
📊 Dataset

The dataset contains passenger information such as:

Passenger class
Sex
Age
Fare
Embarked location
Survival status

The target variable is Survived.

🔍 Data Preprocessing

The following preprocessing steps were performed:

Removed unnecessary columns such as PassengerId, Name, SibSp, Parch, Ticket, and Cabin.
Filled missing Age values using the median age.
Filled missing Embarked values using the mode (S).
Converted categorical variables into dummy variables using pd.get_dummies().

🤖 Machine Learning Model

A Logistic Regression model was trained using:

70% of the data for training
30% of the data for testing
random_state = 42

The model was used to predict passenger survival on the test dataset.

📈 Model Evaluation

The model was evaluated using:

Confusion Matrix
Classification Report
Accuracy Score
🚀 How to Run
Install Python.
Install the required libraries:
pip install pandas scikit-learn
Keep Titanic.csv in the same folder as titanic.py.
Run:
python titanic.py

📁 Project Structure

Titanic-Survival-Prediction/
│
├── README.md
├── titanic.py
└── Titanic.csv
🎯 Learning Outcomes

Through this project, I gained practical experience in:

Data preprocessing
Handling missing data
Categorical variable encoding
Logistic Regression
Train-test splitting
Model evaluation
Python-based machine learning
