import pickle

import pandas as pd

# Read the dataset
data = pd.read_csv("student_performance.csv")

print("Original Dataset Shape:", data.shape)

# 1. Remove duplicate records
data = data.drop_duplicates()

# 2. Replace invalid attendance values with missing values
data.loc[
    (data["Attendance_Percentage"] < 0) |
    (data["Attendance_Percentage"] > 100),
    "Attendance_Percentage"
] = None

# 3. Fill missing numerical values with median
data["Attendance_Percentage"] = data["Attendance_Percentage"].fillna(
    data["Attendance_Percentage"].median()
)

data["Assignment_Score"] = data["Assignment_Score"].fillna(
    data["Assignment_Score"].median()
)

data["Study_Hours_Per_Day"] = data["Study_Hours_Per_Day"].fillna(
    data["Study_Hours_Per_Day"].median()
)
 # 4. Fill missing engagement levels with the most common value
data["Engagement_Level"] = data["Engagement_Level"].fillna(
    data["Engagement_Level"].mode()[0]
)

# Remove repeated Student IDs
data = data.drop_duplicates(subset="Student_ID", keep="first")

# 5. Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# 6. Check duplicates after cleaning
print("\nDuplicate Records After Cleaning:")
print(data.duplicated().sum())

# 7. Display cleaned dataset information
print("\nCleaned Dataset Shape:", data.shape)

# 8. Save cleaned dataset
data.to_csv("cleaned_student_performance.csv", index=False)

print("\nDataset cleaned and saved successfully!")

X = data[[
    "Attendance_Percentage",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Semester_Score",
    "Study_Hours_Per_Day",
    "Engagement_Level"
]].copy()    

# 10. Select target column
y = data["Performance_Band"].copy()

print("\nInput Features:")
print(X.head())

print("\nTarget Categories:")
print(y.value_counts())

print("\nTotal Input Records:", len(X))
print("Total Target Records:", len(y))

from sklearn.preprocessing import LabelEncoder

# Convert Engagement_Level into numbers
encoder = LabelEncoder()

X["Engagement_Level"] = encoder.fit_transform(
    X["Engagement_Level"]
)

print("\nEncoded Engagement Levels:")
print(encoder.classes_)

print("\nUpdated Input Features:")
print(X.head())


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create Decision Tree
dt = DecisionTreeClassifier(random_state=42)

# Parameters to test

parameters = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [4, 6, 8, 10, 12, 15],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "class_weight": [None, "balanced"]
}


# Grid Search with 5-fold cross-validation
grid = GridSearchCV(
    estimator=dt,
    param_grid=parameters,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

# Train and find the best settings
grid.fit(X_train, y_train)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Model 1: Previous best Decision Tree
model1 = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model1.fit(X_train, y_train)

# Model 2: Best model selected by GridSearchCV
model2 = grid.best_estimator_

# Compare both models using the same test data
models = {
    "Previous Decision Tree": model1,
    "GridSearch Decision Tree": model2
}

for name, model in models.items():

    predictions = model.predict(X_test)

    print("\n------------------------------")
    print(name)
    print("------------------------------")

    print("Accuracy:",
          round(accuracy_score(y_test, predictions) * 100, 2), "%")

    print("Macro F1:",
          round(f1_score(y_test, predictions, average="macro"), 3))

    print("Weighted F1:",
          round(f1_score(y_test, predictions, average="weighted"), 3))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))


print("Best Parameters:", grid.best_params_)
print("Best CV Macro F1:", grid.best_score_)

# Select the best Decision Tree
model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# Predict test data
y_pred = model.predict(X_test)

import joblib

joblib.dump(
    {
        "model": model1,
        "encoder": encoder
    },
    "student_model.pkl"
)

print("Final model saved successfully!")
print("Features:", list(model.feature_names_in_))
print("Final model saved successfully!")

with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)