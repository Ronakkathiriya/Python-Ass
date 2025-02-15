from sklearn.tree import DecisionTreeRegressor


import sys

print("Script Name:", sys.argv[0])

if len(sys.argv) > 1:
    print("Arguments Passed:", sys.argv[1:])
else:
    print("No arguments provided.")

import pandas as pd
df = pd.read_csv("data.csv")
df.drop_duplicates(inplace=True)
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
print(df.head())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

data = {
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Ronak', 'Krinay', 'Bhargav', 'Jay', 'Om'],
    'Age': [25, 30, 25, 30, 40],
    'Salary': [50000, 60000, 50000, 55000, 53750],
    'target': [25, 30, 40, 2, 1]  
}


df = pd.DataFrame(data)
X = df.drop(columns=["target", "Name"])  
y = df["target"]

selector = SelectKBest(score_func=f_classif, k=3)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)

import pandas as pd
df["Age_Salary_Product"] = df["Age"] * df["Salary"]
print(df)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {"Category": ["A", "B", "C", "A", "B"]}
df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=["Category"])

print(df_encoded)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv("data.csv")  
    print("Dataset loaded successfully.")
    print("Initial Data:\n", df.head())
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

df['target'] = (df['Salary'] > 55000).astype(int) 
print("\nData with target variable:\n", df.head())

df.dropna(inplace=True) 
print("\nData after handling missing values:\n", df.head())

if 'Name' in df.columns:
    df = pd.get_dummies(df, columns=['Name'], drop_first=True)
    print("\nData after encoding categorical variables:\n", df.head())

numeric_columns = df.select_dtypes(include=['number']).columns
if 'target' in numeric_columns:
    numeric_columns = numeric_columns.drop('target')  

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print("\nData after scaling numeric features:\n", df.head())

X = df.drop(columns=['target', 'Salary'])  
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nData split into training and testing sets:")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel training completed.")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV, LeaveOneOut

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=LeaveOneOut())
grid_search.fit(X_train, y_train)
print("\nBest Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy after Tuning:", accuracy)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")
df['target'] = (df['Salary'] > 55000).astype(int)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
df = pd.concat([df, df[df['target'] == 1]], ignore_index=True)
print("Dataset:")
print(df)
print("\nTarget Variable (y):")
print(y.value_counts())
X = df.drop(columns=['target', 'Salary']) 
y = df['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTraining Set (X_train):")
print(X_train)
print("\nTesting Set (X_test):")
print(X_test)
print("\nTraining Labels (y_train):")
print(y_train)
print("\nTesting Labels (y_test):")
print(y_test)
numeric_features = ['Age']  
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_features = ['Name']  
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, request, jsonify

df = pd.read_csv("data.csv")
print("Columns in the dataset:", df.columns)
df['target'] = (df['Salary'] > 50000).astype(int)
X = df.drop(columns=['target', 'ID', 'Name'])  
y = df['target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "trained_model.pkl")
print("Model saved successfully as 'trained_model.pkl'.")
loaded_model = joblib.load("trained_model.pkl")
print("Model loaded successfully.")
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])
    prediction = loaded_model.predict(input_df)
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data.csv")
df['target'] = (df['Salary'] > 55000).astype(int)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
df = pd.concat([df, df[df['target'] == 1]], ignore_index=True)
X = df.drop(columns=['target', 'Salary'])  
y = df['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numeric_features = ['Age']  
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_features = ['Name']  
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
joblib.dump(pipeline, "automated_model.pkl")
print("Model saved successfully.")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")
print("Columns in the dataset:", df.columns)
if 'Age' in df.columns:
    sns.histplot(df['Age'], kde=True)
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column 'Age' not found in the dataset.")
if 'Age' in df.columns and 'Salary' in df.columns:
    sns.scatterplot(x='Age', y='Salary', data=df)
    plt.title('Age vs Salary')
    plt.xlabel('Age')
    plt.ylabel('Salary')
    plt.show()
else:
    print("Columns 'Age' or 'Salary' not found in the dataset.")
numeric_columns = df.select_dtypes(include=['number']).columns
if len(numeric_columns) > 0:
    corr = df[numeric_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
else:
    print("No numeric columns found for correlation heatmap.")
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data.csv")
df['target'] = (df['Salary'] > 50000).astype(int)
X = df.drop(columns=['target', 'ID', 'Name'])  
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

from flask import Flask, request, jsonify
import joblib
import pandas as pd

loaded_model = joblib.load("trained_model.pkl")
print("Model loaded successfully.")
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])
    prediction = loaded_model.predict(input_df)
    return jsonify({'prediction': int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)