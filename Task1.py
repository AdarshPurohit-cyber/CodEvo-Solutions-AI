import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)

# Split data into features and target
X, y = mnist['data'], mnist['target'].astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', random_state=42)

# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
import joblib

# Save the trained model to a file
joblib.dump(model, 'logistic_regression_mnist_model.pkl')
report = f"""
MNIST Image Classification using Logistic Regression

Model: Logistic Regression
Dataset: MNIST (Handwritten Digits)

Preprocessing:
- Images were flattened from 28x28 to 784 features.
- Pixel values were normalized using StandardScaler.

Model Training:
- The model was trained on 80% of the data.
- The training process took approximately 1000 iterations to converge.

Results:
- Accuracy on the test set: {accuracy:.4f}

Conclusion:
- Logistic Regression achieved an accuracy
