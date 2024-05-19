import numpy as np
import csv
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Load the model
model = tf.keras.models.load_model('placement_model.h5')

# Example new data to predict
# Make sure the new data has the same number of features as the training data
dtest = []
with open('Test.csv', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            dtest.append(row)
# dtest = dtest[1:]
y_test = np.array([1 if entry[4] == "Placed" else 0 for entry in dtest])
x_test = np.array([[float(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])] for entry in dtest])


# Make predictions
y_pred = model.predict(x_test)

# Convert predictions to binary labels (0 or 1)
y_pred = (y_pred > 0.5).astype(int)
print("Real | vs | Prediction:")
for i in range(len(y_test)):
    print(f"{y_test[i]}    | vs | {y_pred[i]}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

# Print confusion matrix and metrics
print('Confusion Matrix:')
print(conf_matrix)
print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

# Classification Report
report = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"])
print(report)

