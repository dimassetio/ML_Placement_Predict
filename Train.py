import csv
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Load data from CSV file
data = []
with open('Placement_Data.csv', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if there is one
    for row in reader:
        data.append(row)

# Preprocess your data
features = np.array([[float(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])] for entry in data])
labels = np.array([1 if entry[4] == "Placed" else 0 for entry in data])

# Split data into training and testing sets (80% training, 20% testing)
split_index = int(0.8 * len(features))
x_train, x_test = features[:split_index], features[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

model.save('placement_model.h5')

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).flatten()

# Print y_test and y_pred values side by side
print("y_test vs y_pred:")
for i in range(len(y_test)):
    print(f"y_test[{i}]: {y_test[i]}, y_pred[{i}]: {y_pred[i]}")

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
