import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the  classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Use this line to get you started on adding a validation dataset
# X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=10, random_state=0)

neural_net_model = MLPClassifier( hidden_layer_sizes=(40),random_state=42,tol=0.005)

neural_net_model.fit(X_train, y_train)
# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Training set size: {len(y_train)}")
print(f"Layer sizes: {layer_size_str}")


# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1


# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test)*100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")


cm = confusion_matrix(y_test, y_pred)

# Get label to letter mapping (0 → A, 1 → B, ..., 24 → Z minus J/Z if excluded)
import string
labels = list(string.ascii_uppercase)
labels.remove('J')  # J is not included in the dataset
labels.remove('Z')  # Z is also excluded (dynamic gesture)

# Find misidentification counts
mis_id_counts = {}

for true_idx in range(len(cm)):
    for pred_idx in range(len(cm)):
        if true_idx != pred_idx:
            mis_id_counts[(true_idx, pred_idx)] = cm[true_idx][pred_idx]

# Sort most common misidentifications
top_mis = sorted(mis_id_counts.items(), key=lambda x: x[1], reverse=True)[:3]

# For better formatting
print("Confusion Matrix:")
for i, row in enumerate(cm):
    print(f"Class {i}:", " ".join(f"{num:5d}" for num in row))

print("Most commonly confused letter pairs:")
for (true_idx, pred_idx), count in top_mis:
    print(f"{labels[true_idx]} → {labels[pred_idx]}: {count} times")
